from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any, Callable

from .db import connection
from .feature_tuning import load_complete_race_ids
from .hashed_feature_dataset import HashedRaceDataset, load_hashed_dataset
from .listwise.model import evaluate_range, fit_scaler, train_listwise_model


MODEL = "genetic_listwise_island_v1"
GENOME_VERSION = 1
TARGETS = ("winner", "top3_pl")


@dataclass(frozen=True)
class Genome:
    target: str
    alpha: float
    learning_rate: float
    epochs: int
    ev_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "genome_version": GENOME_VERSION,
            "target": self.target,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "ev_threshold": self.ev_threshold,
        }


def genome_from_dict(value: dict[str, Any]) -> Genome:
    target = str(value.get("target") or "")
    if target not in TARGETS:
        raise ValueError(f"unsupported genetic target: {target}")
    alpha = float(value["alpha"])
    learning_rate = float(value["learning_rate"])
    epochs = int(value["epochs"])
    ev_threshold = float(value["ev_threshold"])
    if not 1e-7 <= alpha <= 1e-2:
        raise ValueError("genetic alpha must be between 1e-7 and 1e-2")
    if not 0.001 <= learning_rate <= 0.2:
        raise ValueError("genetic learning rate must be between 0.001 and 0.2")
    if not 1 <= epochs <= 3:
        raise ValueError("genetic epochs must be between 1 and 3")
    if not 1.0 <= ev_threshold <= 3.0:
        raise ValueError("genetic EV threshold must be between 1.0 and 3.0")
    return Genome(target, alpha, learning_rate, epochs, ev_threshold)


def random_genome(rng: random.Random) -> Genome:
    return Genome(
        target=rng.choice(TARGETS),
        alpha=10 ** rng.uniform(-6.5, -2.5),
        learning_rate=10 ** rng.uniform(math.log10(0.004), math.log10(0.08)),
        epochs=rng.choice((1, 1, 1, 2)),
        ev_threshold=rng.uniform(1.05, 1.8),
    )


def crossover(left: Genome, right: Genome, rng: random.Random) -> Genome:
    return Genome(
        target=rng.choice((left.target, right.target)),
        alpha=math.sqrt(left.alpha * right.alpha),
        learning_rate=math.sqrt(left.learning_rate * right.learning_rate),
        epochs=rng.choice((left.epochs, right.epochs)),
        ev_threshold=(left.ev_threshold + right.ev_threshold) / 2.0,
    )


def mutate(genome: Genome, rng: random.Random, rate: float = 0.35) -> Genome:
    target = genome.target
    alpha = genome.alpha
    learning_rate = genome.learning_rate
    epochs = genome.epochs
    ev_threshold = genome.ev_threshold
    if rng.random() < rate:
        target = rng.choice(TARGETS)
    if rng.random() < rate:
        alpha = min(1e-2, max(1e-7, alpha * math.exp(rng.gauss(0.0, 0.8))))
    if rng.random() < rate:
        learning_rate = min(
            0.2,
            max(0.001, learning_rate * math.exp(rng.gauss(0.0, 0.35))),
        )
    if rng.random() < rate:
        epochs = rng.choice((1, 1, 2, 3))
    if rng.random() < rate:
        ev_threshold = min(3.0, max(1.0, ev_threshold + rng.gauss(0.0, 0.12)))
    return Genome(target, alpha, learning_rate, epochs, ev_threshold)


def speculative_fitness(metrics: dict[str, Any], genome: Genome) -> float:
    """Rank candidates on a selection window; never use this for promotion."""
    ranking = float(metrics["ranking_log_loss"])
    entry = float(metrics["entry_log_loss"])
    winner = float(metrics["winner_top1_accuracy"])
    top5 = float(metrics["trifecta_top5_hit_rate"])
    complexity = 0.002 * max(0, genome.epochs - 1)
    return -ranking - 0.35 * entry + 0.35 * winner + 0.20 * top5 - complexity


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _fitness_distribution(ranked: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["fitness"]) for row in ranked)
    return {
        "min_fitness": values[0],
        "q1_fitness": _percentile(values, 0.25),
        "median_fitness": _percentile(values, 0.5),
        "q3_fitness": _percentile(values, 0.75),
        "max_fitness": values[-1],
        "std_fitness": statistics.pstdev(values),
    }


def evolve_island(
    *,
    rng: random.Random,
    population_size: int,
    local_generations: int,
    elite_count: int,
    evaluator: Callable[[Genome], dict[str, Any]],
    immigrants: list[Genome] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    elite_count = min(max(1, elite_count), population_size // 2)
    population = list(immigrants or [])[:elite_count]
    population.extend(random_genome(rng) for _ in range(population_size - len(population)))
    history: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    for local_generation in range(local_generations):
        unique = {json.dumps(row.as_dict(), sort_keys=True): row for row in population}
        while len(unique) < population_size:
            genome = random_genome(rng)
            unique[json.dumps(genome.as_dict(), sort_keys=True)] = genome
        genomes = list(unique.values())[:population_size]
        with ThreadPoolExecutor(max_workers=min(2, population_size)) as executor:
            metrics_rows = list(executor.map(evaluator, genomes))
        ranked = sorted(
            (
                {
                    "genome": genome.as_dict(),
                    "metrics": metrics,
                    "fitness": speculative_fitness(metrics, genome),
                }
                for genome, metrics in zip(genomes, metrics_rows)
            ),
            key=lambda row: float(row["fitness"]),
            reverse=True,
        )
        history.append({
            "local_generation": local_generation,
            "best_fitness": ranked[0]["fitness"],
            "best_genome": ranked[0]["genome"],
            **_fitness_distribution(ranked),
        })
        elites = [genome_from_dict(row["genome"]) for row in ranked[:elite_count]]
        population = list(elites)
        while len(population) < population_size:
            parents = (
                rng.sample(elites, 2)
                if len(elites) > 1
                else [elites[0], elites[0]]
            )
            population.append(mutate(crossover(*parents, rng), rng))
    return ranked[:elite_count], history


def _slice_dataset(
    dataset: HashedRaceDataset,
    *,
    train_races: int,
    validation_races: int,
) -> tuple[HashedRaceDataset, int]:
    total = train_races + validation_races
    if dataset.race_count < total:
        raise ValueError(f"genetic cache has {dataset.race_count} races; {total} required")
    start = dataset.race_count - total
    sliced = HashedRaceDataset(
        matrix=dataset.matrix[dataset.row_slice(start, dataset.race_count)].tocsr(),
        race_keys=dataset.race_keys[start:],
        ranks=dataset.ranks[start:],
        n_features=dataset.n_features,
        drop_feature_groups=dataset.drop_feature_groups,
        hasher_settings=dataset.hasher_settings,
        feature_schema_version=dataset.feature_schema_version,
    )
    return sliced, train_races


def run(args: argparse.Namespace) -> dict[str, Any]:
    previous = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = args.evaluation_date
    try:
        with connection(args.db) as conn:
            race_keys = load_complete_race_ids(conn)
    finally:
        if previous is None:
            os.environ.pop("BOATRACE_EVAL_MAX_RACE_DATE", None)
        else:
            os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] = previous
    dataset = load_hashed_dataset(
        args.cache_prefix,
        race_keys=race_keys,
        n_features=args.n_features,
        drop_feature_groups=("base_pastlog",),
    )
    if dataset is None:
        raise ValueError("daily genetic feature cache is absent, stale, or incompatible")
    dataset, train_end = _slice_dataset(
        dataset,
        train_races=args.train_races,
        validation_races=args.validation_races,
    )
    scaler = fit_scaler(dataset, race_end=train_end, batch_rows=args.batch_races * 6)

    def evaluate(genome: Genome) -> dict[str, Any]:
        model, history = train_listwise_model(
            dataset,
            train_race_end=train_end,
            target=genome.target,
            alpha=genome.alpha,
            learning_rate=genome.learning_rate,
            epochs=genome.epochs,
            batch_races=args.batch_races,
            scaler=scaler,
        )
        metrics, _ = evaluate_range(
            dataset,
            model,
            race_start=train_end,
            race_end=dataset.race_count,
            batch_races=args.batch_races,
        )
        metrics["final_training_ranking_log_loss"] = history[-1][
            "training_ranking_log_loss"
        ]
        return metrics

    immigrants = [genome_from_dict(row) for row in args.immigrants]
    elites, history = evolve_island(
        rng=random.Random(args.seed),
        population_size=args.population_size,
        local_generations=args.local_generations,
        elite_count=args.elite_count,
        evaluator=evaluate,
        immigrants=immigrants,
    )
    result = {
        "status": "completed",
        "model": (
            f"{MODEL}-{args.cohort}-g{args.generation:02d}"
            f"-i{args.island_id:02d}"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "speculative_recent_window_not_promotion_evidence",
        "promotion_eligible": False,
        "cohort": args.cohort,
        "generation": args.generation,
        "island_id": args.island_id,
        "island_count": args.island_count,
        "max_generations": args.max_generations,
        "seed": args.seed,
        "population_size": args.population_size,
        "local_generations": args.local_generations,
        "elite_count": args.elite_count,
        "train_races": args.train_races,
        "validation_races": args.validation_races,
        "cache_prefix": str(args.cache_prefix),
        "cache_race_count": len(race_keys),
        "champion": elites[0],
        "elites": elites,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one speculative genetic model island")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--evaluation-date", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--island-id", type=int, required=True)
    parser.add_argument("--island-count", type=int, default=4)
    parser.add_argument("--max-generations", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--local-generations", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--train-races", type=int, default=12_000)
    parser.add_argument("--validation-races", type=int, default=3_000)
    parser.add_argument("--batch-races", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=8192)
    parser.add_argument("--immigrants-json", default="[]")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    immigrants = json.loads(args.immigrants_json)
    if not isinstance(immigrants, list) or not all(isinstance(row, dict) for row in immigrants):
        raise ValueError("immigrants-json must be a list of genome objects")
    args.immigrants = immigrants
    result = run(args)
    print(json.dumps(result["champion"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
