from __future__ import annotations

import argparse
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
from .genetic_search import GeneticSearchSettings, evolve_population
from .hashed_feature_dataset import HashedRaceDataset, load_hashed_dataset
from .listwise.model import evaluate_range, fit_scaler, train_listwise_model


# Artifact v4 uses genome v3. Legacy v2/v3 artifacts remain valid immigrants:
# their discrete target is mapped exactly to a blend endpoint.
MODEL = "genetic_listwise_island_v4"
ARTIFACT_VERSION = 4
GENOME_VERSION = 3
TARGETS = ("winner", "top3_pl")
POLICY_EV_THRESHOLD = 1.20
VALIDATION_SEGMENT_COUNT = 3
MIN_EMBARGO_DAYS = 1
MAX_EPOCHS = 6
WORST_RANKING_PENALTY_WEIGHT = 0.50
WINNER_STABILITY_PENALTY_WEIGHT = 0.25
TOP5_STABILITY_PENALTY_WEIGHT = 0.20
EPOCH_COMPLEXITY_PENALTY_WEIGHT = 0.004
EPOCH_BOUNDARY_PENALTY = 0.002


@dataclass(frozen=True)
class Genome:
    target: str
    alpha: float
    learning_rate: float
    epochs: int
    ev_threshold: float
    loss_blend: float | None = None

    def __post_init__(self) -> None:
        if self.target not in (*TARGETS, "blended"):
            raise ValueError(f"unsupported genetic target: {self.target}")
        blend = self.loss_blend
        if blend is None:
            if self.target == "blended":
                raise ValueError("blended target requires loss_blend")
            blend = 0.0 if self.target == "winner" else 1.0
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("genetic loss_blend must be between 0 and 1")
        if self.target == "winner" and blend != 0.0:
            raise ValueError("winner target requires loss_blend=0")
        if self.target == "top3_pl" and blend != 1.0:
            raise ValueError("top3_pl target requires loss_blend=1")
        object.__setattr__(self, "loss_blend", blend)

    def as_dict(self) -> dict[str, Any]:
        return {
            "genome_version": GENOME_VERSION,
            "target": self.target,
            "loss_blend": self.loss_blend,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "ev_threshold": self.ev_threshold,
        }


@dataclass(frozen=True)
class TemporalSplit:
    train_end: int
    embargo_start: int
    validation_start: int
    validation_end: int
    train_dates: tuple[str, ...]
    embargo_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]


def genome_from_dict(value: dict[str, Any]) -> Genome:
    target = str(value.get("target") or "")
    blend_value = value.get("loss_blend")
    if blend_value is None:
        if target not in TARGETS:
            raise ValueError(f"unsupported genetic target: {target}")
        blend = 0.0 if target == "winner" else 1.0
    else:
        blend = float(blend_value)
        target = _target_for_blend(blend)
    if target not in (*TARGETS, "blended"):
        raise ValueError(f"unsupported genetic target: {target}")
    alpha = float(value["alpha"])
    learning_rate = float(value["learning_rate"])
    epochs = int(value["epochs"])
    ev_threshold = POLICY_EV_THRESHOLD
    if not 1e-7 <= alpha <= 1e-2:
        raise ValueError("genetic alpha must be between 1e-7 and 1e-2")
    if not 0.001 <= learning_rate <= 0.2:
        raise ValueError("genetic learning rate must be between 0.001 and 0.2")
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"genetic epochs must be between 1 and {MAX_EPOCHS}")
    if not 1.0 <= ev_threshold <= 3.0:
        raise ValueError("genetic EV threshold must be between 1.0 and 3.0")
    return Genome(target, alpha, learning_rate, epochs, ev_threshold, blend)


def _target_for_blend(blend: float) -> str:
    if float(blend) == 0.0:
        return "winner"
    if float(blend) == 1.0:
        return "top3_pl"
    return "blended"


def _random_blend(rng: random.Random) -> float:
    endpoint = rng.random()
    if endpoint < 0.20:
        return 0.0
    if endpoint < 0.40:
        return 1.0
    return rng.random()


def random_genome(rng: random.Random) -> Genome:
    blend = _random_blend(rng)
    return Genome(
        target=_target_for_blend(blend),
        alpha=10 ** rng.uniform(-6.5, -2.5),
        learning_rate=10 ** rng.uniform(math.log10(0.004), math.log10(0.08)),
        epochs=rng.choice((1, 1, 2, 2, 3, 4, 5, 6)),
        ev_threshold=POLICY_EV_THRESHOLD,
        loss_blend=blend,
    )


def crossover(left: Genome, right: Genome, rng: random.Random) -> Genome:
    blend = (
        rng.choice((left.loss_blend, right.loss_blend))
        if rng.random() < 0.5
        else (float(left.loss_blend) + float(right.loss_blend)) / 2.0
    )
    return Genome(
        target=_target_for_blend(blend),
        alpha=math.sqrt(left.alpha * right.alpha),
        learning_rate=math.sqrt(left.learning_rate * right.learning_rate),
        epochs=rng.choice((left.epochs, right.epochs)),
        ev_threshold=POLICY_EV_THRESHOLD,
        loss_blend=blend,
    )


def mutate(genome: Genome, rng: random.Random, rate: float = 0.35) -> Genome:
    blend = float(genome.loss_blend)
    alpha = genome.alpha
    learning_rate = genome.learning_rate
    epochs = genome.epochs
    if rng.random() < rate:
        if rng.random() < 0.25:
            blend = rng.choice((0.0, 1.0))
        else:
            blend = min(1.0, max(0.0, blend + rng.gauss(0.0, 0.20)))
    if rng.random() < rate:
        alpha = min(1e-2, max(1e-7, alpha * math.exp(rng.gauss(0.0, 0.8))))
    if rng.random() < rate:
        learning_rate = min(
            0.2,
            max(0.001, learning_rate * math.exp(rng.gauss(0.0, 0.35))),
        )
    if rng.random() < rate:
        epochs = rng.randint(1, MAX_EPOCHS)
    return Genome(
        _target_for_blend(blend),
        alpha,
        learning_rate,
        epochs,
        POLICY_EV_THRESHOLD,
        blend,
    )


def _aggregate_speculative_score(metrics: dict[str, Any], genome: Genome) -> float:
    ranking = float(metrics["ranking_log_loss"])
    entry = float(metrics["entry_log_loss"])
    winner = float(metrics["winner_top1_accuracy"])
    top5 = float(metrics["trifecta_top5_hit_rate"])
    effective_epochs = int(metrics.get("effective_epochs") or genome.epochs)
    complexity = EPOCH_COMPLEXITY_PENALTY_WEIGHT * max(0, effective_epochs - 1)
    if genome.epochs == MAX_EPOCHS:
        complexity += EPOCH_BOUNDARY_PENALTY
    return -ranking - 0.35 * entry + 0.35 * winner + 0.20 * top5 - complexity


def fitness_components(metrics: dict[str, Any], genome: Genome) -> dict[str, float]:
    """Score chronological validation segments; never use this for promotion."""
    segments = metrics.get("validation_segments")
    if not isinstance(segments, list) or not segments:
        base = _aggregate_speculative_score(metrics, genome)
        return {
            "segment_mean_score": base,
            "worst_segment_score": base,
            "worst_segment_ranking_log_loss": float(metrics["ranking_log_loss"]),
            "worst_segment_entry_log_loss": float(metrics["entry_log_loss"]),
            "worst_segment_winner_top1_accuracy": float(
                metrics["winner_top1_accuracy"]
            ),
            "worst_segment_trifecta_top5_hit_rate": float(
                metrics["trifecta_top5_hit_rate"]
            ),
            "ranking_worst_segment_penalty": 0.0,
            "winner_stability_std": 0.0,
            "winner_stability_penalty": 0.0,
            "top5_stability_std": 0.0,
            "top5_stability_penalty": 0.0,
            "stability_penalty": 0.0,
            "fitness": base,
        }

    segment_scores = [_aggregate_speculative_score(row, genome) for row in segments]
    ranking_losses = [float(row["ranking_log_loss"]) for row in segments]
    winner_rates = [float(row["winner_top1_accuracy"]) for row in segments]
    top5_rates = [float(row["trifecta_top5_hit_rate"]) for row in segments]
    segment_mean = statistics.fmean(segment_scores)
    worst_ranking = max(ranking_losses)
    ranking_penalty = WORST_RANKING_PENALTY_WEIGHT * max(
        0.0, worst_ranking - statistics.fmean(ranking_losses)
    )
    winner_std = statistics.pstdev(winner_rates)
    top5_std = statistics.pstdev(top5_rates)
    winner_penalty = WINNER_STABILITY_PENALTY_WEIGHT * winner_std
    top5_penalty = TOP5_STABILITY_PENALTY_WEIGHT * top5_std
    stability_penalty = ranking_penalty + winner_penalty + top5_penalty
    return {
        "segment_mean_score": segment_mean,
        "worst_segment_score": min(segment_scores),
        "worst_segment_ranking_log_loss": worst_ranking,
        "worst_segment_entry_log_loss": max(
            float(row["entry_log_loss"]) for row in segments
        ),
        "worst_segment_winner_top1_accuracy": min(winner_rates),
        "worst_segment_trifecta_top5_hit_rate": min(top5_rates),
        "ranking_worst_segment_penalty": ranking_penalty,
        "winner_stability_std": winner_std,
        "winner_stability_penalty": winner_penalty,
        "top5_stability_std": top5_std,
        "top5_stability_penalty": top5_penalty,
        "stability_penalty": stability_penalty,
        "fitness": segment_mean - stability_penalty,
    }


def speculative_fitness(metrics: dict[str, Any], genome: Genome) -> float:
    return fitness_components(metrics, genome)["fitness"]


def chronological_validation_segments(
    race_start: int,
    race_end: int,
    *,
    race_keys: list[tuple[str, str, str, int]] | None = None,
    segment_count: int = VALIDATION_SEGMENT_COUNT,
) -> list[tuple[int, int]]:
    """Partition an ordered range without splitting a date when keys are supplied."""
    race_count = race_end - race_start
    if segment_count < 1:
        raise ValueError("segment_count must be positive")
    if race_count < segment_count:
        raise ValueError(
            f"validation requires at least {segment_count} races; got {race_count}"
        )
    if race_keys is None:
        quotient, remainder = divmod(race_count, segment_count)
        segments: list[tuple[int, int]] = []
        start = race_start
        for index in range(segment_count):
            stop = start + quotient + (1 if index < remainder else 0)
            segments.append((start, stop))
            start = stop
        return segments

    groups = _date_groups(race_keys, race_start=race_start, race_end=race_end)
    if len(groups) < segment_count:
        raise ValueError(
            f"validation requires at least {segment_count} complete days; "
            f"got {len(groups)}"
        )
    quotient, remainder = divmod(len(groups), segment_count)
    segments: list[tuple[int, int]] = []
    group_start = 0
    for index in range(segment_count):
        group_stop = group_start + quotient + (1 if index < remainder else 0)
        segments.append((groups[group_start][1], groups[group_stop - 1][2]))
        group_start = group_stop
    return segments


def _date_groups(
    race_keys: list[tuple[str, str, str, int]],
    *,
    race_start: int = 0,
    race_end: int | None = None,
) -> list[tuple[str, int, int]]:
    stop = len(race_keys) if race_end is None else min(len(race_keys), race_end)
    start = max(0, race_start)
    if start >= stop:
        return []
    dates = [str(row[1]) for row in race_keys[start:stop]]
    if dates != sorted(dates):
        raise ValueError("genetic races must be ordered chronologically by race date")
    groups: list[tuple[str, int, int]] = []
    group_start = start
    current = dates[0]
    for index in range(start + 1, stop):
        race_date = str(race_keys[index][1])
        if race_date != current:
            groups.append((current, group_start, index))
            current = race_date
            group_start = index
    groups.append((current, group_start, stop))
    return groups


def evolve_island(
    *,
    rng: random.Random,
    population_size: int,
    local_generations: int,
    elite_count: int,
    evaluator: Callable[[Genome], dict[str, Any]],
    immigrants: list[Genome] | None = None,
    mutation_rate: float = 0.35,
    random_injections: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    elite_count = min(max(1, elite_count), population_size // 2)
    ranked, generic_history = evolve_population(
        settings=GeneticSearchSettings(
            population_size=population_size,
            generations=local_generations,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            random_injections=max(0, random_injections),
            max_workers=min(2, population_size),
            seed=rng.randrange(0, 2**31),
        ),
        evaluator=evaluator,
        fitness=lambda metrics, genome: speculative_fitness(
            dict(metrics), genome
        ),
        random_candidate=random_genome,
        crossover=crossover,
        mutate=lambda genome, local_rng, rate: mutate(
            genome, local_rng, rate=rate
        ),
        candidate_key=lambda genome: json.dumps(
            genome.as_dict(), sort_keys=True
        ),
        serialize=lambda genome: genome.as_dict(),
        immigrants=tuple(immigrants or ()),
    )
    candidate_rows = [
        {
            "genome": row.candidate.as_dict(),
            "metrics": dict(row.metrics),
            "fitness": row.fitness,
            "fitness_components": fitness_components(
                dict(row.metrics), row.candidate
            ),
        }
        for row in ranked[:elite_count]
    ]
    history = [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {
                    "generation", "best_candidate", "unique_candidates"
                }
            },
            "local_generation": row["generation"],
            "best_genome": row["best_candidate"],
            "unique_genomes": row["unique_candidates"],
        }
        for row in generic_history
    ]
    return candidate_rows, history


def _slice_dataset(
    dataset: HashedRaceDataset,
    *,
    train_races: int,
    validation_races: int,
    embargo_days: int = MIN_EMBARGO_DAYS,
) -> tuple[HashedRaceDataset, TemporalSplit]:
    if embargo_days < MIN_EMBARGO_DAYS:
        raise ValueError(
            f"genetic validation requires at least {MIN_EMBARGO_DAYS} embargo day"
        )
    if train_races < 1 or validation_races < 1:
        raise ValueError("train_races and validation_races must be positive")
    groups = _date_groups(dataset.race_keys)
    if len(groups) < embargo_days + VALIDATION_SEGMENT_COUNT + 1:
        raise ValueError("genetic cache does not contain enough complete days")

    def suffix_start(rows: list[tuple[str, int, int]], required: int) -> int:
        count = 0
        for index in range(len(rows) - 1, -1, -1):
            count += rows[index][2] - rows[index][1]
            if count >= required:
                return index
        raise ValueError(
            f"genetic cache has only {count} eligible races; {required} required"
        )

    validation_group_start = min(
        suffix_start(groups, validation_races),
        len(groups) - VALIDATION_SEGMENT_COUNT,
    )
    embargo_group_start = validation_group_start - embargo_days
    if embargo_group_start <= 0:
        raise ValueError(
            "genetic cache lacks a full training window before the embargo"
        )
    training_groups = groups[:embargo_group_start]
    train_group_start = suffix_start(training_groups, train_races)
    start = groups[train_group_start][1]
    train_end_absolute = groups[embargo_group_start][1]
    validation_start_absolute = groups[validation_group_start][1]
    sliced = HashedRaceDataset(
        matrix=dataset.matrix[dataset.row_slice(start, dataset.race_count)].tocsr(),
        race_keys=dataset.race_keys[start:],
        ranks=dataset.ranks[start:],
        n_features=dataset.n_features,
        drop_feature_groups=dataset.drop_feature_groups,
        hasher_settings=dataset.hasher_settings,
        feature_schema_version=dataset.feature_schema_version,
    )
    train_end = train_end_absolute - start
    validation_start = validation_start_absolute - start
    split = TemporalSplit(
        train_end=train_end,
        embargo_start=train_end,
        validation_start=validation_start,
        validation_end=sliced.race_count,
        train_dates=tuple(row[0] for row in groups[train_group_start:embargo_group_start]),
        embargo_dates=tuple(
            row[0] for row in groups[embargo_group_start:validation_group_start]
        ),
        validation_dates=tuple(row[0] for row in groups[validation_group_start:]),
    )
    return sliced, split


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
    dataset, split = _slice_dataset(
        dataset,
        train_races=args.train_races,
        validation_races=args.validation_races,
        embargo_days=getattr(args, "embargo_days", MIN_EMBARGO_DAYS),
    )
    train_end = split.train_end
    scaler = fit_scaler(dataset, race_end=train_end, batch_rows=args.batch_races * 6)

    validation_segments = chronological_validation_segments(
        split.validation_start,
        split.validation_end,
        race_keys=dataset.race_keys,
    )

    def evaluate(genome: Genome) -> dict[str, Any]:
        model, history = train_listwise_model(
            dataset,
            train_race_end=train_end,
            target=genome.target,
            loss_blend=genome.loss_blend,
            alpha=genome.alpha,
            learning_rate=genome.learning_rate,
            epochs=genome.epochs,
            batch_races=args.batch_races,
            scaler=scaler,
            early_stopping_patience=1,
            early_stopping_min_delta=1e-4,
        )
        metrics, _ = evaluate_range(
            dataset,
            model,
            race_start=split.validation_start,
            race_end=split.validation_end,
            batch_races=args.batch_races,
        )
        selected_epoch = next(
            row for row in history if int(row["epoch"]) == model.epochs
        )
        metrics["final_training_ranking_log_loss"] = selected_epoch[
            "training_ranking_log_loss"
        ]
        metrics["requested_epochs"] = genome.epochs
        metrics["effective_epochs"] = model.epochs
        metrics["observed_epochs"] = len(history)
        metrics["early_stopped"] = bool(history[-1].get("early_stopped"))
        metrics["loss_blend"] = genome.loss_blend
        metrics["validation_segments"] = []
        for segment_index, (segment_start, segment_end) in enumerate(
            validation_segments, start=1
        ):
            segment_metrics, _ = evaluate_range(
                dataset,
                model,
                race_start=segment_start,
                race_end=segment_end,
                batch_races=args.batch_races,
            )
            metrics["validation_segments"].append({
                "segment_index": segment_index,
                "race_start": segment_start,
                "race_end": segment_end,
                "start_race_date": str(dataset.race_keys[segment_start][1]),
                "end_race_date": str(dataset.race_keys[segment_end - 1][1]),
                "effective_epochs": model.epochs,
                **segment_metrics,
            })
        return metrics

    immigrants = [genome_from_dict(row) for row in args.immigrants]
    elites, history = evolve_island(
        rng=random.Random(args.seed),
        population_size=args.population_size,
        local_generations=args.local_generations,
        elite_count=args.elite_count,
        evaluator=evaluate,
        immigrants=immigrants,
        mutation_rate=args.mutation_rate,
        random_injections=args.random_injections,
    )
    result = {
        "status": "completed",
        "artifact_version": ARTIFACT_VERSION,
        "model": (
            f"{MODEL}-{args.cohort}-g{args.generation:02d}"
            f"-i{args.island_id:02d}"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "speculative_recent_window_not_promotion_evidence",
        "promotion_eligible": False,
        "validation_protocol": {
            "kind": "three_full_day_contiguous_chronological_segments",
            "segment_count": VALIDATION_SEGMENT_COUNT,
            "shuffled": False,
            "day_boundary_aligned": True,
            "embargo_days": len(split.embargo_dates),
            "embargo_dates": list(split.embargo_dates),
            "train_date_range": [split.train_dates[0], split.train_dates[-1]],
            "validation_date_range": [
                split.validation_dates[0],
                split.validation_dates[-1],
            ],
            "segment_date_ranges": [
                {
                    "segment_index": index,
                    "start_race_date": str(dataset.race_keys[start][1]),
                    "end_race_date": str(dataset.race_keys[stop - 1][1]),
                    "race_count": stop - start,
                }
                for index, (start, stop) in enumerate(validation_segments, start=1)
            ],
            "formal_365d_validation_separate": True,
            "fitness": (
                "mean_segment_speculative_score_minus_worst_ranking_and_"
                "winner_top5_stability_penalties"
            ),
        },
        "genome_scope": "prediction_hyperparameters_only",
        "excluded_policy_genes": ["ev_threshold"],
        "policy_optimization_stage": "selection_window_before_formal_holdout",
        "cohort": args.cohort,
        "generation": args.generation,
        "island_id": args.island_id,
        "island_count": args.island_count,
        "max_generations": args.max_generations,
        "seed": args.seed,
        "population_size": args.population_size,
        "local_generations": args.local_generations,
        "elite_count": args.elite_count,
        "base_mutation_rate": args.mutation_rate,
        "random_injections_per_local_generation": args.random_injections,
        "configured_random_injections": args.random_injections,
        "train_races": args.train_races,
        "validation_races": args.validation_races,
        "actual_train_races": split.train_end,
        "actual_embargo_races": split.validation_start - split.embargo_start,
        "actual_validation_races": split.validation_end - split.validation_start,
        "embargo_days": len(split.embargo_dates),
        "embargo_dates": list(split.embargo_dates),
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
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--random-injections", type=int, default=1)
    parser.add_argument("--train-races", type=int, default=12_000)
    parser.add_argument("--validation-races", type=int, default=3_000)
    parser.add_argument("--embargo-days", type=int, default=MIN_EMBARGO_DAYS)
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
