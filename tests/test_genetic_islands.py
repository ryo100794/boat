from __future__ import annotations

import random

import pytest

from boatrace_ai.genetic_islands import (
    Genome,
    evolve_island,
    genome_from_dict,
    speculative_fitness,
)


def _metrics(genome: Genome) -> dict[str, float]:
    quality = -abs(genome.learning_rate - 0.02)
    return {
        "ranking_log_loss": 1.3 - quality,
        "entry_log_loss": 0.34 - quality,
        "winner_top1_accuracy": 0.56 + quality,
        "trifecta_top5_hit_rate": 0.31 + quality,
    }


def test_genome_validation_rejects_unsafe_search_ranges() -> None:
    with pytest.raises(ValueError, match="alpha"):
        genome_from_dict({
            "target": "winner",
            "alpha": 1.0,
            "learning_rate": 0.02,
            "epochs": 1,
            "ev_threshold": 1.2,
        })


def test_speculative_fitness_is_not_a_promotion_flag() -> None:
    genome = Genome("winner", 1e-4, 0.02, 1, 1.2)
    better = _metrics(genome)
    worse = dict(better, ranking_log_loss=1.5, winner_top1_accuracy=0.50)

    assert speculative_fitness(better, genome) > speculative_fitness(worse, genome)


def test_island_evolution_is_reproducible_and_preserves_immigrant() -> None:
    immigrant = Genome("top3_pl", 1e-5, 0.02, 1, 1.15)
    first = evolve_island(
        rng=random.Random(17),
        population_size=6,
        local_generations=2,
        elite_count=2,
        evaluator=_metrics,
        immigrants=[immigrant],
    )
    second = evolve_island(
        rng=random.Random(17),
        population_size=6,
        local_generations=2,
        elite_count=2,
        evaluator=_metrics,
        immigrants=[immigrant],
    )

    assert first == second
    elites, history = first
    assert len(elites) == 2
    assert len(history) == 2
    assert elites[0]["fitness"] >= elites[1]["fitness"]
