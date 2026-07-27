from __future__ import annotations

import random

import pytest

from boatrace_ai.genetic_islands import (
    Genome,
    POLICY_EV_THRESHOLD,
    chronological_validation_segments,
    evolve_island,
    fitness_components,
    genome_from_dict,
    mutate,
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


def _segments(*rows: tuple[float, float, float]) -> list[dict[str, float]]:
    return [
        dict(
            _metrics(Genome("winner", 1e-4, 0.02, 1, 1.2)),
            ranking_log_loss=ranking,
            winner_top1_accuracy=winner,
            trifecta_top5_hit_rate=top5,
        )
        for ranking, winner, top5 in rows
    ]


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


def test_chronological_segments_are_contiguous_complete_and_not_shuffled() -> None:
    segments = chronological_validation_segments(5, 15)

    assert segments == [(5, 9), (9, 12), (12, 15)]
    assert [stop - start for start, stop in segments] == [4, 3, 3]
    with pytest.raises(ValueError, match="at least 3 races"):
        chronological_validation_segments(5, 7)


def test_temporal_instability_lowers_fitness_at_equal_segment_means() -> None:
    genome = Genome("winner", 1e-4, 0.02, 1, 1.2)
    aggregate = _metrics(genome)
    stable = dict(
        aggregate,
        validation_segments=_segments(
            (1.2, 0.56, 0.31),
            (1.2, 0.56, 0.31),
            (1.2, 0.56, 0.31),
        ),
    )
    unstable = dict(
        aggregate,
        validation_segments=_segments(
            (0.8, 0.70, 0.20),
            (1.2, 0.56, 0.31),
            (1.6, 0.42, 0.42),
        ),
    )

    stable_components = fitness_components(stable, genome)
    unstable_components = fitness_components(unstable, genome)

    assert unstable_components["segment_mean_score"] == pytest.approx(
        stable_components["segment_mean_score"]
    )
    assert stable_components["stability_penalty"] == 0.0
    assert unstable_components["ranking_worst_segment_penalty"] > 0.0
    assert unstable_components["winner_stability_penalty"] > 0.0
    assert unstable_components["top5_stability_penalty"] > 0.0
    assert speculative_fitness(stable, genome) > speculative_fitness(unstable, genome)


def test_policy_ev_threshold_is_not_evolved_with_prediction_genome() -> None:
    source = Genome("winner", 1e-4, 0.02, 1, 2.75)

    mutated = mutate(source, random.Random(3), rate=1.0)
    restored = genome_from_dict({**source.as_dict(), "ev_threshold": 2.5})

    assert mutated.ev_threshold == POLICY_EV_THRESHOLD
    assert restored.ev_threshold == POLICY_EV_THRESHOLD


def test_v2_immigrant_remains_loadable_by_v3_island() -> None:
    immigrant = genome_from_dict({
        "genome_version": 2,
        "target": "top3_pl",
        "alpha": 1e-5,
        "learning_rate": 0.02,
        "epochs": 2,
        "ev_threshold": 1.8,
    })

    assert immigrant == Genome("top3_pl", 1e-5, 0.02, 2, POLICY_EV_THRESHOLD)
    assert immigrant.as_dict()["genome_version"] == 2


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
    assert history[0]["min_fitness"] <= history[0]["q1_fitness"]
    assert history[0]["q1_fitness"] <= history[0]["median_fitness"]
    assert history[0]["median_fitness"] <= history[0]["q3_fitness"]
    assert history[0]["q3_fitness"] <= history[0]["max_fitness"]
    assert history[0]["std_fitness"] >= 0
    assert history[0]["random_injections"] == 1
    assert history[0]["mutation_rate"] >= 0.35
    assert history[0]["unique_genomes"] == 6


def test_island_supports_multiple_random_injections() -> None:
    _elites, history = evolve_island(
        rng=random.Random(23),
        population_size=8,
        local_generations=2,
        elite_count=2,
        evaluator=_metrics,
        random_injections=3,
    )

    assert history[0]["random_injections"] == 3
