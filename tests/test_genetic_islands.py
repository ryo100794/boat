from __future__ import annotations

from contextlib import nullcontext
import random
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import boatrace_ai.genetic_islands as genetic_islands
from boatrace_ai.genetic_islands import (
    ARTIFACT_VERSION,
    GENOME_VERSION,
    MAX_EPOCHS,
    MODEL,
    Genome,
    POLICY_EV_THRESHOLD,
    _slice_dataset,
    chronological_validation_segments,
    evolve_island,
    fitness_components,
    genome_from_dict,
    mutate,
    speculative_fitness,
)
from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai.listwise.model import train_listwise_model


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


def test_v2_and_v3_immigrants_migrate_to_v3_blend_endpoints() -> None:
    immigrant = genome_from_dict({
        "genome_version": 2,
        "target": "top3_pl",
        "alpha": 1e-5,
        "learning_rate": 0.02,
        "epochs": 2,
        "ev_threshold": 1.8,
    })
    v3_artifact_immigrant = genome_from_dict({
        "genome_version": 3,
        "target": "winner",
        "alpha": 2e-5,
        "learning_rate": 0.01,
        "epochs": 3,
    })
    mixed = genome_from_dict({
        "genome_version": 3,
        "target": "winner",
        "loss_blend": 0.4,
        "alpha": 2e-5,
        "learning_rate": 0.01,
        "epochs": 3,
    })

    assert immigrant == Genome("top3_pl", 1e-5, 0.02, 2, POLICY_EV_THRESHOLD)
    assert immigrant.loss_blend == 1.0
    assert v3_artifact_immigrant.loss_blend == 0.0
    assert mixed.target == "blended"
    assert mixed.loss_blend == 0.4
    assert immigrant.as_dict()["genome_version"] == GENOME_VERSION == 3
    assert ARTIFACT_VERSION == 4
    assert MODEL == "genetic_listwise_island_v4"


def _dated_dataset(day_counts: tuple[int, ...]) -> HashedRaceDataset:
    keys = []
    ranks = []
    matrix = []
    race = 0
    for day, count in enumerate(day_counts, start=1):
        for _ in range(count):
            keys.append((
                f"2026-01-{day:02d}-01-{race % 12 + 1:02d}",
                f"2026-01-{day:02d}",
                "01",
                race % 12 + 1,
            ))
            ranks.append([1, 2, 3, 4, 5, 6])
            for lane in range(6):
                matrix.append([float(lane), float(race)])
            race += 1
    return HashedRaceDataset(
        matrix=sparse.csr_matrix(np.asarray(matrix)),
        race_keys=keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=2,
        drop_feature_groups=(),
    )


def test_temporal_split_aligns_full_days_and_embargoes_one_complete_day() -> None:
    sliced, split = _slice_dataset(
        _dated_dataset((2, 3, 1, 4, 2, 5, 3)),
        train_races=5,
        validation_races=6,
        embargo_days=1,
    )

    assert split.train_dates == ("2026-01-01", "2026-01-02", "2026-01-03")
    assert split.embargo_dates == ("2026-01-04",)
    assert split.validation_dates == ("2026-01-05", "2026-01-06", "2026-01-07")
    assert split.train_end == 6
    assert split.validation_start == 10
    assert sliced.race_keys[split.train_end - 1][1] == "2026-01-03"
    assert sliced.race_keys[split.train_end][1] == "2026-01-04"
    assert sliced.race_keys[split.validation_start][1] == "2026-01-05"


def test_embargo_and_validation_values_cannot_change_training() -> None:
    sliced, split = _slice_dataset(
        _dated_dataset((2, 3, 1, 4, 2, 5, 3)),
        train_races=5,
        validation_races=6,
        embargo_days=1,
    )
    altered_matrix = sliced.matrix.toarray()
    altered_matrix[split.train_end * 6 :] += 1_000_000.0
    altered_ranks = sliced.ranks.copy()
    altered_ranks[split.train_end :] = altered_ranks[split.train_end :, ::-1]
    altered = HashedRaceDataset(
        matrix=sparse.csr_matrix(altered_matrix),
        race_keys=sliced.race_keys,
        ranks=altered_ranks,
        n_features=sliced.n_features,
        drop_feature_groups=sliced.drop_feature_groups,
    )
    arguments = dict(
        train_race_end=split.train_end,
        target="blended",
        loss_blend=0.45,
        alpha=1e-4,
        learning_rate=0.01,
        epochs=2,
        batch_races=2,
    )

    first, first_history = train_listwise_model(sliced, **arguments)
    second, second_history = train_listwise_model(altered, **arguments)

    np.testing.assert_array_equal(first.weights, second.weights)
    assert first_history == second_history


def test_validation_segments_never_split_a_day() -> None:
    dataset = _dated_dataset((2, 5, 1, 4, 3, 6))
    segments = chronological_validation_segments(
        0,
        dataset.race_count,
        race_keys=dataset.race_keys,
    )

    assert segments == [(0, 7), (7, 12), (12, 21)]
    for left, right in zip(segments, segments[1:]):
        assert left[1] == right[0]
        assert dataset.race_keys[left[1] - 1][1] != dataset.race_keys[right[0]][1]


def test_epoch_search_accepts_six_and_penalizes_ungained_boundary_capacity() -> None:
    five = Genome("top3_pl", 1e-4, 0.02, 5, POLICY_EV_THRESHOLD)
    six = genome_from_dict({
        "genome_version": 3,
        "target": "top3_pl",
        "loss_blend": 1.0,
        "alpha": 1e-4,
        "learning_rate": 0.02,
        "epochs": MAX_EPOCHS,
    })
    metrics = _metrics(five)

    assert six.epochs == 6
    assert speculative_fitness(metrics, six) < speculative_fitness(metrics, five)
    with pytest.raises(ValueError, match="between 1 and 6"):
        genome_from_dict({**six.as_dict(), "epochs": 7})


def test_v4_artifact_audits_embargo_segments_blend_and_worst_fold(
    monkeypatch, tmp_path
) -> None:
    dataset = _dated_dataset((2, 3, 1, 4, 2, 5, 3))
    monkeypatch.setattr(
        genetic_islands,
        "connection",
        lambda _db: nullcontext(object()),
    )
    monkeypatch.setattr(
        genetic_islands,
        "load_complete_race_ids",
        lambda _conn: dataset.race_keys,
    )
    monkeypatch.setattr(
        genetic_islands,
        "load_hashed_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    output = tmp_path / "ga-v4.json"
    result = genetic_islands.run(SimpleNamespace(
        db="unused",
        evaluation_date="2026-01-07",
        cache_prefix=tmp_path / "cache",
        n_features=2,
        train_races=5,
        validation_races=6,
        embargo_days=1,
        batch_races=2,
        seed=17,
        population_size=4,
        local_generations=1,
        elite_count=1,
        immigrants=[],
        mutation_rate=0.35,
        random_injections=0,
        cohort="test",
        generation=0,
        island_id=0,
        island_count=2,
        max_generations=1,
        output=output,
    ))

    protocol = result["validation_protocol"]
    champion = result["champion"]
    assert result["artifact_version"] == 4
    assert result["model"].startswith("genetic_listwise_island_v4-")
    assert protocol["embargo_dates"] == ["2026-01-04"]
    assert protocol["segment_date_ranges"] == [
        {
            "segment_index": 1,
            "start_race_date": "2026-01-05",
            "end_race_date": "2026-01-05",
            "race_count": 2,
        },
        {
            "segment_index": 2,
            "start_race_date": "2026-01-06",
            "end_race_date": "2026-01-06",
            "race_count": 5,
        },
        {
            "segment_index": 3,
            "start_race_date": "2026-01-07",
            "end_race_date": "2026-01-07",
            "race_count": 3,
        },
    ]
    assert 0.0 <= champion["genome"]["loss_blend"] <= 1.0
    assert "worst_segment_score" in champion["fitness_components"]
    assert "worst_segment_ranking_log_loss" in champion["fitness_components"]
    assert output.exists()


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
