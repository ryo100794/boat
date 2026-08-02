from __future__ import annotations

from boatrace_ai.genetic_search import GeneticSearchSettings, evolve_population


def _run(seed: int):
    evaluations: dict[str, int] = {}

    def key(value: float) -> str:
        return f"{value:.8f}"

    def evaluate(value: float):
        evaluations[key(value)] = evaluations.get(key(value), 0) + 1
        return {"distance": abs(value - 0.73)}

    ranked, history = evolve_population(
        settings=GeneticSearchSettings(
            population_size=8,
            generations=4,
            elite_count=2,
            random_injections=1,
            max_workers=4,
            seed=seed,
        ),
        evaluator=evaluate,
        fitness=lambda metrics, _value: -float(metrics["distance"]),
        random_candidate=lambda rng: rng.random(),
        crossover=lambda left, right, rng: (
            (blend := rng.random()) * left + (1.0 - blend) * right
        ),
        mutate=lambda value, rng, rate: min(
            1.0,
            max(
                0.0,
                value
                + (rng.gauss(0.0, 0.1) if rng.random() < rate else 0.0),
            ),
        ),
        candidate_key=key,
        serialize=lambda value: {"value": value},
        immigrants=(0.10, 0.90),
    )
    return ranked, history, evaluations


def test_genetic_search_is_seeded_cached_and_records_convergence() -> None:
    first, first_history, first_evaluations = _run(17)
    second, second_history, second_evaluations = _run(17)

    assert first[0].candidate == second[0].candidate
    assert first[0].fitness == second[0].fitness
    assert first_history == second_history
    assert first_evaluations == second_evaluations
    assert all(count == 1 for count in first_evaluations.values())
    assert len(first_evaluations) < 8 * 4
    assert all(row["unique_candidates"] == 8 for row in first_history)
    assert all("cached_evaluations" in row for row in first_history)


def test_genetic_search_changes_population_with_seed() -> None:
    first, first_history, _ = _run(17)
    second, second_history, _ = _run(18)

    assert first_history[0]["best_candidate"] != second_history[0]["best_candidate"]
    assert 0.0 <= first[0].candidate <= 1.0
    assert 0.0 <= second[0].candidate <= 1.0
