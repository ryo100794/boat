import pytest

from boatrace_ai.listwise.cluster_bootstrap import paired_cluster_mean_bootstrap


def test_cluster_bootstrap_resamples_whole_days() -> None:
    result = paired_cluster_mean_bootstrap(
        [-0.2, -0.2, 0.1, 0.1, 0.1],
        ["day-1", "day-1", "day-2", "day-2", "day-2"],
        samples=2_000,
        seed=7,
    )

    assert result["observations"] == 5
    assert result["clusters"] == 2
    assert result["mean_difference"] == pytest.approx(-0.02)
    assert result["ci95_lower"] <= -0.2
    assert result["ci95_upper"] >= 0.1


def test_single_cluster_is_reported_but_not_fabricated() -> None:
    result = paired_cluster_mean_bootstrap(
        [-0.1, -0.2],
        ["day-1", "day-1"],
        samples=200,
    )

    assert result["clusters"] == 1
    assert result["ci95_lower"] == pytest.approx(-0.15)
    assert result["ci95_upper"] == pytest.approx(-0.15)


@pytest.mark.parametrize(
    ("differences", "labels"),
    [([], []), ([0.1], []), ([0.1], [["not-hashable"]])],
)
def test_cluster_bootstrap_rejects_invalid_inputs(differences, labels) -> None:
    with pytest.raises(ValueError):
        paired_cluster_mean_bootstrap(differences, labels)
