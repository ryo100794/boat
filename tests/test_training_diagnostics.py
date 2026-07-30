from boatrace_ai.training_diagnostics import classifier_training_diagnostics


class _Classifier:
    loss_curve_ = [0.8, 0.6, 0.5]
    n_iter_ = 3


def test_summarizes_classifier_loss_curve() -> None:
    result = classifier_training_diagnostics(_Classifier())

    assert result == {
        "optimizer_updates": 3,
        "reported_iterations": 3,
        "finite_loss_curve": True,
        "loss_start": 0.8,
        "loss_end": 0.5,
        "loss_min": 0.5,
        "loss_reduction_fraction": 0.37500000000000006,
        "recent_loss_change": -0.09999999999999998,
    }


def test_handles_classifier_without_loss_curve() -> None:
    result = classifier_training_diagnostics(object())

    assert result["optimizer_updates"] == 0
    assert result["loss_end"] is None
