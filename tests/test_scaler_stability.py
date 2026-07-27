import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from boatrace_ai.calibrated_shadow_model import stabilize_sparse_scaler


def test_stabilize_sparse_scaler_repairs_invalid_variance_only() -> None:
    scaler = StandardScaler(with_mean=False)
    scaler.mean_ = np.asarray([2.0, 3.0, 4.0])
    scaler.var_ = np.asarray([4.0, -1e-12, np.nan])
    scaler.scale_ = np.asarray([2.0, np.nan, np.nan])

    result = stabilize_sparse_scaler(scaler)

    assert result is scaler
    np.testing.assert_array_equal(scaler.var_, [4.0, 0.0, 0.0])
    np.testing.assert_array_equal(scaler.scale_, [2.0, 1.0, 1.0])


def test_stabilize_sparse_scaler_rejects_invalid_mean() -> None:
    scaler = StandardScaler(with_mean=False)
    scaler.mean_ = np.asarray([np.inf])
    scaler.var_ = np.asarray([1.0])
    scaler.scale_ = np.asarray([1.0])

    with pytest.raises(ValueError, match="non-finite sparse scaler mean"):
        stabilize_sparse_scaler(scaler)
