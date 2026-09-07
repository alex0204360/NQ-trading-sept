"""Behavioral contract for completed-bar, gap-reset causal features."""

import numpy as np
import pandas as pd
import pytest

from nqpatterns.features import compute_features


def bars(n=180, start="2023-01-03T14:30:00Z"):
    i = np.arange(n, dtype=float)
    close = 12000 + 0.03 * i + 2 * np.sin(i / 5)
    open_ = close - 0.3 * np.cos(i / 3)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="min"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.5 + 0.1 * np.sin(i) ** 2,
            "low": np.minimum(open_, close) - 0.4 - 0.1 * np.cos(i) ** 2,
            "close": close,
            "volume": 100 + (np.arange(n) * 13) % 101,
        }
    )


def test_preserves_index_and_requires_sixty_prior_contiguous_bars():
    frame = bars(100)
    frame.index = pd.Index(np.arange(100) * 3, name="source_index")
    original = frame.copy(deep=True)
    features = compute_features(frame)
    assert features.index.equals(frame.index)
    assert features["valid"].dtype == bool
    assert not features["valid"].iloc[:60].any()
    assert features["valid"].iloc[60:].all()
    numeric = features.drop(columns="valid").select_dtypes(include="number")
    assert np.isfinite(numeric.loc[features["valid"]].to_numpy()).all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("stop", [0, 1, 60, 61, 107, 180])
def test_features_are_prefix_invariant(stop):
    frame = bars()
    expected = compute_features(frame).iloc[:stop]
    actual = compute_features(frame.iloc[:stop])
    pd.testing.assert_frame_equal(actual, expected)


def test_future_price_volume_and_contract_perturbation_cannot_change_the_past():
    frame = bars()
    altered = frame.copy()
    altered.loc[120:, ["open", "high", "low", "close"]] *= 3
    altered.loc[120:, "volume"] *= 100
    altered["contract"] = "NQH23"
    altered.loc[120:, "contract"] = "NQM23"
    pd.testing.assert_frame_equal(
        compute_features(frame).iloc[:120], compute_features(altered).iloc[:120]
    )


@pytest.mark.parametrize("boundary", ["gap", "contract"])
def test_gap_and_contract_change_restart_the_entire_warmup(boundary):
    frame = bars(160)
    if boundary == "gap":
        frame.loc[80:, "timestamp"] += pd.Timedelta(minutes=7)
    else:
        frame["contract"] = ["NQH23"] * 80 + ["NQM23"] * 80
    features = compute_features(frame)
    assert features["valid"].iloc[79]
    assert not features["valid"].iloc[80:140].any()
    assert features["valid"].iloc[140:].all()
    pd.testing.assert_frame_equal(
        features.iloc[80:], compute_features(frame.iloc[80:])
    )


def test_features_need_no_more_than_the_current_and_sixty_prior_bars():
    frame = bars()
    pd.testing.assert_frame_equal(
        compute_features(frame).iloc[[-1]],
        compute_features(frame.iloc[-61:]).iloc[[-1]],
        rtol=1e-8,
        atol=1e-8,
    )


def test_simple_candle_geometry_and_prior_extrema_exclude_current_bar():
    frame = bars(61)
    frame.loc[:, ["open", "high", "low", "close"]] = [100, 102, 98, 101]
    frame.loc[60, ["open", "high", "low", "close"]] = [101, 107, 99, 105]
    last = compute_features(frame).iloc[-1]
    assert last["valid"]
    assert last["body_fraction"] == pytest.approx(0.5)
    assert last["upper_wick_fraction"] == pytest.approx(0.25)
    assert last["lower_wick_fraction"] == pytest.approx(0.25)
    assert last["prior_high20"] == 102
    assert last["prior_high60"] == 102
    assert last["prior_low20"] == 98
    assert last["prior_low60"] == 98
    assert last["position60"] == pytest.approx(1.75)
    assert last["breakout_up20_atr"] > 0
    assert last["breakout_down20_atr"] < 0


def test_price_state_uses_trailing_population_statistics():
    frame = bars(61)
    close = 100 + np.arange(61, dtype=float)
    frame["open"] = close
    frame["close"] = close
    frame["high"] = close + 1
    frame["low"] = close - 1
    last = compute_features(frame).iloc[-1]
    assert last["price_z20"] == pytest.approx(
        (close[-1] - close[-20:].mean()) / close[-20:].std(ddof=0)
    )
    assert last["trend20_60"] == pytest.approx(
        (close[-20:].mean() - close[-60:].mean()) / last["atr14"]
    )
    assert last["atr14_fraction"] == pytest.approx(last["atr14"] / close[-1])


def test_flat_bars_and_zero_volume_have_finite_neutral_states():
    frame = bars(80)
    frame.loc[:, ["open", "high", "low", "close"]] = 100.0
    frame["volume"] = 0.0
    features = compute_features(frame)
    valid = features.loc[features["valid"]]
    assert np.isfinite(valid.drop(columns="valid").to_numpy(dtype=float)).all()
    for column in [
        "body_fraction", "upper_wick_fraction", "lower_wick_fraction",
        "lag1_return_to_range", "price_z20", "trend20_60", "volume_z20",
        "range_z20", "range_to_atr14",
    ]:
        assert (valid[column] == 0).all()


def test_volume_standardization_uses_only_observed_trailing_volumes():
    frame = bars(61)
    frame["volume"] = np.arange(61, dtype=float)
    last = compute_features(frame).iloc[-1]
    observed = frame["volume"].iloc[-20:].to_numpy()
    assert last["volume_z20"] == pytest.approx(
        (observed[-1] - observed.mean()) / observed.std(ddof=0)
    )


@pytest.mark.parametrize("defect", ["naive", "descending", "duplicate", "missing", "nonfinite", "negative_volume", "bad_ohlc"])
def test_rejects_invalid_downstream_bar_contract(defect):
    frame = bars(80)
    if defect == "naive":
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    elif defect == "descending":
        frame = frame.iloc[::-1]
    elif defect == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif defect == "missing":
        frame = frame.drop(columns="volume")
    elif defect == "nonfinite":
        frame.loc[10, "close"] = np.nan
    elif defect == "negative_volume":
        frame.loc[10, "volume"] = -1
    else:
        frame.loc[10, "high"] = frame.loc[10, "low"] - 1
    with pytest.raises((ValueError, TypeError)):
        compute_features(frame)
