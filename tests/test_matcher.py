"""Feed-independent matching must reproduce the historical detection schedule."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from nqpatterns.matcher import LiveMatcher


def library(status="confirmed"):
    return {
        "schema_version": 1,
        "instrument": "NQ",
        "bar_interval_minutes": 1,
        "timestamp_convention": "interval_start",
        "tick_size": 0.25,
        "patterns": [
            {
                "pattern_id": "synthetic-flat-price-state",
                "family": "price_state",
                "signature": {
                    "version": 1,
                    "kind": "rules",
                    "conditions": [{"feature": "price_z20", "op": "ge", "value": 0.0}],
                },
                "direction": "up",
                "magnitude_points": 4.0,
                "horizon_bars": 12,
                "status": status,
                "historical_stats": {"sample_size": 1234, "hit_rate": 0.61},
            }
        ],
    }


def bars(n=160, start="2023-01-03T14:30:00Z"):
    return [
        {
            "timestamp": stamp.isoformat(),
            "open": 100.0,
            "high": 100.25,
            "low": 99.75,
            "close": 100.0,
            "volume": 100.0,
        }
        for stamp in pd.date_range(start, periods=n, freq="min")
    ]


def emitted_times(results):
    return [pd.Timestamp(result["detection_timestamp"]) for result in results]


def test_empty_confirmed_library_accepts_standard_bars_without_matches():
    frozen = library()
    frozen["patterns"] = []
    assert LiveMatcher(frozen).match_bars(bars(65)) == []


def test_first_detection_requires_sixty_prior_completed_bars_and_45_minute_cooldown():
    source = bars()
    matcher = LiveMatcher(library())
    assert matcher.match_bars(source[:60]) == []
    results = matcher.match_bars(source[60:])
    assert emitted_times(results) == [pd.Timestamp(source[i]["timestamp"]) for i in (60, 105, 150)]
    assert all(item["historical_stats"]["sample_size"] == 1234 for item in results)
    assert all(item["status"] == "confirmed" for item in results)
    assert all(item["historical_population"] is True for item in results)
    assert all(
        pd.Timestamp(item["available_at"])
        == pd.Timestamp(item["detection_timestamp"]) + pd.Timedelta(minutes=1)
        for item in results
    )
    json.dumps(results, allow_nan=False)


def test_batch_and_streaming_are_identical_for_iterators_and_chunked_calls():
    source = bars(125)
    streamed = LiveMatcher(library())
    by_update = [item for bar in source for item in streamed.update(bar)]
    assert LiveMatcher(library()).match_bars(iter(source)) == by_update
    chunked = LiveMatcher(library())
    assert chunked.match_bars(source[:77]) + chunked.match_bars(source[77:]) == by_update


def test_matches_share_the_causal_historical_feature_and_signature_code():
    from nqpatterns.candidates import match_candidate
    from nqpatterns.features import compute_features

    source = bars(135)
    for i, bar in enumerate(source):
        bar["close"] = 100 + round(np.sin(i / 7) * 12) / 4
        bar["open"] = bar["close"]
        bar["high"] = bar["close"] + 0.25
        bar["low"] = bar["close"] - 0.25
    frame = pd.DataFrame(source)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    raw = match_candidate(compute_features(frame), library()["patterns"][0]["signature"])
    expected = []
    last = None
    for index in np.flatnonzero(raw):
        stamp = frame["timestamp"].iloc[index]
        if last is None or stamp - last >= pd.Timedelta(minutes=45):
            expected.append(stamp)
            last = stamp
    assert emitted_times(LiveMatcher(library()).match_bars(source)) == expected


@pytest.mark.parametrize("boundary", ["gap", "contract"])
def test_gap_or_contract_change_restarts_warmup_and_does_not_compress_elapsed_time(boundary):
    source = bars(135)
    if boundary == "gap":
        for bar in source[70:]:
            bar["timestamp"] = (pd.Timestamp(bar["timestamp"]) + pd.Timedelta(minutes=3)).isoformat()
    else:
        for i, bar in enumerate(source):
            bar["contract"] = "NQH23" if i < 70 else "NQM23"
    results = LiveMatcher(library()).match_bars(source)
    assert emitted_times(results) == [pd.Timestamp(source[i]["timestamp"]) for i in (60, 130)]


def test_provisional_matches_are_opt_in_and_suppressed_matches_are_marked_diagnostic():
    source = bars(63)
    assert LiveMatcher(library("provisional")).match_bars(source) == []
    results = LiveMatcher(library("provisional"), diagnostic=True).match_bars(source)
    assert [item["status"] for item in results] == ["provisional", "suppressed", "suppressed"]
    assert [item["historical_population"] for item in results] == [True, False, False]
    assert all(item["pattern_status"] == "provisional" for item in results)
    assert all(item["suppression_reason"] == "45_minute_cooldown" for item in results[1:])


def test_two_outcome_variants_share_one_signature_detection_schedule():
    frozen = library()
    second = deepcopy(frozen["patterns"][0])
    second.update(pattern_id="same-signature-down", direction="down", horizon_bars=3)
    frozen["patterns"].append(second)
    results = LiveMatcher(frozen).match_bars(bars(108))
    assert len(results) == 4
    assert len(set(emitted_times(results))) == 2
    assert {result["pattern_id"] for result in results} == {
        "synthetic-flat-price-state", "same-signature-down"
    }


@pytest.mark.parametrize(
    "defect",
    ["naive", "off_minute", "duplicate", "descending", "missing", "nan", "ohlc",
     "negative_volume", "nonpositive", "off_tick", "incomplete", "invalid_contract"],
)
def test_invalid_bar_raises_without_consuming_timestamp_warmup_or_cooldown(defect):
    source = bars(62)
    matcher = LiveMatcher(library())
    matcher.match_bars(source[:60])
    invalid = deepcopy(source[60])
    if defect == "naive":
        invalid["timestamp"] = "2023-01-03T15:30:00"
    elif defect == "off_minute":
        invalid["timestamp"] = "2023-01-03T15:30:01Z"
    elif defect == "duplicate":
        invalid["timestamp"] = source[59]["timestamp"]
    elif defect == "descending":
        invalid["timestamp"] = source[10]["timestamp"]
    elif defect == "missing":
        del invalid["high"]
    elif defect == "nan":
        invalid["high"] = float("nan")
    elif defect == "ohlc":
        invalid["high"] = 99.0
    elif defect == "negative_volume":
        invalid["volume"] = -1
    elif defect == "nonpositive":
        invalid.update(open=0, low=0, close=0)
    elif defect == "off_tick":
        invalid["close"] = 100.1
    elif defect == "incomplete":
        invalid["completed"] = False
    else:
        invalid["contract"] = []
    with pytest.raises((ValueError, TypeError)):
        matcher.update(invalid)
    result = matcher.update(source[60])
    assert len(result) == 1
    assert emitted_times(result) == [pd.Timestamp(source[60]["timestamp"])]


def test_library_json_roundtrip_and_caller_mutation_cannot_change_frozen_model(tmp_path):
    frozen = library()
    path = tmp_path / "library.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    matcher = LiveMatcher(frozen)
    expected = LiveMatcher(path).match_bars(bars(108))
    frozen["patterns"][0]["historical_stats"]["sample_size"] = -1
    frozen["patterns"][0]["signature"]["conditions"][0]["value"] = 999
    first = matcher.match_bars(bars(61))
    assert first == expected[:1]
    first[0]["historical_stats"]["sample_size"] = -2
    assert matcher.match_bars(bars(108)[61:]) == expected[1:]


@pytest.mark.parametrize(
    ("key", "value"),
    [("schema_version", 2), ("instrument", "MNQ"), ("bar_interval_minutes", 15),
     ("timestamp_convention", "interval_end"), ("tick_size", 0.5)],
)
def test_unverified_library_metadata_cannot_masquerade_as_this_nq_model(key, value):
    frozen = library()
    frozen[key] = value
    with pytest.raises(ValueError):
        LiveMatcher(frozen)


@pytest.mark.parametrize(
    ("key", "value"),
    [("horizon_bars", 0), ("horizon_bars", 46), ("horizon_bars", 2.5),
     ("magnitude_points", 0), ("magnitude_points", 0.1), ("direction", "sideways"),
     ("status", "winning"), ("signature", {"kind": "future"}),
     ("historical_stats", {"hit_rate": float("nan")})],
)
def test_invalid_or_out_of_scope_library_entry_is_rejected_even_if_inactive(key, value):
    frozen = library("provisional")
    frozen["patterns"][0][key] = value
    with pytest.raises((ValueError, TypeError)):
        LiveMatcher(frozen)


def test_duplicate_pattern_ids_are_rejected():
    frozen = library()
    frozen["patterns"].append(deepcopy(frozen["patterns"][0]))
    with pytest.raises(ValueError, match="(?i)duplicate"):
        LiveMatcher(frozen)


def test_feature_buffer_is_bounded_to_sixty_prior_bars(monkeypatch):
    import nqpatterns.matcher as module

    actual = module.compute_features
    observed_lengths = []

    def inspect_frame(frame):
        observed_lengths.append(len(frame))
        return actual(frame)

    monkeypatch.setattr(module, "compute_features", inspect_frame)
    LiveMatcher(library()).match_bars(bars(120))
    assert observed_lengths
    assert max(observed_lengths) <= 61


def test_timezone_offsets_normalize_to_utc_without_changing_the_matches():
    utc = bars(108)
    eastern = deepcopy(utc)
    for bar in eastern:
        bar["timestamp"] = pd.Timestamp(bar["timestamp"]).tz_convert("America/New_York").isoformat()
    assert LiveMatcher(library()).match_bars(eastern) == LiveMatcher(library()).match_bars(utc)
