"""Synthetic source-quality cases; no historical market data is opened here."""

import json

import numpy as np
import pandas as pd
import pytest

from nqpatterns.data import normalize_bars


def bars(timestamps, **overrides):
    count = len(timestamps)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
            "volume": [10.0] * count,
        }
    )
    for name, values in overrides.items():
        frame[name] = values
    return frame


def exclusions(audit):
    return {item["source_row"]: set(item["reasons"]) for item in audit["exclusions"]}


def test_normalization_preserves_input_and_audits_reordering_and_source_rows():
    raw = bars(["2024-01-02 09:31", "2024-01-02 09:30"])
    original = raw.copy(deep=True)
    normalized, audit = normalize_bars(raw)

    pd.testing.assert_frame_equal(raw, original)
    assert normalized.timestamp.tolist() == [
        pd.Timestamp("2024-01-02T14:30:00Z"),
        pd.Timestamp("2024-01-02T14:31:00Z"),
    ]
    assert normalized.source_row.tolist() == [2, 1]
    assert isinstance(normalized.index, pd.RangeIndex)
    assert audit["sort"]["reordered"] is True
    assert audit["sort"]["moved_row_ids"] == [1, 2]
    assert audit["input_rows"] == audit["output_rows"] == 2
    assert audit["excluded_rows"] == 0
    assert audit["calendar_status"] == "unverified"
    json.dumps(audit, allow_nan=False)


def test_custom_timestamp_column_csv_and_explicit_end_convention(tmp_path):
    raw = bars(["2024-07-02 09:31"]).rename(columns={"timestamp": "DateTime"})
    source = tmp_path / "synthetic.csv"
    raw.to_csv(source, index=False)
    normalized, audit = normalize_bars(
        source, timestamp_column="DateTime", timestamp_convention="end"
    )
    assert normalized.timestamp.iloc[0] == pd.Timestamp("2024-07-02T13:30:00Z")
    assert audit["timestamp_normalization"]["source_timezone"] == "America/New_York"
    assert audit["timestamp_normalization"]["end_to_start_rows"] == 1


def test_aware_offsets_are_respected_and_explicit_utc_source_is_supported():
    raw = bars(["2024-11-03T01:30:00-04:00", "2024-11-03T01:30:00-05:00"])
    normalized, _ = normalize_bars(raw)
    assert normalized.timestamp.tolist() == [
        pd.Timestamp("2024-11-03T05:30:00Z"),
        pd.Timestamp("2024-11-03T06:30:00Z"),
    ]
    utc, _ = normalize_bars(bars(["2024-01-02 14:30"]), source_tz="UTC")
    assert utc.timestamp.iloc[0] == pd.Timestamp("2024-01-02T14:30:00Z")


def test_invalid_dates_dst_and_nonminute_rows_are_quarantined_without_guesses():
    raw = bars(
        [
            "2024-02-30 09:30",
            "2024-11-03 01:30",
            "2024-03-10 02:30",
            None,
            "2024-01-02 09:30:01",
            "2024-01-02 09:31",
        ]
    )
    normalized, audit = normalize_bars(raw)
    assert normalized.source_row.tolist() == [6]
    assert exclusions(audit) == {
        1: {"invalid_timestamp"},
        2: {"ambiguous_local_time"},
        3: {"nonexistent_local_time"},
        4: {"invalid_timestamp"},
        5: {"not_minute_aligned"},
    }
    assert audit["excluded_row_ids"] == [1, 2, 3, 4, 5]
    assert audit["reason_counts"]["invalid_timestamp"] == 2
    assert audit["input_rows"] == audit["output_rows"] + audit["excluded_rows"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("open", 0, "invalid_open"),
        ("high", np.inf, "invalid_high"),
        ("low", -1, "invalid_low"),
        ("close", "bad", "invalid_close"),
        ("volume", np.nan, "invalid_volume"),
        ("volume", -1, "invalid_volume"),
        ("high", 99.75, "high_below_ohlc"),
        ("low", 100.25, "low_above_ohlc"),
    ],
)
def test_invalid_ohlcv_has_raw_row_reasons_without_imputation(field, value, reason):
    raw = bars(["2024-01-02 09:30", "2024-01-02 09:31"])
    raw[field] = raw[field].astype(object)
    raw.loc[0, field] = value
    normalized, audit = normalize_bars(raw)
    assert normalized.source_row.tolist() == [2]
    assert reason in exclusions(audit)[1]
    assert audit["excluded_rows"] == 1
    json.dumps(audit, allow_nan=False)


def test_identical_duplicates_collapse_and_every_conflicting_row_is_quarantined():
    raw = bars(
        [
            "2024-01-02 09:30", "2024-01-02 09:30",
            "2024-01-02 09:31", "2024-01-02 09:31",
            "2024-01-02 09:32",
        ],
        close=[100, 100, 100, 100.25, 100],
    )
    normalized, audit = normalize_bars(raw)
    assert normalized.source_row.tolist() == [1, 5]
    assert exclusions(audit) == {
        2: {"identical_duplicate"},
        3: {"conflicting_duplicate"},
        4: {"conflicting_duplicate"},
    }
    assert normalized.segment_id.tolist() == [0, 1]
    assert audit["input_rows"] == audit["output_rows"] + audit["excluded_rows"]


def test_invalid_duplicate_cannot_leave_a_favorable_valid_version():
    raw = bars(["2024-01-02 09:30"] * 2, volume=[10, -1])
    normalized, audit = normalize_bars(raw)
    assert normalized.empty
    assert "conflicting_duplicate" in exclusions(audit)[1]
    assert exclusions(audit)[2] == {"invalid_volume", "conflicting_duplicate"}


def test_contract_disagreement_is_a_conflicting_duplicate():
    raw = bars(["2024-01-02 09:30"] * 2, contract=["NQH4", "NQM4"])
    normalized, audit = normalize_bars(raw)
    assert normalized.empty
    assert audit["reason_counts"]["conflicting_duplicate"] == 2


def test_missing_minutes_and_contract_changes_start_segments_without_filling():
    raw = bars(
        ["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:33", "2024-01-02 09:34"],
        contract=["NQH4", "NQH4", "NQH4", "NQM4"],
        volume=[0, 10, 20, 30],
    )
    normalized, audit = normalize_bars(raw)
    assert len(normalized) == 4
    assert normalized.volume.tolist() == [0, 10, 20, 30]
    assert normalized.contract.tolist() == raw.contract.tolist()
    assert normalized.segment_id.tolist() == [0, 0, 1, 2]
    assert [(gap["source_row"], gap["missing_minutes"], gap["contract_change"])
            for gap in audit["boundaries"]] == [(3, 1, False), (4, 0, True)]
    assert audit["segment_count"] == 3
    assert audit["calendar_status"] == "unverified"


def test_unresolved_source_row_breaks_an_otherwise_consecutive_window():
    raw = bars(["2024-01-02 09:30", "unresolved", "2024-01-02 09:31"])
    normalized, audit = normalize_bars(raw)
    assert normalized.segment_id.tolist() == [0, 1]
    assert audit["boundaries"][0]["unresolved_source_rows"] == [2]


def test_tick_grid_violations_are_reported_and_retained():
    normalized, audit = normalize_bars(bars(["2024-01-02 09:30"], close=[100.1]))
    assert len(normalized) == 1
    assert audit["tick_grid_violations"] == [{"source_row": 1, "fields": ["close"]}]


def test_extreme_flags_use_previous_sixty_true_ranges_and_retain_rows():
    times = pd.date_range("2024-01-02T14:30:00Z", periods=64, freq="min")
    raw = bars(times)
    raw.loc[60, ["high", "low"]] = [125, 75]
    raw.loc[62, ["open", "high", "low", "close"]] = [150, 151, 149, 150]
    normalized, audit = normalize_bars(raw)
    prefix, _ = normalize_bars(raw.iloc[:61])
    assert len(normalized) == 64
    assert normalized.extreme_range.tolist() == [False] * 60 + [True, False, False, False]
    assert normalized.extreme_jump.iloc[62]
    assert normalized.warmup_available.tolist() == [False] * 60 + [True] * 4
    pd.testing.assert_frame_equal(normalized.iloc[:61].reset_index(drop=True), prefix)
    assert audit["warmup_unavailable_rows"] == 60
    assert audit["outliers"][0]["source_row"] == 61


def test_extreme_baseline_and_warmup_reset_at_gap():
    times = list(pd.date_range("2024-01-02T14:30:00Z", periods=60, freq="min"))
    times.append(pd.Timestamp("2024-01-02T15:31:00Z"))
    normalized, _ = normalize_bars(bars(times, high=[101] * 60 + [200]))
    assert not normalized.warmup_available.any()
    assert not normalized.extreme_range.any()


def test_empty_input_has_typed_canonical_columns_and_valid_audit():
    normalized, audit = normalize_bars(bars([]))
    assert normalized.empty
    assert str(normalized.timestamp.dtype) == "datetime64[ns, UTC]"
    assert audit["segment_count"] == 0
    assert audit["input_rows"] == audit["output_rows"] == 0
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"source_tz": None}, {"source_tz": "Unknown/Zone"},
                                     {"timestamp_convention": "unknown"}])
def test_unresolved_timestamp_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        normalize_bars(bars(["2024-01-02 09:30"]), **kwargs)


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="volume"):
        normalize_bars(bars(["2024-01-02 09:30"]).drop(columns="volume"))


def test_unverified_timestamp_interpretation_remains_an_explicit_assumption():
    _, audit = normalize_bars(bars(["2024-01-02 09:30"]))
    assert audit["timestamp_normalization"]["source_timezone_verified"] is False
    assert audit["timestamp_normalization"]["source_convention_verified"] is False
    assert audit["confirmatory_metadata_ready"] is False


def test_missing_contract_rows_are_quarantined_with_audit():
    normalized, audit = normalize_bars(
        bars(["2024-01-02 09:30", "2024-01-02 09:31"], contract=[None, "NQH4"])
    )
    assert normalized.source_row.tolist() == [2]
    assert exclusions(audit) == {1: {"invalid_contract"}}


def test_duplicate_extra_source_field_disagreement_is_not_exact_duplication():
    raw = bars(["2024-01-02 09:30"] * 2, Vwap_RTH=[100.0, 100.25])
    normalized, audit = normalize_bars(raw)
    assert normalized.empty
    assert audit["reason_counts"]["conflicting_duplicate"] == 2
    assert audit["ignored_source_columns"] == ["Vwap_RTH"]


@pytest.mark.parametrize("tick_size", [0, -1, np.nan, np.inf])
def test_invalid_tick_grid_definition_is_rejected(tick_size):
    with pytest.raises(ValueError, match="tick_size"):
        normalize_bars(bars([]), tick_size=tick_size)


def test_duplicate_source_column_names_are_rejected():
    source = bars(["2024-01-02 09:30"])
    source = pd.concat([source, source[["volume"]]], axis=1)
    with pytest.raises(ValueError, match="unique"):
        normalize_bars(source)


def test_nonstring_timezone_is_a_type_error():
    with pytest.raises(TypeError, match="source_tz"):
        normalize_bars(bars([]), source_tz=123)
