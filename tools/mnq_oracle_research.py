"""Frozen, preliminary MNQ oracle imitation; private row-level artifacts stay local."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from tools.mnq_oracle_prepare import FEATURE_NAMES, load_candles, causal_features, add_teacher_labels, sha256
from tools.mnq_oracle_execution import alma_signals, replay

PART_HASHES = ["3430116d80a7e4e2b9e2b2ed34315dc5255d2433620b737c96055556312176f2", "a7a64421e57fa8142a7c07f4415c719981c7d6020d50c068749806a0aff3eb34", "514249caa15cc110b289c5c48b2c8afa44c1d602d2174e5b7c393bf67a4c4a4b", "a218f1228416cd191d38d1841e66a75b2eb677f87448f42b8f9a120388816933", "b2777b42a4ef96037c868e4664e9142aac8ecb010abf4e4e65d2b59b6a28bff6", "bcf54e4b620996c1f4c28cda92f442f473573e042319c9eb9e1e810dc15a1114"]
TEACHER_HASH = "f5831d15be6ea0a16b8d3ffe8eb0792ec92d89ca9423f379f8cb2223404c75d5"
CLASSES = np.array([-1, 0, 1])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(directory):
    directory = Path(directory)
    parts = [directory/f"MNQ_Yahoo_1m_20260913_20261012_part{number:02d}.csv" for number in range(1, 7)]
    for file, expected in zip(parts, PART_HASHES):
        if sha256(file) != expected:
            raise ValueError("Candle hash mismatch: "+file.name)
    teacher_path = directory/"mnq_oracle_teacher_labels.csv"
    if sha256(teacher_path) != TEACHER_HASH:
        raise ValueError("Teacher hash mismatch")
    bars, audit = load_candles(parts)
    if not (bars.timestamp.dt.year == 2026).all() or audit["duplicate_rows_identical_removed"]:
        raise ValueError("Only nonoverlapping 2026 MNQ source rows allowed")
    teacher = pd.read_csv(teacher_path)
    features, label_audit = add_teacher_labels(causal_features(bars), teacher)
    audit.update(label_audit)
    expected_counts = {"raw_rows": 28312, "valid_minute_rows": 28311, "nonminute_rows_rejected_as_partial": 1, "interruptions_excluding_first_row": 75, "teacher_entry_timestamp_matches": 866, "labels_on_feature_ready_rows": 813}
    for field, expected in expected_counts.items():
        if audit[field] != expected:
            raise ValueError("Audit count mismatch: "+field)
    for column in ("entry", "exit"):
        teacher[column+"_utc"] = pd.to_datetime(teacher[column+"_time_display_no_timezone"]).dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    audit["input_files"] = [{"name": file.name, "sha256": expected} for file, expected in zip(parts, PART_HASHES)]
    audit["teacher_sha256"] = TEACHER_HASH
    audit["matched_price_equals_close"] = int((features.entry_price.to_numpy() == bars.close.to_numpy()).sum())
    audit["independent_timestamp_verification"] = False
    audit["independent_contract_roll_verification"] = False
    audit["sealed_nq_2025_files_opened"] = 0
    gaps = bars.timestamp.diff().dt.total_seconds().div(60)
    audit["gap_review"] = {"over_one_minute": int((gaps > 1).sum()), "over_one_day": int((gaps > 1440).sum()), "at_most_one_day": int(((gaps > 1) & (gaps <= 1440)).sum()), "status": "No independent exchange calendar or contract-specific feed reconciliation; durations alone do not establish whether candles are missing."}
    return bars, features, teacher, audit


def split_masks(features, teacher, plan):
    decision_time = features.timestamp_utc + pd.Timedelta(minutes=1)
    event_end = features.timestamp_utc.map(teacher.set_index("entry_utc").exit_utc)
    ready = features.feature_ready
    start, validation, final, end = [pd.Timestamp(plan[key]) for key in ("train_start", "validation_start", "final_start", "coverage_end_exclusive")]
    purge = pd.Timedelta(minutes=plan["purge_minutes"])
    train = ready & (features.timestamp_utc >= start) & (decision_time < validation-purge) & (event_end.isna() | (event_end < validation-purge))
    development = ready & (features.timestamp_utc >= validation) & (decision_time < final-purge) & (event_end.isna() | (event_end < final-purge))
    heldout = ready & (features.timestamp_utc >= final) & (features.timestamp_utc < end)
    return {"train": train.to_numpy(), "development": development.to_numpy(), "final": heldout.to_numpy()}


def make_models(seed):
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=seed)),
        "forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=50, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
        "histogram": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1, random_state=seed, early_stopping=False),
    }


def signals_from_probabilities(probabilities, threshold):
    directions = np.where(probabilities[:, 2] >= probabilities[:, 0], 1, -1)
    confidence = np.maximum(probabilities[:, 0], probabilities[:, 2])
    return np.where((confidence >= threshold) & (confidence > probabilities[:, 1]), directions, 0)


def classification_metrics(labels, probabilities, predictions):
    matrix = confusion_matrix(labels, predictions, labels=CLASSES)
    onehot = labels[:, None] == CLASSES[None, :]
    directional = labels != 0
    output = {"rows": len(labels), "prevalence": {str(label): int(np.sum(labels == label)) for label in CLASSES}, "class_order": [-1, 0, 1], "confusion": matrix.tolist(), "directional_event_recall": float(np.mean(predictions[directional] == labels[directional])) if directional.any() else None, "multiclass_brier": float(np.mean(np.sum((probabilities-onehot)**2, axis=1))), "directions": {}}
    for column, label in ((0, -1), (2, 1)):
        target = (labels == label).astype(int)
        precision, recall, thresholds = precision_recall_curve(target, probabilities[:, column])
        picked = np.unique(np.linspace(0, len(precision)-1, min(51, len(precision))).astype(int))
        bins = []
        for lower in np.arange(0., 1., .1):
            mask = (probabilities[:, column] >= lower) & (probabilities[:, column] < lower+.1 if lower < .9 else probabilities[:, column] <= 1)
            bins.append({"lower": float(lower), "count": int(mask.sum()), "mean_probability": float(probabilities[mask, column].mean()) if mask.any() else None, "event_rate": float(target[mask].mean()) if mask.any() else None})
        output["directions"][str(label)] = {"average_precision": float(average_precision_score(target, probabilities[:, column])) if target.any() else None, "precision_recall_curve_sample": [{"precision": float(precision[index]), "recall": float(recall[index]), "threshold": float(thresholds[index]) if index < len(thresholds) else None} for index in picked], "calibration": bins}
    return output


def source_identity(plan_path):
    paths = [Path(__file__), Path(__file__).with_name("mnq_oracle_execution.py"), Path(__file__).with_name("mnq_oracle_prepare.py"), Path(plan_path)]
    return {path.name: sha256(path) for path in paths}


def model_probabilities(model, features, mask):
    if list(model.classes_) != [-1, 0, 1]:
        raise ValueError("Training sample must contain all three classes")
    probabilities = np.zeros((len(features), 3))
    probabilities[:, 1] = 1
    probabilities[mask] = model.predict_proba(features.loc[mask, FEATURE_NAMES].to_numpy(float))
    return probabilities


def develop(data, out, plan_path):
    out = Path(out)
    if (out/"selection.json").exists():
        raise ValueError("Development is already frozen; choose a separate output for a reproducibility rerun")
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_path).read_text())
    save_json(out/"frozen_plan.json", plan)
    save_json(out/"source_identity.json", source_identity(plan_path))
    bars, features, teacher, audit = load_inputs(data)
    masks = split_masks(features, teacher, plan)
    save_json(out/"audit.json", audit)
    save_json(out/"environment.json", {"python": platform.python_version(), **{package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}})
    labels = features.teacher_direction.to_numpy(int)
    atr = features.atr14.to_numpy(float)
    train, dev = masks["train"], masks["development"]
    development_end = (pd.Timestamp(plan["final_start"])-pd.Timedelta(minutes=plan["purge_minutes"])).isoformat()
    candidates, metrics, failures = [], {}, []
    for name, model in make_models(plan["seed"]).items():
        try:
            with threadpool_limits(limits=1):
                arguments = {"sample_weight": compute_sample_weight("balanced", labels[train])} if name == "histogram" else {}
                model.fit(features.loc[train, FEATURE_NAMES].to_numpy(float), labels[train], **arguments)
                probabilities = model_probabilities(model, features, dev)
            joblib.dump(model, out/(name+".joblib"))
            for threshold in plan["thresholds"]:
                signals = signals_from_probabilities(probabilities, threshold)
                for rule in plan["exit_rules"]:
                    summary, _, _ = replay(bars, signals, atr, plan["validation_start"], development_end, rule, plan["costs"]["base"])
                    candidates.append({"model": name, "threshold": threshold, "rule": rule, "eligible": summary["trades"] >= 20, "metrics": summary})
            metrics[name] = classification_metrics(labels[dev], probabilities[dev], signals_from_probabilities(probabilities[dev], .5))
        except Exception as error:
            failures.append({"model": name, "error_type": type(error).__name__, "message": str(error)})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.sort(key=lambda candidate: (-candidate["metrics"]["net"], candidate["metrics"]["trades"], candidate["model"], candidate["threshold"], candidate["rule"]))
    best = eligible[0] if eligible else None
    selected = best if best and best["metrics"]["net"] > 0 else {"model": "no_trade", "threshold": None, "rule": "time10"}
    per_model = {}
    for name in metrics:
        options = [candidate for candidate in eligible if candidate["model"] == name]
        per_model[name] = options[0] if options else {"model": name, "threshold": .7, "rule": "time10", "eligible": False}
    save_json(out/"development.json", {"train_rows": int(train.sum()), "train_class_counts": {str(label): int(np.sum(labels[train] == label)) for label in CLASSES}, "development_rows": int(dev.sum()), "candidates": candidates, "classification": metrics, "failures": failures})
    selection = {"selected": selected, "per_model": per_model, "best_trading_attempt": best, "model_hashes": {name: sha256(out/(name+".joblib")) for name in metrics}, "plan_sha256": sha256(plan_path), "source_identity": source_identity(plan_path), "final_metrics_opened": False}
    save_json(out/"selection.json", selection)
    return {"stage": "development_frozen", "train_rows": int(train.sum()), "development_rows": int(dev.sum()), "selected": selected["model"], "candidate_count": len(candidates), "failures": failures}


def evaluate(data, out, plan_path):
    out = Path(out)
    if (out/"final_opened.json").exists():
        raise ValueError("Final segment already opened; do not retune or overwrite this run")
    selection = json.loads((out/"selection.json").read_text())
    if selection["source_identity"] != source_identity(plan_path) or selection["plan_sha256"] != sha256(plan_path):
        raise ValueError("Frozen source/plan changed")
    plan = json.loads(Path(plan_path).read_text())
    bars, features, teacher, audit = load_inputs(data)
    mask = split_masks(features, teacher, plan)["final"]
    labels = features.teacher_direction.to_numpy(int)
    atr = features.atr14.to_numpy(float)
    save_json(out/"final_opened.json", {"selection_sha256": sha256(out/"selection.json"), "plan_sha256": sha256(plan_path), "policy": "No subsequent tuning on this final segment"})
    ready = features.feature_ready.to_numpy()
    signals = {"no_trade": (np.zeros(len(bars), int), "time10"), "momentum": (np.where(ready & (features.r8_atr.abs().to_numpy() >= 1), np.sign(features.r8_atr.fillna(0).to_numpy()), 0).astype(int), "atr1_target1.5_time30"), "causal_alma_surrogate": (np.where(ready, alma_signals(bars), 0), "atr1_target1.5_time30"), "NONCAUSAL_alma_lookahead_surrogate": (np.where(ready, alma_signals(bars, lookahead=True), 0), "atr1_target1.5_time30"), "NONCAUSAL_teacher_timing_fixed_exit": (np.where(ready, labels, 0), "atr1_target1.5_time30")}
    classification = {}
    for name, candidate in selection["per_model"].items():
        model_path = out/(name+".joblib")
        if sha256(model_path) != selection["model_hashes"][name]:
            raise ValueError("Model changed after selection")
        model = joblib.load(model_path)
        with threadpool_limits(limits=1):
            probabilities = model_probabilities(model, features, mask)
        decisions = signals_from_probabilities(probabilities, candidate["threshold"])
        signals[name] = (decisions, candidate["rule"])
        classification[name] = classification_metrics(labels[mask], probabilities[mask], decisions[mask])
        pd.DataFrame({"timestamp": features.timestamp_utc[mask], "teacher_target": labels[mask], "p_short": probabilities[mask, 0], "p_wait": probabilities[mask, 1], "p_long": probabilities[mask, 2], "decision": decisions[mask]}).to_csv(out/(name+"_predictions_PRIVATE.csv"), index=False)
    results = {}
    for name, (decisions, rule) in signals.items():
        results[name] = {}
        for cost_name, costs in plan["costs"].items():
            summary, trades, events = replay(bars, decisions, atr, plan["final_start"], plan["coverage_end_exclusive"], rule, costs)
            results[name][cost_name] = summary
            pd.DataFrame(trades).to_csv(out/(name+"_"+cost_name+"_trades_PRIVATE.csv"), index=False)
            save_json(out/(name+"_"+cost_name+"_events_PRIVATE.json"), events)
    oracle = teacher.loc[(teacher.entry_utc >= pd.Timestamp(plan["final_start"])) & (teacher.entry_utc < pd.Timestamp(plan["coverage_end_exclusive"])) & (teacher.exit_utc < pd.Timestamp(plan["coverage_end_exclusive"]))]
    oracle_points = (oracle.exit_price-oracle.entry_price)*oracle.direction_code
    teacher_reference = {"trades": len(oracle), "gross_reported_price_reference": float((oracle_points*2).sum()), "win_rate_before_costs": float((oracle_points > 0).mean()), "warning": "Hindsight trade prices, potentially different feed; NOT executable, NOT a mathematical upper bound, and NOT the same-feed replay above."}
    closes = bars.set_index("timestamp").close
    matched_entry = oracle.entry_utc.map(closes)
    matched_exit = oracle.exit_utc.map(closes)
    matched = matched_entry.notna() & matched_exit.notna()
    oracle_feed_pnl = (matched_exit[matched]-matched_entry[matched])*oracle.loc[matched, "direction_code"]*2
    teacher_reference["same_feed_hindsight_close_reference"] = {"matched_pairs": int(matched.sum()), "unmatched_pairs_excluded": int((~matched).sum()), "gross": float(oracle_feed_pnl.sum()), "posthoc_cost_sensitivity": {name: float(oracle_feed_pnl.sum()-matched.sum()*(2*cost["commission_per_side"]+cost["slippage_ticks_per_side"])) for name, cost in plan["costs"].items()}, "warning": "Both entry and exit timestamps chosen with future information; observed minute closes, not certified fills. Descriptive oracle-only comparison, not a causal backtest."}
    report = {"status": "preliminary_no_verified_edge", "plan": plan, "selection": selection, "audit": audit, "final_feature_rows": int(mask.sum()), "classification": classification, "backtests": results, "teacher_reference": teacher_reference, "selected_policy": selection["selected"]["model"], "limitations": ["Only roughly two weeks of exported oracle labels; final segment is approximately two sessions including partial Monday.", "Timestamp convention, TradingView versus Yahoo feed alignment, and continuous-contract roll mapping lack independent verification.", "ALMA offset/sigma/UTC anchor are prespecified assumptions, not verified exact proprietary Pine settings.", "Oracle labels are future-informed supervision. Same-row features are available one minute after the assumed teacher timestamp; results are delayed imitation, not foresight.", "Negative labels assume the exported teacher trade coverage is complete; missing teacher bars and feature warm-up remove events.", "Bar-close drawdown misses intrabar excursions; OHLC cannot certify fills, queue priority or bid/ask spreads. Cost scenarios are assumptions, not broker quotes.", "Gap liquidations at the next observed open are explicitly unresolved exposure, not reconstructed tradable paths.", "No independent market-source verification or live deployment approval; original NQ 2025 remains unopened."]}
    save_json(out/"results.json", report)
    files = sorted(file for file in out.iterdir() if file.is_file() and file.name != "manifest.json")
    save_json(out/"manifest.json", {"files_sha256": {file.name: sha256(file) for file in files}, "source_identity": source_identity(plan_path), "complete": True})
    return {"stage": "final_complete", "selected_policy": report["selected_policy"], "results": {name: {cost: {key: value[key] for key in ("trades", "net", "max_drawdown_bar_close")} for cost, value in costs.items()} for name, costs in results.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "final"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("docs/MNQ_ORACLE_EXPERIMENT_PLAN.json"))
    args = parser.parse_args()
    result = develop(args.data, args.out, args.plan) if args.stage == "development" else evaluate(args.data, args.out, args.plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
