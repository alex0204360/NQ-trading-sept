"""Publish aggregate-only artifacts; raw bars, predictions and fills remain private."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from tools.mnq_oracle_prepare import sha256
from tools.mnq_oracle_research import save_json


def money(value):
    return "—" if value is None else f"${value:,.2f}"


def chart(series):
    values = [0.] + list(series.values())
    minimum, maximum = min(values), max(values)
    span = max(1., maximum-minimum)
    points = " ".join(f"{30+index*640/max(1,len(values)-1):.2f},{170-(value-minimum)*140/span:.2f}" for index, value in enumerate(values))
    return f'<svg viewBox="0 0 700 200" role="img" aria-label="Daily closing liquidatable equity"><polyline points="{points}" fill="none" stroke="#087f8c" stroke-width="3"/><text x="30" y="195">Daily closing equity · {money(values[-1])}</text><text x="30" y="18">Range {money(minimum)} to {money(maximum)}</text></svg>'


def render(report, development):
    rows, details = [], []
    for name, costs in report["backtests"].items():
        if name.startswith("NONCAUSAL"):
            continue
        base = costs["base"]
        rows.append(f'<tr><th>{html.escape(name)}</th><td>{base["trades"]}</td><td>{money(base["gross_price_move"])}</td><td>{money(base["commissions"]+base["slippage"])}</td><td>{money(base["net"])}</td><td>{money(costs["stress6"]["net"])}</td><td>{money(costs["stress10"]["net"])}</td><td>{money(base["max_drawdown_bar_close"])}</td></tr>')
        days = "".join(f'<tr><td>{html.escape(key)}</td><td>{value["trades"]}</td><td>{money(value["net"])}</td></tr>' for key, value in base["day_session"].items())
        details.append(f'<details><summary>{html.escape(name)}: equity and session results</summary>{chart(base["daily_equity"])}<table><tr><th>Day / session</th><th>Trades</th><th>Net</th></tr>{days}</table><p>Ambiguous stop-first fills: {base["ambiguous_trades"]}; unresolved gap exits: {base["gap_unresolved_trades"]}. Drawdown is bar-close, not intrabar.</p></details>')
    candidate_rows = "".join(f'<tr><td>{candidate["model"]}</td><td>{candidate["threshold"]}</td><td>{candidate["rule"]}</td><td>{candidate["metrics"]["trades"]}</td><td>{money(candidate["metrics"]["net"])}</td></tr>' for candidate in development["candidates"])
    classifications = []
    for name, metrics in report["classification"].items():
        classifications.append(f'<details><summary>{name}: imitation quality, not trading accuracy</summary><p>Directional event recall: {metrics["directional_event_recall"]:.1%}; Brier score: {metrics["multiclass_brier"]:.4f}. Short AP: {metrics["directions"]["-1"]["average_precision"]:.4f}; long AP: {metrics["directions"]["1"]["average_precision"]:.4f}.</p><p>Confusion matrix: rows actual, columns predicted; order short / wait / long.</p><pre>{html.escape(json.dumps(metrics["confusion"], indent=2))}</pre><p>Full prevalence, calibration bins and precision–recall data are in results.json.</p></details>')
    noncausal = "".join(f'<li>{html.escape(name)}: base {money(costs["base"]["net"])}. Uses future information; never eligible for selection.</li>' for name, costs in report["backtests"].items() if name.startswith("NONCAUSAL"))
    limits = "".join(f'<li>{html.escape(item)}</li>' for item in report["limitations"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MNQ oracle — causal reality check</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#18283b;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:36px 24px}}header{{background:#132c43;color:white;border-radius:18px;padding:30px}}h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:10px 0}}h2{{margin-top:0}}section{{background:white;border:1px solid #dbe4ed;border-radius:14px;padding:24px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}}.card{{padding:18px;background:#e9f4f2;border-radius:10px}}.card strong{{display:block;font-size:24px}}.warning{{background:#fff3d9;border-left:5px solid #b17a13;padding:16px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;text-align:right;font-variant-numeric:tabular-nums}}th,td{{padding:12px;border-bottom:1px solid #dbe4ed;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{background:#eef3f8}}details{{border:1px solid #dbe4ed;padding:15px;margin:12px 0;border-radius:8px}}summary{{cursor:pointer;font-weight:650}}svg{{width:100%;max-width:700px}}svg text{{font:13px system-ui;fill:#345}}pre{{overflow:auto}}a{{color:#087f8c}}.eyebrow{{letter-spacing:.12em;font-size:12px;text-transform:uppercase}}@media(max-width:700px){{main{{padding:14px}}section{{padding:16px}}.cards{{grid-template-columns:1fr}}}}
    </style></head><body><main><header><div class="eyebrow">ChartChico research · MNQ · 2026 only</div><h1>Can a causal student learn<br>the oracle's decisions?</h1><p>Completed baseline experiment. Historical teacher labels are supervision—not knowledge of the future.</p></header>
    <div class="cards"><div class="card">Policy selected before final evaluation<strong>No trade</strong></div><div class="card">Frozen development comparisons<strong>3 models / 18 configurations</strong></div><div class="card">Verified tradable edge<strong>Not established</strong></div></div>
    <section><h2>The answer so far</h2><p>Development rejected trading in favor of abstaining. A forest challenger made a small base-cost final-period gain, but lost under higher costs and was not the preselected policy. It cannot be promoted after inspecting this result.</p><p class="warning">The oracle-style benchmarks benefit from future information. Their profits are not executable evidence. Timestamp and contract-roll provenance remain unresolved.</p></section>
    <section><h2>Final-period costed results</h2><p>October 8, 22:00 UTC → October 12, 14:30 UTC. One MNQ contract; $2 per point. Includes a partial Monday. $4 / $6 / $10 modeled round-trip costs; stress fills are resimulated.</p><div class="scroll"><table><thead><tr><th>Policy</th><th>Trades</th><th>Gross move</th><th>Friction</th><th>Net $4</th><th>Net $6</th><th>Net $10</th><th>Base max DD</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>Gross minus commissions and nominal slippage equals net. No-trade is a risk-control baseline, not a profitable strategy.</p>{''.join(details)}</section>
    <section><h2>Why the hindsight result is different</h2><ul>{noncausal}</ul><p>Same-feed historical entry/exit-close reference and unmatched pairs are recorded separately in results.json. It is not a mathematical profit upper bound.</p></section>
    <section><h2>All development attempts—including failures</h2><p>{development["train_rows"]:,} training rows; {development["development_rows"]:,} development rows. Thresholds and exits selected here, not on final outcomes.</p><div class="scroll"><table><tr><th>Model</th><th>Threshold</th><th>Exit</th><th>Trades</th><th>Development net</th></tr>{candidate_rows}</table></div></section>
    <section><h2>Did the models imitate the labels?</h2>{''.join(classifications)}</section>
    <section><h2>Data and limitations</h2><p>28,312 source bars → 28,311 retained; 866 matched oracle entries → 813 feature-ready examples. Nine missing teacher timestamps, 75 gaps, 26 zero-volume bars and one partial candle are audited. Original NQ research and its sealed 2025 holdout are unchanged.</p><ul>{limits}</ul></section>
    <section><h2>Smallest next experiment</h2><p>First verify the teacher's exact non-lookahead settings, interval timing and contract mapping against a licensed contract-specific feed. Then collect a new, untouched period and preregister the same frozen students for forward paper evaluation. Do not select the forest retrospectively or keep tuning this final sample.</p><p>Reproduction instructions and failed checks are in MNQ_ORACLE_RESULTS.md. Full per-trade records and predictions are private; this page contains aggregate performance only.</p></section>
    <footer>Research only · no live trading · no synthetic data · no animation · draft evidence, not financial advice</footer></main></body></html>'''


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/"manifest.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("Incomplete research run")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha256(source/name) != expected:
            raise ValueError("Artifact checksum mismatch: "+name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "development.json", "selection.json", "audit.json", "environment.json"):
        save_json(destination/name, json.loads((source/name).read_text()))
    save_json(destination/"private_artifact_hashes.json", manifest)
    report = json.loads((source/"results.json").read_text())
    development = json.loads((source/"development.json").read_text())
    (destination/"dashboard.html").write_text(render(report, development), encoding="utf-8")
    return {"published_aggregate_directory": str(destination), "private_raw_files_copied": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination)))
