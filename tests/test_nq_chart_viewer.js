"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const {csvFields, easternTimestamp, parseHistoricalCSV, HistoricalReplay} = require("../examples/nq_chart_viewer.js");

test("ET end stamps convert to UTC start stamps across DST", () => {
  assert.equal(new Date(easternTimestamp("12/26/2022 18:01")).toISOString(), "2022-12-26T23:00:00.000Z");
  assert.equal(new Date(easternTimestamp("7/10/2024 09:31")).toISOString(), "2024-07-10T13:30:00.000Z");
  assert.throws(() => easternTimestamp("3/10/2024 02:30"));
  assert.throws(() => easternTimestamp("11/3/2024 01:30"));
  assert.throws(() => easternTimestamp("2/30/2024 09:30"));
  assert.deepEqual(csvFields('"a,b",1,"c""d"'), ["a,b", "1", 'c"d']);
});

test("CSV rejects malformed geometry, off-tick prices, duplicates and naive times", async () => {
  const csv = "timestamp,open,high,low,close,volume\n" + [
    "2024-01-02T14:30:00Z,16000,16001,15999,16000.25,20",
    "2024-01-02T14:31:00Z,16000,15999,15998,16000,20",
    "2024-01-02T14:32:00Z,16000.1,16001,15999,16000,20",
    "2024-01-02T14:30:00Z,16000,16001,15999,16000,20",
    "2024-01-02T14:33:00,16000,16001,15999,16000,20",
    "2025-01-02T14:30:00Z,16000,16001,15999,16000,20",
  ].join("\n");
  const data = await parseHistoricalCSV(csv);
  assert.equal(data.count, 1);
  assert.equal(data.report.invalid, 4);
  assert.equal(data.report.holdout, 1);
});

test("historical replay clamps seeking and never slices unrevealed candles", () => {
  const packed = new Float64Array(300 * 6);
  for (let index = 0; index < 300; index++) packed.set([index * 60000, 16000, 16001, 15999, 16000.25, index], index * 6);
  const replay = new HistoricalReplay({packed, count: 300});
  assert.equal(replay.candles.length, 160);
  assert.equal(replay.candles.slice(150, 300).length, 10);
  replay.seek(250); replay.stepCandles(100);
  assert.equal(replay.cursor, 300);
  assert.equal(replay.candles.slice(-1)[0].v, 299);
  replay.seek(-100); assert.equal(replay.cursor, 1);
});

test("standalone HTML embeds identical viewer and audited exclusion metadata", () => {
  const html = fs.readFileSync(path.join(__dirname, "../examples/orderbook_market_simulator.html"), "utf8");
  assert.equal(html.match(/<script id="chartViewer">([\s\S]*?)<\/script>/)[1].trim(), fs.readFileSync(path.join(__dirname, "../examples/nq_chart_viewer.js"), "utf8").trim());
  const meta = JSON.parse(html.match(/<script id="historicalAudit" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "../reports/data_quality.json")));
  assert.deepEqual(meta.excludedRows, [...new Set([...audit.mechanical.excluded_row_ids, ...audit.calendar.outside_scheduled_source_rows])].sort((first, second) => first - second));
});

test("full checksum-pinned Kaggle CSV reproduces main's 706,008 training bars", {skip: !process.env.NQ_RAW_CSV}, async () => {
  const bytes = fs.readFileSync(process.env.NQ_RAW_CSV);
  assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"), "1577e60a7feab411e49da7a56c7052a64738cd1757cfd60aa11fd783ff43b60b");
  const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "../reports/data_quality.json")));
  const data = await parseHistoricalCSV(bytes.toString("utf8"), {pinned: true, excludedRows: [...audit.mechanical.excluded_row_ids, ...audit.calendar.outside_scheduled_source_rows]});
  assert.equal(data.count, 706008);
  assert.equal(data.report.sourceRows, 1048575);
  assert.equal(data.report.invalid, 0);
  assert.equal(data.report.holdout, 330516);
  assert.equal(data.report.quarantined, 12051);
  const replay = new HistoricalReplay(data);
  assert.equal(new Date(replay.bar(0).time).toISOString(), "2022-12-26T23:00:00.000Z");
  assert.equal(new Date(replay.bar(data.count - 1).time).toISOString(), "2024-12-31T21:59:00.000Z");
  assert.deepEqual(replay.bar(0), {time: Date.parse("2022-12-26T23:00:00Z"), o: 13759, h: 13794.75, l: 13759, c: 13788.5, v: 540});
});
