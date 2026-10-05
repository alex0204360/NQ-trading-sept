"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {NQSim, SCENARIOS, RateClock, examples} = require("../examples/nq_synthetic.js");

const profile = {
  schema_version: "nq-synthetic-profile-v1", calibrated: true,
  sigma_log_return: 0.0003, return_ar1: 0.02, volatility_persistence: 0.9,
  upper_wick_fraction_mean: 0.0001, lower_wick_fraction_mean: 0.0001,
  log_volume_mean: 6, log_volume_std: 0.8, tick_size: 0.25, point_value: 20,
};

test("playback uses elapsed seconds independent of refresh rate", () => {
  for (const refresh of [30, 60, 144]) {
    for (const rate of [1, 10, 100]) {
      const clock = new RateClock();
      let total = clock.due(0, rate, true);
      for (let frame = 1; frame <= refresh * 10; frame++) total += clock.due(frame * 1000 / refresh, rate, true);
      assert.equal(total, rate * 10);
    }
  }
});

test("pause and hidden-tab delays cannot create a fast catchup burst", () => {
  const clock = new RateClock();
  clock.due(0, 10, true);
  assert.equal(clock.due(50, 10, true), 0);
  assert.equal(clock.due(60, 10, false), 0);
  assert.equal(clock.due(100000, 10, true), 0);
  assert.equal(clock.due(100100, 10, true), 1);
  assert.equal(clock.due(200000, 10, true), 1);
});

for (const scenario of [...SCENARIOS, "mixed"]) {
  test(`${scenario} is seeded, prefix invariant, and respects NQ geometry/ticks`, () => {
    const full = new NQSim(profile, {seed: 41, scenario});
    const prefix = new NQSim(profile, {seed: 41, scenario});
    full.stepCandles(1000);
    prefix.stepCandles(80);
    assert.deepEqual(prefix.candles, full.candles.slice(0, 80));
    for (const bar of full.candles) {
      assert.ok(Object.values(bar).every(Number.isFinite));
      for (const value of [bar.o, bar.h, bar.l, bar.c]) {
        assert.ok(value > 0);
        assert.equal(value * 4, Math.round(value * 4));
      }
      assert.ok(bar.h >= Math.max(bar.o, bar.c));
      assert.ok(bar.l <= Math.min(bar.o, bar.c));
      assert.ok(Number.isInteger(bar.v) && bar.v >= 0);
    }
  });
}

test("features are unchanged by future prices and labels enter at the next open", () => {
  const simulation = new NQSim(profile, {seed: 13});
  simulation.stepCandles(120);
  const rows = simulation.rows("sample", "train");
  const initial = examples(rows)[0];
  const changed = structuredClone(rows);
  for (let index = 61; index < changed.length; index++) {
    for (const key of ["open", "high", "low", "close"]) changed[index][key] += 1000;
    changed[index].volume *= 3;
  }
  assert.deepEqual(examples(changed)[0].features, initial.features);
  const difference = rows[70].close - rows[61].open;
  assert.equal(initial.targets.forward_points, difference);
  assert.equal(initial.targets.long_net_dollars, difference * 20 - 14.5);
  assert.equal(initial.targets.entry_timestamp, rows[61].timestamp);
  assert.equal(initial.targets.exit_timestamp, rows[70].timestamp);
  assert.ok(!("scenario" in initial.features));
  assert.ok(!("seed" in initial.features));
  assert.equal(examples(rows.slice(0, 80))[0].targets.forward_points, difference);
});

test("samples crossing sequence, split, or minute gaps are excluded", () => {
  const simulation = new NQSim(profile, {seed: 13});
  simulation.stepCandles(120);
  for (const defect of ["sequence_id", "split", "timestamp"]) {
    const rows = simulation.rows("sample", "train");
    rows[70][defect] = defect === "timestamp" ? "2024-02-01T14:30:00Z" : "different";
    assert.equal(examples(rows).filter(sample => sample.timestamp === rows[60].timestamp).length, 0);
  }
});

test("flat features stay finite and transaction costs label hold", () => {
  const simulation = new NQSim(profile, {seed: 13});
  simulation.stepCandles(100);
  const rows = simulation.rows("flat", "train");
  for (const row of rows) Object.assign(row, {open: 24000, high: 24000, low: 24000, close: 24000, volume: 0});
  const sample = examples(rows)[0];
  assert.ok(Object.values(sample.features).every(Number.isFinite));
  assert.equal(sample.targets.action, "hold");
  assert.equal(sample.targets.long_net_dollars, -14.5);
  assert.equal(sample.targets.short_net_dollars, -14.5);
});

test("manual NQ trades include both slippage and round-trip commission", () => {
  const simulation = new NQSim(profile, {seed: 13});
  simulation.buy();
  assert.equal(simulation.realized + simulation.floating(), -14.5);
  simulation.sell();
  assert.equal(simulation.realized, -14.5);
  assert.equal(simulation.position, 0);
});

test("invalid seeds, calibration, costs, and horizons fail explicitly", () => {
  assert.throws(() => new NQSim({...profile, calibrated: false}));
  assert.throws(() => new NQSim(profile, {seed: 0}));
  assert.throws(() => new NQSim(profile, {scenario: "magic"}));
  assert.throws(() => new NQSim(profile, {roundTripCommission: -1}));
  assert.throws(() => examples([], {horizon: 46}));
  assert.throws(() => examples([], {slippageTicks: NaN}));
});

test("self-contained HTML embeds the same engine/profile and changes playback in all modes", () => {
  const html = fs.readFileSync(path.join(__dirname, "../examples/orderbook_market_simulator.html"), "utf8");
  const engine = html.match(/<script id="nqEngine">([\s\S]*?)<\/script>/)[1];
  assert.equal(engine.trim(), fs.readFileSync(path.join(__dirname, "../examples/nq_synthetic.js"), "utf8").trim());
  const profileText = html.match(/<script id="nqProfile" type="application\/json">([\s\S]*?)<\/script>/)[1];
  assert.deepEqual(JSON.parse(profileText), JSON.parse(fs.readFileSync(path.join(__dirname, "../examples/nq_calibration.json"))));
  const nodes = {};
  const defaults = {mode: "nq", seed: "12345", startPrice: "24000", commission: "4.5", slippage: "1", horizon: "10", scenario: "mixed", speed: "0", visible: "160"};
  const context = new Proxy({}, {get: () => () => {}});
  const document = {
    hidden: false, addEventListener() {},
    getElementById(id) {
      return nodes[id] ??= {value: defaults[id] ?? "", textContent: id === "nqProfile" ? profileText : "", getContext: () => context, getBoundingClientRect: () => ({width: 1000, height: 800}), clientWidth: 1000, clientHeight: 800};
    },
  };
  let callback;
  const browser = vm.createContext({document, devicePixelRatio: 1, addEventListener() {}, requestAnimationFrame(next) {callback = next;}, setTimeout, Blob, URL});
  for (const script of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
    if (!script[1].includes('type="application/json"')) vm.runInContext(script[2], browser);
  }
  let timestamp = 0;
  for (const mode of ["nq", "exact", "corrected"]) {
    nodes.mode.value = mode;
    nodes.speed.value = "0";
    nodes.speed.oninput();
    nodes.mode.onchange();
    callback(timestamp);
    for (let frame = 1; frame <= 600; frame++) callback(timestamp + frame * 1000 / 60);
    timestamp += 10000;
    assert.equal(vm.runInContext("sim.candles.length", browser), 10);
    nodes.speed.value = "1";
    nodes.speed.oninput();
    callback(timestamp);
    for (let frame = 1; frame <= 600; frame++) callback(timestamp + frame * 1000 / 60);
    timestamp += 10000;
    assert.equal(vm.runInContext("sim.candles.length", browser), 110);
  }
});
