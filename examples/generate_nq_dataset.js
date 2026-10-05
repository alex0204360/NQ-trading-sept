"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const {NQSim, SCENARIOS, examples, validateProfile} = require("./nq_synthetic.js");

function main() {
  const argumentsByName = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    if (!key.startsWith("--") || process.argv[index + 1] === undefined) throw new Error("Use --name value arguments");
    if (Object.hasOwn(argumentsByName, key)) throw new Error(`Duplicate argument ${key}`);
    argumentsByName[key] = process.argv[index + 1];
  }
  const accepted = ["--profile", "--output", "--sequences", "--bars", "--seed", "--horizon", "--start-price", "--commission", "--slippage"];
  for (const key of Object.keys(argumentsByName)) if (!accepted.includes(key)) throw new Error(`Unknown argument ${key}`);
  const profilePath = argumentsByName["--profile"] ?? path.join(__dirname, "nq_calibration.json");
  const output = argumentsByName["--output"];
  if (!output) throw new Error("--output must name a new dataset directory");
  if (fs.existsSync(output)) throw new Error("Refusing to overwrite an existing dataset");
  const profileBytes = fs.readFileSync(profilePath);
  const profile = JSON.parse(profileBytes);
  validateProfile(profile);
  const sequences = Number(argumentsByName["--sequences"] ?? 100);
  const bars = Number(argumentsByName["--bars"] ?? 1000);
  const seed = Number(argumentsByName["--seed"] ?? 10001);
  const horizon = Number(argumentsByName["--horizon"] ?? 10);
  const startPrice = Number(argumentsByName["--start-price"] ?? 24000);
  const roundTripCommission = Number(argumentsByName["--commission"] ?? 4.5);
  const slippageTicks = Number(argumentsByName["--slippage"] ?? 1);
  if (!Number.isInteger(sequences) || sequences < 20 || sequences > 10000) throw new Error("Use 20–10,000 sequences");
  if (!Number.isInteger(horizon) || horizon < 1 || horizon > 45) throw new Error("Use a 1–45 bar horizon");
  if (!Number.isInteger(bars) || bars < 61 + horizon || bars > 10000) throw new Error("Invalid bars per sequence");
  if (!Number.isInteger(seed) || seed < 1 || seed + sequences > 4294967295) throw new Error("Invalid seed range");
  new NQSim(profile, {seed, startPrice, roundTripCommission, slippageTicks});
  const trainEnd = Math.floor(sequences * 0.70);
  const validationEnd = Math.floor(sequences * 0.85);
  fs.mkdirSync(output, {recursive: true});
  const streams = {};
  const counts = {};
  for (const split of ["train", "validation", "test"]) {
    streams[split] = {
      bars: fs.openSync(path.join(output, `${split}_bars.jsonl`), "wx"),
      examples: fs.openSync(path.join(output, `${split}_examples.jsonl`), "wx"),
    };
    counts[split] = {sequences: 0, bars: 0, examples: 0, actions: {long: 0, short: 0, hold: 0}};
  }
  const audit = fs.openSync(path.join(output, "scenario_audit.jsonl"), "wx");
  const records = (descriptor, rows) => fs.writeSync(descriptor, rows.map(row => JSON.stringify(row)).join("\n") + "\n");
  try {
    for (let ordinal = 0; ordinal < sequences; ordinal++) {
      const split = ordinal < trainEnd ? "train" : ordinal < validationEnd ? "validation" : "test";
      const sequenceId = `nq-${seed + ordinal}`;
      const scenario = ordinal % 3 === 0 ? "mixed" : SCENARIOS[ordinal % SCENARIOS.length];
      const simulation = new NQSim(profile, {seed: seed + ordinal, scenario, startPrice, roundTripCommission, slippageTicks});
      simulation.stepCandles(bars);
      const rows = simulation.rows(sequenceId, split);
      const samples = examples(rows, {horizon, roundTripCommission, slippageTicks});
      records(streams[split].bars, rows);
      records(streams[split].examples, samples);
      records(audit, [{sequence_id: sequenceId, split, seed: seed + ordinal, scenario}]);
      counts[split].sequences++;
      counts[split].bars += rows.length;
      counts[split].examples += samples.length;
      for (const sample of samples) counts[split].actions[sample.targets.action]++;
    }
  } finally {
    for (const stream of Object.values(streams)) {
      fs.closeSync(stream.bars);
      fs.closeSync(stream.examples);
    }
    fs.closeSync(audit);
  }
  const hashes = {};
  for (const filename of fs.readdirSync(output)) {
    hashes[filename] = crypto.createHash("sha256").update(fs.readFileSync(path.join(output, filename))).digest("hex");
  }
  const manifest = {
    schema_version: "nq-synthetic-dataset-v1", synthetic: true,
    profile_sha256: crypto.createHash("sha256").update(profileBytes).digest("hex"),
    engine_sha256: crypto.createHash("sha256").update(fs.readFileSync(path.join(__dirname, "nq_synthetic.js"))).digest("hex"),
    profile, seed, sequences, bars_per_sequence: bars, start_price: startPrice,
    horizon_bars: horizon, round_trip_commission_dollars: roundTripCommission,
    slippage_ticks_per_side: slippageTicks, counts, file_sha256: hashes,
    feature_policy: "Only the features object is model input. Targets are future labels. Scenario audit is diagnostic only.",
    split_policy: "Whole independent seeded sequences assigned 70/15/15; never split overlapping windows across datasets.",
    execution_assumption: "Completed bar t; enter open[t+1], exit close[t+horizon]; one NQ contract, two-sided slippage plus round-trip commission.",
    timestamp_policy: "Each independent sequence uses an artificial contiguous minute grid; not an exchange calendar simulation.",
    validation_policy: "Synthetic test paths test simulator generalization only. Evaluate on untouched real NQ data before claiming an edge.",
  };
  fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", {flag: "wx"});
  console.log(JSON.stringify({output, counts}));
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
