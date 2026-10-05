"use strict";

(function expose(root) {
  const SCENARIOS = ["noise", "trend_up", "trend_down", "range", "compression", "breakout", "failed_breakout", "shock"];

  function finite(value, name, minimum, maximum) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      throw new Error(`${name} must be finite and between ${minimum} and ${maximum}`);
    }
    return value;
  }

  class SeededRandom {
    constructor(seed) {
      finite(seed, "seed", 1, 4294967295);
      if (!Number.isInteger(seed)) throw new Error("seed must be an integer");
      this.state = seed >>> 0;
    }
    uniform() {
      let state = this.state;
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      this.state = state >>> 0;
      return (this.state + 0.5) / 4294967296;
    }
    normal() {
      return Math.sqrt(-2 * Math.log(this.uniform())) * Math.cos(2 * Math.PI * this.uniform());
    }
    student() {
      const numerator = this.normal();
      let denominator = 0;
      for (let count = 0; count < 5; count++) denominator += this.normal() ** 2;
      return numerator / Math.sqrt(denominator / 5) * Math.sqrt(3 / 5);
    }
  }

  class RateClock {
    constructor() {
      this.reset();
    }
    reset() {
      this.previous = null;
      this.pending = 0;
    }
    due(timestamp, rate, playing) {
      finite(rate, "candles per second", 0.1, 500);
      if (!playing) {
        this.reset();
        return 0;
      }
      if (this.previous === null) {
        this.previous = timestamp;
        return 0;
      }
      const elapsed = Math.min(100, Math.max(0, timestamp - this.previous));
      this.previous = timestamp;
      this.pending += elapsed / 1000 * rate;
      const due = Math.floor(this.pending + 1e-9);
      this.pending -= due;
      return due;
    }
  }

  function validateProfile(profile) {
    if (profile.schema_version !== "nq-synthetic-profile-v1" || profile.calibrated !== true) {
      throw new Error("A calibrated nq-synthetic-profile-v1 profile is required");
    }
    finite(profile.sigma_log_return, "sigma_log_return", 1e-8, 0.05);
    finite(profile.return_ar1, "return_ar1", -0.5, 0.5);
    finite(profile.volatility_persistence, "volatility_persistence", 0, 0.999);
    finite(profile.upper_wick_fraction_mean, "upper wick", 0, 0.05);
    finite(profile.lower_wick_fraction_mean, "lower wick", 0, 0.05);
    finite(profile.log_volume_mean, "log volume", 0, 30);
    finite(profile.log_volume_std, "log volume deviation", 0, 10);
    if (profile.tick_size !== 0.25 || profile.point_value !== 20) throw new Error("Expected NQ contract settings");
  }

  class NQSim {
    constructor(profile, options = {}) {
      validateProfile(profile);
      this.profile = JSON.parse(JSON.stringify(profile));
      this.rng = new SeededRandom(options.seed ?? 12345);
      this.scenario = options.scenario ?? "mixed";
      if (this.scenario !== "mixed" && !SCENARIOS.includes(this.scenario)) throw new Error("Unknown scenario");
      this.lastPrice = this.tick(finite(options.startPrice ?? 24000, "start price", 100, 1000000));
      this.start = Date.parse(options.startTimestamp ?? "2024-01-02T14:30:00Z");
      if (!Number.isFinite(this.start) || this.start % 60000) throw new Error("Start timestamp must be minute-aligned");
      this.mode = "nq";
      this.candles = [];
      this.candleIndex = 0;
      this.previousReturn = 0;
      this.logVolatility = 0;
      this.episodeAge = 0;
      this.episodeLength = 0;
      this.regime = "noise";
      this.anchor = this.lastPrice;
      this.position = 0;
      this.entry = 0;
      this.realized = 0;
      this.roundTripCommission = finite(options.roundTripCommission ?? 4.5, "round-trip commission", 0, 1000);
      this.slippageTicks = finite(options.slippageTicks ?? 1, "slippage per side", 0, 100);
      this.book = {last: this.lastPrice, buy: new Map(), sell: new Map(), imbalance: 0};
    }
    tick(price) {
      return Math.max(0.25, Math.round(price * 4) / 4);
    }
    stepCandles(count) {
      if (!Number.isInteger(count) || count < 0 || count > 100000) throw new Error("Invalid batch size");
      for (let ordinal = 0; ordinal < count; ordinal++) this.step();
    }
    step() {
      if (this.episodeAge >= this.episodeLength) {
        this.regime = this.scenario === "mixed" ? SCENARIOS[Math.floor(this.rng.uniform() * SCENARIOS.length)] : this.scenario;
        this.episodeLength = 80 + Math.floor(this.rng.uniform() * 140);
        this.episodeAge = 0;
        this.anchor = this.lastPrice;
        this.direction = this.rng.uniform() < 0.5 ? -1 : 1;
      }
      const progress = this.episodeAge / this.episodeLength;
      const sigma = this.profile.sigma_log_return;
      const persistence = this.profile.volatility_persistence;
      this.logVolatility = persistence * this.logVolatility + 0.10 * this.rng.normal();
      let scale = Math.exp(Math.max(-1.5, Math.min(1.5, this.logVolatility)));
      let drift = 0;
      if (this.regime === "trend_up") drift = sigma * 0.20;
      if (this.regime === "trend_down") drift = -sigma * 0.20;
      if (this.regime === "range") drift = 0.035 * Math.log(this.anchor / this.lastPrice);
      if (this.regime === "compression") scale *= 0.35;
      if (this.regime === "breakout" || this.regime === "failed_breakout") {
        if (progress < 0.4) {
          scale *= 0.35;
          drift = 0.035 * Math.log(this.anchor / this.lastPrice);
        } else {
          drift = this.direction * sigma * 0.40;
          if (this.regime === "failed_breakout" && progress > 0.65) {
            drift = 0.06 * Math.log(this.anchor / this.lastPrice);
          }
        }
      }
      if (this.regime === "shock" && progress > 0.40 && progress < 0.55) scale *= 3.5;
      const innovation = sigma * scale * this.rng.student();
      const rawReturn = drift + this.profile.return_ar1 * this.previousReturn + innovation;
      const logReturn = Math.max(-0.03, Math.min(0.03, rawReturn));
      const open = this.lastPrice;
      const close = this.tick(open * Math.exp(logReturn));
      const upper = -Math.log(this.rng.uniform()) * this.profile.upper_wick_fraction_mean * open * scale;
      const lower = -Math.log(this.rng.uniform()) * this.profile.lower_wick_fraction_mean * open * scale;
      const high = this.tick(Math.max(open, close) + upper);
      const low = this.tick(Math.max(0.25, Math.min(open, close) - lower));
      const volume = Math.max(0, Math.round(Math.expm1(Math.min(20,
        this.profile.log_volume_mean + this.profile.log_volume_std * this.rng.normal() + 0.35 * Math.log(scale)))));
      const candle = {o: open, h: high, l: low, c: close, v: volume};
      this.candles.push(candle);
      this.lastPrice = close;
      this.book.last = close;
      this.previousReturn = logReturn;
      this.candleIndex++;
      this.episodeAge++;
      return candle;
    }
    trade(direction) {
      const slippage = this.slippageTicks * 0.25;
      if (!this.position) {
        this.position = direction;
        this.entry = this.lastPrice + direction * slippage;
        this.realized -= this.roundTripCommission / 2;
      } else if (this.position !== direction) {
        const exit = this.lastPrice - this.position * slippage;
        this.realized += (exit - this.entry) * this.position * 20 - this.roundTripCommission / 2;
        this.position = 0;
      }
    }
    buy() { this.trade(1); }
    sell() { this.trade(-1); }
    floating() {
      return this.position ? (this.lastPrice - this.position * this.slippageTicks * 0.25 - this.entry) * this.position * 20 - this.roundTripCommission / 2 : 0;
    }
    rows(sequenceId = "browser", split = "preview") {
      return this.candles.map((bar, index) => ({
        timestamp: new Date(this.start + index * 60000).toISOString(),
        open: bar.o, high: bar.h, low: bar.l, close: bar.c, volume: bar.v,
        completed: true, synthetic: true, sequence_id: sequenceId, split,
      }));
    }
  }

  function mean(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
  }
  function examples(rows, options = {}) {
    const horizon = options.horizon ?? 10;
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > 45) throw new Error("Horizon must be 1–45 bars");
    const commission = finite(options.roundTripCommission ?? 4.5, "round-trip commission", 0, 1000);
    const slippage = finite(options.slippageTicks ?? 1, "slippage ticks per side", 0, 100);
    const costPoints = commission / 20 + slippage * 0.25 * 2;
    const samples = [];
    for (let index = 60; index + horizon < rows.length; index++) {
      const history = rows.slice(index - 60, index + 1);
      const current = rows[index];
      const future = rows.slice(index + 1, index + horizon + 1);
      if (history.concat(future).some((row, ordinal, window) => row.sequence_id !== current.sequence_id ||
          row.split !== current.split || (ordinal > 0 && Date.parse(row.timestamp) - Date.parse(window[ordinal - 1].timestamp) !== 60000))) continue;
      const ranges = history.slice(-14).map((bar, ordinal) => {
        const prior = history[history.length - 15 + ordinal].close;
        return Math.max(bar.high - bar.low, Math.abs(bar.high - prior), Math.abs(bar.low - prior));
      });
      const atr = Math.max(0.25, mean(ranges));
      const closes20 = history.slice(-20).map(bar => bar.close);
      const closeMean = mean(closes20);
      const closeStd = Math.sqrt(mean(closes20.map(close => (close - closeMean) ** 2)));
      const prior20 = history.slice(-21, -1);
      const volumes = history.slice(-20).map(bar => bar.volume);
      const volumeMean = mean(volumes);
      const volumeStd = Math.sqrt(mean(volumes.map(volume => (volume - volumeMean) ** 2)));
      const range = Math.max(0.25, current.high - current.low);
      const movement = future[future.length - 1].close - future[0].open;
      const longNet = (movement - costPoints) * 20;
      const shortNet = (-movement - costPoints) * 20;
      samples.push({
        sequence_id: current.sequence_id, split: current.split, timestamp: current.timestamp,
        features: {
          body_fraction: (current.close - current.open) / range,
          upper_wick_fraction: (current.high - Math.max(current.open, current.close)) / range,
          lower_wick_fraction: (Math.min(current.open, current.close) - current.low) / range,
          return_1_atr: (current.close - history[59].close) / atr,
          return_5_atr: (current.close - history[55].close) / atr,
          return_20_atr: (current.close - history[40].close) / atr,
          range_to_atr14: range / atr,
          atr14_fraction: atr / current.close,
          price_z20: closeStd ? (current.close - closeMean) / closeStd : 0,
          volume_z20: volumeStd ? (current.volume - volumeMean) / volumeStd : 0,
          breakout_up20_atr: (current.close - Math.max(...prior20.map(bar => bar.high))) / atr,
          breakout_down20_atr: (Math.min(...prior20.map(bar => bar.low)) - current.close) / atr,
        },
        targets: {
          horizon_bars: horizon, entry_timestamp: future[0].timestamp,
          exit_timestamp: future[future.length - 1].timestamp,
          forward_points: movement, long_net_dollars: longNet, short_net_dollars: shortNet,
          action: longNet > 0 ? "long" : shortNet > 0 ? "short" : "hold",
        },
      });
    }
    return samples;
  }

  const api = {SCENARIOS, SeededRandom, RateClock, NQSim, examples, validateProfile};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NQSynthetic = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
