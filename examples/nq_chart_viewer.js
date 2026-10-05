"use strict";

function csvFields(line) {
  const fields = [];
  let field = "", quoted = false;
  for (let index = 0; index < line.length; index++) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {field += '"'; index++;}
      else quoted = !quoted;
    } else if (character === "," && !quoted) {fields.push(field); field = "";}
    else field += character;
  }
  fields.push(field);
  return fields;
}

function easternTimestamp(value) {
  const match = /^(\d{1,2})\/(\d{1,2})\/(\d{4}) (\d{1,2}):(\d{2})$/.exec(value);
  if (!match) throw new Error("Expected Kaggle timestamp ET: MM/DD/YYYY HH:mm");
  const [month, day, year, hour, minute] = match.slice(1).map(Number);
  if (year < 2007 || year > 2099 || month < 1 || month > 12 || hour > 23 || minute > 59) throw new Error("Invalid ET timestamp");
  const base = Date.UTC(year, month - 1, day);
  if (new Date(base).getUTCDate() !== day) throw new Error("Invalid calendar date");
  const spring = 8 + (7 - new Date(Date.UTC(year, 2, 8)).getUTCDay()) % 7;
  const autumn = 1 + (7 - new Date(Date.UTC(year, 10, 1)).getUTCDay()) % 7;
  if (month === 3 && day === spring && hour === 2) throw new Error("Nonexistent ET time");
  if (month === 11 && day === autumn && hour === 1) throw new Error("Ambiguous ET time");
  const daylight = month > 3 && month < 11 || month === 3 && (day > spring || day === spring && hour >= 3) || month === 11 && (day < autumn || day === autumn && hour < 1);
  return base + ((hour + (daylight ? 4 : 5)) * 60 + minute - 1) * 60000;
}

async function parseHistoricalCSV(text, options = {}, progress = () => {}) {
  const newline = text.indexOf("\n");
  if (newline < 0) throw new Error("CSV requires a header and data rows");
  const header = csvFields(text.slice(0, newline).replace(/^\uFEFF/, "").trim());
  const kaggle = header.includes("timestamp ET");
  const names = [kaggle ? "timestamp ET" : "timestamp", "open", "high", "low", "close", "volume"];
  const columns = names.map(name => header.indexOf(name));
  if (columns.some(index => index < 0)) throw new Error("CSV requires timestamp/open/high/low/close/volume");
  const excluded = new Set(options.excludedRows || []);
  const cutoff = Date.parse("2025-01-01T05:00:00Z");
  let capacity = options.pinned ? 706008 : 100000, packed = new Float64Array(capacity * 6);
  let count = 0, sourceRows = 0, invalid = 0, quarantined = 0, holdout = 0;
  let start = newline + 1, previous = -Infinity;
  while (start < text.length) {
    const end = text.indexOf("\n", start);
    const line = text.slice(start, end < 0 ? text.length : end).trim();
    start = end < 0 ? text.length : end + 1;
    if (!line) continue;
    sourceRows++;
    if (options.pinned && excluded.has(sourceRows)) {quarantined++; continue;}
    try {
      const values = csvFields(line);
      const stamp = values[columns[0]];
      if (!kaggle && !/(?:Z|[+-]\d{2}:?\d{2})$/.test(stamp)) throw new Error("Canonical timestamps must be timezone-aware");
      const time = kaggle ? easternTimestamp(stamp) : Date.parse(stamp);
      if (!Number.isFinite(time) || time % 60000) throw new Error("Invalid minute timestamp");
      if (time >= cutoff) {holdout++; continue;}
      const numeric = columns.slice(1).map(index => values[index] === "" ? NaN : Number(values[index]));
      const [open, high, low, close, volume] = numeric;
      if (!numeric.every(Number.isFinite) || Math.min(open, high, low, close) <= 0 || volume < 0 || high < Math.max(open, close, low) || low > Math.min(open, close, high)) throw new Error("Invalid OHLCV");
      if (![open, high, low, close].every(price => Math.abs(price * 4 - Math.round(price * 4)) < 1e-7)) throw new Error("Invalid NQ tick grid");
      if (time <= previous) throw new Error("Timestamps must be strictly increasing");
      if (count === capacity) {
        capacity *= 2;
        const grown = new Float64Array(capacity * 6); grown.set(packed); packed = grown;
      }
      packed.set([time, ...numeric], count * 6); count++; previous = time;
    } catch (error) {if (error instanceof RangeError) throw error; invalid++;}
    if (sourceRows % 25000 === 0) {progress(Math.round(start / text.length * 100)); await new Promise(resolve => setTimeout(resolve, 0));}
  }
  if (!count) throw new Error("No valid pre-2025 NQ bars found");
  if (options.pinned && count !== 706008) throw new Error(`Pinned audit replay expected 706,008 bars, received ${count}`);
  return {packed: packed.slice(0, count * 6), count, report: {sourceRows, quarantined, invalid, holdout, pinned: !!options.pinned, convention: kaggle ? "ET end stamp shifted back one minute" : "aware interval start"}};
}

class HistoricalReplay {
  constructor(dataset) {
    this.dataset = dataset;
    this.mode = "real";
    this.maxBars = dataset.count;
    this.cursor = Math.min(160, dataset.count);
    this.position = 0;
    this.realized = 0;
    this.book = {last: this.bar(this.cursor - 1).c, buy: new Map(), sell: new Map(), imbalance: 0};
    const replay = this;
    this.candles = {get length() {return replay.cursor;}, slice(start, end = replay.cursor) {
      if (start < 0) start = Math.max(0, replay.cursor + start);
      end = Math.min(end, replay.cursor);
      return Array.from({length: Math.max(0, end - start)}, (_, offset) => replay.bar(start + offset));
    }};
  }
  bar(index) {
    const offset = index * 6, values = this.dataset.packed;
    return {time: values[offset], o: values[offset + 1], h: values[offset + 2], l: values[offset + 3], c: values[offset + 4], v: values[offset + 5]};
  }
  seek(cursor) {this.cursor = Math.max(1, Math.min(this.maxBars, Math.round(cursor))); this.book.last = this.bar(this.cursor - 1).c;}
  stepCandles(count) {this.seek(this.cursor + count);}
  floating() {return 0;}
  buy() {}
  sell() {}
}

function installChartViewer() {
  const view = {end: null, hover: null, drag: null, width: 160};
  let historical = null;
  const originalReset = reset;
  const meta = JSON.parse($("historicalAudit").textContent);
  let zone, formatter;
  const format = time => {
    if (zone !== $("chartZone").value) {zone = $("chartZone").value; formatter = new Intl.DateTimeFormat("en-US", {timeZone: zone, year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23"});}
    return formatter.format(time);
  };
  const barAt = index => sim.mode === "real" ? sim.bar(index) : {...sim.candles[index], time: (sim.start || Date.UTC(2024, 0, 2, 14, 30)) + index * 60000};
  const display = price => sim.mode === "nq" || sim.mode === "real" ? price : price / 100;
  reset = function resetViewer() {
    view.end = null; view.hover = null;
    if ($("mode").value !== "real") {$("play").disabled = false; originalReset(); $("realSettings").hidden = true; $("buy").disabled = false; $("sell").disabled = false; return;}
    pause(); $("realSettings").hidden = false; $("nqSettings").hidden = true; $("nqNote").hidden = true; $("originalNote").hidden = true; $("bookPanel").hidden = true;
    $("buy").disabled = true; $("sell").disabled = true;
    if (!historical) {sim = null; $("play").disabled = true; ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight); $("price").textContent = "—"; $("count").textContent = "0"; $("internal").textContent = "No CSV loaded"; $("bookStat").textContent = "No real bars loaded"; $("candleInfo").textContent = "Load the real CSV to start replay"; $("status").textContent = "Choose the downloaded Kaggle CSV. Parsing runs in a worker; audited pre-2025 training prices only."; return;}
    $("play").disabled = false;
    sim = new HistoricalReplay(historical); rateClock.reset();
    $("seekBar").max = String(historical.count); $("seekBar").value = String(sim.cursor);
    $("status").textContent = historical.count.toLocaleString()+" real training bars loaded. "+(historical.report.pinned ? "Pinned source checksum and main audit verified." : "Custom CSV: basic validation only; exchange calendar is unverified.")+" 2025 holdout excluded.";
    draw();
  };
  $("mode").onchange = reset; $("reset").onclick = reset;
  $("visible").oninput = () => {view.width = +$("visible").value; draw();};
  $("chartZone").onchange = () => draw();
  $("latest").onclick = () => {if (!sim) return; if (sim.mode === "real") {pause(); sim.seek(sim.maxBars);} view.end = null; draw();};
  $("seekBar").oninput = () => {if (sim && sim.mode === "real") {pause(); sim.seek(+$("seekBar").value); view.end = null; draw();}};
  $("jump").onclick = () => {
    if (!sim || sim.mode !== "real") return;
    const target = Date.parse($("jumpTime").value + "Z");
    if (!Number.isFinite(target)) {showError(new Error("Enter a UTC date and time")); return;}
    let lower = 0, upper = sim.maxBars;
    while (lower < upper) {const middle = Math.floor((lower + upper) / 2); if (sim.bar(middle).time < target) lower = middle + 1; else upper = middle;}
    pause(); sim.seek(lower + 1); view.end = null; draw();
  };
  $("csvFile").onchange = async () => {
    const file = $("csvFile").files[0]; if (!file) return;
    pause(); $("csvFile").disabled = true; $("status").textContent = "Reading and verifying CSV…";
    const source = `${csvFields.toString()}\n${easternTimestamp.toString()}\n${parseHistoricalCSV.toString()}\nonmessage=async(event)=>{try{const {file,meta}=event.data;const bytes=await file.arrayBuffer();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),value=>value.toString(16).padStart(2,'0')).join('');const pinned=hash===meta.sha256;const result=await parseHistoricalCSV(new TextDecoder().decode(bytes),{pinned,excludedRows:meta.excludedRows},percent=>postMessage({progress:percent}));result.report.sha256=hash;postMessage(result,[result.packed.buffer]);}catch(error){postMessage({error:error.message})}};`;
    const url = URL.createObjectURL(new Blob([source], {type: "text/javascript"}));
    const worker = new Worker(url);
    worker.onmessage = event => {
      if (event.data.progress !== undefined) {$("status").textContent = "Validating CSV: "+event.data.progress+"%"; return;}
      worker.terminate(); URL.revokeObjectURL(url); $("csvFile").disabled = false;
      if (event.data.error) {showError(new Error(event.data.error)); return;}
      historical = event.data; reset();
    };
    worker.onerror = event => {worker.terminate(); URL.revokeObjectURL(url); $("csvFile").disabled = false; showError(new Error(event.message));};
    worker.postMessage({file, meta});
  };
  let geometry;
  draw = function drawMarketChart() {
    if (!sim) return;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    ctx.fillStyle = "#070916"; ctx.fillRect(0, 0, width, height);
    const left = 12, right = Math.max(50, width - 80), top = 40, bottom = height * 0.72, volumeTop = bottom + 25, volumeBottom = height - 35;
    const end = Math.min(sim.candles.length, view.end ?? sim.candles.length);
    const start = Math.max(0, end - view.width);
    const bars = Array.from({length: end - start}, (_, index) => barAt(start + index));
    if (!bars.length) return;
    let low = Infinity, high = -Infinity, maxVolume = 1;
    for (const bar of bars) {low = Math.min(low, display(bar.l)); high = Math.max(high, display(bar.h)); maxVolume = Math.max(maxVolume, bar.v);}
    const padding = Math.max((high - low) * 0.08, sim.mode === "nq" || sim.mode === "real" ? 0.5 : 0.01);
    low -= padding; high += padding;
    const candleWidth = (right - left) / Math.max(view.width, bars.length);
    const py = price => top + (high - price) / (high - low) * (bottom - top);
    ctx.font = "11px system-ui"; ctx.lineWidth = 1;
    for (let grid = 0; grid <= 5; grid++) {
      const price = low + (high - low) * grid / 5, vertical = py(price);
      ctx.strokeStyle = "#1d2638"; ctx.beginPath(); ctx.moveTo(left, vertical); ctx.lineTo(right, vertical); ctx.stroke();
      ctx.fillStyle = "#9aabc5"; ctx.fillText(price.toFixed(2), right + 7, vertical + 4);
    }
    ctx.fillStyle = "#9aabc5"; ctx.fillText("Volume", left, volumeTop - 7); ctx.fillText(maxVolume.toLocaleString(), right + 5, volumeTop + 8); ctx.fillText("0", right + 5, volumeBottom);
    for (let index = 0; index < bars.length; index++) {
      const bar = bars[index], horizontal = left + (index + 0.5) * candleWidth;
      const color = bar.c >= bar.o ? "#27c7a1" : "#ed6675";
      ctx.strokeStyle = color; ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(horizontal, py(display(bar.h))); ctx.lineTo(horizontal, py(display(bar.l))); ctx.stroke();
      ctx.fillRect(horizontal - candleWidth * 0.32, py(display(Math.max(bar.o, bar.c))), Math.max(1, candleWidth * 0.64), Math.max(1, py(display(Math.min(bar.o, bar.c))) - py(display(Math.max(bar.o, bar.c)))));
      const volumeHeight = bar.v / maxVolume * Math.max(10, volumeBottom - volumeTop);
      ctx.globalAlpha = 0.65; ctx.fillRect(horizontal - candleWidth * 0.32, volumeBottom - volumeHeight, Math.max(1, candleWidth * 0.64), volumeHeight); ctx.globalAlpha = 1;
    }
    const timeStep = Math.max(1, Math.ceil(bars.length / Math.max(1, Math.floor((right - left) / 145))));
    for (let index = 0; index < bars.length; index += timeStep) {ctx.fillStyle = "#9aabc5"; ctx.fillText(format(bars[index].time), left + index * candleWidth, height - 13);}
    const selected = Math.max(0, Math.min(bars.length - 1, view.hover === null ? bars.length - 1 : Math.floor((view.hover.x - left) / candleWidth)));
    const bar = bars[selected];
    const info = `${sim.mode.toUpperCase()} · ${format(bar.time)} ${$("chartZone").value} · O ${display(bar.o).toFixed(2)} H ${display(bar.h).toFixed(2)} L ${display(bar.l).toFixed(2)} C ${display(bar.c).toFixed(2)} · Vol ${bar.v.toLocaleString()}`;
    $("candleInfo").textContent = info;
    ctx.fillStyle = "#bcc9e1"; ctx.fillText(sim.mode === "real" ? "NQ · real Kaggle training prices" : "Simulation · artificial one-minute timeline", left, 21);
    const last = display(bars[bars.length - 1].c);
    ctx.fillStyle = "#29496a"; ctx.fillRect(right, py(last) - 9, 80, 19); ctx.fillStyle = "#ffffff"; ctx.fillText(last.toFixed(2), right + 5, py(last) + 4);
    if (view.hover) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "#768aa9";
      const horizontal = left + (selected + 0.5) * candleWidth;
      ctx.beginPath(); ctx.moveTo(horizontal, top); ctx.lineTo(horizontal, volumeBottom); ctx.stroke();
      if (view.hover.y >= top && view.hover.y <= bottom) {
        ctx.beginPath(); ctx.moveTo(left, view.hover.y); ctx.lineTo(right, view.hover.y); ctx.stroke();
        const price = high - (view.hover.y - top) / (bottom - top) * (high - low);
        ctx.fillStyle = "#465570"; ctx.fillRect(right, view.hover.y - 9, 80, 19); ctx.fillStyle = "white"; ctx.fillText(price.toFixed(2), right + 5, view.hover.y + 4);
      }
      ctx.setLineDash([]);
    }
    geometry = {left, right, candleWidth, end};
    $("price").textContent = display(sim.book.last).toFixed(2) + (sim.mode === "real" || sim.mode === "nq" ? " pts" : " USD");
    $("internal").textContent = sim.mode === "real" ? "Historical CSV · completed minute bars" : "Synthetic / reconstructed prices";
    $("count").textContent = sim.candles.length.toLocaleString();
    $("bookStat").textContent = sim.mode === "real" ? "of "+sim.maxBars.toLocaleString()+" loaded training bars" : "Artificial one-minute timeline";
    $("position").textContent = sim.position === 1 ? "Long" : sim.position === -1 ? "Short" : "Flat";
    $("pnl").textContent = "PnL: "+(sim.realized + sim.floating()).toFixed(2);
    if (sim.mode === "real") $("seekBar").value = String(sim.cursor);
  };
  canvas.addEventListener("pointermove", event => {
    if (!sim) return;
    const rectangle = canvas.getBoundingClientRect();
    view.hover = {x: event.clientX - rectangle.left, y: event.clientY - rectangle.top};
    if (view.drag && geometry) {view.end = Math.max(1, Math.min(sim.candles.length, Math.round(view.drag.end - (view.hover.x - view.drag.x) / geometry.candleWidth)));}
    draw();
  });
  canvas.addEventListener("pointerleave", () => {view.hover = null; if (!view.drag) draw();});
  canvas.addEventListener("pointerdown", event => {if (!sim) return; const rectangle = canvas.getBoundingClientRect(); view.drag = {x: event.clientX - rectangle.left, end: view.end ?? sim.candles.length}; canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener("pointerup", () => {view.drag = null;});
  canvas.addEventListener("wheel", event => {event.preventDefault(); view.width = Math.max(20, Math.min(2000, Math.round(view.width * (event.deltaY > 0 ? 1.2 : 0.8)))); draw();}, {passive: false});
  reset();
}

if (typeof module !== "undefined" && module.exports) module.exports = {csvFields, easternTimestamp, parseHistoricalCSV, HistoricalReplay};
else installChartViewer();
