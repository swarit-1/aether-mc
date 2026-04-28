#!/usr/bin/env node
// Optimal-strategy finder for the AETHER_CRYSTAL options challenge.
// Uses antithetic GBM Monte Carlo + analytic Black-Scholes cross-check.
// Score = mean PnL × contract size (3000), so the optimum is: take max
// volume on every product whose edge clearly clears MC noise.
'use strict';

const TRADING_DAYS_PER_YEAR = 252;
const STEPS_PER_DAY = 4;
const STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY;
const STEPS_2W = 40;
const STEPS_3W = 60;
const CONTRACT_SIZE = 3000;
const T_2W = (STEPS_2W / STEPS_PER_DAY) / TRADING_DAYS_PER_YEAR;
const T_3W = (STEPS_3W / STEPS_PER_DAY) / TRADING_DAYS_PER_YEAR;

const PRODUCTS = [
  { id: 'AC',         kind: 'underlying',  bid: 49.975, ask: 50.025, maxVol: 200 },
  { id: 'AC_50_P',    kind: 'put',  K: 50, T: STEPS_3W, Tyrs: T_3W, bid: 12,    ask: 12.05, maxVol: 50 },
  { id: 'AC_50_C',    kind: 'call', K: 50, T: STEPS_3W, Tyrs: T_3W, bid: 12,    ask: 12.05, maxVol: 50 },
  { id: 'AC_35_P',    kind: 'put',  K: 35, T: STEPS_3W, Tyrs: T_3W, bid: 4.33,  ask: 4.35,  maxVol: 50 },
  { id: 'AC_40_P',    kind: 'put',  K: 40, T: STEPS_3W, Tyrs: T_3W, bid: 6.5,   ask: 6.55,  maxVol: 50 },
  { id: 'AC_45_P',    kind: 'put',  K: 45, T: STEPS_3W, Tyrs: T_3W, bid: 9.05,  ask: 9.1,   maxVol: 50 },
  { id: 'AC_60_C',    kind: 'call', K: 60, T: STEPS_3W, Tyrs: T_3W, bid: 8.8,   ask: 8.85,  maxVol: 50 },
  { id: 'AC_50_P_2',  kind: 'put',  K: 50, T: STEPS_2W, Tyrs: T_2W, bid: 9.7,   ask: 9.75,  maxVol: 50 },
  { id: 'AC_50_C_2',  kind: 'call', K: 50, T: STEPS_2W, Tyrs: T_2W, bid: 9.7,   ask: 9.75,  maxVol: 50 },
  { id: 'AC_50_CO',   kind: 'chooser',    K: 50,         bid: 22.2, ask: 22.3,  maxVol: 50 },
  { id: 'AC_40_BP',   kind: 'binary_put', K: 40,         bid: 5,    ask: 5.1,   maxVol: 50 },
  { id: 'AC_45_KO',   kind: 'ko_put',     K: 45,         bid: 0.15, ask: 0.175, maxVol: 500 },
];

// ---------- Analytic Black-Scholes (for vanilla cross-check) ----------
function normCdf(x) {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);
  return 0.5 * (1 + sign * y);
}
function bsCall(S, K, T, sigma) {
  const sd = sigma * Math.sqrt(T);
  const d1 = (Math.log(S/K) + 0.5*sigma*sigma*T) / sd;
  return S*normCdf(d1) - K*normCdf(d1 - sd);
}
function bsPut(S, K, T, sigma) {
  const sd = sigma * Math.sqrt(T);
  const d1 = (Math.log(S/K) + 0.5*sigma*sigma*T) / sd;
  return K*normCdf(-(d1 - sd)) - S*normCdf(-d1);
}

// ---------- RNG ----------
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeGaussian(rng) {
  let cached = null;
  return function () {
    if (cached !== null) { const v = cached; cached = null; return v; }
    let u, v;
    do { u = rng(); } while (u < 1e-12);
    v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    const a = 2 * Math.PI * v;
    cached = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

// ---------- Antithetic GBM ----------
function simulatePaths({ N, S0 = 50, sigma = 2.51, mu = 0, seed = 1337 }) {
  if (N % 2 === 1) N++;
  const half = N / 2;
  const rng = mulberry32(seed);
  const gauss = makeGaussian(rng);
  const dt = 1 / STEPS_PER_YEAR;
  const drift = (mu - 0.5 * sigma * sigma) * dt;
  const diff = sigma * Math.sqrt(dt);

  const S14  = new Float64Array(N);
  const S21  = new Float64Array(N);
  const minS = new Float64Array(N);
  const Zs   = new Float64Array(STEPS_3W);

  for (let pair = 0; pair < half; pair++) {
    for (let t = 0; t < STEPS_3W; t++) Zs[t] = gauss();

    // Path A
    let Sa = S0, ma = S0, s14a = S0;
    // Path B (antithetic: replace Z_t with -Z_t)
    let Sb = S0, mb = S0, s14b = S0;
    for (let t = 1; t <= STEPS_3W; t++) {
      const z = Zs[t - 1];
      Sa *= Math.exp(drift + diff * z);
      Sb *= Math.exp(drift - diff * z);
      if (Sa < ma) ma = Sa;
      if (Sb < mb) mb = Sb;
      if (t === STEPS_2W) { s14a = Sa; s14b = Sb; }
    }
    const i = 2 * pair;
    S14[i]   = s14a; S21[i]   = Sa; minS[i]   = ma;
    S14[i+1] = s14b; S21[i+1] = Sb; minS[i+1] = mb;
  }
  return { S14, S21, minS, N };
}

function payoffsFor(paths, bpPayoff, koBarrier) {
  const { S14, S21, minS, N } = paths;
  const p = {};
  for (const prod of PRODUCTS) p[prod.id] = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const s14 = S14[i], s21 = S21[i], mn = minS[i];
    p.AC[i]        = s21;
    p.AC_50_P[i]   = Math.max(50 - s21, 0);
    p.AC_50_C[i]   = Math.max(s21 - 50, 0);
    p.AC_35_P[i]   = Math.max(35 - s21, 0);
    p.AC_40_P[i]   = Math.max(40 - s21, 0);
    p.AC_45_P[i]   = Math.max(45 - s21, 0);
    p.AC_60_C[i]   = Math.max(s21 - 60, 0);
    p.AC_50_P_2[i] = Math.max(50 - s14, 0);
    p.AC_50_C_2[i] = Math.max(s14 - 50, 0);
    p.AC_50_CO[i]  = (s14 >= 50) ? Math.max(s21 - 50, 0) : Math.max(50 - s21, 0);
    p.AC_40_BP[i]  = (s21 < 40) ? bpPayoff : 0;
    p.AC_45_KO[i]  = (mn < koBarrier) ? 0 : Math.max(45 - s21, 0);
  }
  return p;
}

function meanStd(arr) {
  const N = arr.length;
  let m = 0; for (let i = 0; i < N; i++) m += arr[i]; m /= N;
  let v = 0; for (let i = 0; i < N; i++) v += (arr[i] - m) ** 2; v /= N;
  return { mean: m, std: Math.sqrt(v) };
}

function fairValuesWithSE(payoffs, N) {
  const out = {};
  for (const prod of PRODUCTS) {
    const { mean, std } = meanStd(payoffs[prod.id]);
    out[prod.id] = { fair: mean, sd: std, se: std / Math.sqrt(N) };
  }
  return out;
}

function bsAnalytic(prod, S0 = 50, sigma = 2.51) {
  if (prod.kind === 'call') return bsCall(S0, prod.K, prod.Tyrs, sigma);
  if (prod.kind === 'put')  return bsPut (S0, prod.K, prod.Tyrs, sigma);
  if (prod.kind === 'chooser') return bsCall(S0, prod.K, T_2W, sigma) + bsPut(S0, prod.K, T_3W, sigma);
  if (prod.kind === 'underlying') return S0;
  return null;
}

function decide({ fair, se }, prod, noiseMult = 2) {
  const buyEdge  = fair - prod.ask;
  const sellEdge = prod.bid - fair;
  const noise    = noiseMult * se; // require edge > noise to act
  let action = 'SKIP', edge = 0, vol = 0, confident = false;
  if (buyEdge > 0 && buyEdge >= sellEdge) {
    action = 'BUY'; edge = buyEdge; vol = prod.maxVol;
    confident = buyEdge > noise;
  } else if (sellEdge > 0) {
    action = 'SELL'; edge = sellEdge; vol = prod.maxVol;
    confident = sellEdge > noise;
  }
  return { action, edge, vol, buyEdge, sellEdge, confident, noise };
}

function buildResult(paths, payoffs, fairs, decisions) {
  const N = paths.N;
  const totalPnL = new Float64Array(N);
  let totalEV = 0;
  for (const prod of PRODUCTS) {
    const d = decisions[prod.id];
    if (d.action === 'SKIP') continue;
    const sign = d.action === 'BUY' ? 1 : -1;
    const ref  = d.action === 'BUY' ? prod.ask : prod.bid;
    const arr = payoffs[prod.id];
    for (let i = 0; i < N; i++) totalPnL[i] += sign * (arr[i] - ref) * d.vol * CONTRACT_SIZE;
    totalEV += d.edge * d.vol * CONTRACT_SIZE;
  }
  const { mean, std } = meanStd(totalPnL);
  const sorted = Array.from(totalPnL).sort((a, b) => a - b);
  const pct = q => sorted[Math.min(N - 1, Math.max(0, Math.floor(N * q)))];
  return {
    totalEV, totalPnL,
    mean, std,
    p05: pct(0.05), p50: pct(0.5), p95: pct(0.95),
    pMin: sorted[0], pMax: sorted[N - 1],
    se100: std / Math.sqrt(100),
  };
}

// ---------- Pretty printing ----------
const fmt = (v, d = 0) => v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const pad = (s, n) => { s = String(s); return s.length >= n ? s : s + ' '.repeat(n - s.length); };
const rpad = (s, n) => { s = String(s); return s.length >= n ? s : ' '.repeat(n - s.length) + s; };

function printPricing(label, paths, payoffs, fairs) {
  console.log(`\n=== ${label} ===`);
  console.log(
    pad('Product', 12) + rpad('Bid', 8) + rpad('Ask', 8) + rpad('MC fair', 9) + rpad('±2·SE', 9) +
    rpad('BS fair', 9) + rpad('BuyEdge', 10) + rpad('SellEdge', 10) + ' ' + pad('Action', 9) + rpad('Vol', 6) + rpad('EV ($)', 14)
  );
  let totalEV = 0;
  for (const prod of PRODUCTS) {
    const f = fairs[prod.id];
    const d = decide(f, prod, 2);
    const bs = bsAnalytic(prod);
    const bsStr = bs === null ? '   —   ' : bs.toFixed(3);
    const flag = d.action === 'SKIP' ? '' : (d.confident ? '  ✓' : '  ?');
    const ev = d.edge * d.vol * CONTRACT_SIZE;
    totalEV += ev;
    console.log(
      pad(prod.id, 12) +
      rpad(prod.bid.toFixed(3), 8) + rpad(prod.ask.toFixed(3), 8) +
      rpad(f.fair.toFixed(3), 9) + rpad('±' + (2 * f.se).toFixed(3), 9) +
      rpad(bsStr, 9) +
      rpad(d.buyEdge.toFixed(3), 10) + rpad(d.sellEdge.toFixed(3), 10) + ' ' +
      pad(d.action + flag, 9) + rpad(d.vol, 6) + rpad(fmt(ev), 14)
    );
  }
  console.log('-'.repeat(108));
  console.log(`Total greedy-max-EV (✓ + ?): $${fmt(totalEV)}`);
}

// ---------- Run ----------
const N = parseInt(process.argv[2] || '2000000');
const baseBP = 10, baseKO = 35;

console.log(`AETHER_CRYSTAL strategy search`);
console.log(`σ = 251% (annual), μ = 0, S₀ = 50, contract size = 3,000`);
console.log(`MC: ${N.toLocaleString()} antithetic paths, seed = 1337`);
console.log(`Defaults for unspecified exotics: BP payout = $${baseBP}, KO barrier = $${baseKO}\n`);

const tStart = Date.now();
const paths = simulatePaths({ N, seed: 1337 });
console.log(`Simulated ${N.toLocaleString()} paths in ${((Date.now() - tStart) / 1000).toFixed(2)} s`);

const payoffs = payoffsFor(paths, baseBP, baseKO);
const fairs = fairValuesWithSE(payoffs, N);
printPricing(`Pricing & decisions  (✓ = edge clears 2·SE noise band, ? = within noise)`, paths, payoffs, fairs);

// Build greedy strategy & risk profile
const decisions = {};
for (const prod of PRODUCTS) decisions[prod.id] = decide(fairs[prod.id], prod, 2);
const greedyAll = buildResult(paths, payoffs, fairs, decisions);

// Confident-only strategy (skip non-confident)
const confDec = {};
for (const prod of PRODUCTS) {
  const d = decide(fairs[prod.id], prod, 2);
  confDec[prod.id] = d.confident ? d : { ...d, action: 'SKIP', edge: 0, vol: 0 };
}
const confidentOnly = buildResult(paths, payoffs, fairs, confDec);

console.log(`\n--- Risk profile of "greedy max-EV" portfolio ---`);
console.log(`Total EV          : $${fmt(greedyAll.totalEV)}`);
console.log(`Realized μ / σ    : $${fmt(greedyAll.mean)} / $${fmt(greedyAll.std)}`);
console.log(`5 / 50 / 95 %ile  : $${fmt(greedyAll.p05)} / $${fmt(greedyAll.p50)} / $${fmt(greedyAll.p95)}`);
console.log(`Worst / Best sim  : $${fmt(greedyAll.pMin)} / $${fmt(greedyAll.pMax)}`);
console.log(`SE of 100-sim avg : $${fmt(greedyAll.se100)}   (95% CI ≈ ±$${fmt(1.96 * greedyAll.se100)})`);

console.log(`\n--- Risk profile of "confident only" portfolio (✓ trades only) ---`);
console.log(`Total EV          : $${fmt(confidentOnly.totalEV)}`);
console.log(`Realized μ / σ    : $${fmt(confidentOnly.mean)} / $${fmt(confidentOnly.std)}`);
console.log(`5 / 50 / 95 %ile  : $${fmt(confidentOnly.p05)} / $${fmt(confidentOnly.p50)} / $${fmt(confidentOnly.p95)}`);
console.log(`Worst / Best sim  : $${fmt(confidentOnly.pMin)} / $${fmt(confidentOnly.pMax)}`);
console.log(`SE of 100-sim avg : $${fmt(confidentOnly.se100)}   (95% CI ≈ ±$${fmt(1.96 * confidentOnly.se100)})`);

// ----- Sensitivity sweeps -----
console.log(`\n=== Sensitivity: Binary-Put payout (strike = 40) ===`);
console.log(pad('Payout', 9) + rpad('Fair', 9) + rpad('±2·SE', 9) + rpad('BuyEdge', 10) + rpad('SellEdge', 10) + ' ' + pad('Action', 9) + rpad('EV (leg)', 16) + rpad('Total EV', 16));
for (const bp of [4, 6, 8, 9, 10, 10.5, 11, 12, 14, 16, 20]) {
  const po = payoffsFor(paths, bp, baseKO);
  const fr = fairValuesWithSE(po, N);
  let tot = 0;
  for (const prod of PRODUCTS) {
    const d = decide(fr[prod.id], prod, 2);
    tot += d.edge * d.vol * CONTRACT_SIZE;
  }
  const f = fr.AC_40_BP;
  const prod = PRODUCTS.find(x => x.id === 'AC_40_BP');
  const d = decide(f, prod, 2);
  console.log(
    pad('$' + bp, 9) + rpad(f.fair.toFixed(3), 9) + rpad('±' + (2 * f.se).toFixed(3), 9) +
    rpad(d.buyEdge.toFixed(3), 10) + rpad(d.sellEdge.toFixed(3), 10) + ' ' +
    pad(d.action + (d.confident ? ' ✓' : (d.action !== 'SKIP' ? ' ?' : '')), 9) +
    rpad('$' + fmt(d.edge * d.vol * CONTRACT_SIZE), 16) + rpad('$' + fmt(tot), 16)
  );
}

console.log(`\n=== Sensitivity: KO barrier (strike = 45 put) ===`);
console.log(pad('Barrier', 10) + rpad('Fair', 10) + rpad('±2·SE', 10) + rpad('BuyEdge', 10) + rpad('SellEdge', 10) + ' ' + pad('Action', 9) + rpad('EV (leg)', 16) + rpad('Total EV', 16));
for (const ko of [10, 15, 20, 25, 30, 32.5, 35, 37.5, 40, 42, 44]) {
  const po = payoffsFor(paths, baseBP, ko);
  const fr = fairValuesWithSE(po, N);
  let tot = 0;
  for (const prod of PRODUCTS) {
    const d = decide(fr[prod.id], prod, 2);
    tot += d.edge * d.vol * CONTRACT_SIZE;
  }
  const f = fr.AC_45_KO;
  const prod = PRODUCTS.find(x => x.id === 'AC_45_KO');
  const d = decide(f, prod, 2);
  console.log(
    pad('$' + ko, 10) + rpad(f.fair.toFixed(4), 10) + rpad('±' + (2 * f.se).toFixed(4), 10) +
    rpad(d.buyEdge.toFixed(4), 10) + rpad(d.sellEdge.toFixed(4), 10) + ' ' +
    pad(d.action + (d.confident ? ' ✓' : (d.action !== 'SKIP' ? ' ?' : '')), 9) +
    rpad('$' + fmt(d.edge * d.vol * CONTRACT_SIZE), 16) + rpad('$' + fmt(tot), 16)
  );
}

console.log(`\nTotal time: ${((Date.now() - tStart) / 1000).toFixed(2)} s`);
