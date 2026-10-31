// Browser inference for the exported gradient-boosted trees (see src/train.py).
const $ = (id) => document.getElementById(id);
const usd = (n) => "$" + Math.round(n).toLocaleString("en-US");
const short = (n) => (n >= 1000 ? `$${Math.round(n / 1000)}k` : usd(n));

const [web, explore] = await Promise.all([fetch("model.json").then((r) => r.json()), fetch("explore.json").then((r) => r.json())]);

function encode(p) {
  const v = web.countries.map((c) => (p.country === c ? 1 : 0));
  v.push(...web.roles.map((r) => (p.role === r ? 1 : 0)));
  v.push(...web.remote.map((r) => (p.remote === r ? 1 : 0)));
  v.push(web.education.indexOf(p.education), p.years_pro);
  return v;
}

function run(model, x) {
  let total = model.init;
  for (const t of model.trees) {
    let n = 0;
    while (t.l[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    total += model.lr * t.v[n];
  }
  return Math.exp(total);
}

function predict(p) {
  const x = encode(p);
  const mid = run(web.models.median, x);
  return { mid, lo: Math.min(run(web.models.p25, x), mid), hi: Math.max(run(web.models.p75, x), mid) };
}
window.predictSalary = predict; // handy for checking parity from the console

/* ---------- form ---------- */
function fill(select, options, selected) {
  for (const o of options) select.add(new Option(o, o, false, o === selected));
}
fill($("country"), web.countries, "United States");
fill($("role"), web.roles, "Back-end developer");
fill($("education"), web.education, "Bachelor's");
fill($("remote"), web.remote, "Hybrid");

const profile = () => ({
  country: $("country").value, role: $("role").value, education: $("education").value,
  remote: $("remote").value, years_pro: Number($("years").value),
});

const SCALE_MAX = 300000;
function render() {
  const p = profile();
  $("years-out").textContent = p.years_pro;
  const { mid, lo, hi } = predict(p);
  $("mid").textContent = usd(mid);
  $("range").textContent = `${usd(lo)} to ${usd(hi)}`;
  const pct = (v) => `${Math.min(100, (v / SCALE_MAX) * 100)}%`;
  $("band-bar").style.left = pct(lo);
  $("band-bar").style.width = `calc(${pct(hi)} - ${pct(lo)})`;
  $("band-dot").style.left = `${((mid - lo) / Math.max(1, hi - lo)) * 100}%`;
  drawCurve(p);
}

/* ---------- experience curve (hand-drawn SVG) ---------- */
function drawCurve(p) {
  const svg = $("curve");
  const W = 640, H = 240, L = 54, R = 12, T = 12, B = 30;
  const xs = Array.from({ length: 31 }, (_, i) => i);
  const pts = xs.map((y) => ({ y, ...predict({ ...p, years_pro: y }) }));
  const top = Math.max(...pts.map((d) => d.hi)) * 1.08;
  const sx = (y) => L + (y / 30) * (W - L - R);
  const sy = (v) => T + (1 - v / top) * (H - T - B);
  const line = (k) => pts.map((d, i) => `${i ? "L" : "M"}${sx(d.y).toFixed(1)},${sy(d[k]).toFixed(1)}`).join("");
  const area = line("hi") + [...pts].reverse().map((d) => `L${sx(d.y).toFixed(1)},${sy(d.lo).toFixed(1)}`).join("") + "Z";
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top);
  const now = predict(p);
  svg.innerHTML = `
    ${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" class="grid"/><text x="${L - 8}" y="${sy(v) + 4}" class="ax" text-anchor="end">${short(v)}</text>`).join("")}
    ${[0, 5, 10, 15, 20, 25, 30].map((y) => `<text x="${sx(y)}" y="${H - 8}" class="ax" text-anchor="middle">${y}${y === 30 ? "+" : ""}</text>`).join("")}
    <path d="${area}" class="area"/>
    <path d="${line("mid")}" class="line"/>
    <line x1="${sx(Math.min(30, p.years_pro))}" x2="${sx(Math.min(30, p.years_pro))}" y1="${T}" y2="${H - B}" class="now"/>
    <circle cx="${sx(Math.min(30, p.years_pro))}" cy="${sy(now.mid)}" r="5" class="pt"/>`;
}

/* ---------- explore bars ---------- */
function bars(el, rows, key) {
  const max = Math.max(...rows.map((r) => r.median));
  el.innerHTML = rows
    .map((r) => `<div class="bar-row"><span class="lab">${r[key]}</span><span class="track"><i style="width:${(r.median / max) * 100}%"></i></span><span class="val">${short(r.median)}</span></div>`)
    .join("");
}
bars($("by-country"), explore.by_country, "country");
bars($("by-role"), explore.by_role, "role");
$("rows").textContent = explore.rows.toLocaleString("en-US");

const m = web.metrics;
$("metrics").innerHTML = `
  <tr><th>Held-out test rows</th><td>${m.rows_test.toLocaleString("en-US")}</td></tr>
  <tr><th>Mean absolute error</th><td>${usd(m.mae_usd)}</td></tr>
  <tr><th>Median absolute error</th><td>${usd(m.median_abs_error_usd)}</td></tr>
  <tr><th>Baseline (country median only), mean absolute error</th><td>${usd(m.baseline_country_median_mae_usd)}</td></tr>
  <tr><th>R² on log salary</th><td>${m.r2_log_salary}</td></tr>
  <tr><th>Actual salaries inside the 25th–75th band</th><td>${Math.round(m.share_inside_p25_p75 * 100)}%</td></tr>`;

$("form").addEventListener("input", render);
render();
