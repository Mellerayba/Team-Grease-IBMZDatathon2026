// Playback of the real replay exported by scripts/export_demo.py (public/data/demo.json + zones.geojson).
// Nothing here is simulated in the browser: unit timelines, call lifecycles and zone forecasts all come from the Python replay
// and the forecaster's output. This file only looks up "where is unit N at minute t".
export type Pt = [number, number];                       // [lat, lon]
// leg: [t0, t1, kind, fromLat, fromLon, toLat, toLon, callIdx, whyIdx?, path?]   (minutes from the start of the window)
export type Leg = [number, number, number, number, number, number, number, number, number?, (Pt[] | null)?];
export type UnitTrack = {id: number; lat: number; lon: number; legs: Leg[]};
export type Move = {t: number; unit: number; why: number; to: string; kind: string};
export type PolicyData = {label: string; units: UnitTrack[]; calls: Record<string, [number, number, number, number]>; why: string[]; moves: Move[]};
export type Call = {t: number; zone: string; acuity: 'c' | 'o'; lat: number; lon: number};
export type Cycle = {t: number; crit: number[]; other: number[]; risk: number[]; actual: number[]};
export type Demo = {
  meta: {day: string; start: string; minutes: number; units: number; cycle_min: number; target_min: number; day_incidents: number; window_incidents: number; note: string};
  zone_ids: string[];
  zones: Record<string, {label: string; borough: string; lat: number; lon: number; vuln: number}>;
  hospitals: {name: string; borough: string; lat: number; lon: number}[];
  cycles: Cycle[];
  calls: Call[];
  policies: Record<'static' | 'forecast', PolicyData>;
  day_summary: Record<string, number | string>[];
  by_borough: Record<string, Record<string, number>>;
};
export type ZonesGeo = GeoJSON.FeatureCollection;

export const KIND = {reposition: 0, respond: 1, scene: 2, transport: 3, hospital: 4, home: 5} as const;
export type Phase = 'idle' | 'reposition' | 'respond' | 'scene' | 'transport' | 'hospital' | 'home';
export type UnitView = {id: number; pos: Pt; phase: Phase; dest: Pt | null; ahead: Pt[] | null; call: number; why: number};

export async function loadDemo(): Promise<{demo: Demo; geo: ZonesGeo}> {
  const base = import.meta.env.BASE_URL;
  const [d, g] = await Promise.all([fetch(base + 'data/demo.json'), fetch(base + 'data/zones.geojson')]);
  if (!d.ok || !g.ok) throw new Error('Demo data not found. Run: python scripts/export_demo.py');
  return {demo: await d.json(), geo: await g.json()};
}

const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

function along(path: Pt[], f: number): {pos: Pt; ahead: Pt[]} {
  const seg: number[] = [];
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    const d = Math.hypot(path[i][0] - path[i - 1][0], (path[i][1] - path[i - 1][1]) * 0.76);
    seg.push(d); total += d;
  }
  let want = f * total;
  for (let i = 0; i < seg.length; i++) {
    if (want <= seg[i] || i === seg.length - 1) {
      const k = seg[i] > 0 ? Math.min(1, want / seg[i]) : 1;
      const pos: Pt = [lerp(path[i][0], path[i + 1][0], k), lerp(path[i][1], path[i + 1][1], k)];
      return {pos, ahead: [pos, ...path.slice(i + 1)]};
    }
    want -= seg[i];
  }
  return {pos: path[path.length - 1], ahead: []};
}

const PHASE: Phase[] = ['reposition', 'respond', 'scene', 'transport', 'hospital', 'home'];

/** Where unit `u` is at minute `t`, what it is doing, and where it is heading. */
export function unitView(u: UnitTrack, t: number): UnitView {
  const L = u.legs;
  let lo = 0, hi = L.length - 1, hit = -1;               // last leg that has started
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (L[mid][0] <= t) { hit = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (hit < 0) return {id: u.id, pos: [u.lat, u.lon], phase: 'idle', dest: null, ahead: null, call: -1, why: -1};
  const g = L[hit];
  if (t >= g[1]) return {id: u.id, pos: [g[5], g[6]], phase: 'idle', dest: null, ahead: null, call: -1, why: -1};
  const f = g[1] > g[0] ? (t - g[0]) / (g[1] - g[0]) : 1;
  const path = g[9] ?? null;
  let phase = PHASE[g[2]];
  if (g[2] === KIND.home && g[8] !== undefined) phase = 'reposition';   // a freed unit sent to a post by the policy
  const moving = phase === 'reposition' || phase === 'respond' || phase === 'transport' || phase === 'home';
  let pos: Pt = [lerp(g[3], g[5], f), lerp(g[4], g[6], f)];
  let ahead: Pt[] | null = moving ? [pos, [g[5], g[6]]] : null;
  if (path && path.length > 1 && moving) {
    const r = along([[g[3], g[4]], ...path.slice(1, -1), [g[5], g[6]]], f);
    pos = r.pos; ahead = r.ahead;
  }
  return {id: u.id, pos, phase, dest: moving ? [g[5], g[6]] : null, ahead, call: g[7], why: g[8] ?? -1};
}

export type CallStats = {mean: number; p90: number; within: number; critWithin: number; waiting: number; enRoute: number; onScene: number; served: number};
export function callStats(demo: Demo, policy: 'static' | 'forecast', t: number): CallStats {
  const info = demo.policies[policy].calls;
  const resp: number[] = [], crit: number[] = [];
  let waiting = 0, enRoute = 0, onScene = 0;
  demo.calls.forEach((c, i) => {
    if (c.t > t) return;
    const x = info[i];
    if (!x || x[0] > t) { waiting++; return; }
    if (x[1] > t) { enRoute++; return; }
    const m = x[1] - c.t;
    resp.push(m);
    if (c.acuity === 'c') crit.push(m);
    if (x[1] + 15 > t) onScene++;
  });
  const sorted = [...resp].sort((a, b) => a - b);
  const target = demo.meta.target_min;
  return {
    mean: resp.length ? resp.reduce((a, b) => a + b, 0) / resp.length : 0,
    p90: sorted.length ? sorted[Math.max(0, Math.ceil(0.9 * sorted.length) - 1)] : 0,
    within: resp.length ? 100 * resp.filter(m => m <= target).length / resp.length : 0,
    critWithin: crit.length ? 100 * crit.filter(m => m <= target).length / crit.length : 0,
    waiting, enRoute, onScene, served: resp.length,
  };
}

export function cycleAt(demo: Demo, t: number) {
  const i = Math.max(0, Math.min(demo.cycles.length - 1, Math.floor(t / demo.meta.cycle_min)));
  const since = t - demo.cycles[i].t;                   // minutes since this forecast was issued
  return {i, cur: demo.cycles[i], prev: demo.cycles[Math.max(0, i - 1)], since};
}

export type Metric = 'risk' | 'forecast' | 'actual';
/** One number per zone for the shading: the model's risk index (0-100), expected calls next hour, or what actually happened. */
export function zoneValues(demo: Demo, t: number, metric: Metric, blendMin = 4): number[] {
  const {cur, prev, since} = cycleAt(demo, t);
  const pick = (c: Cycle, k: number) => metric === 'risk' ? c.risk[k] : metric === 'forecast' ? c.crit[k] + c.other[k] : c.actual[k];
  const f = Math.min(1, since / blendMin);               // the new forecast "lands" over a few minutes so the change is visible
  return demo.zone_ids.map((_, k) => lerp(pick(prev, k), pick(cur, k), f));
}

export const hhmm = (demo: Demo, t: number) => {
  const d = new Date(Date.parse(demo.meta.start + 'Z') + t * 60000);
  return d.toISOString().slice(11, 16);
};
