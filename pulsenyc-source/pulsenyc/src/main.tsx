import {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import 'leaflet/dist/leaflet.css';
import './style.css';
import {DemoMap, PHASE_COLOR, PHASE_LABEL} from './DemoMap';
import {Demo, ZonesGeo, Metric, Phase, loadDemo, unitView, callStats, cycleAt, hhmm} from './replay';

type View = 'compare' | 'forecast' | 'static';
const SPEEDS = [1, 2, 4, 8];                              // simulated minutes per real second
const f1 = (n: number) => n.toFixed(1);

function useTicker(demo: Demo | null) {
  const q = new URLSearchParams(location.search);           // ?t=120 opens the replay at minute 120 (handy for screenshots)
  const [t, setT] = useState(Number(q.get('t')) || 0), [playing, setPlaying] = useState(false), [speed, setSpeed] = useState(4);
  useEffect(() => {
    if (!playing || !demo) return;
    const id = setInterval(() => setT(x => {
      const n = x + speed * 0.05;
      if (n >= demo.meta.minutes) { setPlaying(false); return demo.meta.minutes; }
      return n;
    }), 50);
    return () => clearInterval(id);
  }, [playing, speed, demo]);
  return {t, setT, playing, setPlaying, speed, setSpeed};
}

/** Idle units (and units already driving to a new post) counted by the zone they are in, or heading to. */
function unitsByZone(demo: Demo, policy: 'static' | 'forecast', t: number) {
  const zs = demo.zone_ids.map(z => demo.zones[z]);
  const counts = new Array(zs.length).fill(0);
  const phases: Record<Phase, number> = {idle: 0, reposition: 0, respond: 0, scene: 0, transport: 0, hospital: 0, home: 0};
  for (const u of demo.policies[policy].units) {
    const v = unitView(u, t);
    phases[v.phase]++;
    if (v.phase !== 'idle' && v.phase !== 'reposition') continue;
    const [la, lo] = v.dest ?? v.pos;
    let best = 0, bd = 1e9;
    zs.forEach((z, i) => { const d = (z.lat - la) ** 2 + ((z.lon - lo) * 0.76) ** 2; if (d < bd) { bd = d; best = i; } });
    counts[best]++;
  }
  return {counts, phases};
}

function App() {
  const [data, setData] = useState<{demo: Demo; geo: ZonesGeo} | null>(null), [err, setErr] = useState('');
  useEffect(() => { loadDemo().then(setData).catch(e => setErr(String(e.message ?? e))); }, []);
  const demo = data?.demo ?? null;
  const {t, setT, playing, setPlaying, speed, setSpeed} = useTicker(demo);
  const [view, setView] = useState<View>('compare'), [metric, setMetric] = useState<Metric>('risk');
  const [zones, setZones] = useState(true), [routes, setRoutes] = useState(true), [calls, setCalls] = useState(true), [cover, setCover] = useState(false);

  const tq = Math.floor(t * 4) / 4;                         // panels refresh four times per simulated minute
  const info = useMemo(() => {
    if (!demo) return null;
    const stat = callStats(demo, 'static', tq), model = callStats(demo, 'forecast', tq);
    const us = unitsByZone(demo, 'static', tq), um = unitsByZone(demo, 'forecast', tq);
    const {cur, since} = cycleAt(demo, tq);
    const hot = demo.zone_ids.map((z, k) => ({z, k, risk: cur.risk[k], fc: cur.crit[k] + cur.other[k], act: cur.actual[k]}))
      .sort((a, b) => b.risk - a.risk || b.fc - a.fc).slice(0, 7);
    return {stat, model, us, um, cur, since, hot};
  }, [demo, tq]);

  if (err) return <div className="min-h-screen bg-[#07101e] text-slate-100 p-8"><h1 className="text-xl font-bold mb-2">Demo data missing</h1><p className="text-sm text-slate-400">{err}</p></div>;
  if (!demo || !data || !info) return <div className="min-h-screen bg-[#07101e] text-slate-300 p-8">Loading replay…</div>;

  const run = () => { setT(0); setView('compare'); setPlaying(true); };
  const mp = demo.policies.forecast, M = info.model, S = info.stat;
  const clock = hhmm(demo, t);
  const date = new Date(Date.parse(demo.meta.start + 'Z')).toUTCString().slice(0, 16);
  const maps = view === 'compare' ? (['static', 'forecast'] as const) : ([view] as const);
  const height = view === 'compare' ? 540 : 640;
  const summary = Object.fromEntries(demo.day_summary.map(r => [String(r.policy), r]));
  const logRows = mp.moves.filter(m => m.t <= t).sort((a, b) => b.t - a.t).slice(0, 14);

  const parse = (why: string) => {
    const m = why.match(/^U\d+ (.*?): (\d+) -> (\d+) \((\d+) min drive\), \+([\d.]+) weighted/);
    return m ? {kind: m[1] === 'relocation' ? 'Repositioned' : 'Sent to post', from: m[2], to: m[3], drive: m[4], gain: m[5]} : null;
  };

  return (
    <div className="min-h-screen bg-[#07101e] text-slate-100 p-3 md:p-5">
      <header className="flex flex-wrap justify-between items-center gap-3 border-b border-slate-700 pb-4 mb-4">
        <div>
          <h1 className="text-2xl font-black tracking-tight">✚ PROJECT <span className="text-cyan-400">DISPATCH</span> <span className="text-slate-500 font-normal text-sm">/ COMMAND</span></h1>
          <p className="text-[10px] tracking-[.2em] text-slate-400">FORECAST-DRIVEN EMS UNIT POSITIONING · NEW YORK CITY</p>
        </div>
        <div className="flex gap-3 items-center flex-wrap">
          <span className="text-emerald-300 bg-emerald-900/30 rounded px-2 py-1 text-xs">REAL CALLS · {date.toUpperCase()}</span>
          <span className="text-amber-300 bg-amber-900/30 rounded px-2 py-1 text-xs">MODELLED RESPONSE TIMES</span>
          <span className="text-xs text-slate-400">{playing ? '● LIVE REPLAY' : 'PAUSED'}</span>
          <span className="font-mono text-lg">{clock}</span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[210px_minmax(0,1fr)_260px] gap-3">
        <aside className="space-y-3">
          <section className="panel">
            <h2 className="heading">REPLAY CONTROL</h2>
            <button className="btn-primary w-full mb-2" onClick={run}>▶ RUN DEMO</button>
            <div className="flex gap-2">
              <button className="btn flex-1" onClick={() => setPlaying(!playing)}>{playing ? 'Ⅱ Pause' : '▶ Play'}</button>
              <button className="btn flex-1" onClick={() => { setPlaying(false); setT(0); }}>↻ Reset</button>
            </div>
            <label className="field">Replay speed (minutes of city time per second)</label>
            <div className="grid grid-cols-4 gap-1">{SPEEDS.map(v => <button key={v} className={'btn !px-1 ' + (speed === v ? 'chosen' : '')} onClick={() => setSpeed(v)}>{v}×</button>)}</div>
            <label className="field">Time {clock} ({Math.round(t)} / {demo.meta.minutes} min)</label>
            <input className="w-full accent-cyan-400" type="range" min="0" max={demo.meta.minutes} step="1" value={Math.round(t)} onChange={e => setT(+e.target.value)} />
          </section>
          <section className="panel">
            <h2 className="heading">VIEW</h2>
            <div className="grid grid-cols-3 gap-1">{([['compare', 'Compare'], ['forecast', 'Model'], ['static', 'Static']] as const).map(([v, l]) => <button key={v} className={'btn !px-1 text-[10px] ' + (view === v ? 'chosen' : '')} onClick={() => setView(v)}>{l}</button>)}</div>
            <label className="field">Zone shading shows</label>
            <div className="grid grid-cols-1 gap-1">{([['risk', 'Model risk index'], ['forecast', 'Forecast calls, next hour'], ['actual', 'What actually happened']] as const).map(([v, l]) => <button key={v} className={'btn text-left ' + (metric === v ? 'chosen' : '')} onClick={() => setMetric(v)}>{l}</button>)}</div>
            {([['Zone shading', zones, setZones], ['Destination lines', routes, setRoutes], ['Incoming calls', calls, setCalls], ['Coverage rings', cover, setCover]] as const).map(([label, val, set]) =>
              <label key={label} className="flex justify-between mt-3 text-xs text-slate-300">{label}<input type="checkbox" checked={val} onChange={e => set(e.target.checked)} className="accent-cyan-400" /></label>)}
          </section>
          <section className="panel">
            <h2 className="heading">MODEL-GUIDED FLEET · {demo.meta.units} UNITS</h2>
            {(['idle', 'reposition', 'respond', 'scene', 'transport', 'hospital', 'home'] as Phase[]).map(p => info.um.phases[p] || p === 'idle' ? <div key={p} className="flex justify-between py-0.5 text-xs"><span style={{color: PHASE_COLOR[p]}}>● {PHASE_LABEL[p]}</span><b>{info.um.phases[p]}</b></div> : null)}
            <div className="flex justify-between py-0.5 text-xs mt-2 border-t border-slate-700 pt-2"><span className="text-rose-400">● Call waiting</span><span className="text-orange-400">● Unit on the way</span></div>
          </section>
        </aside>

        <main className="min-w-0 space-y-3">
          <section className="panel !p-0 overflow-hidden">
            <div className="flex flex-wrap justify-between items-center p-3 border-b border-slate-700 gap-2">
              <div>
                <h2 className="font-bold">NYC Operations Map</h2>
                <p className="muted">Forecast issued {hhmm(demo, info.cur.t)} · {info.since < 4 ? 'new forecast just landed' : `${Math.round(info.since)} min ago`} · refreshed every {demo.meta.cycle_min} min · shading: {metric === 'risk' ? 'model risk index' : metric === 'forecast' ? 'forecast calls next hour' : 'actual calls next hour'}</p>
              </div>
            </div>
            <div className={view === 'compare' ? 'grid grid-cols-1 md:grid-cols-2' : 'grid grid-cols-1'}>
              {maps.map(p => <div key={p} className="relative min-w-0 border-r border-slate-700">
                <div className="absolute top-3 left-12 z-[500] rounded bg-slate-950/90 border border-slate-700 px-3 py-2 text-xs font-semibold">{p === 'forecast' ? '✦ MODEL-GUIDED' : '◼ STATIC LAYOUT'}</div>
                <DemoMap demo={demo} geo={data.geo} policy={p} t={t} metric={metric} showZones={zones} showRoutes={routes} showCalls={calls} showCoverage={cover} height={height} />
              </div>)}
            </div>
            <div className="flex flex-wrap gap-4 p-3 text-[10px] text-slate-300">
              {(['idle', 'reposition', 'respond', 'scene', 'transport', 'hospital'] as Phase[]).map(p => <span key={p}><span style={{color: PHASE_COLOR[p]}}>●</span> {PHASE_LABEL[p]}</span>)}
              <span><span className="text-rose-500">●</span> Call (red = no unit yet)</span><span><span className="text-violet-300">●</span> Hospital</span><span><span className="text-orange-400">■</span> Zone: higher model risk</span>
            </div>
          </section>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {[['CRITICAL IN 8 MIN · STATIC', f1(S.critWithin) + '%'], ['CRITICAL IN 8 MIN · MODEL', f1(M.critWithin) + '%'], ['MEAN RESPONSE · STATIC', f1(S.mean) + ' min'], ['MEAN RESPONSE · MODEL', f1(M.mean) + ' min']].map(([n, v]) =>
              <div key={n} className="panel"><div className="heading !mb-1">{n}</div><div className={'text-xl md:text-2xl font-bold tracking-tight ' + (n.includes('MODEL') ? 'text-cyan-300' : '')}>{v}</div></div>)}
          </div>

          <section className="panel">
            <h2 className="heading">POLICY PERFORMANCE · SAME CALLS, SAME FLEET</h2>
            <div className="grid grid-cols-4 gap-2 text-xs text-slate-400 border-b border-slate-700 pb-2"><span>Metric</span><span>Static</span><span className="text-cyan-300">Model-guided</span><span>Oracle (full day)</span></div>
            {[
              ['Critical within 8 min · so far', f1(S.critWithin) + '%', f1(M.critWithin) + '%', '—'],
              ['90th percentile · so far', f1(S.p90) + ' min', f1(M.p90) + ' min', '—'],
              ['Calls answered · so far', S.served, M.served, '—'],
              ['Critical within 8 min · full day', f1(Number(summary['static']?.crit_within_target_pct)) + '%', f1(Number(summary['dynamic: forecast']?.crit_within_target_pct)) + '%', f1(Number(summary['dynamic: oracle']?.crit_within_target_pct)) + '%'],
              ['90th percentile · full day', f1(Number(summary['static']?.crit_p90_min)) + ' min', f1(Number(summary['dynamic: forecast']?.crit_p90_min)) + ' min', f1(Number(summary['dynamic: oracle']?.crit_p90_min)) + ' min'],
            ].map(([n, a, b, c]) => <div key={String(n)} className="grid grid-cols-4 gap-2 text-xs border-b border-slate-800 py-2"><span className="text-slate-400">{n}</span><b>{a}</b><b className="text-cyan-300">{b}</b><b className="text-slate-300">{c}</b></div>)}
            <p className="muted mt-3">{demo.meta.note} The oracle is an upper bound that knows the next hour's real demand.</p>
          </section>
        </main>

        <aside className="space-y-3">
          <section className="panel">
            <h2 className="heading">INCOMING CALLS</h2>
            <div className="flex justify-between mb-3">
              <div><div className="text-xl font-bold text-rose-400">{M.waiting}</div><div className="muted">Waiting</div></div>
              <div><div className="text-xl font-bold text-amber-300">{M.enRoute}</div><div className="muted">Unit en route</div></div>
              <div><div className="text-xl font-bold">{M.served}</div><div className="muted">Reached</div></div>
            </div>
            <div className="max-h-40 overflow-y-auto">{demo.calls.map((c, i) => ({c, i})).filter(x => x.c.t <= t).slice(-8).reverse().map(({c, i}) => {
              const x = mp.calls[i]; const state = !x || x[0] > t ? 'waiting' : x[1] > t ? 'unit en route' : `reached in ${f1(x[1] - c.t)} min`;
              return <div key={i} className="text-xs border-t border-slate-700 py-1.5 flex justify-between gap-2"><span>{c.acuity === 'c' ? '🔴' : '🟠'} {hhmm(demo, c.t)} {demo.zones[c.zone].borough.toLowerCase()} {c.zone}</span><span className={state === 'waiting' ? 'text-rose-300' : 'text-slate-400'}>{state}</span></div>;
            })}</div>
          </section>

          <section className="panel">
            <h2 className="heading">HIGHEST-RISK ZONES · NEXT HOUR</h2>
            <div className="grid grid-cols-[1fr_44px_44px_44px] gap-1 text-[10px] text-slate-500 mb-1"><span>ZIP zone</span><span className="text-right">Risk</span><span className="text-right">Static</span><span className="text-right text-cyan-300">Model</span></div>
            {info.hot.map(h => <div key={h.z} className="grid grid-cols-[1fr_44px_44px_44px] gap-1 text-xs py-1 border-t border-slate-800">
              <span title={demo.zones[h.z].label}>{h.z} <span className="text-slate-500 text-[10px]">{demo.zones[h.z].borough.slice(0, 3)}</span></span>
              <b className="text-right text-orange-300">{h.risk.toFixed(0)}</b><span className="text-right">{info.us.counts[h.k]}</span><b className="text-right text-cyan-300">{info.um.counts[h.k]}</b></div>)}
            <p className="muted mt-2">Units parked in (or driving to) each zone's area. The model moves units toward the zones it expects to be busy.</p>
          </section>

          <section className="panel">
            <h2 className="heading">MODEL DECISIONS</h2>
            <div className="max-h-64 overflow-y-auto space-y-2">{logRows.length ? logRows.map((m, i) => {
              const p = parse(mp.why[m.why]);
              return <div key={i} className="border-l-2 border-cyan-600 pl-2 py-1 text-[11px] bg-slate-800/40">
                <span className="text-cyan-300 font-mono">{hhmm(demo, m.t)}</span> <span className="text-slate-100">Unit {m.unit}</span>
                {p ? <p className="text-slate-300">{p.kind}: {p.from} → {p.to} ({demo.zones[p.to]?.borough?.toLowerCase()}), {p.drive} min drive · +{p.gain} expected-call coverage</p> : <p className="text-slate-300">{mp.why[m.why]}</p>}
              </div>;
            }) : <p className="muted">Press RUN DEMO. Each move the model makes appears here with its reason.</p>}</div>
          </section>

          <section className="panel">
            <h2 className="heading">HOW TO READ THIS</h2>
            <p className="text-xs text-slate-400 leading-relaxed">Every call is a real 2022 NYC EMS incident. Every {demo.meta.cycle_min} minutes the LightGBM forecaster predicts calls per ZIP zone for the next hour; zones are shaded by that output, and the policy moves idle units toward the busiest, most vulnerable zones. Both maps replay the same calls with the same fleet. Response times are modelled (assumed fleet, scene and hospital times, traffic); only the difference between layouts is meaningful.</p>
          </section>
        </aside>
      </div>
      <footer className="text-center text-[10px] text-slate-500 py-4">Project Dispatch · Research prototype · Team Grease · IBM Z Datathon 2026 · Not for emergency operations</footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
