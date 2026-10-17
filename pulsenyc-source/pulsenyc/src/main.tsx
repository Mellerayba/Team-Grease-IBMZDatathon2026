import {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import 'leaflet/dist/leaflet.css';
import './style.css';
import {DemoMap, PHASE_COLOR, PHASE_LABEL, CALL_WAITING, CALL_EN_ROUTE} from './DemoMap';
import {Demo, ZonesGeo, Metric, Phase, loadDemo, unitView, callStats, cycleAt, hhmm} from './replay';

type View = 'compare' | 'forecast' | 'static';
const SPEEDS = [1, 2, 4, 8];                              // simulated minutes per real second
const f1 = (n: number) => n.toFixed(1);
const PHASES: Phase[] = ['idle', 'reposition', 'respond', 'scene', 'transport', 'hospital', 'home'];

/** A small flat colour square: the legend and status marker used throughout (no circles, no emoji). */
const Sq = ({c, diamond}: {c: string; diamond?: boolean}) => <span className="sq" style={{background: c, transform: diamond ? 'rotate(45deg) scale(.85)' : undefined}} />;

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

  if (err) return <div className="min-h-screen bg-paper p-8"><h1 className="font-display text-2xl uppercase mb-2">Demo data missing</h1><p className="text-sm text-ink">{err}</p></div>;
  if (!demo || !data || !info) return <div className="min-h-screen bg-paper p-8 text-ink">Loading replay…</div>;

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
    <div className="min-h-screen bg-paper text-navy">
      <header className="bg-navy border-t-[6px] border-signal text-white px-4 md:px-6 py-4">
        <div className="flex flex-wrap justify-between items-end gap-4">
          <div>
            <p className="text-gold text-[11px] tracking-wide">IBM Z Datathon: Theme 3 - Wildcard</p>
            <h1 className="font-display font-bold text-4xl md:text-5xl uppercase leading-none tracking-wide mt-1">Project Dispatch</h1>
            <p className="text-sm text-slate-300 mt-1">A solution to New York EMS response times.</p>
          </div>
          <div className="flex gap-6 items-end flex-wrap">
            <div className="text-xs leading-5"><div><Sq c="#cc0000" />Real calls: {date}</div><div><Sq c="#ffcc00" />Response times are modelled</div></div>
            <div className="text-right"><div className="text-[11px] text-slate-400 uppercase tracking-wider">{playing ? 'Replay running' : 'Paused'}</div><div className="font-display text-4xl leading-none">{clock}</div></div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_270px] gap-4 p-4 md:p-6">
        <aside className="space-y-4">
          <section className="panel">
            <h2 className="heading">Replay control</h2>
            <button className="btn-primary w-full mb-2" onClick={run}>Run demo</button>
            <div className="flex gap-2">
              <button className="btn flex-1" onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
              <button className="btn flex-1" onClick={() => { setPlaying(false); setT(0); }}>Reset</button>
            </div>
            <label className="field">Speed: minutes of city time per second</label>
            <div className="grid grid-cols-4 gap-1">{SPEEDS.map(v => <button key={v} className={'btn !px-1 ' + (speed === v ? 'chosen' : '')} onClick={() => setSpeed(v)}>{v}×</button>)}</div>
            <label className="field">Time {clock} ({Math.round(t)} of {demo.meta.minutes} min)</label>
            <input className="w-full" type="range" min="0" max={demo.meta.minutes} step="1" value={Math.round(t)} onChange={e => setT(+e.target.value)} />
          </section>
          <section className="panel">
            <h2 className="heading">View</h2>
            <div className="grid grid-cols-3 gap-1">{([['compare', 'Compare'], ['forecast', 'Model'], ['static', 'Static']] as const).map(([v, l]) => <button key={v} className={'btn !px-1 ' + (view === v ? 'chosen' : '')} onClick={() => setView(v)}>{l}</button>)}</div>
            <label className="field">Zone shading shows</label>
            <div className="grid grid-cols-1 gap-1">{([['risk', 'Model risk index'], ['forecast', 'Forecast calls, next hour'], ['actual', 'What actually happened']] as const).map(([v, l]) => <button key={v} className={'btn text-left ' + (metric === v ? 'chosen' : '')} onClick={() => setMetric(v)}>{l}</button>)}</div>
            {([['Zone shading', zones, setZones], ['Destination lines', routes, setRoutes], ['Incoming calls', calls, setCalls], ['Coverage areas', cover, setCover]] as const).map(([label, val, set]) =>
              <label key={label} className="flex justify-between mt-3 text-xs text-ink">{label}<input type="checkbox" checked={val} onChange={e => set(e.target.checked)} /></label>)}
          </section>
          <section className="panel">
            <h2 className="heading">Fleet: {demo.meta.units} units</h2>
            {PHASES.map(p => info.um.phases[p] || p === 'idle' ? <div key={p} className="flex justify-between py-0.5 text-xs"><span><Sq c={PHASE_COLOR[p]} />{PHASE_LABEL[p]}</span><b>{info.um.phases[p]}</b></div> : null)}
            <div className="text-xs mt-3 border-t border-rule pt-2 space-y-0.5">
              <div><Sq c={CALL_WAITING} diamond />Call, no unit yet</div>
              <div><Sq c={CALL_EN_ROUTE} diamond />Call, unit on the way</div>
              <div><Sq c="#ffffff" />Hospital</div>
            </div>
          </section>
        </aside>

        <main className="min-w-0 space-y-4">
          <section className="panel !p-0 overflow-hidden">
            <div className="p-4 border-b border-rule">
              <h2 className="heading !mb-1">NYC operations map</h2>
              <p className="muted">Forecast issued {hhmm(demo, info.cur.t)} · {info.since < 4 ? 'new forecast just landed' : `${Math.round(info.since)} min ago`} · refreshed every {demo.meta.cycle_min} min · shading: {metric === 'risk' ? 'model risk index' : metric === 'forecast' ? 'forecast calls next hour' : 'actual calls next hour'}</p>
            </div>
            <div className={view === 'compare' ? 'grid grid-cols-1 md:grid-cols-2 gap-px bg-rule' : 'grid grid-cols-1'}>
              {maps.map(p => <div key={p} className="relative min-w-0 bg-white">
                <div className={'absolute top-3 left-12 z-[500] px-3 py-1.5 font-display uppercase tracking-wider text-sm ' + (p === 'forecast' ? 'bg-signal text-white' : 'bg-navy text-white')}>{p === 'forecast' ? 'Model-guided' : 'Static layout'}</div>
                <DemoMap demo={demo} geo={data.geo} policy={p} t={t} metric={metric} showZones={zones} showRoutes={routes} showCalls={calls} showCoverage={cover} height={height} />
              </div>)}
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-1 p-3 text-[11px] text-ink border-t border-rule">
              {(['idle', 'reposition', 'respond', 'scene', 'transport', 'hospital'] as Phase[]).map(p => <span key={p}><Sq c={PHASE_COLOR[p]} />{PHASE_LABEL[p]}</span>)}
              <span><Sq c={CALL_WAITING} diamond />Call waiting</span><span><Sq c="#e9a15a" />Zone shading: higher risk</span>
            </div>
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-rule p-5">
              <h2 className="font-display uppercase text-lg tracking-wide text-slate-500">Static layout</h2>
              <div className="flex items-baseline gap-6 mt-3">
                <div><div className="font-display font-bold text-5xl text-signal leading-none">{f1(S.critWithin)}<span className="text-2xl">%</span></div><div className="muted mt-1">Critical calls reached within {demo.meta.target_min} min</div></div>
                <div><div className="font-display font-bold text-4xl text-signal leading-none">{f1(S.mean)}<span className="text-lg"> MIN</span></div><div className="muted mt-1">Mean response</div></div>
              </div>
              <p className="text-xs text-slate-500 mt-4"><span className="text-signal font-bold">&times;</span> Fixed posts, reactive deployment.</p>
            </div>
            <div className="bg-navy text-white p-5">
              <h2 className="font-display uppercase text-lg tracking-wide">Model-guided</h2>
              <div className="flex items-baseline gap-6 mt-3">
                <div><div className="font-display font-bold text-5xl text-amber leading-none">{f1(M.critWithin)}<span className="text-2xl">%</span></div><div className="text-[11px] text-slate-300 mt-1">Critical calls reached within {demo.meta.target_min} min</div></div>
                <div><div className="font-display font-bold text-4xl text-amber leading-none">{f1(M.mean)}<span className="text-lg"> MIN</span></div><div className="text-[11px] text-slate-300 mt-1">Mean response</div></div>
              </div>
              <p className="text-xs text-slate-200 mt-4"><span className="text-gold font-bold">&#10003;</span> Units repositioned every {demo.meta.cycle_min} minutes from the forecast.</p>
            </div>
          </div>

          <section className="panel">
            <h2 className="heading">Policy performance: same calls, same fleet</h2>
            <div className="grid grid-cols-4 gap-2 text-xs font-display uppercase tracking-wider text-slate-500 border-b border-rule pb-2"><span>Metric</span><span>Static</span><span className="text-signal">Model-guided</span><span>Oracle (full day)</span></div>
            {[
              ['Critical within 8 min, so far', f1(S.critWithin) + '%', f1(M.critWithin) + '%', '—'],
              ['90th percentile, so far', f1(S.p90) + ' min', f1(M.p90) + ' min', '—'],
              ['Calls answered, so far', S.served, M.served, '—'],
              ['Critical within 8 min, full day', f1(Number(summary['static']?.crit_within_target_pct)) + '%', f1(Number(summary['dynamic: forecast']?.crit_within_target_pct)) + '%', f1(Number(summary['dynamic: oracle']?.crit_within_target_pct)) + '%'],
              ['90th percentile, full day', f1(Number(summary['static']?.crit_p90_min)) + ' min', f1(Number(summary['dynamic: forecast']?.crit_p90_min)) + ' min', f1(Number(summary['dynamic: oracle']?.crit_p90_min)) + ' min'],
            ].map(([n, a, b, c]) => <div key={String(n)} className="grid grid-cols-4 gap-2 text-xs border-b border-rule py-2"><span className="text-ink">{n}</span><b>{a}</b><b className="text-signal">{b}</b><b className="text-slate-600">{c}</b></div>)}
            <p className="muted mt-3">{demo.meta.note} The oracle is an upper bound that knows the next hour's real demand.</p>
          </section>
        </main>

        <aside className="space-y-4">
          <section className="panel">
            <h2 className="heading">Incoming calls</h2>
            <div className="flex justify-between mb-3">
              <div><div className="font-display font-bold text-3xl text-signal leading-none">{M.waiting}</div><div className="muted">Waiting</div></div>
              <div><div className="font-display font-bold text-3xl text-amber leading-none">{M.enRoute}</div><div className="muted">Unit en route</div></div>
              <div><div className="font-display font-bold text-3xl leading-none">{M.served}</div><div className="muted">Reached</div></div>
            </div>
            <div className="max-h-40 overflow-y-auto">{demo.calls.map((c, i) => ({c, i})).filter(x => x.c.t <= t).slice(-8).reverse().map(({c, i}) => {
              const x = mp.calls[i]; const waiting = !x || x[0] > t; const state = waiting ? 'waiting' : x[1] > t ? 'unit en route' : `reached in ${f1(x[1] - c.t)} min`;
              return <div key={i} className="text-xs border-t border-rule py-1.5 flex justify-between gap-2"><span><Sq c={waiting ? CALL_WAITING : CALL_EN_ROUTE} diamond />{hhmm(demo, c.t)} {demo.zones[c.zone].borough.toLowerCase()} {c.zone}{c.acuity === 'c' ? <b className="text-signal"> critical</b> : ''}</span><span className={waiting ? 'text-signal font-semibold' : 'text-slate-500'}>{state}</span></div>;
            })}</div>
          </section>

          <section className="panel">
            <h2 className="heading">Highest-risk zones, next hour</h2>
            <div className="grid grid-cols-[1fr_40px_44px_44px] gap-1 text-[10px] font-display uppercase tracking-wider text-slate-500 mb-1"><span>ZIP zone</span><span className="text-right">Risk</span><span className="text-right">Static</span><span className="text-right text-signal">Model</span></div>
            {info.hot.map(h => <div key={h.z} className="grid grid-cols-[1fr_40px_44px_44px] gap-1 text-xs py-1 border-t border-rule">
              <span title={demo.zones[h.z].label}>{h.z} <span className="text-slate-500 text-[10px]">{demo.zones[h.z].borough.slice(0, 3)}</span></span>
              <b className="text-right">{h.risk.toFixed(0)}</b><span className="text-right">{info.us.counts[h.k]}</span><b className="text-right text-signal">{info.um.counts[h.k]}</b></div>)}
            <p className="muted mt-2">Units parked in, or driving to, each zone's area. The model moves units toward the zones it expects to be busy.</p>
          </section>

          <section className="panel">
            <h2 className="heading">Model decisions</h2>
            <div className="max-h-64 overflow-y-auto space-y-2">{logRows.length ? logRows.map((m, i) => {
              const p = parse(mp.why[m.why]);
              return <div key={i} className="quote text-[11px]">
                <span className="font-display tracking-wider text-signal">{hhmm(demo, m.t)}</span> <span className="font-semibold">Unit {m.unit}</span>
                {p ? <p className="text-ink">{p.kind}: {p.from} to {p.to} ({demo.zones[p.to]?.borough?.toLowerCase()}), {p.drive} min drive. Expected-call coverage +{p.gain}</p> : <p className="text-ink">{mp.why[m.why]}</p>}
              </div>;
            }) : <p className="muted">Press Run demo. Each move the model makes appears here with its reason.</p>}</div>
          </section>

          <section className="panel">
            <h2 className="heading">How to read this</h2>
            <p className="text-xs text-ink leading-relaxed">Every call is a real 2022 NYC EMS incident. Every {demo.meta.cycle_min} minutes the LightGBM forecaster predicts calls per ZIP zone for the next hour; zones are shaded by that output, and the policy moves idle units toward the busiest, most vulnerable zones. Both maps replay the same calls with the same fleet. Response times are modelled (assumed fleet, scene and hospital times, traffic); only the difference between layouts is meaningful.</p>
          </section>
        </aside>
      </div>
      <footer className="bg-navy border-b-[6px] border-signal text-center text-[11px] text-slate-400 py-4">Project Dispatch · Team Grease · IBM Z Datathon 2026 · Research prototype, not for emergency operations</footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
