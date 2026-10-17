import {useEffect, useRef} from 'react';
import {MapContainer, useMap} from 'react-leaflet';
import L from 'leaflet';
import {StreetMap} from './StreetMap';
import {Demo, ZonesGeo, Metric, Phase, unitView, zoneValues, cycleAt, hhmm} from './replay';

export const PHASE_COLOR: Record<Phase, string> = {
  idle: '#34d399', reposition: '#22d3ee', respond: '#fb7185', scene: '#fbbf24', transport: '#60a5fa', hospital: '#a78bfa', home: '#64748b',
};
export const PHASE_LABEL: Record<Phase, string> = {
  idle: 'Available', reposition: 'Moving to a new post', respond: 'Responding to a call', scene: 'On scene', transport: 'To hospital', hospital: 'At hospital', home: 'Returning to base',
};

// Risk index is a 0-100 percentile; calls are expected counts, so they get their own scale (set from the window's own range).
function shade(v: number, metric: Metric, hi: number) {
  const x = Math.max(0, Math.min(1, metric === 'risk' ? v / 100 : v / hi));
  const s = metric === 'risk' ? Math.max(0, (x - 0.5) / 0.5) : x;         // risk: only the upper half of the ranking is coloured
  const colour = s < 0.5 ? mix([253, 224, 71], [249, 115, 22], s / 0.5) : mix([249, 115, 22], [220, 38, 38], (s - 0.5) / 0.5);
  return {fillColor: colour, fillOpacity: s * 0.6, color: colour, weight: s > 0.6 ? 1.2 : 0.4, opacity: 0.2 + s * 0.5};
}
const mix = (a: number[], b: number[], f: number) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(',')})`;

type Props = {
  demo: Demo; geo: ZonesGeo; policy: 'static' | 'forecast'; t: number; metric: Metric;
  showZones: boolean; showRoutes: boolean; showCalls: boolean; showCoverage: boolean; height: number;
};

// When two maps are shown side by side they pan and zoom together.
const linked = new Set<L.Map>();
let syncing = false;
function useLinked(map: L.Map) {
  useEffect(() => {
    linked.add(map);
    const follow = () => {
      if (syncing) return;
      syncing = true;
      linked.forEach(o => { if (o !== map) o.setView(map.getCenter(), map.getZoom(), {animate: false}); });
      syncing = false;
    };
    map.on('moveend', follow);
    return () => { map.off('moveend', follow); linked.delete(map); };
  }, [map]);
}

function Layers({demo, geo, policy, t, metric, showZones, showRoutes, showCalls, showCoverage}: Omit<Props, 'height'>) {
  const map = useMap();
  useLinked(map);
  const S = useRef<{zones: L.GeoJSON; units: L.CircleMarker[]; calls: L.CircleMarker[]; dyn: L.LayerGroup; cover: L.LayerGroup; scale: number} | null>(null);
  const live = useRef({t, metric});
  live.current = {t, metric};

  // build once
  useEffect(() => {
    const idx = new Map(demo.zone_ids.map((z, i) => [z, i]));
    const all = demo.cycles.flatMap(c => [...c.crit.map((v, k) => v + c.other[k]), ...c.actual]);
    const scale = [...all].sort((a, b) => a - b)[Math.floor(all.length * 0.98)] || 1;
    const zones = L.geoJSON(geo, {
      style: () => ({weight: 0.4, fillOpacity: 0}),
      onEachFeature: (f, layer) => {
        const k = idx.get(f.properties!.id)!;
        layer.bindTooltip(() => {
          const {cur} = cycleAt(demo, live.current.t);
          const z = demo.zones[demo.zone_ids[k]];
          return `<b>${z.label}</b> · ${z.borough}<br/>Forecast next hour: ${cur.crit[k].toFixed(1)} critical + ${cur.other[k].toFixed(1)} other<br/>Risk index: ${cur.risk[k].toFixed(0)}/100 · Actually happened: ${cur.actual[k]}`;
        }, {sticky: true});
      },
    }).addTo(map);
    const hosp = L.layerGroup(demo.hospitals.map(h => L.circleMarker([h.lat, h.lon], {radius: 5.5, color: '#5b21b6', weight: 2, fillColor: '#c4b5fd', fillOpacity: 1}).bindTooltip(h.name, {direction: 'right'}))).addTo(map);
    void hosp;
    const cover = L.layerGroup().addTo(map);
    const dyn = L.layerGroup().addTo(map);
    const calls = demo.calls.map(c => L.circleMarker([c.lat, c.lon], {radius: c.acuity === 'c' ? 7 : 5, color: '#fff', weight: 1.2, fillColor: '#ef4444', fillOpacity: 0.95}));
    const units = demo.policies[policy].units.map(u => L.circleMarker([u.lat, u.lon], {radius: 4.2, color: '#061526', weight: 1.2, fillColor: PHASE_COLOR.idle, fillOpacity: 1}).addTo(map));
    S.current = {zones, units, calls, dyn, cover, scale};
    return () => { zones.remove(); hosp.remove(); cover.remove(); dyn.remove(); units.forEach(u => u.remove()); calls.forEach(c => c.remove()); S.current = null; };
  }, [map, demo, geo, policy]);

  // update on every tick
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    const tr = demo.policies[policy];
    // 1. zones shaded by the model's output
    const vals = zoneValues(demo, t, metric);
    s.zones.eachLayer(l => {
      const f = (l as L.GeoJSON & {feature: GeoJSON.Feature}).feature;
      const i = demo.zone_ids.indexOf(f.properties!.id);
      (l as L.Path).setStyle(showZones ? shade(vals[i], metric, s.scale) : {fillOpacity: 0, opacity: 0.15, weight: 0.3, color: '#64748b'});
    });
    // 2. units, and where they are heading
    s.dyn.clearLayers();
    s.cover.clearLayers();
    demo.policies[policy].units.forEach((u, i) => {
      const v = unitView(u, t);
      const m = s.units[i];
      m.setLatLng(v.pos);
      m.setStyle({fillColor: PHASE_COLOR[v.phase]});
      m.setRadius(v.phase === 'reposition' ? 5.6 : 4.2);
      if (showCoverage && v.phase === 'idle') L.circle(v.pos, {radius: 1500, stroke: false, fillColor: '#22d3ee', fillOpacity: 0.03, interactive: false}).addTo(s.cover);
      if (showRoutes && v.ahead && v.ahead.length > 1 && (v.phase === 'reposition' || v.phase === 'respond')) {
        const col = v.phase === 'reposition' ? '#22d3ee' : '#fb7185';
        L.polyline(v.ahead, {color: col, weight: v.phase === 'reposition' ? 2.4 : 1.4, opacity: v.phase === 'reposition' ? 0.95 : 0.55, dashArray: v.phase === 'reposition' ? '7 5' : undefined, interactive: false}).addTo(s.dyn);
        if (v.phase === 'reposition' && v.dest) L.circleMarker(v.dest, {radius: 7, color: '#22d3ee', weight: 2, fill: false, interactive: false}).addTo(s.dyn);
      }
    });
    // 3. calls: shown from the moment they come in until a unit reaches them
    demo.calls.forEach((c, i) => {
      const m = s.calls[i];
      const x = tr.calls[i];
      const open = c.t <= t && (!x || x[1] > t);
      const onMap = map.hasLayer(m);
      if (open && showCalls) {
        const waiting = !x || x[0] > t;
        m.setStyle({fillColor: waiting ? '#dc2626' : '#f97316'});
        m.setRadius((c.acuity === 'c' ? 7 : 5) + (waiting ? 1.5 : 0));
        if (!onMap) m.addTo(map);
        m.bringToFront();
      } else if (onMap) m.remove();
    });
    s.units.forEach(u => u.bringToFront());
  }, [t, demo, policy, metric, showZones, showRoutes, showCalls, showCoverage, map]);
  return null;
}

export function DemoMap(p: Props) {
  return (
    <div className="relative">
      <MapContainer center={[40.7, -73.98]} zoom={10} preferCanvas zoomSnap={0.25} style={{height: p.height, width: '100%', background: '#0a1828'}}>
        <StreetMap />
        <Layers {...p} />
      </MapContainer>
      <div className="absolute bottom-2 left-2 z-[500] rounded bg-slate-950/85 border border-slate-700 px-2 py-1 text-[10px] text-slate-300">{hhmm(p.demo, p.t)} · {p.demo.policies[p.policy].label}</div>
    </div>
  );
}
