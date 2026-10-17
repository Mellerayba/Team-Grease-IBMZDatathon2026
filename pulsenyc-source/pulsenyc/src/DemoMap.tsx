import {useEffect, useRef} from 'react';
import {MapContainer, useMap} from 'react-leaflet';
import L from 'leaflet';
import {StreetMap} from './StreetMap';
import {Demo, ZonesGeo, Metric, Phase, unitView, zoneValues, cycleAt, hhmm} from './replay';

// Colours from the pitch deck: navy, signal red, gold, amber, greys.
export const PHASE_COLOR: Record<Phase, string> = {
  idle: '#0b192c', reposition: '#ffcc00', respond: '#cc0000', scene: '#f5a623', transport: '#5b6b82', hospital: '#aab4c3', home: '#d5dbe3',
};
export const PHASE_LABEL: Record<Phase, string> = {
  idle: 'Available', reposition: 'Moving to a new post', respond: 'Responding to a call', scene: 'On scene', transport: 'To hospital', hospital: 'At hospital', home: 'Returning to base',
};
export const CALL_WAITING = '#cc0000', CALL_EN_ROUTE = '#f5a623';

// Squares and diamonds instead of circles: Leaflet's canvas renderer only knows circles, so teach it two more shapes.
type Pathed = L.CircleMarker & {_point: L.Point; _radius: number; _renderer: unknown; _empty(): boolean};
type Ctx2 = {beginPath(): void; rect(x: number, y: number, w: number, h: number): void; moveTo(x: number, y: number): void; lineTo(x: number, y: number): void; closePath(): void};
type CanvasPatch = {_drawing: boolean; _ctx: Ctx2; _fillStroke(ctx: Ctx2, layer: unknown): void};
(L.Canvas as unknown as {include(o: object): void}).include({
  _updateSquare(this: CanvasPatch, layer: Pathed) {
    if (!this._drawing || layer._empty()) return;
    const p = layer._point, r = Math.max(layer._radius, 1), ctx = this._ctx;
    ctx.beginPath(); ctx.rect(p.x - r, p.y - r, 2 * r, 2 * r); this._fillStroke(ctx, layer);
  },
  _updateDiamond(this: CanvasPatch, layer: Pathed) {
    if (!this._drawing || layer._empty()) return;
    const p = layer._point, r = Math.max(layer._radius, 1) * 1.3, ctx = this._ctx;
    ctx.beginPath(); ctx.moveTo(p.x, p.y - r); ctx.lineTo(p.x + r, p.y); ctx.lineTo(p.x, p.y + r); ctx.lineTo(p.x - r, p.y); ctx.closePath(); this._fillStroke(ctx, layer);
  },
});
class SquareMarker extends L.CircleMarker {
  _updatePath() { (this as unknown as Pathed)._renderer && ((this as unknown as {_renderer: {_updateSquare(l: unknown): void}})._renderer)._updateSquare(this); }
}
class DiamondMarker extends L.CircleMarker {
  _updatePath() { (this as unknown as Pathed)._renderer && ((this as unknown as {_renderer: {_updateDiamond(l: unknown): void}})._renderer)._updateDiamond(this); }
}

// Risk index is a 0-100 percentile; calls are expected counts, so they get their own scale (set from the window's own range).
// Cream to red to dark red, as in the deck's response-time map.
const RAMP = [[251, 234, 181], [233, 161, 90], [204, 0, 0], [122, 12, 12]];
function shade(v: number, metric: Metric, hi: number) {
  const x = Math.max(0, Math.min(1, metric === 'risk' ? v / 100 : v / hi));
  const s = metric === 'risk' ? Math.max(0, (x - 0.5) / 0.5) : x;         // risk: only the upper half of the ranking is coloured
  const seg = Math.min(2, Math.floor(s * 3)), f = s * 3 - seg;
  const colour = mix(RAMP[seg], RAMP[seg + 1], f);
  return {fillColor: colour, fillOpacity: s * 0.62, color: colour, weight: s > 0.6 ? 1.2 : 0.4, opacity: 0.15 + s * 0.6};
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
    const hosp = L.layerGroup(demo.hospitals.map(h => new SquareMarker([h.lat, h.lon], {radius: 6.5, color: '#0b192c', weight: 2.5, fillColor: '#ffffff', fillOpacity: 1}).bindTooltip(h.name, {direction: 'right', className: 'hospital-label'}))).addTo(map);
    const cover = L.layerGroup().addTo(map);
    const dyn = L.layerGroup().addTo(map);
    const calls = demo.calls.map(c => new DiamondMarker([c.lat, c.lon], {radius: c.acuity === 'c' ? 6 : 4.5, color: '#ffffff', weight: 1.2, fillColor: CALL_WAITING, fillOpacity: 1}));
    const units = demo.policies[policy].units.map(u => new SquareMarker([u.lat, u.lon], {radius: 4, color: '#ffffff', weight: 1, fillColor: PHASE_COLOR.idle, fillOpacity: 1}).addTo(map));
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
      (l as L.Path).setStyle(showZones ? shade(vals[i], metric, s.scale) : {fillOpacity: 0, opacity: 0.3, weight: 0.4, color: '#8794a6'});
    });
    // 2. units, and where they are heading
    s.dyn.clearLayers();
    s.cover.clearLayers();
    demo.policies[policy].units.forEach((u, i) => {
      const v = unitView(u, t);
      const m = s.units[i];
      m.setLatLng(v.pos);
      m.setStyle({fillColor: PHASE_COLOR[v.phase], color: v.phase === 'reposition' ? '#0b192c' : '#ffffff'});
      m.setRadius(v.phase === 'reposition' ? 5.4 : 4);
      if (showCoverage && v.phase === 'idle') L.circle(v.pos, {radius: 1500, stroke: false, fillColor: '#0b192c', fillOpacity: 0.04, interactive: false}).addTo(s.cover);
      if (showRoutes && v.ahead && v.ahead.length > 1 && (v.phase === 'reposition' || v.phase === 'respond')) {
        const move = v.phase === 'reposition';
        L.polyline(v.ahead, {color: move ? '#0b192c' : '#cc0000', weight: move ? 2.4 : 1.4, opacity: move ? 0.9 : 0.5, dashArray: move ? '7 5' : undefined, interactive: false}).addTo(s.dyn);
        if (move && v.dest) new SquareMarker(v.dest, {radius: 7, color: '#0b192c', weight: 2, fillColor: '#ffcc00', fillOpacity: 0.35, interactive: false}).addTo(s.dyn);
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
        m.setStyle({fillColor: waiting ? CALL_WAITING : CALL_EN_ROUTE});
        m.setRadius((c.acuity === 'c' ? 6 : 4.5) + (waiting ? 1.5 : 0));
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
      <MapContainer center={[40.7, -73.98]} zoom={10} preferCanvas zoomSnap={0.25} style={{height: p.height, width: '100%'}}>
        <StreetMap />
        <Layers {...p} />
      </MapContainer>
      <div className="absolute bottom-2 left-2 z-[500] bg-navy text-white px-2 py-1 text-[11px] font-display tracking-wider uppercase">{hhmm(p.demo, p.t)} · {p.demo.policies[p.policy].label}</div>
    </div>
  );
}
