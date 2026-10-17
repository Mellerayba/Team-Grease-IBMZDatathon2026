export type Point = [number,number];
export type Policy = 'static'|'ai';
export type Status = 'idle'|'relocating'|'responding'|'scene'|'transporting';
export type Unit = {id:number; pos:Point; home:Point; target:Point|null; status:Status; until:number; incidentId:number|null; moves:number; responseStart:number};
export type Incident = {id:number; at:number; pos:Point; zone:number; status:'pending'|'assigned'|'complete'; assigned:number|null; arrival:number|null; done:number|null};
export type Log = {time:number; text:string};
export type State = {time:number; units:Unit[]; incidents:Incident[]; responses:number[]; relocations:number; log:Log[]; policy:Policy; lastOptimize:number};
export const zones = [
 {name:'Manhattan',pos:[40.758,-73.985] as Point,weight:1.35},
 {name:'Brooklyn',pos:[40.678,-73.944] as Point,weight:1.2},
 {name:'Queens',pos:[40.728,-73.795] as Point,weight:1.0},
 {name:'Bronx',pos:[40.844,-73.875] as Point,weight:.8},
 {name:'Staten Island',pos:[40.579,-74.15] as Point,weight:.45},
];
// Campus reference coordinates from NYC Open Data q6fj-vxf8 and 67g2-p84d.
// These five facilities are a demo subset, not a complete hospital directory.
export const hospitals: {name:string; borough:string; pos:Point}[] = [
 {name:'Bellevue Hospital Center',borough:'Manhattan',pos:[40.73962320748,-73.976572846645]},
 {name:'Woodhull Medical and Mental Health Center',borough:'Brooklyn',pos:[40.70052661478,-73.941643858765]},
 {name:'Elmhurst Hospital Center',borough:'Queens',pos:[40.738710402563,-73.878351155182]},
 {name:'Lincoln Medical and Mental Health Center',borough:'Bronx',pos:[40.817688484049,-73.924200271483]},
 {name:'Staten Island University Hospital — South',borough:'Staten Island',pos:[40.51684177454684,-74.19628294526741]},
];
const rand=(seed:number)=>{let s=seed>>>0;return ()=>{s=(1664525*s+1013904223)>>>0;return s/4294967296;}};
export const distance=(a:Point,b:Point)=>{const lat=(a[0]-b[0])*111;const lon=(a[1]-b[1])*84;return Math.hypot(lat,lon);};
export const travel=(a:Point,b:Point)=>Math.max(1.5,distance(a,b)/.42); // minutes, illustrative urban road speed
export const interpolate=(a:Point,b:Point,f:number):Point=>[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];
export const forecast=(time:number,horizon:number)=>zones.map((z,i)=>({name:z.name,pos:z.pos,count:Math.round(z.weight*(horizon/30)*3*(i===1&&time>=45&&time<=180?3.2:1)),hot:i===1&&time>=45&&time<=180}));
export function makeEvents(seed=91):Incident[]{const r=rand(seed);const events:Incident[]=[];let t=5;for(let id=1;t<250;id++){const surge=t>=65&&t<155;const weights=zones.map((z,i)=>z.weight*(surge&&i===1?4:1));const sum=weights.reduce((a,b)=>a+b,0);let x=r()*sum;let zi=0;for(;zi<weights.length-1;zi++){x-=weights[zi];if(x<=0)break;}const base=zones[zi].pos;const jitter=(r()-.5)*.042;events.push({id,at:t,pos:[base[0]+jitter,base[1]+(r()-.5)*.055],zone:zi,status:'pending',assigned:null,arrival:null,done:null});t+=2+r()*5*(surge?.48:1);}return events;}
export function createState(policy:Policy,count:number,events:Incident[]):State{const units=Array.from({length:count},(_,i)=>{const z=zones[i%zones.length];const a=i*2.399;const home:Point=[z.pos[0]+Math.sin(a)*.014,z.pos[1]+Math.cos(a)*.017];return {id:i+1,pos:home,home,target:null,status:'idle' as Status,until:0,incidentId:null,moves:0,responseStart:0};});return {time:0,units,incidents:events.map(e=>({...e})),responses:[],relocations:0,log:[],policy,lastOptimize:-15};}
const note=(s:State,text:string)=>{s.log.unshift({time:s.time,text});s.log=s.log.slice(0,45);};
function assign(s:State){const pending=s.incidents.filter(x=>x.status==='pending'&&x.at<=s.time);for(const inc of pending){const choices=s.units.filter(u=>u.status==='idle'||u.status==='relocating');if(!choices.length)break;const unit=choices.reduce((a,b)=>travel(a.pos,inc.pos)<=travel(b.pos,inc.pos)?a:b);unit.status='responding';unit.target=inc.pos;unit.until=s.time+travel(unit.pos,inc.pos);unit.incidentId=inc.id;unit.responseStart=s.time;inc.status='assigned';inc.assigned=unit.id;note(s,`Unit A-${String(unit.id).padStart(2,'0')} dispatched to ${zones[inc.zone].name}`);}}
function optimize(s:State){if(s.policy!=='ai')return;const f=forecast(s.time,60);const ranked=f.map((x,i)=>({...x,i})).sort((a,b)=>b.count-a.count);let moved=0;for(const target of ranked.slice(0,2)){const staged=s.units.filter(u=>(u.status==='idle'||u.status==='relocating')&&distance(u.pos,target.pos)<2.7).length;const desired=Math.max(1,Math.round(target.count/5));if(staged>=desired)continue;const eligible=s.units.filter(u=>u.status==='idle'&&u.moves<4&&distance(u.pos,target.pos)>3&&distance(u.pos,target.pos)<22);if(!eligible.length)continue;const unit=eligible.reduce((a,b)=>distance(a.pos,target.pos)<distance(b.pos,target.pos)?a:b);unit.status='relocating';unit.target=[target.pos[0]+(unit.id%3-1)*.005,target.pos[1]+(unit.id%4-1.5)*.005];unit.until=s.time+travel(unit.pos,unit.target);unit.moves++;s.relocations++;moved++;note(s,`AI recommends A-${String(unit.id).padStart(2,'0')} → ${target.name} (${target.count} forecast calls / 60m)`);if(moved>=2)break;}}
export function step(s:State,dt=1){const end=s.time+dt;while(s.time<end-1e-8){const delta=Math.min(1,end-s.time);s.time+=delta;for(const u of s.units){if(u.target&&(u.status==='responding'||u.status==='relocating'||u.status==='transporting')){const remaining=Math.max(.001,u.until-(s.time-delta));u.pos=interpolate(u.pos,u.target,Math.min(1,delta/remaining));}if(u.status==='responding'&&s.time>=u.until){u.pos=u.target!;u.target=null;u.status='scene';u.until=s.time+7;const inc=s.incidents.find(x=>x.id===u.incidentId)!;inc.arrival=s.time;s.responses.push(inc.arrival-inc.at);}else if(u.status==='scene'&&s.time>=u.until){u.status='transporting';u.target=hospitals.reduce((a,b)=>distance(u.pos,a.pos)<distance(u.pos,b.pos)?a:b).pos;u.until=s.time+travel(u.pos,u.target);}else if(u.status==='transporting'&&s.time>=u.until){u.pos=u.target!;u.target=null;u.status='idle';const inc=s.incidents.find(x=>x.id===u.incidentId)!;inc.done=s.time;inc.status='complete';u.incidentId=null;}else if(u.status==='relocating'&&s.time>=u.until){u.pos=u.target!;u.target=null;u.status='idle';}}
if(Math.floor(s.time/15)>Math.floor(s.lastOptimize/15)){s.lastOptimize=s.time;optimize(s);}assign(s);}}
export function metrics(s:State){const r=[...s.responses].sort((a,b)=>a-b);return {mean:r.length?r.reduce((a,b)=>a+b,0)/r.length:0,p90:r.length?r[Math.max(0,Math.ceil(.9*r.length)-1)]:0,within:r.length?100*r.filter(x=>x<=8).length/r.length:0,complete:s.incidents.filter(x=>x.status==='complete').length,waiting:s.incidents.filter(x=>x.at<=s.time&&x.status==='pending').length,active:s.incidents.filter(x=>x.at<=s.time&&x.status==='assigned').length,idle:s.units.filter(x=>x.status==='idle').length,busy:s.units.filter(x=>x.status==='responding'||x.status==='scene'||x.status==='transporting').length,relocations:s.relocations};}
