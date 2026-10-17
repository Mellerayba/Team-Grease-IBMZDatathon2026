import {useEffect, useRef, useState} from 'react';
import {TileLayer, useMap} from 'react-leaflet';

const sources = [
  {url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png', attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'},
  {url:'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'},
];

export function StreetMap(){
  const map = useMap();
  const [source,setSource] = useState(0);
  const [unavailable,setUnavailable] = useState(false);
  const failures = useRef(0);
  useEffect(()=>{
    // Fit all five boroughs, including the South campus on Staten Island.
    map.fitBounds([[40.49,-74.26],[40.92,-73.70]],{padding:[18,18]});
    const observer = new ResizeObserver(()=>map.invalidateSize());
    observer.observe(map.getContainer());
    return ()=>observer.disconnect();
  },[map]);
  return <>
    <TileLayer key={source} {...sources[source]} maxZoom={19} eventHandlers={{
      tileerror:()=>{
        failures.current++;
        if(failures.current<3)return;
        if(source===0){failures.current=0;setSource(1);}
        else setUnavailable(true);
      },
      tileload:()=>{failures.current=0;setUnavailable(false);},
    }}/>
    {unavailable&&<div role="status" className="absolute bottom-6 left-3 right-3 z-[1000] rounded bg-slate-950/95 p-3 text-xs text-amber-200">Street map tiles could not load. Check your internet connection and reload the page.</div>}
  </>;
}
