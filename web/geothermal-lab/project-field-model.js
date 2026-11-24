import {N,DX,THICKNESS,CP,YEAR,networkForWells} from './project-model.js';
export const initialAtDepth=depth=>Math.round((15+.0725*depth)/10)*10;
// Independent equal-conductance layers. A layer circulates only when it has
// at least one injector and one producer; total rate is shared across its wells.
export function createFieldModel({rate=50,injection=70,capacity=5,price=75,availability=.9,efficiency=.12,auxiliary=.1,heatCapacity=2500000}={}) {
 const layers=new Map(),settings={rate,injection,capacity,price,availability,efficiency,auxiliary,heatCapacity};let wells=[],seconds=0,energy=0,revenue=0,extractedJ=0;
 const cellCapacity=DX*DX*THICKNESS*heatCapacity;
 function layer(depth){if(!layers.has(depth))layers.set(depth,{depth,initial:initialAtDepth(depth),temperature:new Float64Array(N*N).fill(initialAtDepth(depth)),injectors:[],producers:[],network:null});return layers.get(depth);}
 function configure(nextWells){
  if(!Array.isArray(nextWells))throw Error('Invalid well list');const ids=nextWells.map(w=>w.id);
  if(new Set(ids).size!==ids.length||nextWells.some(w=>!Number.isInteger(w.id)||w.id<0||w.id>=N*N||!Number.isInteger(w.depth)||w.depth<1500||w.depth>2800||!['injector','producer'].includes(w.type)))throw Error('Invalid or overlapping wells.');
  wells=nextWells.map(w=>({...w}));for(const l of layers.values()){l.injectors=[];l.producers=[];l.network=null;}
  for(const w of wells)layer(w.depth)[w.type==='injector'?'injectors':'producers'].push(w.id);
  for(const l of layers.values())if(l.injectors.length&&l.producers.length)l.network=networkForWells(l.injectors,l.producers,settings.rate);
 }
 function setOperating(changes){
  const next={...settings,...changes};if(!Object.values(next).every(Number.isFinite)||next.rate<0||next.injection<0||next.injection>=120||next.capacity<0||next.price<0||next.availability<0||next.availability>1)throw Error('Invalid operating settings.');
  Object.assign(settings,changes);configure(wells);
 }
 function step(days){
  if(!Number.isFinite(days)||days<0)throw Error('Invalid time step.');let remaining=days*86400;
  while(remaining>0){const active=[...layers.values()].filter(l=>l.network),maxOutgoing=Math.max(0,...active.map(l=>Math.max(...l.network.outgoing)*CP));const limit=maxOutgoing?.8*cellCapacity/maxOutgoing:Infinity,dt=Math.min(remaining,limit,30*86400),factor=settings.availability;let thermal=0;
   for(const l of active){const old=l.temperature.slice(),delta=new Float64Array(N*N),qIn=settings.rate/l.injectors.length,qOut=settings.rate/l.producers.length;
    for(const e of l.network.edges){const heat=e.mass*CP*old[e.from]*dt*factor;delta[e.from]-=heat;delta[e.to]+=heat;}
    for(const i of l.injectors)delta[i]+=qIn*CP*settings.injection*dt*factor;
    for(const i of l.producers){delta[i]-=qOut*CP*old[i]*dt*factor;const power=qOut*CP*(old[i]-settings.injection);thermal+=Math.max(0,power)/1e6;extractedJ+=power*dt*factor;}
    for(let i=0;i<l.temperature.length;i++)l.temperature[i]+=delta[i]/cellCapacity;
   }
   const net=Math.min(settings.capacity,thermal*settings.efficiency*(1-settings.auxiliary));const e=net*dt/3600*factor;energy+=e;revenue+=e*settings.price;seconds+=dt;remaining-=dt;
  }return snapshot();
 }
 function snapshot(viewDepth=2000){const active=[...layers.values()].filter(l=>l.network),p=[];for(const l of active)for(const id of l.producers)p.push({id,depth:l.depth,temperature:l.temperature[id],weight:settings.rate/l.producers.length});const weight=p.reduce((n,w)=>n+w.weight,0),production=weight?p.reduce((n,w)=>n+w.temperature*w.weight,0)/weight:null,thermal=p.reduce((n,w)=>n+Math.max(0,w.weight*CP*(w.temperature-settings.injection))/1e6,0),net=Math.min(settings.capacity,thermal*settings.efficiency*(1-settings.auxiliary));const l=layers.get(viewDepth),initial=l?.initial??initialAtDepth(viewDepth);
  const layerTemperatures=Object.fromEntries([...layers.values()].filter(v=>v.producers.length).map(v=>[v.depth,v.producers.reduce((sum,i)=>sum+v.temperature[i],0)/v.producers.length]));
  return {years:seconds/YEAR,production,thermal,net,energy,revenue,capacity:settings.capacity,injection:settings.injection,rate:settings.rate,extractedJ,temperature:l?Array.from(l.temperature):Array(N*N).fill(initial),initial,depth:viewDepth,activeLayers:active.length,wellTemperatures:p,layerTemperatures,wells:wells.map(w=>({...w}))};}
 return {settings,configure,setOperating,step,snapshot};
}
