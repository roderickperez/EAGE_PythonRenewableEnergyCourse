import { HOURS_PER_SECOND } from './project-game.js';
// Session estimates from the model, not measured or forecast production.
export function createHistory(now=0) {
  let last=now, elapsed=0, current={results:[],freestream:0}, energy={}, points=[];
  function advance(time, active=true) {
    const seconds=Math.max(0,(time-last)/1000);last=time;
    if(active){
      elapsed+=seconds;
      for(const turbine of current.results) energy[turbine.id]=(energy[turbine.id] || 0)+turbine.power*seconds*HOURS_PER_SECOND/1000;
    }
  }
  function snapshot() {
    const speeds=Object.fromEntries(current.results.map(p=>[p.id,p.speed]));
    const total=Object.values(energy).reduce((a,b)=>a+b,0);
    points.push({time:elapsed,total,energy:{...energy},speeds,freestream:current.freestream});
    if(points.length>300) points.shift();
    return {time:elapsed,total,energy:{...energy},points};
  }
  return {
    setModel(result,freestream,time,active=true){advance(time,active);current={results:result.results.map(p=>({...p})),freestream};return snapshot();},
    tick(time,active=true){advance(time,active);return snapshot();},
    reset(time){last=time;elapsed=0;energy={};points=[];current={results:[],freestream:0};return snapshot();},
  };
}

// A repeating teaching scenario, not a real climate record or forecast.
export function seasonalWind(seconds, variation={speed:0,direction:0}) {
  const yearSeconds=120, phase=((Math.max(0,seconds)%yearSeconds)/yearSeconds)*2*Math.PI;
  const speed=Math.round(Math.max(5,Math.min(15.5,10+3*Math.cos(phase)+variation.speed))*2)/2;
  const direction=((Math.round(35*Math.sin(phase)+15*Math.sin(phase*2)+variation.direction)%360)+360)%360;
  const month=Math.floor((phase/(2*Math.PI))*12);
  return {speed,direction,month};
}

// Mean-reverting random variability; future values are generated only as time passes.
export function nextSeasonalVariation(previous, random=Math.random) {
  return {speed:Math.max(-2.5,Math.min(2.5,.86*previous.speed+(random()-.5)*1.4)),
    direction:Math.max(-25,Math.min(25,.86*previous.direction+(random()-.5)*14))};
}
