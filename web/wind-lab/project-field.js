import { FARM, windFieldSampler } from './project-model.js';

// Representative plasma colors, from low speed to undisturbed speed.
export const PLASMA = [[13,8,135],[84,3,160],[139,10,165],[185,50,137],
  [219,92,104],[244,136,73],[254,188,43],[240,249,33]];
export function plasma(t) {
  const v=Math.max(0,Math.min(1,t))*(PLASMA.length-1), i=Math.min(Math.floor(v),PLASMA.length-2), f=v-i;
  return PLASMA[i].map((c,j)=>Math.round(c+(PLASMA[i+1][j]-c)*f));
}

export function wakeRaster(ids, direction, speed, size=160) {
  const pixels=new Uint8Array(size*size*4);
  if (!ids.length || speed <= 0) return {pixels, min:0, max:speed};
  const sample=windFieldSampler(ids,direction,speed);
  const values=new Float64Array(size*size);
  let minimum=speed, hasWake=false;
  for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
    const local=sample((x+.5)*FARM/size,(y+.5)*FARM/size);
    values[y*size+x]=local;
    if(local < speed - 1e-8){minimum=Math.min(minimum,local);hasWake=true;}
  }
  if(!hasWake) return {pixels,min:0,max:speed};
  const span=speed-minimum;
  for(let index=0;index<values.length;index++) {
    const local=values[index];
    if(local >= speed-1e-8) continue;
    pixels.set([...plasma(span>1e-12?(local-minimum)/span:1),220],index*4);
  }
  return {pixels,min:minimum,max:speed};
}
export const wakePixels=(ids,direction,speed,size=160)=>wakeRaster(ids,direction,speed,size).pixels;

export function particleWind(direction, speed) {
  const theta=direction*Math.PI/180;
  // World X points east; world Z points south. Direction is where wind comes from.
  return { x:-Math.sin(theta)*speed*.10, z:Math.cos(theta)*speed*.10,
    angle:-theta, length:0.2 + Math.min(speed,25)*.04 };
}

export function advanceParticle(point, wind, dt) {
  const wrap=v=>((v+10)%20+20)%20-10;
  return {x:wrap(point.x+wind.x*dt), z:wrap(point.z+wind.z*dt)};
}

export function particleAppearance(localSpeed, freeStream) {
  const deficit=freeStream>0 ? Math.max(0,Math.min(1,1-localSpeed/freeStream)) : 0;
  return {deficit, opacity:0.75-0.40*deficit};
}

export function localParticle(point, direction, sample, dt=0, options={}) {
  const speedAt=p=>sample((p.x+10)*FARM/20,(10-p.z)*FARM/20);
  const before=speedAt(point);
  const wind=particleWind(direction,before), appearance=particleAppearance(before,options.freeStream ?? before);
  // Illustrative wake eddies: stronger lateral motion with two smoothly varying scales.
  const phase=(options.elapsed ?? 0)*2.1+(options.phase ?? 0);
  const disturbance=appearance.deficit*3.0*(.75*Math.sin(phase)+.25*Math.sin(phase*1.9+point.x*.6+point.z*.4));
  const theta=direction*Math.PI/180;
  const next=advanceParticle(point,{x:wind.x-Math.cos(theta)*disturbance,z:wind.z-Math.sin(theta)*disturbance},dt);
  const speed=speedAt(next);
  return {...next, speed, wind:particleWind(direction,speed), ...particleAppearance(speed,options.freeStream ?? speed)};
}

// Visual animation, not the turbine's physical RPM or a mechanical model.
export function rotorRate(speed) {
  return speed < 3 || speed > 25 ? 0 : 1.4*Math.min(speed/12,1);
}
