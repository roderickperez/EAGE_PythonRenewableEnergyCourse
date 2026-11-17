import { FARM, windFieldSampler } from './project-model.js';

// Representative plasma colors, from low speed to undisturbed speed.
export const PLASMA = [[13,8,135],[84,3,160],[139,10,165],[185,50,137],
  [219,92,104],[244,136,73],[254,188,43],[240,249,33]];
export function plasma(t) {
  const v=Math.max(0,Math.min(1,t))*(PLASMA.length-1), i=Math.min(Math.floor(v),PLASMA.length-2), f=v-i;
  return PLASMA[i].map((c,j)=>Math.round(c+(PLASMA[i+1][j]-c)*f));
}

export function wakePixels(ids, direction, speed, size=160) {
  const pixels=new Uint8Array(size*size*4);
  if (!ids.length || speed <= 0) return pixels;
  const sample=windFieldSampler(ids,direction,speed);
  for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
    const local=sample((x+.5)*FARM/size,(y+.5)*FARM/size);
    if (local >= speed - 1e-8) continue;
    const offset=(y*size+x)*4, color=plasma(local/speed);
    pixels.set([...color,220],offset);
  }
  return pixels;
}

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
