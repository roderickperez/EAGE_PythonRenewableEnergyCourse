import { sunDirection } from './model.js';

export const CLOUD_HEIGHT = 6;
export const CLOUD_LOBES = [
  [-2.3,-.25,1.9,1.05,.65],[-.8,-.7,2,1.4,.9],[1,-.25,2.1,1.1,.82],
  [2.35,.4,1.4,.8,.65],[-1.3,.8,1.7,.9,.72],[.3,.75,1.9,.9,.8],
];

// Identical smooth density field drives both the cloud texture and the shadow.
export function cloudDensity(x,z) {
  let transmission=1;
  for(const [cx,cz,rx,rz,opacity] of CLOUD_LOBES){
    const r2=((x-cx)/rx)**2+((z-cz)/rz)**2;
    transmission*=1-opacity*Math.exp(-2*r2);
  }
  return 1-transmission;
}
const wrap=x=>((x+22)%44+44)%44-22;

// Fixed wind direction: clouds drift east in straight parallel lines.
export function cloudState(hour,latitude,declination,phase,cloudCover) {
  const sun=sunDirection(hour,latitude,declination);
  const clouds=[[-13,-5],[1,1],[15,7]].map(([x,z])=>({position:[wrap(x+phase*.8),CLOUD_HEIGHT,z]}));
  return {sun,clouds,active:sun[1]>0.02,opacity:Math.min(.82,.42+.48*cloudCover)};
}

export function gridPosition(id){return {x:-9+(id%10)*2,z:9-Math.floor(id/10)*2};}

// Trace a ray from an individual panel or ground point to the cloud plane.
export function cloudBlockage(state,x,z,height=0) {
  if(!state.active)return 0;
  const [east,up,north]=state.sun;
  const rayX=x+east/up*(CLOUD_HEIGHT-height),rayZ=z-north/up*(CLOUD_HEIGHT-height);
  let transmission=1;
  for(const cloud of state.clouds){const [cx,,cz]=cloud.position;transmission*=1-state.opacity*cloudDensity(rayX-cx,rayZ-cz);}
  return 1-transmission;
}
export function applyCloudShadow(weather, blockage) {
  return {...weather,dni:weather.dni*(1-blockage),cloudBlockage:blockage};
}
