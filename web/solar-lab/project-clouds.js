import { sunDirection } from './model.js';

// Clouds travel across the sun direction as seen from the centre of the site.
// This is a shared, illustrative farm-wide shadow, not a local weather forecast.
export function cloudState(hour, latitude, declination, phase, cloudCover) {
  const s=sunDirection(hour,latitude,declination);
  const horizontal=Math.hypot(s[0],s[2]);
  const tangent=horizontal>1e-6?[s[2]/horizontal,0,-s[0]/horizontal]:[1,0,0];
  const wrap=x=>((x+6)%12+12)%12-6;
  const offsets=[0,4,8].map(offset=>wrap(phase*.9+offset));
  const active=s[1]>0;
  const clouds=offsets.map(offset=>({offset,position:[s[0]*12+tangent[0]*offset,s[1]*12+.2,s[2]*-12-tangent[2]*offset]}));
  const overlap=active?Math.max(...offsets.map(offset=>Math.exp(-.5*(offset/1.05)**2))):0;
  const blockage=Math.min(.85,overlap*(.35+.6*cloudCover));
  return {clouds,blockage,active};
}

export function applyCloudShadow(weather, cloud) {
  return {...weather, dni:weather.dni*(1-cloud.blockage), cloudBlockage:cloud.blockage};
}
