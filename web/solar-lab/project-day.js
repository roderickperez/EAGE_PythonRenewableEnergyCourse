import { cloudState, cloudBlockage, gridPosition, applyCloudShadow } from './project-clouds.js';
import { declination, inputs } from './project-model.js';
import { solarPower } from './model.js';

// A complete synthetic day using the current weather and the same cloud paths.
// Cloud phase advances 2.5 units per simulated hour at every playback speed.
export function fullSolarDay(site,date,hour,settings,weather,cloudPhase,ids) {
  const decl=declination(date),startPhase=cloudPhase-hour/.4,steps=144,points=[];
  function sample(time){const clouds=cloudState(time,site.lat,decl,startPhase+time/.4,weather.clouds), values={};
    for(const id of ids){const p=gridPosition(id),shade=cloudBlockage(clouds,p.x,p.z,.68);
      const input=inputs(site,date,{...settings,count:1},applyCloudShadow(weather,shade));
      values[id]=solarPower(input,time).ac;
    }
    return {hour:time,values};
  }
  for(let i=0;i<=steps;i++)points.push(sample(i/6));
  if(hour>0&&hour<24&&Math.abs(hour*6-Math.round(hour*6))>1e-8){const index=Math.ceil(hour*6);points.splice(index,0,sample(hour));}
  let energy=0;
  for(let i=1;i<points.length;i++)for(const id of ids)energy+=(points[i-1].values[id]+points[i].values[id])*.5*(points[i].hour-points[i-1].hour)/1000;
  return {points,energy};
}
