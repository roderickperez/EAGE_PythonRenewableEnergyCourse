import * as T from 'three';
import {HydroPlant} from '../hydro-lab/scene.js';
import {computeHydro, DEFAULT_INPUTS} from '../hydro-lab/model.js';
import {buildWindPlant} from '../wind-lab/scene.js';
import {operatingPoint} from '../wind-lab/model.js';
import {defaults as windDefaults} from '../wind-lab/lab.js';
import {buildSolarPlant} from '../solar-lab/scene.js';
import {defaults as solarDefaults} from '../solar-lab/lab.js';
import {solarPower, RAD} from '../solar-lab/model.js';
import {buildGeothermalPlant} from '../geothermal-lab/scene.js';
import {geothermal, defaults as heatDefaults} from '../geothermal-lab/model.js';

export function installPlant(id, group) {
  let plant, result, configure, advance, control;
  switch(id) {
    case 'hydro':
      plant=new HydroPlant(group);
      plant.build();
      control=['Gross head',5,250,DEFAULT_INPUTS.grossHeadM,'m'];
      configure=head=>{
        result=computeHydro({...DEFAULT_INPUTS,grossHeadM:head});
        plant.update(result);
        return result.plantMW.toFixed(2)+' MW';
      };
      advance=dt=>plant.advance(dt);
      break;
    case 'wind':
      plant=buildWindPlant(group);
      control=['Wind speed',0,30,8,'m/s'];
      configure=speed=>{
        const input={...windDefaults,speed,heightMode:'hub',outputMode:'curve'};
        result=operatingPoint(input);
        plant.head.position.set(plant.towerX,7.85,0);
        plant.head.rotation.y=input.direction*RAD;
        plant.air.position.copy(plant.head.position);
        plant.air.rotation.y=plant.head.rotation.y;
        return result.gridMW.toFixed(2)+' MW';
      };
      advance=dt=>plant.advance(dt,result,true);
      break;
    case 'solar':
      plant=buildSolarPlant(group);
      control=['Local solar time',0,24,10,'h'];
      configure=hour=>{
        result=solarPower(solarDefaults,hour);
        plant.rack.rotation.y=solarDefaults.azimuth*RAD;
        plant.panel.rotation.x=solarDefaults.tilt*RAD;
        plant.sun.position.copy(plant.sunPoint(result.sun));
        plant.sun.visible=result.elevation>0;
        plant.led.color.set(result.dc>0?'#67db9f':'#425a63');
        plant.output.color.set(result.ac>0?'#ffc153':'#56686d');
        const direction=new T.Vector3(...result.sun);
        const segments=[];
        for(const x of [-1.15,0,1.15]) for(const z of [-.7,.7]){
          const cell=group.worldToLocal(plant.panel.localToWorld(new T.Vector3(x,.15,z)));
          segments.push(cell.clone().addScaledVector(direction,3.2),cell);
        }
        plant.rays.geometry.dispose();
        plant.rays.geometry=new T.BufferGeometry().setFromPoints(segments);
        plant.rays.visible=result.beam>0;
        return result.ac.toFixed(0)+' W';
      };
      advance=dt=>plant.advance(dt,result,true);
      break;
    case 'geothermal':
      plant=buildGeothermalPlant(group);
      control=['Brine flow',0,100,heatDefaults.flow,'kg/s'];
      configure=flow=>{
        result=geothermal({...heatDefaults,flow});
        return result.exported.toFixed(2)+' MW';
      };
      advance=dt=>plant.advance(dt,{brine:control[3]/30,cycle:result.thermal/12,gross:result.gross,grid:result.exported},true);
      break;
  }
  function set(value) {control[3]=value; return configure(value);}
  const initial=set(control[3]);
  return {control,set,advance,initial};
}
