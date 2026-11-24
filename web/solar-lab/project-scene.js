import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { sunDirection, RAD } from './model.js';
import { SITES } from './project-model.js';
import land from './natural-earth-land.geojson';
import { cloudBlockage, cloudDensity } from './project-clouds.js';
import { daylightFactor } from './project-light.js';

export function mountProjectScene(host, onPick, onSite) {
  const renderer=new T.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); host.append(renderer.domElement);
  const scene=new T.Scene(),daySky=new T.Color('#e6eff3'),nightSky=new T.Color('#26394f'); scene.background=daySky.clone();
  const camera=new T.OrthographicCamera(-12,12,10,-10,.1,120); camera.position.set(20,22,27);
  const orbit=new OrbitControls(camera,renderer.domElement); orbit.enablePan=false; orbit.minZoom=.7; orbit.maxZoom=2.5; orbit.target.set(0,1,0); orbit.update();
  const sky=new T.HemisphereLight(0xe8f7ff,0x6c795e,2.5);scene.add(sky); const light=new T.DirectionalLight(0xffe3a6,2); light.position.set(-8,18,8); scene.add(light);
  const globe=new T.Group(), farm=new T.Group(); scene.add(globe,farm); farm.visible=false;
  const mat=(color)=>new T.MeshStandardMaterial({color,roughness:.75});
  const mesh=(g,m,p,parent)=>{const o=new T.Mesh(g,m);o.position.set(...p);parent.add(o);return o;};
  // Natural Earth 1:110m land, public domain: https://www.naturalearthdata.com/
  const canvas=document.createElement('canvas');canvas.width=2048;canvas.height=1024;const c=canvas.getContext('2d');
  c.fillStyle='#80b8d0';c.fillRect(0,0,canvas.width,canvas.height);c.fillStyle='#a4c89a';c.strokeStyle='#79a889';c.lineWidth=1;
  for(const feature of land.features){const geom=feature.geometry;const polys=geom.type==='Polygon'?[geom.coordinates]:geom.coordinates;
    for(const poly of polys){c.beginPath();for(const ring of poly){ring.forEach(([lon,lat],i)=>{const x=(lon+180)/360*canvas.width,y=(90-lat)/180*canvas.height;i?c.lineTo(x,y):c.moveTo(x,y);});c.closePath();}c.fill('evenodd');c.stroke();}}
  const tex=new T.CanvasTexture(canvas);tex.colorSpace=T.SRGBColorSpace;tex.wrapS=T.RepeatWrapping;tex.offset.x=.5;
  const sphere=mesh(new T.SphereGeometry(5,64,40),new T.MeshBasicMaterial({map:tex}),[0,0,0],globe);
  function globePoint(lat,lon,r=5.05){const a=lat*RAD,b=lon*RAD;return new T.Vector3(-r*Math.cos(a)*Math.cos(b),r*Math.sin(a),r*Math.cos(a)*Math.sin(b));}
  const markers=SITES.map((site,index)=>{
    const group=new T.Group(), p=globePoint(site.lat,site.lon); group.position.copy(p);group.lookAt(p.clone().multiplyScalar(2));globe.add(group);
    const ring=mesh(new T.RingGeometry(.20,.29,32),new T.MeshBasicMaterial({color:'#fff1a9',side:T.DoubleSide}),[0,0,0],group);
    const dot=mesh(new T.SphereGeometry(.15,16,12),new T.MeshBasicMaterial({color:'#ffbd3f'}),[0,0,.03],group);dot.userData.site=index;ring.userData.site=index;
    return {group,ring,dot};
  });
  const terrain=mesh(new T.BoxGeometry(20,.35,20),mat('#718b6b'),[0,-.18,0],farm);
  const grid=new T.GridHelper(20,10,'#edf5e9','#b8d1bb');grid.position.y=.03;farm.add(grid);
  const ground=mesh(new T.PlaneGeometry(20,20),new T.MeshBasicMaterial({visible:false,side:T.DoubleSide}),[0,.04,0],farm);ground.rotation.x=-Math.PI/2;
  const racks=new T.Group();farm.add(racks);
  const selection=mesh(new T.RingGeometry(.45,.53,32),new T.MeshBasicMaterial({color:'#efbc48',side:T.DoubleSide}),[-9,.09,9],farm);selection.rotation.x=-Math.PI/2;
  const sun=mesh(new T.SphereGeometry(.42,24,16),new T.MeshBasicMaterial({color:'#ffc34c'}),[0,0,0],farm);
  const path=new T.Line(new T.BufferGeometry(),new T.LineDashedMaterial({color:'#d49130',dashSize:.18,gapSize:.13}));farm.add(path);
  const clouds=new T.Group(); farm.add(clouds);
  const cloudPixels=new Uint8Array(128*128*4);
  for(let y=0;y<128;y++)for(let x=0;x<128;x++){const i=(y*128+x)*4, density=cloudDensity((x/127-.5)*10,(.5-y/127)*6);cloudPixels[i]=250;cloudPixels[i+1]=252;cloudPixels[i+2]=253;cloudPixels[i+3]=Math.min(255,Math.round(420*Math.pow(density,.4)));}
  const cloudTexture=new T.DataTexture(cloudPixels,128,128,T.RGBAFormat);cloudTexture.colorSpace=T.SRGBColorSpace;cloudTexture.magFilter=T.LinearFilter;cloudTexture.minFilter=T.LinearFilter;cloudTexture.needsUpdate=true;
  const cloudMat=new T.MeshBasicMaterial({map:cloudTexture,transparent:true,depthWrite:false,side:T.DoubleSide,opacity:.8});
  const cloudGeo=new T.PlaneGeometry(10,6);
  for(let i=0;i<3;i++){const cloud=mesh(cloudGeo,cloudMat,[0,0,0],clouds);cloud.rotation.x=-Math.PI/2;}
  const shadowSize=64,shadowPixels=new Uint8Array(shadowSize*shadowSize*4);
  const shadowTexture=new T.DataTexture(shadowPixels,shadowSize,shadowSize,T.RGBAFormat);shadowTexture.magFilter=T.LinearFilter;shadowTexture.minFilter=T.LinearFilter;shadowTexture.needsUpdate=true;
  const panelShadowPixels=new Uint8Array(shadowSize*shadowSize*4);
  const panelShadowTexture=new T.DataTexture(panelShadowPixels,shadowSize,shadowSize,T.RGBAFormat);panelShadowTexture.magFilter=T.LinearFilter;panelShadowTexture.minFilter=T.LinearFilter;panelShadowTexture.needsUpdate=true;
  const shadowMaterial=new T.MeshBasicMaterial({map:shadowTexture,transparent:true,depthWrite:false,side:T.DoubleSide});
  const shadows=mesh(new T.PlaneGeometry(20,20),shadowMaterial,[0,.025,0],farm);shadows.rotation.x=-Math.PI/2;
  let panelMaterials=new Map(),labelTextures=[];
  const frame=mat('#9daeb4'),panel=mat('#1c4269'),cell=mat('#3975a0'),post=mat('#7d9094');
  function shadePanel(material){
    material.onBeforeCompile=shader=>{
      shader.uniforms.solarCloudShadow={value:panelShadowTexture};
      shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nvarying vec3 solarWorld;').replace('#include <worldpos_vertex>','#include <worldpos_vertex>\nsolarWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;');
      shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nuniform sampler2D solarCloudShadow; varying vec3 solarWorld;').replace('#include <color_fragment>','#include <color_fragment>\nvec2 cloudUV = vec2((solarWorld.x + 10.0) / 20.0, (10.0 - solarWorld.z) / 20.0);\nfloat cloudShade = texture2D(solarCloudShadow, clamp(cloudUV, 0.0, 1.0)).a;\ndiffuseColor.rgb *= 1.0 - 0.75 * cloudShade;');
    };
    material.customProgramCacheKey=()=> 'solar-panel-cloud-shadow-v1';
    return material;
  }
  function updateRacks(ids,tilt,azimuth){
    racks.traverse(o=>{o.geometry?.dispose();if(o.isSprite)o.material.dispose();});for(const texture of labelTextures)texture.dispose();labelTextures=[];for(const pair of panelMaterials.values())pair.forEach(m=>m.dispose());panelMaterials=new Map();racks.clear();
    for(const id of ids){const x=-9+(id%10)*2,z=9-Math.floor(id/10)*2;
      mesh(new T.CylinderGeometry(.045,.055,.6,8),post,[x,.3,z],racks);
      const rack=new T.Group();rack.position.set(x,.68,z);rack.rotation.y=Math.PI-azimuth*RAD;racks.add(rack);
      const face=new T.Group();face.rotation.x=tilt*RAD;rack.add(face);
      const panelMat=shadePanel(panel.clone()),cellMat=shadePanel(cell.clone());panelMaterials.set(id,[panelMat,cellMat]);
      mesh(new T.BoxGeometry(1.52,.10,1.06),frame,[0,0,0],face);mesh(new T.BoxGeometry(1.44,.025,.98),panelMat,[0,.065,0],face);
      for(let a=0;a<4;a++)for(let b=0;b<3;b++)mesh(new T.BoxGeometry(.33,.015,.29),cellMat,[(a-1.5)*.35,.085,(b-1)*.31],face);
      const label=document.createElement('canvas');label.width=128;label.height=64;const ctx=label.getContext('2d');ctx.fillStyle='#f4faf8';ctx.fillRect(4,4,120,56);ctx.fillStyle='#16384c';ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 36px system-ui';ctx.fillText(String(id+1),64,33);
      const labelTexture=new T.CanvasTexture(label);labelTexture.colorSpace=T.SRGBColorSpace;labelTextures.push(labelTexture);const sprite=new T.Sprite(new T.SpriteMaterial({map:labelTexture,depthTest:false}));sprite.position.set(x,1.35,z);sprite.scale.set(.75,.375,1);racks.add(sprite);
    }
  }
  let globeMode=true, transition=0, clock=0, dragging=null, rackKey='', pathKey='';
  camera.zoom=1.75;camera.updateProjectionMatrix();
  const ray=new T.Raycaster(), mouse=new T.Vector2();
  renderer.domElement.addEventListener('pointerdown',e=>{dragging={x:e.clientX,y:e.clientY};});
  renderer.domElement.addEventListener('pointerup',e=>{if(!dragging||Math.hypot(e.clientX-dragging.x,e.clientY-dragging.y)>5)return;
    const r=renderer.domElement.getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);ray.setFromCamera(mouse,camera);
    if(globeMode){const hits=ray.intersectObjects([sphere,...markers.flatMap(m=>[m.dot,m.ring])]);if(hits[0]?.object.userData.site!==undefined)onSite(hits[0].object.userData.site);return;}
    const hit=ray.intersectObject(ground)[0];if(hit){const col=Math.min(9,Math.max(0,Math.floor((hit.point.x+10)/2))),row=Math.min(9,Math.max(0,Math.floor((10-hit.point.z)/2)));onPick(row*10+col);}
  });
  const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const half=Math.max(12,15*h/w);Object.assign(camera,{left:-half*w/h,right:half*w/h,top:half,bottom:-half});camera.updateProjectionMatrix();renderer.setSize(w,h,false);});resize.observe(host);
  renderer.setAnimationLoop((time)=>{const dt=Math.min(.05,(time-clock)/1000);clock=time;if(globeMode){globe.rotation.y+=dt*.14;}else if(transition<1){transition=Math.min(1,transition+dt*1.7);globe.scale.setScalar(Math.max(.001,1-transition));farm.scale.setScalar(Math.max(.001,transition));if(transition===1)globe.visible=false;}renderer.render(scene,camera);});
  return {
    setSite(index){markers.forEach((m,i)=>{m.dot.material.color.set(i===index?'#fff5c7':'#ffbd3f');m.group.scale.setScalar(i===index?1.5:1);});},
    start(){globeMode=false;transition=0;farm.visible=true;farm.scale.setScalar(.001);camera.zoom=1;camera.updateProjectionMatrix();orbit.target.set(0,1,0);orbit.update();},
    showGlobe(){globeMode=true;globe.visible=true;globe.scale.setScalar(1);farm.visible=false;scene.background.copy(daySky);camera.zoom=1.75;camera.updateProjectionMatrix();},
    setPlaying(){},
    update(ids,settings,site,date,hour,dec,selected=0,weather,cloudState,panelResults){
      const nextRackKey=[ids.join(','),settings.tilt,settings.azimuth].join('|');if(nextRackKey!==rackKey){updateRacks(ids,settings.tilt,settings.azimuth);rackKey=nextRackKey;}
      selection.position.set(-9+(selected%10)*2,.09,9-Math.floor(selected/10)*2);
      const nextPathKey=`${site.lat}|${dec}`;if(pathKey!==nextPathKey){const points=[];for(let i=0;i<=288;i++){const s=sunDirection(i/12,site.lat,dec);if(s[1]>=0)points.push(new T.Vector3(s[0]*12,s[1]*12+.2,-s[2]*12));}path.geometry.dispose();path.geometry=new T.BufferGeometry().setFromPoints(points);path.computeLineDistances();pathKey=nextPathKey;}
      cloudMat.opacity=.95;clouds.children.forEach((cloud,i)=>{cloud.visible=cloudState.active;cloud.position.set(...cloudState.clouds[i].position);});
      for(let y=0;y<shadowSize;y++)for(let x=0;x<shadowSize;x++){const i=(y*shadowSize+x)*4, px=-10+20*x/(shadowSize-1),pz=10-20*y/(shadowSize-1),shade=cloudBlockage(cloudState,px,pz),panelShade=cloudBlockage(cloudState,px,pz,.68);shadowPixels[i]=22;shadowPixels[i+1]=38;shadowPixels[i+2]=45;shadowPixels[i+3]=Math.round(shade*230);panelShadowPixels[i+3]=Math.round(panelShade*255);}shadowTexture.needsUpdate=true;panelShadowTexture.needsUpdate=true;
      const s=sunDirection(hour,site.lat,dec),daylight=daylightFactor(s[1]);sun.visible=s[1]>0;sun.position.set(s[0]*12,s[1]*12+.2,-s[2]*12);light.position.copy(sun.position);light.intensity=daylight*1.5*(1-.65*weather.clouds);sky.intensity=.45+2.05*daylight;scene.background.copy(nightSky).lerp(daySky,daylight);
    },
    destroy(){renderer.setAnimationLoop(null);resize.disconnect();orbit.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});for(const texture of labelTextures)texture.dispose();panel.dispose();cell.dispose();tex.dispose();cloudTexture.dispose();shadowTexture.dispose();panelShadowTexture.dispose();renderer.dispose();renderer.domElement.remove();}
  };
}
