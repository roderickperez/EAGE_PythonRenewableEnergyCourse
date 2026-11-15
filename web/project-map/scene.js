import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function mountProjectScene(host) {
  const scene = new T.Scene();
  const daylight = new T.Color('#e6eff3'), twilight = new T.Color('#bdceda');
  scene.background = daylight.clone();
  const render = new T.WebGLRenderer({antialias:true});
  render.setPixelRatio(Math.min(devicePixelRatio,2));
  render.shadowMap.enabled = true;
  render.toneMapping = T.ACESFilmicToneMapping;
  host.append(render.domElement);
  const camera = new T.OrthographicCamera(-14,14,8,-8,.1,100);
  const target = new T.Vector3(0,.6,0);
  const view = new T.Vector3(13,16,25);
  camera.position.copy(view);
  const orbit = new OrbitControls(camera,render.domElement);
  orbit.target.copy(target);
  Object.assign(orbit,{enablePan:false,enableZoom:false,minPolarAngle:.58,maxPolarAngle:1.17,minAzimuthAngle:-.46,maxAzimuthAngle:.65});
  orbit.update();
  const sky = new T.HemisphereLight(0xe8f5ff,0x87825f,2);
  const sun = new T.DirectionalLight(0xffebc5,3.1);
  sun.position.set(-9,17,8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024,1024);
  Object.assign(sun.shadow.camera,{left:-18,right:18,top:17,bottom:-17});
  sun.shadow.normalBias=.05;
  scene.add(sky,sun);
  const mat = (color,metalness=0) => new T.MeshStandardMaterial({color,metalness,roughness:.62});
  const rock=mat('#9a8c7b'), shale=mat('#78858a'), grass=mat('#738767'), dark=mat('#384d54',.5),
    white=mat('#e0e6e6'), concrete=mat('#a9b7af'), steel=mat('#819099',.45), warm=mat('#be6949'),
    water=mat('#2b8ba7'), cool=mat('#4296b5'), purple=mat('#8d66a2'), glass=mat('#235184',.27);
  function mesh(geometry,material,position,parent=scene){
    const item=new T.Mesh(geometry,material);
    item.position.set(...position);
    item.castShadow=item.receiveShadow=true;
    parent.add(item);
    return item;
  }
  const box=(size,material,p,parent)=>mesh(new T.BoxGeometry(...size),material,p,parent);
  const cylinder=(top,bottom,length,material,p,parent)=>mesh(new T.CylinderGeometry(top,bottom,length,18),material,p,parent);
  const sphere=(r,material,p)=>mesh(new T.SphereGeometry(r,18,10),material,p);
  function strut(a,b,r,material){
    const start=new T.Vector3(...a), delta=new T.Vector3(...b).sub(start);
    const item=cylinder(r,r,delta.length(),material,start.addScaledVector(delta,.5).toArray());
    item.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),delta.normalize());
  }
  function curve(points,r,material){
    const path=new T.CatmullRomCurve3(points.map(point=>new T.Vector3(...point)),false,'centripetal');
    mesh(new T.TubeGeometry(path,36,r,8,false),material,[0,0,0]);
    return path;
  }
  const labelCanvases=[];
  function label(title,border,p){
    const canvas=document.createElement('canvas');
    canvas.width=512; canvas.height=104;
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='#f9fcfb'; ctx.strokeStyle=border; ctx.lineWidth=6;
    ctx.beginPath(); ctx.roundRect(6,6,500,92,20); ctx.fill(); ctx.stroke();
    ctx.fillStyle='#173b37'; ctx.font='700 40px system-ui'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(title,256,53,470);
    const texture=new T.CanvasTexture(canvas);
    texture.colorSpace=T.SRGBColorSpace;
    const marker=new T.Sprite(new T.SpriteMaterial({map:texture,depthTest:false,toneMapped:false}));
    marker.position.set(...p);
    marker.scale.set(3.62,.735,1);
    marker.renderOrder=5;
    scene.add(marker);
    labelCanvases.push(texture);
  }
  const terrainWidth=23, terrainDepth=13;
  // Keep one continuous cutaway beneath all the sites.
  box([terrainWidth,.34,terrainDepth],grass,[0,-.15,0]);
  box([terrainWidth,1.18,terrainDepth],rock,[0,-.91,0]);
  box([terrainWidth,.49,terrainDepth],mat('#638896'),[0,-1.745,0]);
  box([terrainWidth,.25,terrainDepth],shale,[0,-2.115,0]);
  // Valley terrain rises behind the wind and hydro sites.
  const ridgeGeometry=new T.PlaneGeometry(22,4.4,56,14);
  ridgeGeometry.rotateX(-Math.PI/2);
  const ridge=ridgeGeometry.attributes.position;
  for(let i=0;i<ridge.count;i++){
    const x=ridge.getX(i), z=ridge.getZ(i);
    const riverValley=Math.max(0,Math.min(1,(7.45-x)/1.6));
    ridge.setY(i,.03+Math.max(0,1-(z/2.2)**2) *
      (2.6*Math.exp(-(((x+4)/4)**2))+2.1*Math.exp(-(((x-5)/3)**2)))*
      riverValley*(1+.08*Math.sin(2*x)*Math.cos(2*z)));
  }
  ridgeGeometry.computeVertexNormals();
  mesh(ridgeGeometry,mat('#869889'),[0,0,-4.3]);
  const treeGreen=mat('#507461'),trunk=mat('#78624b');
  for(const [x,z,s] of [[-10,-2,.75],[-10,0,.86],[-3.4,-2.3,.75],[-1.6,-1.1,.7],[2,-2.6,.9],
    [10.6,-2,.75],[10.72,3.5,.75],[-9,5.35,.76],[-10.2,5.12,1.05],[-1.1,5.45,.7],[6.1,4.92,.66],[7.3,.8,.65]]){
    cylinder(.055,.085,.65*s,trunk,[x,.325*s,z]);
    for(let i=0;i<3;i++)mesh(new T.ConeGeometry((.4-i*.065)*s,.75*s,10),treeGreen,[x,(.73+.25*i)*s,z]);
  }

  // Wind ridge, with exposed rotor, nacelle and stepped pads.
  const rotors=[];
  const bladeShape=new T.Shape();
  bladeShape.moveTo(-.105,.13); bladeShape.quadraticCurveTo(-.28,.47,-.21,.94);
  bladeShape.quadraticCurveTo(-.11,1.48,.07,1.65); bladeShape.quadraticCurveTo(.145,1.22,.155,.63);
  bladeShape.lineTo(.1,.13); bladeShape.closePath();
  const blade=new T.ExtrudeGeometry(bladeShape,{depth:.065,bevelEnabled:true,bevelSize:.025,bevelThickness:.018,bevelSegments:2,steps:1});
  for(const [x,z,scale] of [[-7,-2.65,1],[-5.05,-3.35,.76]]){
    const base=z < -3 ? 1.02 : .65;
    cylinder(.44*scale,.52*scale,.18,concrete,[x,base,z]);
    cylinder(.135*scale,.28*scale,3.38*scale,white,[x,base+1.69*scale,z]);
    box([.45*scale,.37*scale,.85*scale],white,[x,base+3.41*scale,z]);
    const rotor=new T.Group();
    rotor.position.set(x,base+3.41*scale,z+.48*scale);
    rotor.scale.setScalar(scale);
    scene.add(rotor); rotors.push(rotor);
    mesh(new T.SphereGeometry(.23,18,12),white,[0,0,.07],rotor);
    for(let i=0;i<3;i++)mesh(blade,white,[0,0,0],rotor).rotation.z=i*2*Math.PI/3;
  }
  label('Wind ridge','#7595a1',[-6.9,5.83,-2.8]);

  // PV tables, silver frame, central inverter.
  const panelSurface=glass;
  for(let row=0;row<3;row++) for(let col=0;col<3;col++){
    const x=-9.05+col*1.64,z=1.12+row*1.44;
    const rack=new T.Group();
    rack.position.set(x,.57,z);
    rack.rotation.x=.42;
    scene.add(rack);
    box([1.54,.09,1.18],steel,[0,0,0],rack);
    for(let i=0;i<6;i++)for(let j=0;j<4;j++)
      box([.224,.024,.26],panelSurface,[(i-2.5)*.24,.058,(j-1.5)*.275],rack);
    cylinder(.036,.05,.5,dark,[x-.4,.25,z]);
    cylinder(.036,.05,.5,dark,[x+.4,.25,z]);
  }
  box([.65,.62,.55],white,[-4.6,.33,2.6]);
  box([.23,.15,.035],dark,[-4.6,.4,2.89]);
  label('Solar PV','#397ca8',[-7.25,1.9,4.5]);

  // Run-of-river diversion: intake and power branch beside the main channel.
  const river=[[8.4,.06,-6.35],[8.15,.06,-4.25],[8.55,.06,-2.3],[9.4,.06,-.2],[9.5,.06,2.8],[9.35,.06,6.5]];
  const riverPath=new T.CatmullRomCurve3(river.map(p=>new T.Vector3(...p)));
  const banks=[];
  for(let i=0;i<=52;i++){
    const p=riverPath.getPointAt(i/52), t=riverPath.getTangentAt(i/52);
    banks.push(p.x-t.z*.82,p.y,p.z+t.x*.82,p.x+t.z*.82,p.y,p.z-t.x*.82);
  }
  const faces=[];
  for(let i=0;i<52;i++){const a=i*2;faces.push(a,a+2,a+1,a+1,a+2,a+3);}
  const bed=new T.BufferGeometry();
  bed.setAttribute('position',new T.Float32BufferAttribute(banks,3));
  bed.setIndex(faces); bed.computeVertexNormals();
  water.side=T.DoubleSide;
  mesh(bed,water,[0,0,0]);
  const waterDashes=[];
  for(let i=0;i<12;i++){
    const item=box([.055,.02,.37],new T.MeshBasicMaterial({color:'#86d0de'}),[0,0,0]);
    item.castShadow=false; waterDashes.push(item);
  }
  box([1.78,.29,.29],concrete,[8.53,.17,-2.56]);
  for(const x of [7.65,8.28,8.86,9.42]) box([.12,.4,.39],steel,[x,.2,-2.56]);
  box([1.42,.78,1.1],white,[6.43,.41,-1.52]);
  box([1.53,.12,1.2],dark,[6.43,.86,-1.52]);
  box([.72,.28,.05],glass,[6.43,.5,-.95]);
  curve([[7.87,.31,-2.59],[7.44,.22,-2.23],[6.78,.22,-1.94],[6.35,.22,-1.61]],.14,cool);
  curve([[6.56,.21,-1.29],[7.05,.17,-.91],[8.22,.12,-.28],[9.04,.09,.02]],.14,cool);
  label('Run-of-river hydro','#3898aa',[7.65,2,-3.62]);

  // Expose the two well paths on the front face of the shared cutaway.
  box([4.25,.14,2.18],concrete,[1.14,.09,5.15]);
  const wellFace=terrainDepth/2+.06;
  for(const [x,color] of [[-.72,warm],[2.93,cool]]){
    cylinder(.27,.32,.21,steel,[x,.18,6.26]);
    curve([[x,-1.79,wellFace],[x,-1.45,wellFace],[x,.19,wellFace],[x,.42,6.35],[x,.42,5.96]],.095,color);
  }
  curve([[-.72,.42,5.96],[-.72,.5,5.54],[-.08,.5,5.3]],.095,warm);
  curve([[-.08,.27,5.38],[.15,.28,5.89],[2.38,.29,5.91],[2.93,.42,5.96]],.085,cool);
  box([.76,.73,.66],steel,[-.06,.49,5.18]);
  for(let i=0;i<5;i++)box([.8,.033,.7],white,[-.06,.23+i*.116,5.18]);
  const turbine=cylinder(.29,.29,.57,steel,[.74,.55,5.09]); turbine.rotation.z=Math.PI/2;
  const generator=cylinder(.29,.29,.54,warm,[1.47,.55,5.09]); generator.rotation.z=Math.PI/2;
  strut([.9,.55,5.09],[1.47,.55,5.09],.07,dark);
  box([.8,.61,.88],steel,[2.47,.39,5.16]);
  for(const x of [2.31,2.62])cylinder(.12,.12,.057,dark,[x,.73,5.16]);
  curve([[.18,.7,5.15],[.47,.8,5.1],[.63,.66,5.1]],.082,purple);
  curve([[.79,.33,5],[1.13,.22,4.77],[2.27,.22,4.78],[2.46,.29,5.07]],.074,purple);
  curve([[2.44,.29,5.32],[2.02,.25,5.62],[.22,.24,5.57],[.04,.26,5.27]],.064,cool);
  label('Geothermal','#a86d58',[1.34,1.86,6.04]);

  // Four source branches meet at the sole shared bus before the community.
  const bus=[-.78,.43,.42];
  box([2.02,.085,1.73],concrete,[-.8,.06,.45]);
  box([1.43,.59,.81],dark,[-.8,.35,.45]);
  for(let i=0;i<7;i++)box([1.5,.04,.88],steel,[-.8,.11+i*.082,.45]);
  for(const x of [-1.24,-.8,-.36])cylinder(.046,.057,.21,white,[x,.74,.45]);
  const branches=[
    [[-6.77,.1,-2.16],[-5.65,.12,-1.05],[-2.15,.1,-.2],bus],
    [[-4.55,.13,2.59],[-3.75,.13,2.22],[-2.37,.12,1.32],bus],
    [[6.21,.1,-1.15],[4.26,.12,-1.12],[1.48,.1,-1.08],bus],
    [[1.43,.15,4.46],[.07,.14,3.77],[-.42,.12,2.06],bus],
  ];
  for(const branch of branches)curve(branch,.043,dark);
  label('Isolated grid','#7a8f81',[-.72,1.73,.42]);
  for(const [x,z] of [[.98,.64],[2.98,.5]]) {
    for(const sign of [-1,1]) {
      strut([x+sign*.28,.03,z],[x+sign*.09,2.34,z],.029,dark);
      strut([x+sign*.27,.13,z],[x-sign*.16,1.67,z],.019,dark);
    }
    strut([x-.52,2.01,z],[x+.52,2.01,z],.033,dark);
  }
  for(const offset of [-.36,0,.36])curve([[-.8+offset,.86,.45],[.98+offset,1.93,.64],[1.98+offset,1.63,.54],
    [2.98+offset,1.93,.5],[4.05+offset,.63,.65]],.022,dark);
  const windows=new T.MeshStandardMaterial({color:'#7c876d',emissive:'#d5a75b',emissiveIntensity:.25});
  const roofMaterial=mat('#6e7677');
  for(const [x,z,w,h] of [[4.2,1,.78,.76],[5.45,1.12,.95,.86],[4.35,2.45,.9,.77],[5.78,2.64,1.04,1.03],[6.75,2.33,.70,.62]]) {
    box([w,h,.72],white,[x,h/2,z]);
    const roof=mesh(new T.ConeGeometry(w*.72,.37,4),roofMaterial,[x,h+.18,z]);
    roof.rotation.y=Math.PI/4;
    box([.20,.23,.025],windows,[x-.14,h*.6,z+.365]);
    box([.17,.27,.025],dark,[x+.15,.15,z+.365]);
  }
  label('Community load','#9f945b',[5.69,2.35,2.47]);
  const sunDisk=sphere(.37,new T.MeshBasicMaterial({color:'#fac654'}),[-9,5,1.5]);
  sunDisk.castShadow=false;
  let row,playing=false,visible=true,dirty=true,last=0,flowPhase=0;
  const resize=new ResizeObserver(()=>{
    const {clientWidth:w,clientHeight:h}=host;
    if(!w||!h)return;
    const half=Math.max(7.8,13.9*h/w);
    Object.assign(camera,{left:-half*w/h,right:half*w/h,top:half,bottom:-half});
    camera.updateProjectionMatrix();
    render.setSize(w,h,false); dirty=true;
  });
  const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;dirty=true;});
  resize.observe(host); observer.observe(host);
  orbit.addEventListener('change',()=>{dirty=true;});
  render.setAnimationLoop(time=>{
    const dt=Math.min(.05,(time-last)/1000); last=time;
    if(!row||!visible||(!dirty&&!playing))return;
    if(playing){
      // These speeds illustrate measured wind and river inputs. The map has no dispatch model.
      rotors.forEach((rotor,i)=>{rotor.rotation.z-=dt*row.wind_speed_ms*(.065+i*.008);});
      flowPhase=(flowPhase+dt*Math.min(.35,row.river_flow_m3s/120))%1;
    }
    for(let i=0;i<waterDashes.length;i++){
      const u=(i/waterDashes.length+flowPhase)%1;
      const p=riverPath.getPointAt(u), direction=riverPath.getTangentAt(u);
      waterDashes[i].position.copy(p);
      waterDashes[i].position.y+=.02;
      waterDashes[i].position.x+=(i%3-1)*.32;
      waterDashes[i].rotation.y=Math.atan2(direction.x,direction.z);
    }
    render.render(scene,camera); dirty=false;
  });
  return {
    update(observation,animate){
      row=observation; playing=animate;
      const light=Math.min(1,Math.max(0,row.irradiance_wm2/950));
      scene.background.copy(twilight).lerp(daylight,light);
      sunDisk.visible=light>.015;
      sunDisk.position.set(-9,3.7+light*2.1,-.4);
      sun.intensity=.75+2.6*light;
      sky.intensity=1.25+.75*light;
      panelSurface.emissive.set('#235184');
      panelSurface.emissiveIntensity=.16*light;
      windows.emissiveIntensity=Math.min(.95,row.demand_mw/45)*(.7-.3*light);
      dirty=true;
    },
    destroy(){
      render.setAnimationLoop(null);
      resize.disconnect(); observer.disconnect(); orbit.dispose();
      const geometries=new Set(),materials=new Set();
      scene.traverse(item=>{if(item.geometry)geometries.add(item.geometry);if(item.material)materials.add(item.material);});
      geometries.forEach(geometry=>geometry.dispose());
      materials.forEach(material=>material.dispose());
      labelCanvases.forEach(texture=>texture.dispose());
      render.dispose();render.domElement.remove();
    },
  };
}
