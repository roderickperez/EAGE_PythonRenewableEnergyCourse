import * as THREE from 'three';

// Orthographic projection keeps the map markers aligned with their islands.
export function drawMap(host, stations, buttons, activate) {
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-12, 12, 10, -10, .1, 100);
  camera.position.set(0, 23, 17);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.append(renderer.domElement);
  } catch { /* Map navigation also works with the projected HTML markers. */ }
  scene.add(new THREE.HemisphereLight(0xffffef, 0x769a99, 2.3));
  const sun = new THREE.DirectionalLight(0xfff0cb, 2.7);
  sun.position.set(-6, 15, 8);
  scene.add(sun);
  function solid(parent, geometry, color, x, y, z) {
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color, flatShading: true}));
    mesh.position.set(x, y, z);
    parent.add(mesh);
    return mesh;
  }
  function block(parent, color, x, y, z, w, h, d) {
    return solid(parent, new THREE.BoxGeometry(w,h,d), color,x,y,z);
  }
  function island(x,z,r) {
    const group = new THREE.Group();
    group.position.set(x,0,z);
    scene.add(group);
    solid(group,new THREE.CylinderGeometry(r,r*.83,.82,12),0xbd976a,0,-.23,0);
    solid(group,new THREE.CylinderGeometry(r*.94,r,.23,12),0x98ba84,0,.26,0);
    return group;
  }
  function bridge(x,z) {
    const group = new THREE.Group();
    scene.add(group);
    group.rotation.y = Math.atan2(x,z);
    const length = Math.hypot(x,z);
    for(let d=1.4; d<length-1.15; d+=.43) block(group,0xbe9568,0,.24,d,.76,.16,.36);
  }
  island(0,0,2.15);
  for(let i=0;i<3;i++) {
    const x = (i-1)*.68;
    solid(scene,new THREE.CylinderGeometry(.09,.12,.8,6),0x846244,x,.7,.15);
    solid(scene,new THREE.ConeGeometry(.56,1.5,7),0x467e60,x,1.45,.15);
  }
  const targets = [];
  let rotor;
  for(const s of stations) {
    bridge(s.x,s.z);
    const g = island(s.x,s.z,2);
    g.userData.station = s.id;
    targets.push(g);
    if(s.id==='solar') {
      for(const x of [-.75,.75]) {
        block(g,0x6d8489,x,.7,0,.14,.7,.15);
        const panel = block(g,0x244e72,x,1.15,0,1.36,.12,1.6);
        panel.rotation.x=.35;
        for(const offset of [-.45,0,.45]) block(panel,0x86aec4,offset,.07,0,.024,.02,1.55);
      }
    } else if(s.id==='wind') {
      solid(g,new THREE.CylinderGeometry(.10,.18,2.6,9),0xfffbec,0,1.65,0);
      rotor = new THREE.Group();
      rotor.position.set(0,2.9,.18);
      g.add(rotor);
      for(let i=0;i<3;i++) {
        const blade = new THREE.Group();
        blade.rotation.z=i*Math.PI*2/3;
        rotor.add(blade);
        block(blade,0xfffbec,0,.64,0,.16,1.3,.09);
      }
    } else if(s.id==='hydro') {
      block(g,0x60b3c9,0,.41,0,1.45,.12,2.7);
      block(g,0x728f96,0,.85,-.15,2.5,1,.45);
      for(const x of [-.5,.05,.6]) block(g,0x9ae3ef,x,.69,.17,.32,.63,.24);
    } else if(s.id==='geothermal') {
      for(const x of [-.8,.85]) {
        solid(g,new THREE.CylinderGeometry(.35,.59,.9,8),0xb86b4b,x,.84,0);
        solid(g,new THREE.IcosahedronGeometry(.33,0),0xffe8cb,x+.12,1.65,0);
      }
      block(g,0xc59366,0,.8,0,1.7,.17,.17);
    }
  }
  function render() {renderer?.render(scene,camera);}
  const point = new THREE.Vector3();
  function resize() {
    const {width,height} = host.getBoundingClientRect();
    const aspect = width/height;
    const halfWidth = Math.max(11.8,8.8*aspect);
    Object.assign(camera,{left:-halfWidth,right:halfWidth,top:halfWidth/aspect,bottom:-halfWidth/aspect});
    camera.updateProjectionMatrix();
    renderer?.setSize(width,height,false);
    stations.forEach((s,i)=>{
      point.set(s.x,.45,s.z+1.15).project(camera);
      buttons[i].style.left = ((point.x+1)*50)+'%';
      buttons[i].style.top = ((1-point.y)*50)+'%';
    });
    render();
  }
  new ResizeObserver(resize).observe(host);
  resize();
  if(!renderer) return;
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener('click',event=>{
    const rect=host.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,1-(event.clientY-rect.top)/rect.height*2),camera);
    let hit=ray.intersectObjects(targets,true)[0]?.object;
    while(hit && !hit.userData.station) hit=hit.parent;
    if(hit) activate(hit.userData.station);
  });
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  const visibility=new IntersectionObserver(([entry])=>{
    renderer.setAnimationLoop(entry.isIntersecting && !motion.matches ? time=>{
      rotor.rotation.z=time*.0008;
      render();
    } : null);
  });
  visibility.observe(host);
  motion.addEventListener('change',()=>{visibility.unobserve(host);visibility.observe(host);});
}
