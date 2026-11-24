import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function mountCoaxialCutaway(host) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e6eff3');
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.domElement.setAttribute('aria-hidden', 'true');
  host.append(renderer.domElement);

  const camera = new T.OrthographicCamera(-5,5,4,-4,.1,60);
  camera.position.set(7,5,9);
  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls,{enablePan:false,minZoom:.75,maxZoom:2.5});
  scene.add(new T.HemisphereLight(0xe8faff,0x93877b,2.1));
  const light = new T.DirectionalLight(0xfff4de,2);
  light.position.set(4,7,6);
  scene.add(light);

  const assembly = new T.Group();
  assembly.rotation.z = -Math.PI / 2;
  scene.add(assembly);
  const materials = {
    tube: new T.MeshStandardMaterial({ color:'#cf6945', roughness:.54 }),
    wall: new T.MeshStandardMaterial({ color:'#7b8a8f', roughness:.48, metalness:.25 }),
    annulus: new T.MeshStandardMaterial({ color:'#258ab5', transparent:true, opacity:.42, depthWrite:false, side:T.DoubleSide }),
    face: new T.MeshStandardMaterial({ color:'#5bb8ce', roughness:.62, side:T.DoubleSide }),
    casing: new T.MeshStandardMaterial({ color:'#576f7a', roughness:.5, metalness:.3, side:T.DoubleSide }),
  };

  const draw = () => renderer.render(scene,camera);
  const resize = new ResizeObserver(() => {
    const {clientWidth:width,clientHeight:height} = host;
    if (!width || !height) return;
    const aspect = width/height, half = Math.max(3.85,4.25/aspect);
    Object.assign(camera,{left:-half*aspect,right:half*aspect,top:half,bottom:-half});
    camera.updateProjectionMatrix();
    renderer.setSize(width,height,false);
    draw();
  });
  resize.observe(host);
  controls.addEventListener('change',draw);
  controls.update();

  function sleeve(inner, outer, length, material, angle = 0, sweep = Math.PI*2) {
    const points = [[inner,-length/2],[outer,-length/2],[outer,length/2],[inner,length/2],[inner,-length/2]]
      .map(([r,h])=>new T.Vector2(r,h));
    assembly.add(new T.Mesh(new T.LatheGeometry(points,64,angle,sweep),material));
  }

  return {
    update({inner,outer,bore}) {
      for (const item of [...assembly.children]) {
        item.geometry.dispose();
        item.removeFromParent();
      }
      const ri = inner*9, ro = outer*9, rb = bore*9;
      assembly.add(new T.Mesh(new T.CylinderGeometry(ri,ri,4.55,64),materials.tube));
      sleeve(ri,ro,4.4,materials.wall);
      const water = new T.Mesh(new T.CylinderGeometry(rb,rb,4.05,64,1,true),materials.annulus);
      assembly.add(water);
      const face = new T.Mesh(new T.RingGeometry(ro,rb,64),materials.face);
      face.rotation.x = -Math.PI/2;
      face.position.y = 2.026;
      assembly.add(face);
      sleeve(rb,rb+.065,4.04,materials.casing,1.4,Math.PI*1.45);
      draw();
    },
    destroy() {
      resize.disconnect();
      controls.dispose();
      assembly.traverse(item=>item.geometry?.dispose());
      Object.values(materials).forEach(material=>material.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
