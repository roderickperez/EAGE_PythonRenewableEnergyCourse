import * as T from 'three';
import { mountCanvasDiagram } from './diagram-canvas.js';
import { mountDiagramMotion, mountDiagramRenderer } from './diagram-motion.js';
import { createOperationScene } from './operation-scene.js';
import { createPumpedStorageScene } from './pumped-storage-scene.js';

const palette = { water: '#288daa', net: '#398470', loss: '#c17932', shaft: '#98744c', electricity: '#e5a124' };
export { palette };

export function mountBalanceRenderer(host, kind) {
  return mountDiagramRenderer(host,
    kind === 'flow-volume' ? null : () => create3DScene(host, kind),
    () => mountCanvasDiagram(host, kind));
}

export function mountBalanceScene(host, kind, motionButton) {
  const draw = mountBalanceRenderer(host, kind);
  motionButton.hidden = kind === 'net-head';
  const rate = kind === 'flow-volume' || kind === 'hydraulic-loss' ? .00015 : .000045;
  return kind === 'net-head'
    ? mountDiagramMotion(host, motionButton, draw, undefined, 400)
    : mountDiagramMotion(host, motionButton, draw, values =>
      (kind === 'turbine-operation' ? Math.max(values[1], values[2]) : values[1]) * rate);
}

function create3DScene(host, kind) {
  const scene = new T.Scene();
  scene.background = new T.Color('#e9f3f4');
  const camera = new T.OrthographicCamera();
  camera.position.set(7, 6, 14);
  camera.lookAt(0, 2, 0);
  scene.add(new T.HemisphereLight(0xffffff, 0x6e8a82, 2.4));
  const light = new T.DirectionalLight(0xffffff, 2.3);
  light.position.set(-5, 7, 4);
  scene.add(light);
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  (host.querySelector('#drawing') || host).append(renderer.domElement);
  const material = color => new T.MeshStandardMaterial({ color, roughness: .65 });
  const water = material(palette.water);
  const rock = material('#a7bcb2');
  const steel = material('#49666f');

  function mesh(geometry, surface, x, y, z) {
    const object = new T.Mesh(geometry, surface);
    object.position.set(x, y, z);
    scene.add(object);
    return object;
  }
  const box = (width, height, depth, surface, x, y, z) =>
    mesh(new T.BoxGeometry(width, height, depth), surface, x, y, z);
  function tank(x, width, height, depth, bottom = 0) {
    box(width + .16, .14, depth + .16, rock, x, bottom - .07, 0);
    box(width, height, .10, rock, x, bottom + height / 2, -depth / 2);
    for (const side of [-1, 1]) {
      box(.10, height, depth, rock, x + side * width / 2, bottom + height / 2, 0);
    }
    return box(width - .10, 1, depth - .09, water, x, 0, .01);
  }
  function level(object, height, bottom = 0) {
    object.visible = height > 0;
    object.position.y = bottom + height / 2;
    object.scale.y = height;
  }

  let draw;
  if (kind === 'pumped-storage') {
    draw = createPumpedStorageScene({ material, mesh, box, tank, level, steel, rock, palette });
  } else if (kind === 'hydraulic-loss' || kind === 'turbine-operation') {
    draw = createOperationScene(kind, { scene, material, mesh, box, tank, level, steel, rock, palette });
  } else if (kind === 'net-head') {
    const upper = tank(-2.75, 3, 3.75, 2.4);
    const lower = tank(2.75, 3, 1.05, 2.4);
    const lowerLevel = .63;
    level(lower, lowerLevel);
    const net = box(.40, 1, .28, material(palette.net), .32, 0, 1.55);
    const loss = box(.40, 1, .28, material(palette.loss), .32, 0, 1.55);
    box(.84, .045, .36, steel, .32, lowerLevel, 1.55);
    box(.07, 3, .10, rock, .25, lowerLevel + 1.5, 1.25);
    draw = ([gross, lossHead]) => {
      const usable = (gross - lossHead) * .03;
      level(upper, lowerLevel + gross * .03);
      level(net, usable, lowerLevel);
      level(loss, lossHead * .03, lowerLevel + usable);
    };
  } else {
    const tailwater = .75;
    const upper = tank(-2.8, 2.6, 3.9, 2.4);
    const lower = tank(kind === 'electrical-power' ? 1.0 : 3.0, 2.2, .95, 2.4);
    level(lower, tailwater);
    const inlet = new T.Vector3(-2.12, 0, .69);
    const outlet = new T.Vector3(kind === 'electrical-power' ? .95 : 2.42, tailwater - .15, .69);
    const pipe = mesh(new T.CylinderGeometry(.18, .18, 1, 18), steel, 0, 0, 0);
    const marker = new T.SphereGeometry(.09, 12, 8);
    const stream = new T.MeshBasicMaterial({ color: '#8ef2ff' });
    const particles = Array.from({ length: 12 }, () => mesh(marker, stream, 0, 0, 0));
    const direction = new T.Vector3();
    const up = new T.Vector3(0, 1, 0);
    let runner, pulses;
    if (kind === 'electrical-power') {
      const shaft = material(palette.shaft);
      runner = new T.Group();
      runner.position.copy(outlet);
      scene.add(runner);
      const hub = new T.Mesh(new T.CylinderGeometry(.17, .17, .30, 18), shaft);
      hub.rotation.z = Math.PI / 2;
      runner.add(hub);
      const blade = new T.BoxGeometry(.20, .27, .12);
      for (let i = 0; i < 6; i++) {
        const angle = i * Math.PI / 3;
        const vane = new T.Mesh(blade, shaft);
        vane.position.set(0, Math.cos(angle) * .28, Math.sin(angle) * .28);
        vane.rotation.x = angle;
        runner.add(vane);
      }
      const axle = mesh(new T.CylinderGeometry(.085, .085, 1.35, 16), steel, 1.69, outlet.y, outlet.z);
      axle.rotation.z = Math.PI / 2;
      const generator = mesh(new T.CylinderGeometry(.5, .5, 1.2, 28), shaft, 2.77, outlet.y, outlet.z);
      generator.rotation.z = Math.PI / 2;
      box(1.4, .14, 1.32, rock, 2.77, -.06, outlet.z);
      box(1.22, .045, .045, steel, 3.91, outlet.y, outlet.z);
      const glow = new T.MeshBasicMaterial({ color: palette.electricity });
      pulses = Array.from({ length: 4 }, () => mesh(marker, glow, 0, outlet.y, outlet.z));
    }
    draw = ([head, flow, shaftW, electricalW], phase) => {
      const drop = .03 * head;
      level(upper, tailwater + drop);
      inlet.y = outlet.y + drop;
      direction.subVectors(outlet, inlet);
      pipe.position.copy(inlet).lerp(outlet, .5);
      pipe.scale.y = direction.length();
      pipe.quaternion.setFromUnitVectors(up, direction.normalize());
      for (const [index, particle] of particles.entries()) {
        particle.position.copy(inlet).lerp(outlet, (phase + index / particles.length) % 1);
        particle.position.z += .18;
        particle.visible = flow > 0;
      }
      if (runner) {
        if (shaftW > 0) runner.rotation.x = phase * Math.PI * 4;
        for (const [index, pulse] of pulses.entries()) {
          pulse.position.x = 3.4 + 1.08 * ((2 * phase + index / pulses.length) % 1);
          pulse.visible = electricalW > 0;
          pulse.scale.setScalar(.65 + .6 * Math.min(1, electricalW / 3e6));
        }
      }
    };
  }

  let values, phase;
  function render() {
    if (values) draw(values, phase);
    renderer.render(scene, camera);
  }
  new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = host;
    if (!w || !h || !renderer.domElement.isConnected) return;
    const half = Math.max(kind === 'pumped-storage' ? 3.08 : 2.85, 5.2 * h / w);
    Object.assign(camera, { left: -half * w / h, right: half * w / h, top: half, bottom: -half });
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    render();
  }).observe(host);
  return (next, position) => {
    values = next;
    phase = position;
    render();
  };
}
