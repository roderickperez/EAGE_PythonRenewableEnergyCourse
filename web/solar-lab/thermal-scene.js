import * as T from 'three';
import { mountDiagramOrbit } from '../diagram-orbit.js';
import { solarSvgNode } from './resource-scene.js';
import { THERMAL_IRRADIANCE_MAX } from './thermal-model.js';

export const thermalColors = { sun: '#e6a033', inlet: '#368caf', outlet: '#c86b3d' };
const tilt = Math.PI / 7;
const origin = 1.62;
const collectorPoint = (x, depth) => [x, origin - Math.sin(tilt) * depth + .105, Math.cos(tilt) * depth];
const runs = [.88, .45, .02, -.41, -.84];
const pipeRoute = [[-3.18, .2, 1.46], [-2.24, .2, 1.46], [-1.66, .38, 1.17],
  ...runs.flatMap((depth, i) => {
    const near = i % 2 ? 1.42 : -1.42;
    const far = -near;
    return [collectorPoint(near, depth), collectorPoint(near * .85, depth),
      collectorPoint(far * .85, depth), collectorPoint(far, depth)];
  }),
  [2.04, 1.87, -.76], [2.5, .35, -.76], [3.18, .35, -.76],
];
const solarRays = [[-.95, -.58], [-.7, .38], [.3, -.30], [.92, .50]].map(([x, depth]) => {
  const end = collectorPoint(x, depth);
  return [[end[0] - .86, end[1] + 1.95, end[2] + .38], end];
});
const asVectors = points => points.map(point => new T.Vector3(...point));
const pipe = new T.CatmullRomCurve3(asVectors(pipeRoute));

function fluidColor(delta) {
  return new T.Color(thermalColors.inlet).lerp(new T.Color(thermalColors.outlet),
    delta > 0 ? .55 + .45 * Math.min(1, delta / 12) : 0);
}

export function mountThermalThree(host, label) {
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', label);
  const scene = new T.Scene();
  scene.background = new T.Color('#e4eef2');
  const camera = new T.OrthographicCamera(-6, 6, 3, -3, .1, 60);
  camera.position.set(7, 6.8, 11);
  scene.add(new T.HemisphereLight(0xe5f3ff, 0x647957, 2.15));
  const light = new T.DirectionalLight(0xffdf9e, 2.4);
  light.position.set(-4, 7, 5);
  scene.add(light);
  const material = (color, roughness = .64, metalness = 0) => new T.MeshStandardMaterial({ color, roughness, metalness });
  const mesh = (geometry, surface, position, parent = scene) => {
    const object = new T.Mesh(geometry, surface);
    object.position.set(...position);
    parent.add(object);
    return object;
  };
  const box = (size, surface, position, parent) => mesh(new T.BoxGeometry(...size), surface, position, parent);
  const steel = material('#8798a4', .38, .39);
  box([7.4, .24, 4.85], material('#9aaa8e'), [0, -.13, 0]);
  box([7.35, .08, 4.8], material('#687e61'), [0, .02, 0]);
  box([4.15, .10, 2.7], material('#b8b8a7'), [0, .11, 0]);
  for (const x of [-1.58, 1.58]) for (const z of [-.92, .92]) {
    const height = origin - z * Math.tan(tilt);
    box([.11, height, .11], steel, [x, height / 2 + .1, z]);
  }
  const rack = new T.Group();
  rack.rotation.x = tilt;
  rack.position.y = origin;
  scene.add(rack);
  box([3.65, .18, 2.25], steel, [0, -.065, 0], rack);
  box([3.51, .045, 2.10], material('#1b3445', .8), [0, .025, 0], rack);
  for (const x of [-1.73, 1.73]) box([.07, .075, 2.18], steel, [x, .065, 0], rack);

  const tube = new T.TubeGeometry(pipe, 160, .062, 10, false);
  mesh(tube, material('#a9704a', .42, .45), [0, 0, 0]);
  for (const [i, position] of [pipeRoute[0], pipeRoute.at(-1)].entries()) {
    const coupling = mesh(new T.CylinderGeometry(.104, .104, .30, 16),
      material(i ? thermalColors.outlet : thermalColors.inlet, .46, .25), position);
    coupling.rotation.z = Math.PI / 2;
  }

  const pulseShape = new T.SphereGeometry(.075, 12, 8);
  const fluid = Array.from({ length: 18 }, () => mesh(pulseShape, new T.MeshBasicMaterial({ toneMapped: false }), [0, 0, 0]));
  const rayMaterial = new T.LineBasicMaterial({ color: thermalColors.sun, transparent: true });
  const lightColor = new T.MeshBasicMaterial({ color: '#f7bd52', toneMapped: false, transparent: true });
  const sun = mesh(new T.SphereGeometry(.215, 22, 12), lightColor, [-2.05, 3.92, .38]);
  const beam = solarRays.map(([start, end]) => {
    const [a, b] = asVectors([start, end]);
    const line = new T.Line(new T.BufferGeometry().setFromPoints([a, b]), rayMaterial);
    scene.add(line);
    return { a, b, line, pulse: mesh(pulseShape, lightColor, start) };
  });

  const render = () => { if (renderer.domElement.isConnected) renderer.render(scene, camera); };
  const orbit = mountDiagramOrbit(camera, renderer.domElement, [0, 1.62, 0], render, 1.05);
  const resize = new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = host;
    if (!w || !h || !renderer.domElement.isConnected) return;
    const half = Math.max(3.05, 4.9 * h / w);
    Object.assign(camera, { left: -half * w / h, right: half * w / h, top: half, bottom: -half });
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    render();
  });
  host.append(renderer.domElement);
  resize.observe(host);
  renderer.domElement.addEventListener('webglcontextlost', () => {
    orbit.dispose();
    resize.disconnect();
  }, { once: true });

  const cool = new T.Color(thermalColors.inlet);
  const point = new T.Vector3();
  let previous;
  let warm;
  return ({ inputs, result }, phase) => {
    if (result !== previous) {
      warm = fluidColor(result.running ? result.outlet - inputs.inlet : 0);
      fluid.forEach(pulse => { pulse.visible = result.running; });
      sun.visible = inputs.irradiance > 0;
      const opacity = .24 + .76 * inputs.irradiance / THERMAL_IRRADIANCE_MAX;
      rayMaterial.opacity = lightColor.opacity = opacity;
      beam.forEach(({ line, pulse }) => { line.visible = pulse.visible = sun.visible; });
      previous = result;
    }
    for (const [index, pulse] of fluid.entries()) {
      const progress = (phase + index / fluid.length) % 1;
      pipe.getPointAt(progress, point);
      pulse.position.copy(point);
      pulse.position.y += .042;
      pulse.material.color.copy(cool).lerp(warm, progress);
    }
    beam.forEach(({ pulse, a, b }, index) => pulse.position.copy(a).lerp(b, (phase * 3 + index * .24) % 1));
    render();
  };
}

export function mountThermalFallback(svg) {
  const project = ([x, y, z]) => [270 + 29 * (.91 * x - .52 * z), 188 - 34 * (y + .28 * x + .36 * z)];
  const polygon = (coordinates, fill) => solarSvgNode(svg, 'polygon', { points: coordinates.map(project).join(' '), fill });
  polygon([[-3.7, 0, 2.3], [3.7, 0, 2.3], [3.7, 0, -2.3], [-3.7, 0, -2.3]], '#9aaa8e');
  for (const x of [-1.58, 1.58]) {
    const top = collectorPoint(x, .88);
    solarSvgNode(svg, 'polyline', { points: [[top[0], 0, top[2]], top].map(project).join(' '), stroke: '#8798a4', 'stroke-width': 4 });
  }
  polygon([collectorPoint(-1.8, 1.08), collectorPoint(1.8, 1.08), collectorPoint(1.8, -1.08), collectorPoint(-1.8, -1.08)], '#254456');
  const trace = (positions, color, width) => solarSvgNode(svg, 'polyline', {
    points: positions.map(project).join(' '), fill: 'none', stroke: color, 'stroke-width': width,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round',
  });
  const at = progress => pipe.getPointAt(progress).toArray();
  trace(Array.from({ length: 150 }, (_, i) => at(i / 149)), '#b7855f', 3.7);
  const gold = solarRays.map(([start, end]) => ({
    line: trace([start, end], thermalColors.sun, 1.8),
    dot: solarSvgNode(svg, 'circle', { r: 3, fill: thermalColors.sun }),
    start: project(start), end: project(end),
  }));
  const fluid = Array.from({ length: 18 }, () => solarSvgNode(svg, 'circle', { r: 2.8, stroke: 'white', 'stroke-width': .6 }));
  const cool = new T.Color(thermalColors.inlet);
  const color = new T.Color();
  const place = (dot, [x, y]) => { dot.setAttribute('cx', x); dot.setAttribute('cy', y); };
  let previous, warm;
  return ({ inputs, result }, phase) => {
    if (result !== previous) {
      warm = fluidColor(result.running ? result.outlet - inputs.inlet : 0);
      fluid.forEach(dot => { dot.style.display = result.running ? '' : 'none'; });
      const opacity = inputs.irradiance > 0 ? .24 + .76 * inputs.irradiance / THERMAL_IRRADIANCE_MAX : 0;
      gold.forEach(({ line, dot }) => { line.setAttribute('opacity', opacity); dot.setAttribute('opacity', opacity); });
      previous = result;
    }
    fluid.forEach((dot, index) => {
      const progress = (phase + index / fluid.length) % 1;
      place(dot, project(at(progress)));
      dot.setAttribute('fill', `#${color.copy(cool).lerp(warm, progress).getHexString()}`);
    });
    gold.forEach(({ dot, start, end }, index) => {
      const p = (phase * 3 + index * .24) % 1;
      place(dot, start.map((value, i) => value + (end[i] - value) * p));
    });
  };
}
