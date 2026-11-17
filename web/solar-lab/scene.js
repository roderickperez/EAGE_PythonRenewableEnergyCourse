import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RAD, sunDirection } from './model.js';
import { createSolarPanel } from './panel.js';

export function mountScene(host) {
  const scene = new T.Scene();
  const day = new T.Color('#e4eef2');
  const night = new T.Color('#26394f');
  scene.background = day.clone();
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  host.append(renderer.domElement);
  const camera = new T.OrthographicCamera(-8, 8, 6, -6, .1, 100);
  camera.position.set(13, 10, -16);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 2.7, 0);
  Object.assign(orbit, { enablePan: false, maxPolarAngle: Math.PI / 2.1, minPolarAngle: .42,
    minZoom: .8, maxZoom: 1.85, enableZoom: false });
  orbit.update();
  const sky = new T.HemisphereLight(0xe5f3ff, 0x647957, 2);
  scene.add(sky);
  const sunlight = new T.DirectionalLight(0xffdf9e, 3);
  sunlight.castShadow = true;
  sunlight.shadow.mapSize.set(1024, 1024);
  Object.assign(sunlight.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8 });
  sunlight.shadow.normalBias = .035;
  scene.add(sunlight);
  const mat = (color, roughness = .7, metalness = 0) =>
    new T.MeshStandardMaterial({ color, roughness, metalness });
  function mesh(geometry, material, position, parent = scene) {
    const object = new T.Mesh(geometry, material);
    object.position.set(...position);
    object.castShadow = object.receiveShadow = true;
    parent.add(object);
    return object;
  }
  const box = (size, material, position, parent) => mesh(new T.BoxGeometry(...size), material, position, parent);
  const steel = mat('#8798a4', .35, .55);
  box([13, .48, 9], mat('#9aaa8e'), [0, -.26, 0]);
  box([12.85, .08, 8.85], mat('#687e61'), [0, -.025, 0]);
  box([4.4, .12, 3.6], mat('#b8b8a7'), [0, .07, 0]);
  box([.3, 2, .3], steel, [0, 1, 0]);
  box([.8, .16, .8], steel, [0, .19, 0]);
  const rack = new T.Group();
  const panel = createSolarPanel(box, mat, steel);
  rack.add(panel);
  rack.position.y = 2.05;
  scene.add(rack);
  const inverter = box([.95, .72, .59], mat('#dce2e3'), [3.13, .45, -.9]);
  const led = new T.MeshBasicMaterial({ color: '#425a63' });
  box([.43, .14, .018], led, [3.13, .57, -1.20]);
  const terminal = box([.62, .88, .53], mat('#627681'), [5.1, .45, -.9]);
  const output = new T.MeshBasicMaterial({ color: '#56686d' });
  box([.22, .08, .02], output, [5.1, .69, -1.17]);
  const wires = [
    [[0, 1.9, 0], [0, .2, 0], [1.6, .14, -.45], [2.7, .16, -.9], inverter.position.toArray()],
    [inverter.position.toArray(), [3.7, .14, -.9], [4.7, .14, -.9], terminal.position.toArray()],
  ].map(points => {
    const curve = new T.CatmullRomCurve3(points.map(point => new T.Vector3(...point)), false, 'centripetal');
    mesh(new T.TubeGeometry(curve, 36, .044, 8, false), mat('#31414a'), [0, 0, 0]);
    const pulse = mesh(new T.SphereGeometry(.086, 12, 8), new T.MeshBasicMaterial({ color: '#ffc153' }), [0, 0, 0]);
    pulse.castShadow = false;
    return { curve, pulse };
  });
  const sun = mesh(new T.SphereGeometry(.42, 32, 16), new T.MeshBasicMaterial({ color: '#ffbd3e' }), [0, 0, 0]);
  sun.castShadow = false;
  const path = new T.LineSegments(new T.BufferGeometry(), new T.LineDashedMaterial({ color: '#ba8e48', dashSize: .17, gapSize: .11 }));
  const rays = new T.LineSegments(new T.BufferGeometry(), new T.LineBasicMaterial({ color: '#edaa32', transparent: true }));
  scene.add(path, rays);
  const panelCenter = new T.Vector3(0, 2.05, 0);
  const sunPoint = direction => new T.Vector3(...direction).multiplyScalar(6.1).add(panelCenter);
  let lastTrack, phase = 0, previous = 0, power, playing = false, shown = true, dirty = true;
  function update(input, result, animate) {
    playing = animate;
    power = result;
    rack.rotation.y = input.azimuth * RAD;
    panel.rotation.x = input.tilt * RAD;
    const track = `${input.latitude},${input.declination}`;
    if (track !== lastTrack) {
      const points = [];
      for (let i = 0; i < 120; i++) {
        const a = sunDirection(i / 5, input.latitude, input.declination);
        const b = sunDirection((i + 1) / 5, input.latitude, input.declination);
        if (a[1] > 0 && b[1] > 0) points.push(sunPoint(a), sunPoint(b));
      }
      path.geometry.dispose();
      path.geometry = new T.BufferGeometry().setFromPoints(points);
      path.computeLineDistances();
      lastTrack = track;
    }
    sun.position.copy(sunPoint(result.sun));
    sunlight.position.copy(sun.position);
    sunlight.intensity = result.dni / 310;
    sky.intensity = result.elevation > 0 ? 1.4 + result.dhi / 240 : .75;
    scene.background.copy(day).lerp(night, result.elevation > 0 ? 0 : .86);
    sun.visible = result.elevation > 0;
    rays.visible = result.beam > 0;
    rays.material.opacity = Math.min(.9, .2 + result.beam / 1100);
    const segments = [];
    panel.updateWorldMatrix(true, false);
    for (const x of [-1.15, 0, 1.15]) {
      for (const z of [-.7, .7]) {
        const cell = panel.localToWorld(new T.Vector3(x, .15, z));
        segments.push(cell.clone().addScaledVector(new T.Vector3(...result.sun), 3.2), cell);
      }
    }
    rays.geometry.dispose();
    rays.geometry = new T.BufferGeometry().setFromPoints(segments);
    led.color.set(result.dc > 0 ? '#67db9f' : '#425a63');
    output.color.set(result.ac > 0 ? '#ffc153' : '#56686d');
    dirty = true;
  }
  const resize = new ResizeObserver(() => {
    const { clientWidth: width, clientHeight: height } = host;
    if (!width || !height) return;
    const half = Math.max(5.9, 7.8 * height / width);
    camera.left = -half * width / height;
    camera.right = -camera.left;
    camera.top = half;
    camera.bottom = -half;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    dirty = true;
  });
  const visibility = new IntersectionObserver(([entry]) => { shown = entry.isIntersecting; dirty = true; });
  orbit.addEventListener('change', () => { dirty = true; });
  resize.observe(host);
  visibility.observe(host);
  renderer.setAnimationLoop(time => {
    const dt = Math.min(.05, (time - previous) / 1000);
    previous = time;
    if (!shown || !power || (!dirty && !playing)) return;
    if (playing) phase = (phase + dt * .48) % 1;
    wires.forEach(({ curve, pulse }, i) => {
      pulse.visible = (i ? power.ac : power.dc) > 0;
      pulse.position.copy(curve.getPointAt((phase + i * .5) % 1));
    });
    renderer.render(scene, camera);
    dirty = false;
  });
  return { update, destroy() {
    renderer.setAnimationLoop(null);
    resize.disconnect();
    visibility.disconnect();
    orbit.dispose();
    const materials = new Set();
    scene.traverse(object => { object.geometry?.dispose(); if (object.material) materials.add(object.material); });
    materials.forEach(material => material.dispose());
    renderer.dispose();
    renderer.domElement.remove();
  } };
}
