import * as T from 'three';
import { mountDiagramOrbit } from '../diagram-orbit.js';
import { RAD } from './model.js';
import { createSolarPanel } from './panel.js';
import { panelCenter, resourcePaths, solarColors } from './resource-scene.js';

export function mountResourceThree(host, showAngle, label) {
  const renderer = new T.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', label);
  const scene = new T.Scene();
  scene.background = new T.Color('#e4eef2');
  const camera = new T.OrthographicCamera(-6, 6, 3, -3, .1, 60);
  camera.position.set(7, 6, -10);
  scene.add(new T.HemisphereLight(0xe5f3ff, 0x647957, 2.15));
  const light = new T.DirectionalLight(0xffdf9e, 2.1);
  light.position.set(-3, 7, -5);
  scene.add(light);

  const mat = (color, roughness = .7, metalness = 0) => new T.MeshStandardMaterial({ color, roughness, metalness });
  function mesh(geometry, material, position, parent = scene) {
    const object = new T.Mesh(geometry, material);
    object.position.set(...position);
    parent.add(object);
    return object;
  }
  const box = (size, material, position, parent) => mesh(new T.BoxGeometry(...size), material, position, parent);
  const steel = mat('#8798a4', .35, .55);
  box([6, .24, 4], mat('#9aaa8e'), [0, -.18, 0]);
  box([5.96, .06, 3.96], mat('#687e61'), [0, -.03, 0]);
  box([1.3, .09, 1.15], mat('#b8b8a7'), [0, .045, 0]);
  box([.18, 1.23, .18], steel, [0, .65, 0]);
  const rack = new T.Group();
  rack.position.set(...panelCenter);
  scene.add(rack);
  const panel = createSolarPanel(box, mat, steel);
  panel.scale.set(.79, .56, .60);
  rack.add(panel);

  const colorMaterial = color => new T.MeshBasicMaterial({ color, transparent: true, depthWrite: false, toneMapped: false });
  const sun = mesh(new T.SphereGeometry(.23, 24, 12), colorMaterial('#ffbd3e'), panelCenter);
  const makeLine = material => {
    const line = new T.Line(new T.BufferGeometry(), material);
    scene.add(line);
    return line;
  };
  const dashed = color => new T.LineDashedMaterial({ color, dashSize: .11, gapSize: .08 });
  const normal = makeLine(dashed(solarColors.normal));
  const direction = makeLine(dashed(solarColors.beam));
  const arc = makeLine(new T.LineBasicMaterial({ color: '#94602a' }));
  function setLine(line, positions) {
    line.geometry.dispose();
    line.geometry = new T.BufferGeometry().setFromPoints(positions.map(point => new T.Vector3(...point)));
    line.computeLineDistances();
  }
  const up = new T.Vector3(0, 1, 0);
  const tubeGeometry = new T.CylinderGeometry(.013, .013, 1, 8);
  const pulseGeometry = new T.SphereGeometry(.05, 12, 8);
  const rays = ['ground', 'sky', 'beam'].map(key => {
    const tint = colorMaterial(solarColors[key]);
    return { key, tint, set: Array.from({ length: key === 'beam' ? 3 : 2 }, () => ({
      bar: mesh(tubeGeometry, tint, panelCenter),
      pulse: mesh(pulseGeometry, tint, panelCenter),
      start: new T.Vector3(),
      end: new T.Vector3(),
    })) };
  });
  for (const line of [normal, direction, arc]) line.visible = showAngle;
  const render = () => {
    if (renderer.domElement.isConnected) renderer.render(scene, camera);
  };
  const orbit = mountDiagramOrbit(camera, renderer.domElement, [0, 1.6, 0], render, 1.14);
  const resize = new ResizeObserver(() => {
    const { clientWidth: w, clientHeight: h } = host;
    if (!w || !h || !renderer.domElement.isConnected) return;
    const half = Math.max(3.1, 4.25 * h / w);
    Object.assign(camera, { left: -half * w / h, right: half * w / h, top: half, bottom: -half });
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    render();
  });
  resize.observe(host);
  host.append(renderer.domElement);
  renderer.domElement.addEventListener('webglcontextlost', () => {
    orbit.dispose();
    resize.disconnect();
  }, { once: true });

  const delta = new T.Vector3();
  let geometry;
  return (state, phase) => {
    const key = [state.tilt, state.azimuth, ...state.sunDirection].join(',');
    if (geometry !== key) {
      geometry = key;
      rack.rotation.y = state.azimuth * RAD;
      panel.rotation.x = state.tilt * RAD;
      const { source, normalEnd, arcPoints, routes } = resourcePaths(state);
      sun.position.set(...source);
      setLine(normal, [panelCenter, normalEnd]);
      setLine(direction, [panelCenter, source]);
      setLine(arc, arcPoints);
      for (const group of rays) group.set.forEach(({ start, end, bar }, i) => {
        const [from, to] = routes[group.key][i];
        start.set(...from);
        end.set(...to);
        delta.subVectors(end, start);
        bar.position.copy(start).lerp(end, .5);
        bar.scale.y = delta.length();
        bar.quaternion.setFromUnitVectors(up, delta.normalize());
      });
    }
    sun.visible = showAngle || state.beam > 0 || state.sky > 0 || state.ground > 0;
    for (const { key: component, tint, set } of rays) {
      const value = state[component];
      tint.opacity = .34 + .66 * Math.min(1, value / (component === 'beam' ? 780 : 180));
      set.forEach(({ start, end, bar, pulse }, i) => {
        bar.visible = pulse.visible = value > 0;
        pulse.position.copy(start).lerp(end, (phase + i * .24) % 1);
      });
    }
    render();
  };
}
