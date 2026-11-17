import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function mountDiagramOrbit(camera, canvas, target, render, azimuthRange = .75) {
  const orbit = new OrbitControls(camera, canvas);
  orbit.target.set(...target);
  const azimuth = Math.atan2(camera.position.x - target[0], camera.position.z - target[2]);
  Object.assign(orbit, {
    enablePan: false, enableZoom: false, minPolarAngle: .52, maxPolarAngle: 1.43,
    minAzimuthAngle: azimuth - azimuthRange, maxAzimuthAngle: azimuth + azimuthRange,
  });
  orbit.update();
  orbit.addEventListener('change', render);
  canvas.title = 'Drag to orbit';
  canvas.dataset.orbit = 'true';
  canvas.style.cursor = 'grab';
  return orbit;
}
