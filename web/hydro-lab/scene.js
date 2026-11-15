import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = {
  concrete: 0xe3e6e4,
  concreteShade: 0xaeb8b4,
  concreteEdge: 0x758580,
  water: 0x28a9dc,
  waterDeep: 0x167ba8,
  waterLight: 0x9ceaf4,
  grass: 0x55a95c,
  grassDark: 0x28743f,
  earth: 0xa77b4c,
  pipe: 0x4a6269,
  turbine: 0x2ba66f,
  generator: 0xe39135,
  copper: 0xc76a32,
  gridOff: 0x6c7b80,
  gridOn: 0xf5c744,
};

function toon(color, options = {}) {
  return new THREE.MeshToonMaterial({ color, ...options });
}

function makeMesh(geometry, material, position, parent, name = '') {
  const object = new THREE.Mesh(geometry, material);
  object.position.set(...position);
  object.name = name;
  object.castShadow = true;
  object.receiveShadow = true;
  parent.add(object);
  return object;
}

function cylinderBetween(start, end, radius, material, parent, radialSegments = 16) {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = new THREE.Vector3().subVectors(b, a);
  const object = makeMesh(new THREE.CylinderGeometry(radius, radius, direction.length(), radialSegments), material, [0, 0, 0], parent);
  object.position.copy(a).add(b).multiplyScalar(0.5);
  object.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  return object;
}

function beamBetween(start, end, thickness, depth, material, parent) {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = new THREE.Vector3().subVectors(b, a);
  const object = makeMesh(new THREE.BoxGeometry(direction.length(), thickness, depth), material, [0, 0, 0], parent);
  object.position.copy(a).add(b).multiplyScalar(0.5);
  object.rotation.z = Math.atan2(direction.y, direction.x);
  return object;
}

function tube(points, radius, material, parent, segments = 72) {
  const curve = new THREE.CatmullRomCurve3(points.map((point) => new THREE.Vector3(...point)));
  const object = makeMesh(new THREE.TubeGeometry(curve, segments, radius, 14, false), material, [0, 0, 0], parent);
  return { object, curve };
}

function extrudedSection(points, depth, material, parent, z = -2) {
  const shape = new THREE.Shape();
  shape.moveTo(points[0][0], points[0][1]);
  points.slice(1).forEach(([x, y]) => shape.lineTo(x, y));
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
  geometry.computeVertexNormals();
  return makeMesh(geometry, material, [0, 0, z], parent);
}

function transmissionTower(parent, x, z, scale = 1) {
  const group = new THREE.Group();
  group.position.set(x, 0, z);
  group.scale.setScalar(scale);
  const material = toon(COLORS.gridOff);
  cylinderBetween([-0.62, 0, 0], [0, 4.9, 0], 0.07, material, group, 8);
  cylinderBetween([0.62, 0, 0], [0, 4.9, 0], 0.07, material, group, 8);
  for (const y of [1.3, 2.4, 3.45]) {
    cylinderBetween([-0.48 + y * 0.05, y, 0], [0.48 - y * 0.05, y, 0], 0.045, material, group, 8);
  }
  cylinderBetween([-0.95, 3.55, 0], [0.95, 3.55, 0], 0.06, material, group, 8);
  cylinderBetween([-0.7, 4.2, 0], [0.7, 4.2, 0], 0.06, material, group, 8);
  parent.add(group);
  return { group, material };
}

function cable(parent, points, material) {
  return tube(points, 0.025, material, parent, 36).object;
}

export class HydroScene {
  constructor(host, labelLayer, fallback) {
    this.host = host;
    this.labelLayer = labelLayer;
    this.fallback = fallback;
    this.active = true;
    this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.labels = [];
    this.flowPhase = 0;
    this.spillPhase = 0;
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    } catch {
      this.showFallback();
      return;
    }
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-12, 12, 8, -8, 0.1, 120);
    this.camera.position.set(16.8, 11.4, 20.5);
    this.camera.lookAt(0, 2.7, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 2.7, 0);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.minZoom = 0.9;
    this.controls.maxZoom = 1.35;
    this.controls.minPolarAngle = 0.78;
    this.controls.maxPolarAngle = 1.08;
    this.controls.minAzimuthAngle = -0.36;
    this.controls.maxAzimuthAngle = 0.18;

    this.scene.add(new THREE.HemisphereLight(0xf5fcff, 0x4e754a, 2.8));
    const sun = new THREE.DirectionalLight(0xffffff, 3.4);
    sun.position.set(-9, 17, 13);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -18;
    sun.shadow.camera.right = 18;
    sun.shadow.camera.top = 15;
    sun.shadow.camera.bottom = -10;
    this.scene.add(sun);
    this.buildPlant();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.intersectionObserver = new IntersectionObserver(([entry]) => { this.active = entry.isIntersecting; });
    this.intersectionObserver.observe(host);
    this.resize();
    this.animate = this.animate.bind(this);
    this.animationFrame = requestAnimationFrame(this.animate);
  }

  buildPlant() {
    const world = new THREE.Group();
    world.rotation.y = -0.03;
    this.scene.add(world);
    this.world = world;

    makeMesh(new THREE.BoxGeometry(25, 0.55, 9.8), toon(COLORS.grass), [0, -0.5, 0], world);
    makeMesh(new THREE.BoxGeometry(25, 0.32, 2.5), toon(COLORS.earth), [0, -0.85, 2.7], world);
    const hillA = makeMesh(new THREE.ConeGeometry(5.4, 7.2, 7), toon(COLORS.grassDark), [2.5, 2.7, -4.1], world);
    hillA.rotation.z = -0.22;
    const hillB = makeMesh(new THREE.ConeGeometry(4.6, 6, 7), toon(0x3d8d4b), [8.4, 2.1, -3.8], world);
    hillB.rotation.z = 0.24;

    this.reservoir = makeMesh(
      new THREE.BoxGeometry(8.6, 4.6, 6.2),
      toon(COLORS.water, { transparent: true, opacity: 0.76 }),
      [-7.3, 2.0, -0.15],
      world,
      'Reservoir water',
    );
    this.waterTop = makeMesh(
      new THREE.BoxGeometry(8.7, 0.14, 6.3),
      toon(COLORS.waterLight, { transparent: true, opacity: 0.9 }),
      [-7.3, 4.35, -0.15],
      world,
    );

    this.dam = extrudedSection(
      [[-4.0, -0.2], [1.3, -0.2], [0.55, 2.25], [-0.15, 4.35], [-0.75, 6.55], [-2.75, 6.55]],
      4.2,
      toon(COLORS.concrete),
      world,
      -2.35,
    );
    makeMesh(new THREE.BoxGeometry(2.8, 0.55, 4.45), toon(0xf2f3f2), [-1.72, 6.48, -0.18], world);
    makeMesh(new THREE.BoxGeometry(2.25, 0.5, 4.45), toon(COLORS.concreteShade), [-0.65, 5.58, -0.18], world).rotation.z = -0.08;
    makeMesh(new THREE.BoxGeometry(1.8, 0.42, 4.4), toon(0xc9cfcc), [0.1, 4.65, -0.18], world).rotation.z = -0.13;

    this.intake = makeMesh(new THREE.BoxGeometry(0.55, 2.25, 2.0), toon(COLORS.pipe), [-3.78, 4.45, 1.0], world);
    const rackMaterial = toon(0x243f49);
    for (let z = 0.3; z <= 1.75; z += 0.28) {
      makeMesh(new THREE.BoxGeometry(0.12, 1.85, 0.055), rackMaterial, [-4.1, 4.35, z], world);
    }
    for (let y = 3.62; y <= 5.2; y += 0.4) {
      makeMesh(new THREE.BoxGeometry(0.12, 0.07, 1.62), rackMaterial, [-4.1, y, 1.02], world);
    }
    this.gate = makeMesh(new THREE.BoxGeometry(0.24, 1.6, 1.55), toon(0xe6b74d), [-3.45, 4.55, 1.02], world);
    cylinderBetween([-3.45, 5.25, 1.02], [-3.45, 6.2, 1.02], 0.08, toon(0x6e7778), world, 10);

    const penstockPoints = [
      [-3.65, 4.35, 2.3],
      [-2.9, 3.95, 2.3],
      [-2.1, 2.85, 2.3],
      [-1.05, 1.45, 2.3],
      [0.55, 0.72, 2.3],
      [1.65, 0.62, 2.3],
    ];
    tube(penstockPoints, 0.48, toon(COLORS.pipe, { transparent: true, opacity: 0.38, depthWrite: false }), world);
    tube(penstockPoints, 0.34, toon(COLORS.waterDeep), world);

    this.powerhouse = new THREE.Group();
    world.add(this.powerhouse);
    makeMesh(new THREE.BoxGeometry(5.8, 0.35, 4.1), toon(0xd2d7d5), [3.25, 0.02, 0.45], this.powerhouse);
    // Leave the roof and front wall open so the electro-mechanical chain stays visible.
    makeMesh(new THREE.BoxGeometry(0.28, 3.6, 4.1), toon(COLORS.concreteShade), [0.5, 1.85, 0.45], this.powerhouse);
    makeMesh(new THREE.BoxGeometry(5.8, 3.6, 0.22), toon(COLORS.concreteShade), [3.25, 1.85, -1.52], this.powerhouse);
    makeMesh(new THREE.BoxGeometry(0.28, 3.6, 4.1), toon(COLORS.concrete), [6.0, 1.85, 0.45], this.powerhouse);

    this.turbine = new THREE.Group();
    this.turbine.position.set(1.7, 0.62, 2.3);
    world.add(this.turbine);
    const turbineRing = makeMesh(new THREE.TorusGeometry(0.62, 0.18, 10, 28), toon(COLORS.turbine), [0, 0, 0], this.turbine);
    turbineRing.rotation.x = Math.PI / 2;
    for (let index = 0; index < 8; index += 1) {
      const angle = index * Math.PI / 4;
      const blade = makeMesh(new THREE.BoxGeometry(0.52, 0.09, 0.18), toon(0x75d6ae), [Math.cos(angle) * 0.28, 0, Math.sin(angle) * 0.28], this.turbine);
      blade.rotation.y = angle;
    }
    cylinderBetween([1.7, 0.62, 2.3], [1.7, 2.02, 2.3], 0.12, toon(0x525e60), world, 12);
    this.generator = makeMesh(new THREE.CylinderGeometry(0.82, 0.82, 1.2, 18), toon(COLORS.generator), [1.7, 2.3, 2.3], world);
    this.generator.rotation.z = Math.PI / 2;
    makeMesh(new THREE.TorusGeometry(0.62, 0.1, 8, 22), toon(COLORS.copper), [1.7, 2.3, 2.3], world).rotation.y = Math.PI / 2;

    const draftPoints = [
      [1.7, 0.45, 2.3],
      [1.95, -0.05, 2.3],
      [3.25, -0.12, 2.3],
      [4.65, 0.05, 2.3],
      [5.8, 0.12, 2.3],
    ];
    tube(draftPoints, 0.55, toon(COLORS.pipe, { transparent: true, opacity: 0.38, depthWrite: false }), world);
    tube(draftPoints, 0.4, toon(COLORS.water), world);

    this.tailrace = makeMesh(
      new THREE.BoxGeometry(7.2, 0.68, 5.3),
      toon(COLORS.water, { transparent: true, opacity: 0.86 }),
      [8.0, -0.03, 0.45],
      world,
    );
    makeMesh(new THREE.BoxGeometry(7.2, 0.1, 5.35), toon(COLORS.waterLight, { transparent: true, opacity: 0.85 }), [8.0, 0.34, 0.45], world);

    this.transformer = new THREE.Group();
    this.transformer.position.set(5.15, 1.05, -0.35);
    world.add(this.transformer);
    for (const x of [-0.42, 0, 0.42]) {
      makeMesh(new THREE.CylinderGeometry(0.22, 0.22, 1.3, 12), toon(0x77888b), [x, 0, 0], this.transformer);
      for (const y of [-0.35, -0.1, 0.15, 0.4]) {
        makeMesh(new THREE.TorusGeometry(0.28, 0.035, 7, 14), toon(0xaeb9b8), [x, y, 0], this.transformer).rotation.x = Math.PI / 2;
      }
    }
    this.towers = [transmissionTower(world, 8.0, -2.75, 0.86), transmissionTower(world, 11.0, -3.25, 0.68)];
    const cableMaterial = toon(COLORS.gridOff);
    for (const z of [-0.22, 0.22]) {
      cable(world, [[5.4, 2.0, -0.35 + z], [6.5, 3.1, -1.4 + z], [8.0, 3.05, -2.75 + z], [9.5, 2.8, -3.05 + z], [11.0, 2.45, -3.25 + z]], cableMaterial);
    }

    this.spillways = [];
    for (const offset of [-0.34, 0, 0.34]) {
      const strip = beamBetween([0.05 + offset, 6.12, 1.94], [3.25 + offset, 0.42, 1.94], 0.28, 0.24, toon(COLORS.waterLight, { transparent: true, opacity: 0.78 }), world);
      strip.visible = false;
      this.spillways.push(strip);
    }
    this.spillCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.05, 6.08, 2.02),
      new THREE.Vector3(1.05, 4.15, 2.02),
      new THREE.Vector3(2.15, 2.05, 2.02),
      new THREE.Vector3(3.25, 0.4, 2.02),
      new THREE.Vector3(4.65, 0.2, 1.9),
    ]);
    this.spillParticles = Array.from({ length: 14 }, (_, index) => {
      const particle = makeMesh(new THREE.SphereGeometry(0.11, 7, 7), toon(0xffffff), [0, 0, 0], world);
      particle.userData.offset = index / 14;
      particle.visible = false;
      return particle;
    });

    this.flowCurve = new THREE.CatmullRomCurve3([
      ...penstockPoints.map((point) => new THREE.Vector3(...point)),
      new THREE.Vector3(1.7, 0.45, 2.3),
      ...draftPoints.slice(1).map((point) => new THREE.Vector3(...point)),
    ]);
    this.flowParticles = Array.from({ length: 22 }, (_, index) => {
      const arrow = makeMesh(new THREE.ConeGeometry(0.14, 0.42, 7), toon(0xe9fbff), [0, 0, 0], world);
      arrow.userData.offset = index / 22;
      return arrow;
    });

    this.addLabel('Reservoir', this.reservoir, [0.3, 1.9, 1.7], 'water-label');
    this.addLabel('Trash rack', this.intake, [-0.95, -0.5, 1.0]);
    this.addLabel('Control gate', this.gate, [0.1, 1.45, 0.8]);
    this.addLabel('Dam', this.dam, [0.1, 5.1, 4.35]);
    this.addLabel('Penstock', this.dam, [-1.0, 2.05, 4.95], 'flow-label');
    this.addLabel('Spillway', this.dam, [1.75, 5.35, 4.05], 'flow-label');
    this.addLabel('Turbine', this.turbine, [-0.8, -0.55, 0.55]);
    this.addLabel('Generator', this.generator, [0.85, 0.8, 0.25]);
    this.addLabel('Power house', this.powerhouse, [4.15, 3.35, 1.75]);
    this.addLabel('Transformer', this.transformer, [0.3, 1.25, 0.25]);
    this.addLabel('Draft tube', this.powerhouse, [3.6, -0.05, 2.65], 'flow-label');
    this.addLabel('Outflow', this.tailrace, [1.25, 0.65, 1.7], 'water-label');
    this.addLabel('Transmission', this.towers[0].group, [0, 4.8, 0.35]);
  }

  addLabel(text, object, offset, className = '') {
    const element = document.createElement('span');
    element.className = `scene-label ${className}`.trim();
    element.textContent = text;
    this.labelLayer.append(element);
    this.labels.push({ element, object, offset: new THREE.Vector3(...offset) });
  }

  update(result) {
    this.result = result;
    if (!this.renderer) return;
    const waterY = 2.15 + (result.inputs.grossHeadM - 5) / 245 * 4.15;
    const height = Math.max(0.5, waterY + 0.15);
    this.reservoir.scale.y = height / 4.6;
    this.reservoir.position.y = height / 2 - 0.38;
    this.waterTop.position.y = height - 0.34;

    const spillRatio = Math.min(1, result.spillFlowM3s / 12);
    this.spillways.forEach((strip) => {
      strip.visible = result.spillFlowM3s > 0.02;
      strip.scale.z = 0.55 + spillRatio * 0.85;
      strip.material.opacity = 0.35 + spillRatio * 0.58;
    });
    this.spillParticles.forEach((particle) => { particle.visible = result.spillFlowM3s > 0.02; });

    const powerRatio = result.inputs.ratingMW > 0 ? result.plantMW / result.inputs.ratingMW : 0;
    this.generator.material.color.setHex(powerRatio > 0.02 ? COLORS.generator : COLORS.gridOff);
    this.towers.forEach(({ material }) => material.color.setHex(powerRatio > 0.02 ? COLORS.gridOn : COLORS.gridOff));
  }

  resize() {
    if (!this.renderer) return;
    const width = Math.max(320, this.host.clientWidth);
    const height = Math.max(380, this.host.clientHeight);
    const aspect = width / height;
    const vertical = width < 560 ? 11.5 : 7.6;
    this.camera.left = -vertical * aspect;
    this.camera.right = vertical * aspect;
    this.camera.top = vertical;
    this.camera.bottom = -vertical;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  positionLabels() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    for (const { element, object, offset } of this.labels) {
      const point = object.localToWorld(offset.clone()).project(this.camera);
      element.style.transform = `translate(-50%, -50%) translate(${(point.x * 0.5 + 0.5) * width}px, ${(-point.y * 0.5 + 0.5) * height}px)`;
      element.hidden = point.z > 1 || point.x < -1.08 || point.x > 1.08 || point.y < -1.08 || point.y > 1.08;
    }
  }

  animate() {
    this.animationFrame = requestAnimationFrame(this.animate);
    if (!this.renderer || !this.active) return;
    const turbineFlow = this.result?.turbineFlowM3s ?? 0;
    const spillFlow = this.result?.spillFlowM3s ?? 0;
    const flowSpeed = this.reducedMotion ? 0 : Math.min(2.4, turbineFlow / 4.5);
    const spillSpeed = this.reducedMotion ? 0 : Math.min(2.8, spillFlow / 5);
    this.flowPhase = (this.flowPhase + flowSpeed * 0.0045) % 1;
    this.spillPhase = (this.spillPhase + spillSpeed * 0.006) % 1;

    this.flowParticles.forEach((particle) => {
      const t = (particle.userData.offset + this.flowPhase) % 1;
      const point = this.flowCurve.getPointAt(t);
      const tangent = this.flowCurve.getTangentAt(t).normalize();
      particle.position.copy(point);
      particle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
      particle.visible = turbineFlow > 0.02;
    });
    this.spillParticles.forEach((particle) => {
      const t = (particle.userData.offset + this.spillPhase) % 1;
      particle.position.copy(this.spillCurve.getPointAt(t));
    });
    if (!this.reducedMotion) this.turbine.rotation.y += flowSpeed * 0.08;
    this.controls.update();
    this.positionLabels();
    this.renderer.render(this.scene, this.camera);
  }

  showFallback() {
    this.fallback.hidden = false;
    this.labelLayer.hidden = true;
  }

  destroy() {
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver?.disconnect();
    this.intersectionObserver?.disconnect();
    this.controls?.dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }
}
