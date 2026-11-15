import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const colors = {
  concrete: 0xd8ddd9,
  concreteDark: 0x93a5a0,
  water: 0x2f9fd0,
  waterLight: 0x87d9ed,
  earth: 0x88a56b,
  penstock: 0x376f80,
  turbine: 0x17a985,
  generator: 0xf1ae3d,
  gridOff: 0x6e7c80,
  gridOn: 0xffcb45,
};

function toon(color) {
  return new THREE.MeshToonMaterial({ color });
}

function mesh(geometry, material, position, parent) {
  const object = new THREE.Mesh(geometry, material);
  object.position.set(...position);
  object.castShadow = true;
  object.receiveShadow = true;
  parent.add(object);
  return object;
}

function cylinderBetween(start, end, radius, material, parent) {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = new THREE.Vector3().subVectors(b, a);
  const object = mesh(new THREE.CylinderGeometry(radius, radius, direction.length(), 18), material, [0, 0, 0], parent);
  object.position.copy(a).add(b).multiplyScalar(0.5);
  object.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  return object;
}

function tower(parent, x, z) {
  const material = toon(colors.gridOff);
  const group = new THREE.Group();
  group.position.set(x, 0, z);
  cylinderBetween([-0.55, 0, 0], [0, 4.8, 0], 0.08, material, group);
  cylinderBetween([0.55, 0, 0], [0, 4.8, 0], 0.08, material, group);
  cylinderBetween([-0.7, 3.2, 0], [0.7, 3.2, 0], 0.07, material, group);
  cylinderBetween([-0.55, 4.1, 0], [0.55, 4.1, 0], 0.07, material, group);
  parent.add(group);
  return { group, material };
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
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    } catch (error) {
      this.showFallback();
      return;
    }
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-11, 11, 7.5, -7.5, 0.1, 100);
    this.camera.position.set(17, 13, 20);
    this.camera.lookAt(0, 3, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 3, 0);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.minZoom = 0.82;
    this.controls.maxZoom = 1.45;
    this.controls.minPolarAngle = 0.72;
    this.controls.maxPolarAngle = 1.25;
    this.controls.minAzimuthAngle = -0.75;
    this.controls.maxAzimuthAngle = 0.45;

    const hemisphere = new THREE.HemisphereLight(0xeafaff, 0x5c7650, 2.5);
    this.scene.add(hemisphere);
    const sun = new THREE.DirectionalLight(0xffffff, 3.2);
    sun.position.set(-8, 18, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
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
    world.rotation.y = -0.12;
    this.scene.add(world);
    this.world = world;
    mesh(new THREE.BoxGeometry(25, 0.7, 13), toon(colors.earth), [0, -0.45, 0], world);

    this.reservoir = mesh(new THREE.BoxGeometry(8.8, 4, 9), new THREE.MeshToonMaterial({ color: colors.water, transparent: true, opacity: 0.82 }), [-6.8, 2, 0], world);
    mesh(new THREE.BoxGeometry(8.8, 0.25, 9), toon(colors.waterLight), [-6.8, 4.05, 0], world);
    this.waterTop = world.children.at(-1);
    this.tailrace = mesh(new THREE.BoxGeometry(7.5, 0.8, 5.7), new THREE.MeshToonMaterial({ color: colors.water, transparent: true, opacity: 0.85 }), [8.1, 0.18, 0.5], world);

    const dam = new THREE.Group();
    dam.rotation.z = -0.08;
    world.add(dam);
    mesh(new THREE.BoxGeometry(2.6, 8.2, 9.6), toon(colors.concrete), [-1.7, 3.65, 0], dam);
    mesh(new THREE.BoxGeometry(0.55, 7.6, 8.9), toon(colors.concreteDark), [-0.35, 3.45, 0], dam);
    this.dam = dam;

    const intake = mesh(new THREE.BoxGeometry(0.6, 2.1, 2.4), toon(0x355b67), [-3.2, 3.1, 0], world);
    for (let z = -0.85; z <= 0.85; z += 0.42) {
      mesh(new THREE.BoxGeometry(0.1, 1.85, 0.08), toon(0xc6e2e7), [-3.55, 3.1, z], world);
    }
    const pipeMaterial = toon(colors.penstock);
    cylinderBetween([-3.05, 3.2, 0], [1.4, 0.95, 0], 0.42, pipeMaterial, world);
    cylinderBetween([1.4, 0.95, 0], [4.1, 0.65, 0], 0.5, pipeMaterial, world);

    this.turbine = mesh(new THREE.TorusGeometry(0.72, 0.23, 10, 24), toon(colors.turbine), [1.7, 1.15, 0], world);
    this.turbine.rotation.y = Math.PI / 2;
    this.generator = mesh(new THREE.BoxGeometry(2.1, 1.45, 1.8), toon(colors.generator), [3.05, 1.72, 0], world);
    mesh(new THREE.CylinderGeometry(0.14, 0.14, 1.25, 12), toon(0x4c5d62), [2.25, 1.45, 0], world).rotation.z = Math.PI / 2;
    this.transformer = mesh(new THREE.BoxGeometry(1.4, 1.65, 1.4), toon(0x65777a), [5.05, 1.35, -1.25], world);
    this.tower = tower(world, 8.6, -2.1);

    this.spill = mesh(new THREE.BoxGeometry(0.72, 6.8, 1.25), new THREE.MeshToonMaterial({ color: colors.waterLight, transparent: true, opacity: 0.76 }), [-0.1, 3.2, 3.1], world);
    this.spill.rotation.z = -0.18;
    this.spill.visible = false;

    const points = [
      new THREE.Vector3(-3.4, 3.2, 0),
      new THREE.Vector3(-1.1, 2.2, 0),
      new THREE.Vector3(1.55, 1.05, 0),
      new THREE.Vector3(4.2, 0.65, 0),
    ];
    this.flowCurve = new THREE.CatmullRomCurve3(points);
    this.flowParticles = Array.from({ length: 18 }, (_, index) => {
      const particle = mesh(new THREE.SphereGeometry(0.12, 8, 8), toon(0xc9f5ff), [0, 0, 0], world);
      particle.userData.offset = index / 18;
      return particle;
    });
    this.addLabel('Reservoir', this.reservoir, [-1.7, 2.4, 0]);
    this.addLabel('Dam', dam, [-1.1, 7.3, 0]);
    this.addLabel('Intake', intake, [0, 1.55, 0]);
    this.addLabel('Turbine', this.turbine, [0, -1.05, 0]);
    this.addLabel('Generator', this.generator, [0.1, 1.35, 0]);
    this.addLabel('Transformer', this.transformer, [0, 1.35, 0]);
    this.addLabel('Grid', this.tower.group, [0, 5.3, 0]);
    this.addLabel('Tailrace', this.tailrace, [0.6, 0.85, 0]);
  }

  addLabel(text, object, offset) {
    const element = document.createElement('span');
    element.className = 'scene-label';
    element.textContent = text;
    this.labelLayer.append(element);
    this.labels.push({ element, object, offset: new THREE.Vector3(...offset) });
  }

  update(result) {
    this.result = result;
    if (!this.renderer) return;
    const waterY = 2.25 + (result.inputs.grossHeadM - 5) / 245 * 4.7;
    const height = Math.max(0.5, waterY + 0.45);
    this.reservoir.scale.y = height / 4;
    this.reservoir.position.y = height / 2 - 0.35;
    this.waterTop.position.y = height - 0.28;
    const spillRatio = Math.min(1, result.spillFlowM3s / 12);
    this.spill.visible = result.spillFlowM3s > 0.02;
    this.spill.scale.x = 0.45 + spillRatio * 1.2;
    this.spill.material.opacity = 0.35 + spillRatio * 0.5;
    const powerRatio = result.inputs.ratingMW > 0 ? result.plantMW / result.inputs.ratingMW : 0;
    this.generator.material.color.setHex(powerRatio > 0.02 ? colors.generator : colors.gridOff);
    this.tower.material.color.setHex(powerRatio > 0.02 ? colors.gridOn : colors.gridOff);
  }

  resize() {
    if (!this.renderer) return;
    const width = Math.max(320, this.host.clientWidth);
    const height = Math.max(330, this.host.clientHeight);
    const aspect = width / height;
    const vertical = 8;
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
      element.hidden = point.z > 1;
    }
  }

  animate(time) {
    this.animationFrame = requestAnimationFrame(this.animate);
    if (!this.renderer || !this.active) return;
    const turbineFlow = this.result?.turbineFlowM3s ?? 0;
    const speed = this.reducedMotion ? 0 : Math.min(2.4, turbineFlow / 5);
    this.flowPhase = (this.flowPhase + speed * 0.006) % 1;
    this.flowParticles.forEach((particle) => {
      const t = (particle.userData.offset + this.flowPhase) % 1;
      particle.position.copy(this.flowCurve.getPointAt(t));
      particle.visible = turbineFlow > 0.02;
    });
    if (!this.reducedMotion) this.turbine.rotation.x += speed * 0.075;
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
