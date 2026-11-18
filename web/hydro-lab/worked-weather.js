import * as T from 'three';

export class CatchmentWeather {
  constructor(scene, { clouds, maxRainfall = 2000, dropHeight = 2.9 }) {
    this.maxRainfall = maxRainfall;
    const shape = new T.IcosahedronGeometry(1, 2);
    const colors = ['#cbd4e0', '#e9eef5', '#f5f8fb'].map(color =>
      new T.MeshStandardMaterial({ color, roughness: 1, flatShading: true })
    );
    const lobes = [
      [-.97, -.07, .05, .65, .37, .58], [-.38, .22, -.17, .78, .62, .62],
      [.46, .25, -.05, .96, .65, .71], [.97, -.08, .19, .67, .36, .57],
      [-.18, -.13, .21, 1.3, .37, .66],
    ];
    this.drops = [];
    for (const [x, y, z, scale] of clouds) {
      const cloud = new T.Group();
      cloud.position.set(x, y, z);
      cloud.scale.setScalar(scale);
      lobes.forEach(([px, py, pz, sx, sy, sz], index) => {
        const lobe = new T.Mesh(shape, colors[index === 4 ? 0 : 1 + index % 2]);
        lobe.position.set(px, py, pz);
        lobe.scale.set(sx, sy, sz);
        cloud.add(lobe);
      });
      scene.add(cloud);
      for (let i = 0; i < 60; i++) {
        const spread = (i * .61803398875) % 1;
        const depth = (i * .75487766625) % 1;
        this.drops.push({
          x: x + (spread - .5) * 2.7 * scale,
          z: z + (depth - .5) * 1.25 * scale,
          top: y - .34 * scale,
          fall: dropHeight * (.9 + .1 * depth),
          phase: (i * .56984029099 + x) % 1,
        });
      }
    }
    // Alternate between clouds so light rain is visible under every cloud.
    const interleaved = [];
    for (let i = 0; i < 60; i++) {
      for (let j = 0; j < clouds.length; j++) interleaved.push(this.drops[j * 60 + i]);
    }
    this.drops = interleaved;
    this.positions = new T.BufferAttribute(new Float32Array(this.drops.length * 6), 3);
    this.positions.setUsage(T.DynamicDrawUsage);
    this.geometry = new T.BufferGeometry().setAttribute('position', this.positions);
    this.material = new T.LineBasicMaterial({ color: '#3888b7', transparent: true, depthWrite: false });
    const rain = new T.LineSegments(this.geometry, this.material);
    rain.frustumCulled = false;
    scene.add(rain);
    this.count = 0;
    this.elapsed = 0;
  }

  setPrecipitation(mm) {
    const intensity = T.MathUtils.clamp(mm / this.maxRainfall, 0, 1);
    this.count = Math.round(intensity * this.drops.length);
    this.material.opacity = .56 + intensity * .38;
    this.geometry.setDrawRange(0, this.count * 2);
    this.step(0);
  }

  step(delta) {
    this.elapsed += delta * .96;
    for (let i = 0; i < this.count; i++) {
      const { x, z, top, fall, phase } = this.drops[i];
      const t = T.MathUtils.euclideanModulo(phase + this.elapsed, 1);
      const y = top - t * fall;
      this.positions.setXYZ(i * 2, x + t * .29, y, z);
      this.positions.setXYZ(i * 2 + 1, x + t * .29 - .041, Math.min(top, y + .35), z);
    }
    this.positions.needsUpdate = true;
  }
}
