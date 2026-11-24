import * as T from 'three';

const cold = new T.Color('#b1efff');
const hot = new T.Color('#ffd097');
const lineCold = new T.Color('#75cde7');
const lineHot = new T.Color('#e0a16f');

// Project selected streamlines onto the cutaway face. The dots, rather than
// solid cylinders, distinguish flow in the formation from flow in the wells.
export function createReservoirFlow(scene, inlet, outlet, { fractured = false } = {}) {
  const paths = new T.Group();
  const through = new T.Group();
  const branches = new T.Group();
  const interrupted = new T.Group();
  interrupted.visible = false;
  paths.add(through, branches, interrupted);
  scene.add(paths);

  const [x0, y0, z] = inlet;
  const [x1, y1] = outlet;
  const point = (fraction, rise) =>
    new T.Vector3(T.MathUtils.lerp(x0, x1, fraction), T.MathUtils.lerp(y0, y1, fraction) + rise, z);
  const beadGeometry = new T.SphereGeometry(.065, 10, 8);
  const color = new T.Color();

  function trace(points, parent, opacity) {
    const geometry = new T.BufferGeometry().setFromPoints(points);
    const colors = points.flatMap(p =>
      color.copy(lineCold).lerp(lineHot, T.MathUtils.clamp((p.x - x0) / (x1 - x0), 0, 1)).toArray());
    geometry.setAttribute('color', new T.Float32BufferAttribute(colors, 3));
    const material = new T.LineDashedMaterial({
      color: 0xffffff, vertexColors: true, transparent: true, opacity,
      depthWrite: false, dashSize: fractured ? .26 : .12, gapSize: .1,
    });
    const line = new T.Line(geometry, material);
    line.computeLineDistances();
    parent.add(line);
  }

  const lanes = fractured ? [
    [[0,0],[.15,-.12],[.3,.18],[.42,-.20],[.55,.18],[.72,-.28],[.88,-.09],[1,0]],
    [[0,0],[.13,.23],[.27,.42],[.42,-.20],[.59,-.38],[.73,-.40],[.88,-.09],[1,0]],
    [[0,0],[.15,-.12],[.31,-.48],[.44,-.44],[.55,.18],[.71,.34],[.86,.27],[1,0]],
  ] : [-.33, 0, .40].map(rise =>
    [[0,0],[.18,rise*.55],[.4,rise],[.62,rise*.8],[.84,rise*.6],[1,0]]);

  const routes = lanes.map(lane => {
    const points = lane.map(([fraction, rise]) => point(fraction, rise));
    const curve = fractured
      ? new T.CurvePath()
      : new T.CatmullRomCurve3(points, false, 'centripetal');
    if (fractured) {
      for (let i=1; i<points.length; i++) curve.add(new T.LineCurve3(points[i-1], points[i]));
      trace(points.slice(0, 4), interrupted, .86);
      trace(points.slice(5), interrupted, .86);
    }
    trace(curve.getPoints(90), through, fractured ? .86 : .52);
    const beads = Array.from({length: fractured ? 6 : 8}, () => {
      const bead = new T.Mesh(beadGeometry, new T.MeshBasicMaterial());
      through.add(bead);
      return bead;
    });
    return { curve, beads, distance: curve.getLength() };
  });

  if (fractured) {
    for (const branch of [
      [[.13,.23],[.14,.43],[.21,.49]], [[.27,.42],[.25,.58],[.36,.62]],
      [[.31,-.48],[.28,-.58],[.39,-.58]], [[.71,.34],[.65,.47],[.72,.55]],
      [[.86,.27],[.87,.43],[.92,.48]], [[.73,-.40],[.78,-.54],[.88,-.56]],
    ]) trace(branch.map(([fraction,rise]) => point(fraction,rise)), branches, .65);
  }

  return {
    setConnected(connected) {
      through.visible = connected;
      interrupted.visible = !connected;
    },
    draw(distance, flowing) {
      routes.forEach(({curve, beads, distance: length}, index) => {
        beads.forEach((bead, i) => {
          bead.visible = flowing;
          if (!flowing) return;
          const progress = (distance / length + i / beads.length + index * .13) % 1;
          bead.position.copy(curve.getPointAt(progress));
          bead.position.z += .035;
          bead.material.color.copy(cold).lerp(hot, progress);
        });
      });
    },
  };
}
