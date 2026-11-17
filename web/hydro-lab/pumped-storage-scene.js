import * as T from 'three';
import { MAX_CYCLE_VOLUME_M3, pumpedCycle } from './balance-model.js';

export function createPumpedStorageScene({ material, mesh, box, tank, level, steel, rock, palette }) {
  const bench = 2.3;
  box(9.4, .18, 3.65, rock, 0, -.21, 0);
  box(2.3, bench, 2.45, rock, -3.03, bench / 2 - .12, 0);
  const upper = tank(-3.03, 2.3, 1.5, 2.25, bench);
  const lower = tank(3.02, 2.3, 1.5, 2.25);

  const curve = coordinates => new T.CatmullRomCurve3(coordinates.map(point => new T.Vector3(...point)));
  const waterRoute = curve([
    [-2.98, 2.55, .74], [-1.89, 2.55, .74], [-1.56, 2.43, .74],
    [-.41, .81, .74], [.05, .58, .74], [.75, .50, .74], [1.87, .25, .74], [2.75, .25, .74],
  ]);
  const electricalRoute = curve([[.28, .83, .34], [.34, 1.04, -.25], [.56, 1.07, -.86], [1.02, 1.36, -1.26]]);
  mesh(new T.TubeGeometry(waterRoute, 64, .14, 16), material(palette.water), 0, 0, 0);
  mesh(new T.TubeGeometry(electricalRoute, 24, .031, 8), steel, 0, 0, 0);
  const bronze = material(palette.shaft);
  const housing = mesh(new T.CylinderGeometry(.40, .40, .65, 24), bronze, .28, .56, .74);
  housing.rotation.z = Math.PI / 2;
  for (const x of [.05, .51]) {
    const rim = mesh(new T.CylinderGeometry(.414, .414, .05, 24), steel, x, .56, .74);
    rim.rotation.z = Math.PI / 2;
  }
  box(.92, .14, .93, rock, .28, .07, .74);
  box(.90, .62, .81, rock, 1.02, .31, -1.29);
  box(.61, .55, .47, steel, 1.02, .90, -1.29);
  box(.61, .13, .47, bronze, 1.02, 1.22, -1.29);
  for (const z of [-1.47, -1.29, -1.11]) box(.065, .24, .065, steel, 1.02, 1.36, z);

  const dots = (count, radius, color) => {
    const geometry = new T.SphereGeometry(radius, 12, 8);
    const glow = new T.MeshBasicMaterial({ color });
    return Array.from({ length: count }, () => mesh(geometry, glow, 0, 0, 0));
  };
  const water = dots(13, .073, '#beeffb');
  const electricity = dots(5, .066, palette.electricity);

  function move(markers, path, phase, active, reverse, frontOffset) {
    markers.forEach((marker, index) => {
      marker.visible = active;
      if (!active) return;
      const progress = (phase * 8 + index / markers.length) % 1;
      path.getPointAt(reverse ? 1 - progress : progress, marker.position);
      marker.position.z += frontOffset;
    });
  }

  return ([volume, head], phase) => {
    const active = volume > 0 && head > 0;
    const { pumping, upper: fraction } = pumpedCycle(active ? phase : 0);
    const fill = volume / MAX_CYCLE_VOLUME_M3;
    level(upper, .32 + fill * fraction, bench);
    level(lower, .32 + fill * (1 - fraction));
    move(water, waterRoute, phase, active, pumping, .20);
    move(electricity, electricalRoute, phase, active, pumping, .02);
  };
}
