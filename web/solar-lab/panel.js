import { Group } from 'three';

export function createSolarPanel(box, material, steel) {
  const panel = new Group();
  box([3.8, .16, 2.85], steel, [0, 0, 0], panel);
  box([3.66, .045, 2.71], material('#142740', .28), [0, .083, 0], panel);
  const cells = ['#173963', '#1a416e', '#204a77'].map(color => material(color, .27, .26));
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 8; col++) {
      const x = (col - 3.5) * .45;
      const z = (row - 2.5) * .442;
      box([.426, .019, .418], cells[(row + col * 2) % 3], [x, .114, z], panel);
    }
  }
  return panel;
}
