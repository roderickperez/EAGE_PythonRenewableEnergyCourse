// Independent teaching implementation of a Jensen-style top-hat wake model.
export const FARM = 2000;
export const GRID = 10;
export const DIAMETER = 40;
export const MIN_SPACING = 200;
export const HUB = 60;
export const ROUGHNESS = 0.3;
export const CT = 0.88;
export const INDUCTION = (1 - Math.sqrt(1 - CT)) / 2;
export const DECAY = 0.5 / Math.log(HUB / ROUGHNESS);
export const INITIAL_WAKE = (DIAMETER / 2) * Math.sqrt((1 - INDUCTION) / (1 - 2 * INDUCTION));

export function gridPoint(id) {
  if (!Number.isInteger(id) || id < 0 || id >= GRID * GRID) throw new RangeError('Invalid grid point');
  return { id, x: (id % GRID + 0.5) * FARM / GRID, y: (Math.floor(id / GRID) + 0.5) * FARM / GRID };
}

export function powerKw(speed) {
  if (speed < 3 || speed > 25) return 0;
  return speed < 12 ? 0.3 * speed ** 3 : 518.4;
}

export function windFieldSampler(ids, fromDegrees, speed) {
  if (!Number.isFinite(speed) || speed < 0 || !Number.isFinite(fromDegrees)) throw new RangeError('Invalid wind');
  if (new Set(ids).size !== ids.length) throw new Error('Duplicate turbines');
  const theta = fromDegrees * Math.PI / 180;
  const down = { x: -Math.sin(theta), y: -Math.cos(theta) };
  const cross = { x: Math.cos(theta), y: -Math.sin(theta) };
  const turbines = ids.map(gridPoint).map(p => ({ along:p.x * down.x + p.y * down.y, across:p.x * cross.x + p.y * cross.y }));
  return (x, y) => {
    const along = x * down.x + y * down.y, across = x * cross.x + y * cross.y;
    let sum = 0;
    for (const other of turbines) {
      const distance = along - other.along;
      if (distance > 1e-8 && Math.abs(across - other.across) < INITIAL_WAKE + DECAY * distance) {
        const deficit = 2 * INDUCTION / (1 + DECAY * distance / INITIAL_WAKE) ** 2;
        sum += deficit ** 2;
      }
    }
    return Math.max(0, speed * (1 - Math.sqrt(sum)));
  };
}

export function evaluate(ids, fromDegrees, speed) {
  const sample = windFieldSampler(ids, fromDegrees, speed);
  const results = ids.map(gridPoint).map(p => {
    const effective = sample(p.x, p.y);
    return { ...p, speed: effective, power: powerKw(effective) };
  });
  const total = results.reduce((sum, p) => sum + p.power, 0);
  const ideal = ids.length * powerKw(speed);
  return { results, total, ideal, loss: ideal - total, efficiency: ideal > 0 ? 100 * total / ideal : null };
}

export function validCandidate(id, ids, spacing) {
  const p = gridPoint(id);
  return !ids.includes(id) && ids.every(i => {
    const q = gridPoint(i);
    return Math.hypot(p.x - q.x, p.y - q.y) + 1e-9 >= spacing;
  });
}

export function rankCandidates(ids, direction, speed, spacing) {
  const baseline = evaluate(ids, direction, speed).total;
  return Array.from({ length: GRID * GRID }, (_, i) => i)
    .filter(id => validCandidate(id, ids, spacing))
    .map(id => ({ id, total: evaluate([...ids, id], direction, speed).total }))
    .map(p => ({ ...p, gain: p.total - baseline }))
    .sort((a, b) => b.total - a.total || a.id - b.id);
}
