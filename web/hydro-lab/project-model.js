export const CAPACITY = 50e6, DEAD = 5e6, DT = 86400, HORIZON = 365;
export function initialState() {
  return { day: 0, storage: 30e6, energy: 0, spill: 0, shortage: 0 };
}
export function step(state, requestedFlow) {
  const inflow = 4.57 * (1 + .55 * Math.sin(2 * Math.PI * (state.day - 30) / 365));
  const efficiency = Math.max(.75, .91 - .0001 * state.day);
  const head = 60 + 40 * state.storage / CAPACITY;
  let available = state.storage + inflow * DT;
  const environmental = Math.min(DT, available);
  available -= environmental;
  const turbine = Math.min(Math.max(0, requestedFlow) * DT, Math.max(0, available - DEAD));
  available -= turbine;
  const spill = Math.max(0, available - CAPACITY);
  const storage = available - spill;
  const power = 1000 * 9.81 * (turbine / DT) * head * efficiency * .95 * .99 / 1e6;
  return { day: state.day + 1, storage, energy: state.energy + power * 24,
    spill: state.spill + spill, shortage: state.shortage + DT - environmental,
    inflow, turbineFlow: turbine / DT, environmental: environmental / DT, head, efficiency, power, spillFlow: spill / DT };
}
