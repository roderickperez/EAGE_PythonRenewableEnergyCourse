export const MAX_FLOW_M3S = 8;
export const MAX_INTERVAL_MINUTES = 240;
export const OUTLET_PRESSURE_HEAD_M = 2;
export const FITTING_POSITION = .8;
export const MAX_PIPE_FLOW_M3S = 16;

export function netHead(gross, loss) {
  return gross - loss;
}

export function flowBalance(flow, minutes, density) {
  const seconds = minutes * 60;
  return { seconds, volume: flow * seconds, massFlow: density * flow };
}

export function powerBalance(flow, head, density, gravity, turbineEfficiency = 1, generatorEfficiency = 1) {
  const hydraulicW = density * gravity * flow * head;
  const shaftW = hydraulicW * turbineEfficiency;
  return { hydraulicW, shaftW, electricalW: shaftW * generatorEfficiency };
}
export function pipeLoss(flow, length, diameter, frictionFactor, fittingK, gravity) {
  const area = Math.PI * diameter ** 2 / 4;
  const velocity = flow / area;
  const velocityHead = velocity ** 2 / (2 * gravity);
  const friction = frictionFactor * length / diameter * velocityHead;
  const fittings = fittingK * velocityHead;
  return { area, velocity, friction, fittings, total: friction + fittings };
}

export function turbineOperation(flow, head, efficiency, ratingMW, density, gravity) {
  const wattsPerFlow = efficiency * density * gravity * head;
  const ratedFlow = wattsPerFlow > 0 ? ratingMW * 1e6 / wattsPerFlow : null;
  const turbineFlow = ratedFlow === null ? 0 : Math.min(flow, ratedFlow);
  return {
    ratedFlow,
    turbineFlow,
    bypassFlow: flow - turbineFlow,
    outputMW: turbineFlow * wattsPerFlow / 1e6,
  };
}

export function pipeHeadProfile(loss, length) {
  const location = length * FITTING_POSITION;
  const inlet = OUTLET_PRESSURE_HEAD_M + loss.total;
  const beforeFitting = inlet - loss.friction * FITTING_POSITION;
  return [[0, inlet], [location, beforeFitting],
    [location, beforeFitting - loss.fittings], [length, OUTLET_PRESSURE_HEAD_M]];
}
