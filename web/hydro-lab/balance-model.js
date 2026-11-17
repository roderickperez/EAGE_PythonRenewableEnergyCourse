export const MAX_FLOW_M3S = 8;
export const MAX_INTERVAL_MINUTES = 240;

export function netHead(gross, loss) {
  return gross - loss;
}

export function flowBalance(flow, minutes, density) {
  const seconds = minutes * 60;
  return { seconds, volume: flow * seconds, massFlow: density * flow };
}
