export const THERMAL_AMBIENT_C = 20;
export const THERMAL_ABSORPTION = .78;
export const THERMAL_IRRADIANCE_MAX = 1200;

export function thermalCollector({ irradiance, area, flow, heatCapacity, inlet, lossCoefficient }) {
  const incident = irradiance * area;
  const absorbed = incident * THERMAL_ABSORPTION;
  const ua = lossCoefficient * area;
  const heatingAboveInlet = absorbed - ua * (inlet - THERMAL_AMBIENT_C);
  const running = flow > 0 && heatingAboveInlet > 0;
  if (!running) {
    return { running, incident, absorbed, useful: 0, thermalLoss: null, outlet: null,
      efficiency: incident > 0 ? 0 : null };
  }
  // Solve ṁ cₚ ΔT = absorbed − UA (T_in + ΔT/2 − T_air) in one step.
  const temperatureRise = heatingAboveInlet / (flow * heatCapacity + ua / 2);
  const useful = flow * heatCapacity * temperatureRise;
  return { running, incident, absorbed, useful, thermalLoss: absorbed - useful,
    outlet: inlet + temperatureRise, efficiency: useful / incident };
}
