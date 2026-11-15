// Schematic vertical scale: a fixed submerged intake stays below the minimum
// operating surface. Gross head is still calculated in SI units by model.js.
export const INTAKE_CENTER_Y = 1.2;
export const INTAKE_RADIUS = 0.4;
export function visualState(result) {
  return {
    surfaceY: 1.85 + (result.inputs.grossHeadM - 5) / 245 * 4.45,
    turbineFlow: result.turbineFlowM3s,
    bypassFlow: result.spillFlowM3s,
    load: result.plantMW / result.inputs.ratingMW,
    energized: result.plantMW > 0,
  };
}
