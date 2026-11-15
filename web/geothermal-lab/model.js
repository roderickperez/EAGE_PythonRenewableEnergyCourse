export const defaults = {
  reservoir: 150, wellLoss: 0, reinjection: 70, flow: 50, cp: 4180, depth: 3,
  sink: 25, efficiency: 12, parasitic: 10, pressure: 2, density: 1000, pumpEfficiency: 75,
  auxiliaries: .05, rating: 5, hours: 4, availability: 95, decline: 2, years: 25,
  surface: 15, gradient: 35, conductivity: 2.5, permeability: 50, viscosity: .5,
  pressureGradient: 10, volume: 5, porosity: 12, rockDensity: 2600, rockCp: 900,
  heatPumpPower: .5, cop: 3.5, loadMode: 'fraction',
};

export function geothermal(input) {
  const production = input.reservoir - input.wellLoss;
  const reinjection = Math.min(production, Math.max(input.sink, input.reinjection));
  const delta = production - reinjection;
  const hotK = production + 273.15;
  const returnK = reinjection + 273.15;
  const coldK = input.sink + 273.15;
  const carnot = Math.max(0, 1 - coldK / hotK);
  // Reversible work from a constant-cp stream cooling toward a fixed sink.
  const streamLimit = delta > 0 ? Math.max(0, 1 - coldK * Math.log1p(delta / returnK) / delta) : 0;
  const conversion = Math.min(input.efficiency / 100, streamLimit);
  const enthalpy = input.cp * delta;
  const thermal = input.flow * enthalpy / 1e6;
  const gross = thermal * conversion;
  const volumeFlow = input.flow / input.density;
  const pump = input.pressure * volumeFlow / (input.pumpEfficiency / 100);
  const parasitic = input.loadMode === 'explicit' ? pump + (input.flow > 0 ? input.auxiliaries : 0) : gross * input.parasitic / 100;
  const net = gross - parasitic;
  const exported = Math.min(input.rating, Math.max(net, 0));
  const imported = Math.max(0, -net);
  const annual = Array.from({ length: input.years }, (_, i) =>
    exported * (1 - input.decline / 100) ** i * 8760 * input.availability / 100);
  const depthTemp = input.surface + input.depth * input.gradient;
  const heatFlux = input.conductivity * input.gradient; // mW/m² with gradient in K/km
  const darcy = input.permeability * 9.869233e-16 / (input.viscosity / 1000) * input.pressureGradient * 1000;
  const pore = input.porosity / 100;
  const stored = input.volume * 1e6 * ((1 - pore) * input.rockDensity * input.rockCp + pore * input.density * input.cp)
    * (input.reservoir - input.surface) / 3.6e12; // GWh thermal
  return { production, reinjection, delta, carnot, streamLimit, conversion, enthalpy, thermal, gross, volumeFlow, pump,
    parasitic, net, exported, imported, rejected: thermal - gross, curtailed: Math.max(0, net - exported),
    interval: exported * input.hours, annual, lifetime: annual.reduce((sum, year) => sum + year, 0),
    cf: annual[0] / (input.rating * 8760), depthTemp, heatFlux, darcy, stored };
}
