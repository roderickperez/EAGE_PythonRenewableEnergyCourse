// Rounded readings from the centers of the four overlapping model traces
// in White et al. (2024), Figure 3 (DOI: 10.1016/j.geothermics.2023.102852).
// The near-zero startup transient is omitted because the published curves overlap the axis.
export const benchmark = [
  { label:'11.8 kg/s · 1,174 m insulation', color:'#c45a3c' },
  { label:'38 kg/s · 875 m insulation', color:'#257d9d' },
];
export const outletReadings = [
  [.25,165,93], [.5,158,88], [1,151,85], [2,147,81], [3,144,79],
  [5,141,77], [10,136,74], [15,134,73], [20,133,73], [25,132,72],
  [30,131,71], [35,130,71], [40,129,70],
];

export function outletAt(year) {
  if (year <= outletReadings[0][0]) return outletReadings[0].slice(1);
  if (year >= outletReadings.at(-1)[0]) return outletReadings.at(-1).slice(1);
  const next = outletReadings.findIndex(([time]) => time >= year);
  const [t0,a0,b0] = outletReadings[next-1], [t1,a1,b1] = outletReadings[next];
  const fraction = (year-t0)/(t1-t0);
  return [a0+(a1-a0)*fraction,b0+(b1-b0)*fraction];
}

// Table 3 reports the selected electricity designs with independent settings.
export const scenarios = [
  { geometry:'U-shaped', fluid:'Water', flow:43, length:20, outlet:205, thermal:26, electric:3.9, lcoe:96.6 },
  { geometry:'Coaxial', fluid:'Water', flow:39.2, length:20, outlet:203, thermal:23.8, electric:3.5, lcoe:93.5 },
  { geometry:'U-shaped', fluid:'Supercritical CO₂', flow:100, length:20, outlet:190, thermal:22.2, electric:5, lcoe:83 },
  { geometry:'Coaxial', fluid:'Supercritical CO₂', flow:69.6, length:13, outlet:174, thermal:14.1, electric:2.8, lcoe:86.9 },
];

export function levelizedCost({ capital, operating, generation, decline, discount, years }) {
  let presentCost = capital*1e6, presentEnergy = 0;
  const timeline = [];
  for (let year=1; year<=years; year++) {
    const discountFactor = (1+discount/100)**year;
    presentCost += operating*1e6/discountFactor;
    const electricity = generation*1000*(1-decline/100)**(year-1);
    presentEnergy += electricity/discountFactor;
    timeline.push({ year, cost:presentCost, energy:presentEnergy, lcoe:presentCost/presentEnergy });
  }
  return timeline;
}
