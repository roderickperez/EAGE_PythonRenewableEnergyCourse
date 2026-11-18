const f = (n, places=1) => Number(n).toFixed(places);

// Both figures are redrawn from the same values as the numerical calculations.
export function channelFigure({width, depth, velocity, channelFlow, riverFlow, slope, catchment}) {
  const plotWidth=width/18*210, plotHeight=depth/5*70, left=28+(210-plotWidth)/2, bottom=95;
  const max=Math.max(100, channelFlow, riverFlow), scale=210/max;
  return `<svg viewBox="0 0 380 220" role="img" aria-label="Channel cross section ${f(width)} metres wide and ${f(depth)} metres deep. Channel flow ${f(channelFlow)} and mean river flow ${f(riverFlow)} cubic metres per second.">
    <title>Channel geometry and flow comparison</title>
    <text x="12" y="18" class="title">Channel cross section</text>
    <path d="M${left} 25 V${bottom} H${left+plotWidth} V25" class="wall"/>
    <rect x="${left+1}" y="${bottom-plotHeight}" width="${Math.max(0,plotWidth-2)}" height="${plotHeight}" class="water"/>
    <path d="M${left+4} ${bottom-plotHeight+8} H${left+plotWidth-4}" class="current"/>
    <text x="${left+plotWidth/2}" y="111" text-anchor="middle">${f(width)} m width</text>
    <text x="250" y="43">Water: ${f(depth)} m deep</text><text x="250" y="62">Slope: ${f(slope,3)}%</text><text x="250" y="81">Velocity: ${f(velocity,2)} m/s</text>
    <text x="12" y="137" class="title">Flow comparison (m³/s)</text>
    <text x="12" y="158">Channel</text><rect x="80" y="147" width="${channelFlow*scale}" height="12" rx="4" class="water"/><text x="300" y="158">${f(channelFlow)}</text>
    <text x="12" y="180">River mean</text><rect x="80" y="169" width="${riverFlow*scale}" height="12" rx="4" class="green"/><text x="300" y="180">${f(riverFlow)}</text>
    <text x="12" y="207">Inferred catchment: ${f(catchment,0)} km²</text>
  </svg>`;
}

export function reservoirFigure({mean, spill, ecological, usable, shortfall, meanHead, designHead, annual, power}) {
  const scale=mean>0?260/mean:0, x=24, maxHead=Math.max(150, meanHead, designHead), headScale=110/maxHead;
  const spillW=spill*scale, ecoW=ecological*scale, useW=usable*scale;
  return `<svg viewBox="0 0 380 220" role="img" aria-label="Mean inflow ${f(mean,2)} cubic metres per second: spill ${f(spill,2)}, ecological release ${f(ecological,2)}, main generation ${f(usable,2)}; ecological shortfall ${f(shortfall,2)}.">
    <title>Mean water allocation and head</title>
    <text x="12" y="18" class="title">Mean inflow: ${f(mean,2)} m³/s</text>
    <rect x="${x}" y="30" width="${spillW}" height="22" class="amber"/><rect x="${x+spillW}" y="30" width="${ecoW}" height="22" class="green"/><rect x="${x+spillW+ecoW}" y="30" width="${useW}" height="22" class="water"/>
    <text x="12" y="72" class="amber-text">■ Spill ${f(spill,2)}</text><text x="132" y="72" class="green-text">■ Ecological ${f(ecological,2)}</text><text x="266" y="72" class="blue-text">■ Main ${f(usable,2)}</text>
    <text x="12" y="94">Ecological shortfall: ${f(shortfall,2)} m³/s</text>
    <text x="12" y="120" class="title">Net head (m)</text>
    <text x="12" y="144">Mean</text><rect x="80" y="133" width="${meanHead*headScale}" height="12" rx="4" class="green"/><text x="200" y="144">${f(meanHead)}</text>
    <text x="12" y="166">Design</text><rect x="80" y="155" width="${designHead*headScale}" height="12" rx="4" class="water"/><text x="200" y="166">${f(designHead)}</text>
    <text x="12" y="194">Annual runoff: ${f(annual/1e6,1)} hm³</text><text x="12" y="212">Design generator: ${f(power/1000,2)} MW</text>
  </svg>`;
}
