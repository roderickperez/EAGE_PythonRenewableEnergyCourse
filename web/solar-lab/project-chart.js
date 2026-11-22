export const COLORS=['#007b95','#c0632d','#8457a7','#28845d','#bb4675','#6c75bc','#917018','#3f6f99'];
export const panelColor=id=>COLORS[id%COLORS.length];

export function chartMarkup(points,ids,hour,currentPanels) {
  let maximum=.1;for(const point of points)for(const value of Object.values(point.values))maximum=Math.max(maximum,value/1000);for(const panel of currentPanels)maximum=Math.max(maximum,panel.current.ac/1000);const max=maximum*1.15;
  const x=t=>35+t/24*250,y=v=>140-v/1000/max*115;
  let markup=`<path d="M35 12V140H285" fill="none" stroke="#9aafb8"/><path data-now d="M${x(hour)} 12V140" stroke="#db9e26" stroke-width="2" stroke-dasharray="5 4"/><text x="31" y="17" text-anchor="end">${max.toFixed(2)}</text><text x="31" y="143" text-anchor="end">0</text><text x="35" y="158">00</text><text x="160" y="158" text-anchor="middle">12 · solar time</text><text x="285" y="158" text-anchor="end">24</text><text x="40" y="25">kW / panel</text>`;
  for(const id of ids){let active=false,d='';for(const p of points){const value=p.values[id];if(value===undefined){active=false;continue;}d+=`${active?'L':'M'}${x(p.hour).toFixed(2)} ${y(value).toFixed(2)} `;active=true;}
    markup+=`<path d="${d}" fill="none" stroke="${panelColor(id)}" stroke-width="2"/>`;
  }
  return markup;
}
