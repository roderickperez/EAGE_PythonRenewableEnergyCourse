const blue = '#247fae', hot = '#bd522e', green = '#267762', gold = '#916515';
const ink = '#183b45', muted = '#4c6670', grid = '#d2e1e4';
const fmt = (n, d = 2) => Number(n.toFixed(d)).toLocaleString('en-US', { maximumFractionDigits: d });
const label = (x,y,value,opts='') => `<text x="${x}" y="${y}" ${opts}>${value}</text>`;
const middle = (x,y,value) => label(x,y,value,'text-anchor="middle"');
const small = (x,y,value) => label(x,y,value,`fill="${muted}" font-size="12"`);
const line = (x1,y1,x2,y2,color=grid,extra='') =>
  `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${color}" fill="none" ${extra}/>`;
const box = (x,y,w,h,color,r=5) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${color}"/>`;
const dot = (x,y,r,color) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const arrow = (x1,y1,x2,y2,color,width=2) =>
  line(x1,y1,x2,y2,color,`stroke-width="${width}" marker-end="url(#${color.slice(1)})"`);
const markers = [blue,hot,green,gold].map(c => `<marker id="${c.slice(1)}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${c}"/></marker>`).join('');
const flowD = [
  'M65 123 C155 90 195 121 259 105 S365 117 493 107',
  'M65 123 C155 154 195 129 259 149 S365 141 493 107',
  'M65 123 C185 140 255 110 334 125 S432 126 493 107',
];

function bars(rows, max, unit) {
  const left=145, width=320, top=49, step=43;
  let html='';
  for(const value of [0,max/2,max]){
    const x=left+width*value/max;
    html+=line(x,top-12,x,top+step*(rows.length-1)+26)+middle(x,top+step*(rows.length-1)+45,`${fmt(value)} ${unit}`);
  }
  rows.forEach(([name,value,color],i)=>{
    const y=top+step*i, w=width*value/max;
    html+=label(18,y+17,name)+box(left,y,Math.max(0,w),23,color,3);
    html+=label(left+w+7,y+17,`${fmt(value,3)}`);
  });
  return html;
}

const drawings = {
  darcy(input, { flux }, motion) {
    const duration = 10 / (1 + Math.log1p(flux * 1e6));
    const head = Math.max(0,input.pressureGradient / 30) * 30;
    const particles = flux > 0 ? flowD.map((d,lane) =>
      Array.from({length: 4}, (_,i)=>{
        const phase=(i+lane*.26)/4;
        return `<circle r="4.1" fill="${blue}"><animateMotion path="${d}" dur="${duration}s" begin="-${phase*duration}s" repeatCount="indefinite"/></circle>`;
      }).join('')).join('') : '';
    return {
      title:'Pressure gradient and flow through connected pores',
      html: middle(280,24,'Pressure falls in the direction of flow') +
        line(65,70-head,493,70,blue,'stroke-width="3"') +
        small(65,84,'Inlet') + label(493,84,'Outlet','text-anchor="end" font-size="12"') +
        box(43,94,470,77,'#e9ddcb') +
        flowD.map(d=>`<path d="${d}" fill="none" stroke="#c2e8f1" stroke-width="9" stroke-linecap="round"/>`).join('') +
        box(117,105,19,12,'#a6957d') + box(204,120,24,14,'#a6957d') + box(329,104,27,11,'#a6957d') + box(402,141,21,16,'#a6957d') +
        particles + (flux > 0 ? arrow(187,201,372,201,blue,3) : '') +
        middle(280,226,flux>0 ? `Darcy flux magnitude · ${flux.toExponential(2)} m/s` : 'Zero pressure gradient · no flow') +
        small(43,254,'Pore paths are schematic. Dot speed shows relative flux.'),
      paused: !motion
    };
  },
  gradient(input,{temperature, flux}) {
    const depthY=42+input.depth/6*174;
    const x=t=>64+t/500*445;
    return {title:'Temperature profile with depth and upward conductive heat flux',html:
      line(64,42,64,218)+line(64,218,510,218)+
      [0,100,200,300,400,500].map(t=>line(x(t),42,x(t),218)+middle(x(t),237,fmt(t))).join('')+
      [0,2,4,6].map(d=>small(21,46+29*d,`${d} km`)).join('')+
      line(x(input.surface),42,x(input.surface+6*input.gradient),216,hot,'stroke-width="3"')+
      line(64,depthY,x(temperature),depthY,muted,'stroke-dasharray="4 4"')+
      dot(x(temperature),depthY,6,hot)+
      arrow(470,180,470,180-75*flux/.375,blue,3)+
      small(330,195,`↑ ${fmt(flux,4)} W/m²`)+middle(280,261,'Temperature (°C)')};
  },
  heat(input,{power, enthalpy}) {
    const x=t=>62+t/230*444, y=96;
    return {title:'Production and return temperature, and heat removed at the heat exchanger',html:
      middle(280,29,'Temperature drop across the heat exchanger')+
      line(62,y,506,y)+[0,50,100,150,200].map(t=>line(x(t),y,x(t),y+8)+middle(x(t),y+26,fmt(t))).join('')+
      arrow(x(input.reservoir),y-13,x(input.reinjection),y-13,hot,3)+
      dot(x(input.reservoir),y,6,hot)+dot(x(input.reinjection),y,6,blue)+
      middle(280,147,'Temperature (°C)')+
      box(36,168,488,62,'#e1eef0')+
      middle(280,192,`${fmt(input.flow)} kg/s × ${fmt(enthalpy,1)} kJ/kg`)+
      middle(280,215,`Heat recovered · ${fmt(power,3)} MW thermal`)+
      middle(280,252,`Production ${input.reservoir} °C · Return ${input.reinjection} °C`)};
  },
  carnot(input,{efficiency}) {
    const width=440, start=60, work=width*efficiency;
    return {title:'Ideal work and rejected heat as fractions of heat input',html:
      box(48,30,210,59,'#f5dfd4')+box(302,30,210,59,'#dcedf5')+
      middle(153,53,'Hot reservoir')+middle(153,76,`${fmt(input.reservoir+273.15)} K`)+
      middle(407,53,'Cold reservoir')+middle(407,76,`${fmt(input.sink+273.15)} K`)+
      middle(280,128,'Share of input heat in the ideal cycle')+
      box(start,150,width,38,'#badbe7',0)+box(start,150,work,38,green,0)+
      middle(start+work/2,174,`${fmt(efficiency*100,1)}%`).replace('<text ','<text fill="#fff" ')+
      middle(start+work+(width-work)/2,174,`${fmt((1-efficiency)*100,1)}%`)+
      dot(73,219,6,green)+label(86,224,'Maximum work')+
      dot(286,219,6,blue)+label(300,224,'Heat rejected')};
  },
  conversion(input,{thermal,gross,aux,net}) {
    return {title:'Thermal input and electrical output including auxiliary loads',html:
      bars([['Thermal input',thermal,hot],['Gross electricity',gross,blue],['Auxiliary loads',aux,gold],['Net electricity',net,green]],150,'MW')};
  },
  worked(input,{thermal,gross,aux,net}) {
    const rows = [['01 · Heat recovered',thermal,'MW thermal'],['02 · Electricity before loads',gross,'MW'],
      ['03 · Auxiliary demand',aux,'MW'],['04 · Net energy over 4 h',net*4,'MWh']];
    return {title:'Worked calculation from heat extraction to four hours of net energy',html:
      rows.map(([name,value,unit],i)=>{
        const y=17+i*60;
        return box(35,y,490,48,i%2?'#edf4f6':'#e4eeea')+
          label(51,y+30,name)+label(507,y+30,`${fmt(value,3)} ${unit}`,'text-anchor="end" font-weight="700"');
      }).join('')};
  },
  pump(input,{volumeFlow,power}) {
    const pressureWidth=195*input.pressure/15;
    return {title:'Pump pressure rise, water volume flow and required shaft power',html:
      middle(280,38,`Flow through pump · ${fmt(volumeFlow,4)} m³/s`)+
      arrow(40,102,218,102,blue,3)+arrow(344,102,518,102,blue,3)+
      box(218,70,125,64,'#d6e8ed')+middle(280,100,'Pump')+middle(280,121,`η = ${input.pumpEfficiency}%`)+
      small(40,161,'Pressure rise (0–15 MPa)')+
      box(40,174,195,20,'#e0eaec')+box(40,174,pressureWidth,20,blue)+
      small(40,219,`Δp = ${fmt(input.pressure,1)} MPa`)+
      arrow(280,209,280,145,gold,3)+label(303,195,`${fmt(power,3)} MW`)+small(303,217,'Shaft power required')};
  },
  heatpump(input,{source, delivered}) {
    const width=395, x=130, elec=width*input.heatPumpPower/30, src=width*source/30;
    return {title:'Heat pump energy balance, showing electricity plus ground heat',html:
      label(28,41,'Inputs and delivered heat (0–30 MW)')+
      small(24,91,'Into heat pump')+box(x,72,width,25,'#e4edef')+
      box(x,72,elec,25,gold,0)+box(x+elec,72,src,25,blue,0)+
      small(24,140,'Delivered heat')+box(x,121,width,25,'#e4edef')+box(x,121,width*delivered/30,25,green)+
      dot(31,184,5,gold)+small(44,188,`Electrical input · ${fmt(input.heatPumpPower)} MW`)+
      dot(31,207,5,blue)+small(44,211,`Ground heat · ${fmt(source)} MW`)+
      dot(31,230,5,green)+small(44,234,`Delivered · ${fmt(delivered)} MW · COP ${fmt(input.cop,1)}`)};
  },
  storage(input,{rock,fluid}) {
    const width=120+95*Math.cbrt(input.volume/30), frac=input.porosity/100;
    const x=36,y=88, right=x+width;
    return {title:'Schematic reservoir volume and heat stored in rock and fluid',html:
      `<path d="M${x} ${y}l50 -29h${width}l-50 29Z" fill="#dcc2a3"/><path d="M${right} ${y}l50 -29v99l-50 29Z" fill="#8c7967"/>`+
      box(x,y,width,99,'#ae9074',0)+box(x,y+99*(1-frac),width,99*frac,blue,0)+
      small(34,218,`Volume ${fmt(input.volume,1)} million m³`)+small(34,240,`Fluid fraction ${fmt(input.porosity)}%`)+
      label(327,62,'Heat by material')+
      dot(331,94,5,hot)+small(343,98,`Rock · ${fmt(rock,1)} GWh`)+
      dot(331,126,5,blue)+small(343,130,`Fluid · ${fmt(fluid,1)} GWh`)+
      small(327,170,`Temperature rise ${fmt(input.reservoir-input.surface)} K`)+
      small(327,196,'Relative to reference temperature')};
  },
  decline(input,{power}) {
    const x=y=>61+(y-1)/39*448, y=p=>214-p/20*175;
    const points=Array.from({length:40},(_,i)=>`${x(i+1)},${y(input.initial*(1-input.decline/100)**i)}`).join(' ');
    return {title:'Annual power decline curve with selected operating year',html:
      [0,5,10,15,20].map(p=>line(61,y(p),509,y(p))+small(18,y(p)+4,`${p} MW`)).join('')+
      [1,10,20,30,40].map(n=>middle(x(n),236,n)).join('')+
      `<polyline points="${points}" fill="none" stroke="${green}" stroke-width="3"/>`+
      line(x(input.year),y(power),x(input.year),214,blue,'stroke-dasharray="4 4"')+
      dot(x(input.year),y(power),6,blue)+middle(280,260,'Operating year')};
  },
  energy(input,{operatingHours,capacityFactor}) {
    const x=54,w=460,y0=209,height=166;
    const on=w*input.availability/100, output=height*input.power/20, rating=height*input.rating/20;
    return {title:'Power over the selected period with operating hours and nameplate rating',html:
      [0,5,10,15,20].map(p=>line(x,y0-height*p/20,x+w,y0-height*p/20)+small(5,y0-height*p/20+4,`${p} MW`)).join('')+
      box(x,y0-output,on,output,'#95c8da',0)+
      line(x,y0-rating,x+w,y0-rating,hot,'stroke-dasharray="6 5" stroke-width="2"')+
      small(54,27,`Dashed line · ${fmt(input.rating)} MW nameplate`)+
      middle(x,230,'0 h')+middle(x+w,230,`${fmt(input.hours)} h`)+
      middle(280,257,`${fmt(operatingHours,1)} h operating · ${fmt(capacityFactor*100,1)}% capacity factor`)};
  },
};

export function renderEquationVisual(name, svg, inputs, values, motion) {
  const { title,html,paused } = drawings[name](inputs,values,motion);
  svg.setAttribute('aria-label',title);
  svg.innerHTML = `<defs>${markers}</defs><g font-size="14" fill="${ink}">${html}</g>`;
  if (paused) svg.pauseAnimations();
  else svg.unpauseAnimations();
}
