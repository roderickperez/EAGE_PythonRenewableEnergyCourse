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

function movingDots(path, color, count, duration, endColor = color) {
  return Array.from({length:count}, (_,i) =>
    `<circle r="3.8" fill="${color}"><animateMotion path="${path}" dur="${duration}s" begin="-${i*duration/count}s" repeatCount="indefinite"/>${
      color === endColor ? '' : `<animate attributeName="fill" values="${color};${color};${endColor};${endColor}" keyTimes="0;.26;.73;1" dur="${duration}s" begin="-${i*duration/count}s" repeatCount="indefinite"/>`
    }</circle>`).join('');
}

function rotor(x,y,r,color,duration,moving = true) {
  const blades = Array.from({length:6}, (_,i) =>
    `<path d="M${x} ${y}q${r*.7} 0 ${r*.58} ${-r}l6 3q-2 ${r*.84} ${-r*.58-6} ${r-3}Z" fill="${color}" transform="rotate(${i*60} ${x} ${y})"/>`).join('');
  const animation = moving ? `<animateTransform attributeName="transform" type="rotate" from="0 ${x} ${y}" to="360 ${x} ${y}" dur="${duration}s" repeatCount="indefinite"/>` : '';
  return `<g>${blades}${dot(x,y,4,color)}${animation}</g>`;
}

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
    const duration = 12 / (1 + Math.log1p(flux * 1e6));
    // Aperture here is a visual cue for permeability, independent of the reference pressure drop.
    const poreWidth = 2.5 + 11.5*Math.sqrt((input.permeability-1)/299);
    const radius = Math.min(3.6,poreWidth*.65);
    const inlet = 'M65 45V123';
    const producer = 'V45';
    const particles = flux > 0 ? flowD.map((path,lane) =>
      Array.from({length: 5}, (_,i)=>{
        const phase = (i+lane*.31)/5;
        return `<circle r="${radius}" fill="#155974" stroke="#ecfbff" stroke-width=".8">
          <animateMotion path="${inlet} ${path.replace(/^M65 123 /,'')} ${producer}" dur="${duration}s" begin="-${phase*duration}s" repeatCount="indefinite"/>
        </circle>`;
      }).join('')).join('') : '';
    return {
      title:'Blue inlet, connected pores, and red production outlet',
      html: middle(280,25,'Flow through connected pore paths') +
        label(80,67,'Blue inlet','fill="#247fae"') +
        label(479,67,'Red producer','text-anchor="end" fill="#bd522e"') +
        box(43,95,470,79,'#e9ddcb') +
        flowD.map(d=>`<path d="${d}" fill="none" stroke="#8dcce0" stroke-width="${poreWidth}" stroke-linecap="round"/>`).join('') +
        box(118,103,19,12,'#a6957d') + box(205,142,24,14,'#a6957d') +
        box(329,99,27,11,'#a6957d') + box(403,142,21,16,'#a6957d') +
        line(65,45,65,123,blue,'stroke-width="13" stroke-linecap="round"') +
        line(493,45,493,107,hot,'stroke-width="13" stroke-linecap="round"') +
        particles + (flux > 0 ? arrow(187,200,372,200,blue,3) : '') +
        middle(280,226,flux>0 ? `Darcy flux magnitude · ${flux.toExponential(2)} m/s` : 'Zero pressure gradient · no flow') +
        small(43,254,'Path width shows relative permeability. Dot speed shows relative flux.'),
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
  heat({ flow, reservoir, reinjection }, { power, enthalpy }, motion) {
    return {
      title: 'Cutaway heat exchanger with hot brine entering, cooled brine returning, and heat moving into a separate circuit',
      paused: !motion,
      html: box(123,85,310,108,'#cbdde0',12) +
        '<path d="M123 85l22 -17h287l19 17Z" fill="#e3edef"/><path d="M433 85l18 -11v99l-18 20Z" fill="#8aa7ad"/>' +
        box(130,90,296,94,'#eff6f7',8) +
        '<path d="M401 104H183Q160 104 160 126V143Q160 169 183 169H400" fill="none" stroke="#2a8c83" stroke-width="13"/>' +
        '<path d="M33 138H512" stroke="url(#brine-temperature)" stroke-width="17" stroke-linecap="round"/>' +
        `<defs><linearGradient id="brine-temperature"><stop offset="25%" stop-color="${hot}"/><stop offset="72%" stop-color="${blue}"/></linearGradient></defs>` +
        middle(93,34,'Produced brine') + middle(93,55,`${reservoir} °C`) +
        middle(474,34,'Cooled return') + middle(474,55,`${reinjection} °C`) +
        (flow > 0 ? movingDots('M32 138H512', '#fff2cc', 8, 7/(1+flow/55), blue) : '') +
        (flow > 0 ? [210,251,292,333].map(x => arrow(x,132,x,113,hot,Math.min(5,1+enthalpy/96))).join('') : '') +
        (power > 0 ? movingDots('M396 104H212', '#ffe2a2', 4, 4/(1+power/45)) : '') +
        middle(280,215,'Heat passes through the exchanger wall') +
        middle(280,243,`${fmt(flow)} kg/s × ${fmt(enthalpy,1)} kJ/kg = ${fmt(power,3)} MW thermal`) +
        small(32,126,'Inlet') + label(518,126,'Outlet','font-size="12" text-anchor="end"'),
    };
  },
  carnot({ reservoir, sink }, { efficiency }, motion) {
    const rejected = 1-efficiency;
    return {
      title: 'Ideal heat engine receiving heat from the hot reservoir, delivering work and rejecting heat to the cold reservoir',
      paused: !motion,
      html: box(24,44,150,59,'#f5dfd4') + box(24,169,150,59,'#dcedf5') +
        middle(99,68,'Hot reservoir') + middle(99,89,`${fmt(reservoir+273.15)} K`) +
        middle(99,193,'Cold reservoir') + middle(99,214,`${fmt(sink+273.15)} K`) +
        '<path d="M174 74H254Q278 74 278 96" stroke="#bd522e" stroke-width="13" fill="none"/>' +
        `<path d="M278 178Q278 199 254 199H174" stroke="${blue}" stroke-width="${13*rejected}" fill="none"/>` +
        line(319,137,471,137,green,`stroke-width="${3+15*efficiency}"`) +
        box(238,95,81,83,'#9ebbbd',8) + box(238,95,70,79,'#e4eef0',8) +
        dot(273,135,29,'#365862') + dot(273,135,24,'#c5dee1') +
        rotor(273,135,19,'#648d94',3.4) +
        arrow(458,137,483,137,green,4) +
        movingDots('M174 74H254Q278 74 278 95', '#fff0c7', 3, 2.4) +
        movingDots('M278 179Q278 199 254 199H175', '#e9ffff', 3, 2.4/rejected) +
        movingDots('M320 137H475', '#e5f8dc', 2, 2/efficiency) +
        small(182,58,'Heat in') + small(181,221,'Heat rejected') +
        middle(403,103,'Maximum work') + middle(401,163,`${fmt(100*efficiency,1)}% of input heat`) +
        middle(278,250,'Heat in = work out + heat rejected'),
    };
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
  pump({ pressure, pumpEfficiency }, { volumeFlow, power }, motion) {
    const outletWidth = 10+pressure*.55;
    return {
      title: 'Cutaway centrifugal pump. The motor turns the impeller and the water exits at higher pressure.',
      paused: !motion,
      html: box(138,214,289,12,'#9ab0b3',3) +
        box(155,178,16,37,'#779299',3) + box(296,192,17,23,'#779299',3) +
        box(154,134,157,59,'#557981',16) + box(141,132,69,64,'#789aa1',14) +
        [156,168,180,192].map(x => line(x,139,x,189,'#b6cbd0','stroke-width="3"')).join('') +
        '<path d="M270 143Q270 91 312 91h173v25H346q10 20 8 40a49 49 0 1 1-84-13Z" fill="#81a2a9" stroke="#50777e" stroke-width="4"/>' +
        line(324,102,506,102,blue,`stroke-width="${outletWidth}" stroke-linecap="round"`) +
        dot(306,159,39,'#ccdee1') + dot(306,159,33,'#458994') +
        rotor(306,159,26,'#ddedea',1.9,volumeFlow > 0) +
        line(44,159,268,159,blue,'stroke-width="15" stroke-linecap="round"') +
        dot(306,159,8,'#d5e9ec') +
        (volumeFlow > 0 ? movingDots('M44 159H306Q331 159 331 129V113Q331 102 343 102H506', '#e7fcff', 8, 7/(1+volumeFlow*19)) : '') +
        label(38,122,'Suction inlet') + label(506,74,'Discharge outlet','text-anchor="end"') +
        label(506,136,`+ ${fmt(pressure,1)} MPa`,'text-anchor="end"') +
        small(153,111,`Q = ${fmt(volumeFlow,4)} m³/s`) +
        line(335,175,379,194,muted) + small(385,199,'Impeller') +
        label(174,254,`Motor shaft → ${fmt(power,3)} MW at ${pumpEfficiency}% efficiency`),
    };
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
    return {title:'Schematic reservoir volume with water-filled pores spread through the rock',html:
      `<path d="M${x} ${y}l50 -29h${width}l-50 29Z" fill="#dcc2a3"/><path d="M${right} ${y}l50 -29v99l-50 29Z" fill="#8c7967"/>`+
      box(x,y,width,99,'#ae9074',0)+
      (frac>0 ? Array.from({length:20}, (_,i)=>dot(x+width*(.09+(i%5)*.2),y+15+22*Math.floor(i/5)+3*Math.sin(i),2+12*frac,blue)).join('') : '') +
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


export function renderDarcyPressure(svg, gradient) {
  const x = distance => 49+distance*199;
  const y = pressure => 185-pressure/30*130;
  const ticks = [0,10,20,30].map(pressure =>
    line(x(0),y(pressure),x(1),y(pressure)) +
    label(42,y(pressure)+4,pressure,'text-anchor="end"')).join('');
  svg.setAttribute('aria-label',`Pressure from ${gradient} kPa above the outlet at the inlet to zero at the outlet over one metre`);
  svg.innerHTML = `<g font-size="12" fill="${ink}">
    ${label(49,25,'Pressure above outlet (kPa)','font-weight="650"')}
    ${ticks}
    ${line(x(0),55,x(0),185,muted)}
    ${line(x(0),185,x(1),185,muted)}
    ${line(x(0),y(gradient),x(1),y(0),blue,'stroke-width="3"')}
    ${dot(x(0),y(gradient),5,blue)}${dot(x(1),y(0),5,hot)}
    ${label(x(0),207,'Inlet','text-anchor="middle" fill="#247fae"')}
    ${label(x(1),207,'Outlet','text-anchor="middle" fill="#bd522e"')}
    ${middle(149,229,'Distance through rock (0–1 m)')}
  </g>`;
}



const equipmentPlots = {
  heat({ reservoir, reinjection }) {
    return {
      title: `Brine enters at ${reservoir} degrees Celsius and leaves at ${reinjection} degrees Celsius`,
      html: middle(142,27,'Brine temperature (°C)') +
        compactBars([['Production inlet',reservoir,hot],['Cooled return',reinjection,blue]], 240, 60) +
        middle(142,219,`Temperature drop ΔT = ${reservoir-reinjection} K`),
    };
  },
  carnot(input, { efficiency }) {
    const work = efficiency*100, rejected = 100-work;
    return {
      title: `Of 100 units of heat input, ${fmt(work,1)} become work and ${fmt(rejected,1)} are rejected`,
      html: middle(142,27,'Share of input heat (%)') +
        compactBars([['Maximum work',Number(work.toFixed(1)),green],['Heat rejected',Number(rejected.toFixed(1)),blue]], 100, 25) +
        middle(142,219,`${fmt(work,1)} + ${fmt(rejected,1)} = 100%`),
    };
  },
  storage({ volume, porosity }, { rock, fluid }) {
    const step = Math.max(10, 10**Math.floor(Math.log10(Math.max(rock,fluid)))/2);
    const max = Math.ceil(Math.max(rock,fluid)/(step*4))*step*4;
    return {
      title: `Rock stores ${fmt(rock,1)} gigawatt-hours and pore fluid stores ${fmt(fluid,1)} gigawatt-hours at ${porosity}% porosity`,
      html: middle(142,27,'Heat stored by material (GWh)') +
        compactBars([['Rock matrix',rock,hot],['Pore water',fluid,blue]], max, max/4) +
        middle(142,218,`${fmt(volume*(1-porosity/100),2)} rock + ${fmt(volume*porosity/100,2)} water`) +
        middle(142,235,'Volumes in million m³'),
    };
  },
  pump({ pressure }, { power, volumeFlow }) {
    const water = pressure*volumeFlow, loss = Math.max(0,power-water);
    const magnitude = power > 0 ? 10**Math.floor(Math.log10(power/4)) : .02;
    const step = [1,2,5,10].map(n=>n*magnitude).find(n=>n>=power/4);
    const max = step*4;
    return {
      title: `Shaft input ${fmt(power,3)} megawatts, hydraulic output ${fmt(water,3)} megawatts, and ${fmt(loss,3)} megawatts of losses`,
      html: middle(142,27,'Pump power (MW)') +
        compactBars([['Shaft input',power,gold],['Into the water',water,blue]], max, max/4) +
        middle(142,219,`Pump losses = ${fmt(loss,3)} MW`),
    };
  },
};

function compactBars(rows, max, step) {
  const left = 30, width = 225, axisY = 162;
  const ticks = Array.from({length:5}, (_,i) => {
    const x = left+width*i/4;
    return line(x,63,x,axisY) + middle(x,182,fmt(i*step,4));
  });
  const bars = rows.map(([name,value,color],i) => {
    const y = 48+i*55;
    return label(left,y,name) +
      label(left+width,y,fmt(value,3),'text-anchor="end" font-weight="700"') +
      box(left,y+8,width*value/max,17,color,3);
  });
  return ticks.join('') + bars.join('') + line(left,axisY,left+width,axisY,muted);
}

export function renderEquipmentPlot(name, svg, input, values) {
  const { title,html } = equipmentPlots[name](input,values);
  svg.setAttribute('aria-label',title);
  svg.innerHTML = `<g fill="${ink}" font-size="12">${html}</g>`;
}
