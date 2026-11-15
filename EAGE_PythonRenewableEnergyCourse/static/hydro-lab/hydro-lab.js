var q=`:host {
  --ink: #153b36;
  --muted: #5f716e;
  --line: #cbdad5;
  --surface: #ffffff;
  --paper: #f4f8f6;
  --green: #0f826c;
  --blue: #237fb6;
  --cyan: #48b7d5;
  --amber: #d78b23;
  --danger: #ad4d35;
  color: var(--ink);
  display: block;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow: hidden;
  font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
button, input { font: inherit; }
.hydro-lab { width: 100%; max-width: 100%; min-width: 0; border: 1px solid var(--line); border-radius: 18px; background: var(--paper); overflow: hidden; box-shadow: 0 16px 42px rgb(27 64 55 / 10%); }
.lab-head { padding: 22px 24px 14px; background: linear-gradient(135deg, #e8f6f1, #edf7fb); border-bottom: 1px solid var(--line); }
.lab-kicker { color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
h2 { margin: 4px 0 5px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(24px, 4vw, 36px); font-weight: 500; line-height: 1.15; overflow-wrap: anywhere; }
.lab-head p { max-width: 850px; margin: 0; color: var(--muted); }
.controls { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; padding: 18px 22px; background: var(--surface); border-bottom: 1px solid var(--line); }
.control { min-width: 0; }
.control-line { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.control label { font-weight: 700; color: var(--ink); }
.control output { font: 700 13px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: nowrap; }
input[type="range"] { width: 100%; accent-color: var(--blue); cursor: pointer; }
input[type="range"]:focus-visible { outline: 3px solid #65aee0; outline-offset: 3px; border-radius: 8px; }
.stage-grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(310px, .7fr); gap: 16px; padding: 18px; }
.stage-grid > *, .charts > *, .lab-head, .controls, .lab-foot { min-width: 0; max-width: 100%; }
.scene-card, .metrics-card, .chart-card { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); overflow: hidden; }
.scene-card { min-height: 590px; position: relative; background: linear-gradient(180deg, #e8f7fb 0 58%, #d9eddd 58% 100%); }
.scene { position: absolute; inset: 0; overflow: hidden; }
.scene canvas { width: 100%; height: 100%; display: block; touch-action: none; }
.scene-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.scene-label { position: absolute; left: 0; top: 0; padding: 4px 8px; border: 1px solid rgb(22 65 57 / 22%); border-radius: 6px; background: rgb(255 255 255 / 92%); color: var(--ink); box-shadow: 0 2px 8px rgb(20 50 45 / 14%); font-size: 10px; font-weight: 800; white-space: nowrap; }
.scene-label::after { content: ""; position: absolute; left: 50%; top: 100%; width: 1px; height: 10px; background: rgb(21 59 54 / 45%); }
.scene-label.flow-label { color: #075e83; border-color: rgb(35 127 182 / 35%); }
.scene-label.water-label { color: #08769d; background: rgb(235 251 255 / 94%); }
.fallback { width: 100%; height: 100%; object-fit: contain; }
.scene-help { position: absolute; right: 12px; bottom: 10px; z-index: 2; margin: 0; padding: 5px 9px; border-radius: 8px; background: rgb(255 255 255 / 82%); color: var(--muted); font-size: 11px; pointer-events: none; }
.head-ruler { position: absolute; left: 12px; bottom: 10px; z-index: 2; width: 132px; padding: 9px; border-radius: 10px; background: rgb(255 255 255 / 90%); border: 1px solid var(--line); font-size: 11px; }
.head-ruler strong { display: block; font-size: 12px; }
.head-track { height: 75px; width: 10px; margin: 7px 8px 6px 3px; float: left; border-radius: 8px; background: linear-gradient(to top, var(--amber) 0 var(--loss-pct), var(--blue) var(--loss-pct) 100%); }
.head-ruler span { display: block; color: var(--muted); }
.metrics-card { padding: 16px; }
.power-card { padding: 14px; border-radius: 12px; background: #eaf5fb; border: 1px solid #c5e1ef; }
.power-card .eyebrow { color: var(--muted); font-size: 12px; }
.power-card strong { display: block; color: var(--blue); font: 800 clamp(28px, 5vw, 42px)/1.1 ui-monospace, SFMono-Regular, Consolas, monospace; }
.status { display: inline-block; margin-top: 7px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; }
.status.clipped { color: #7b331f; background: #ffe3d8; }
.status.open { color: #14684f; background: #d9f3e8; }
.metric-list { display: grid; gap: 0; margin: 13px 0; }
.metric { display: flex; justify-content: space-between; gap: 12px; padding: 9px 2px; border-bottom: 1px solid #e2ebe7; }
.metric span { color: var(--muted); }
.metric strong { text-align: right; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.equations { margin-top: 15px; }
.equations h3 { margin: 0 0 8px; font-size: 15px; }
.equation { margin: 6px 0; padding: 8px 10px; border-left: 3px solid var(--green); background: #f5faf8; font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
.assumptions { margin: 13px 0 0; color: var(--muted); font-size: 12px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 18px 18px; }
.chart-card { padding: 13px; }
.chart-card h3 { margin: 0 0 2px; font-size: 15px; }
.chart-card p { margin: 0 0 4px; color: var(--muted); font-size: 12px; }
.chart-card svg { display: block; width: 100%; height: auto; min-height: 230px; overflow: visible; }
.grid { stroke: #e1e9e6; stroke-width: 1; }
.axis { stroke: #80938e; stroke-width: 1.3; }
.tick, .axis-label, .legend, .bar-label, .bar-value { fill: var(--muted); font-size: 12px; }
.axis-label, .legend, .bar-value { font-weight: 700; }
.curve { fill: none; stroke-width: 3; }
.curve.uncapped { stroke: var(--blue); }
.curve.cap { stroke: var(--danger); stroke-dasharray: 7 5; }
.live-point { fill: var(--amber); stroke: #fff; stroke-width: 3; }
.live-label { fill: var(--ink); font-size: 12px; font-weight: 800; }
.cap-text { fill: var(--danger); }
.bar-track { fill: #e5ece9; }
.bar.hydraulic { fill: var(--cyan); }
.bar.uncapped { fill: var(--green); }
.bar.plant { fill: var(--amber); }
.energy-band { fill: #e7f3ed; stroke: #c3ddd1; }
.energy-label { fill: var(--ink); font-size: 14px; font-weight: 800; }
.lab-foot { padding: 12px 20px 16px; color: var(--muted); font-size: 12px; text-align: center; }
.lab-foot a { color: var(--blue); font-weight: 700; }
@media (max-width: 900px) {
  .controls { grid-template-columns: 1fr 1fr; }
  .stage-grid { grid-template-columns: 1fr; }
  .scene-card { min-height: 560px; }
  .charts { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .lab-head { padding: 18px 16px 12px; }
  .controls { display: block; padding: 14px 16px; }
  .control + .control { margin-top: 16px; }
  .stage-grid, .charts { display: block; width: 100%; padding: 12px; }
  .scene-card, .metrics-card, .chart-card { width: 100%; max-width: 100%; min-width: 0; }
  .metrics-card, .chart-card { margin-top: 12px; }
  .charts { padding-top: 0; }
  .scene-card { min-height: 500px; }
  .scene-label { font-size: 9px; padding: 3px 6px; }
  .scene-label::after { height: 7px; }
  .head-ruler { transform: scale(.88); transform-origin: bottom left; }
  .scene-help { max-width: 150px; text-align: right; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation: none !important; transition: none !important; }
}
@media print {
  .controls, .scene-help { display: none; }
  .hydro-lab { box-shadow: none; }
}

.cutaway-card {
  container-type: inline-size;
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
  overflow: hidden;
}
.plant-cutaway { display: block; width: 100%; height: auto; }
.component-target { outline: none; cursor: help; }
.component-hit { fill: transparent; stroke: transparent; stroke-width: 2; pointer-events: all; }
.component-tooltip { opacity: 0; pointer-events: none; transition: opacity .12s; }
.component-tooltip rect { fill: #173745; stroke: #e1edf1; stroke-width: 1; }
.component-tooltip text { fill: white; font: 500 18px Inter, sans-serif; }
.component-target:hover .component-tooltip,
.component-target:focus-visible .component-tooltip,
.component-target[aria-pressed="true"] .component-tooltip { opacity: 1; }
.component-target:hover .component-hit,
.component-target:focus-visible .component-hit { fill: #bce6f4; fill-opacity: .12; stroke: #63a8bb; }
.cutaway-card figcaption { padding: 12px 16px 0; color: var(--muted); font-size: 13px; }
.cutaway-card .head-ruler {
  position: static;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 15px;
  align-items: baseline;
  width: auto;
  margin: 10px 16px 14px;
  padding: 9px 11px;
  transform: none;
  background: #f2f8f7;
}
.cutaway-card .head-track { width: 25px; height: 5px; margin: 0; align-self: center; }
.cutaway-label { fill: #18343b; }
.cutaway-label path { fill: none; stroke: #38535a; stroke-width: 1.6; }
.cutaway-label text {
  paint-order: stroke;
  stroke: #f3f6ee;
  stroke-width: 4px;
  stroke-linejoin: round;
  font-size: 19px;
  font-weight: 750;
}
.cutaway-marker, .cutaway-key { display: none; }
.cutaway-marker circle { fill: #f7fbf9; stroke: #164655; stroke-width: 2.5; }
.cutaway-marker text {
  fill: #163640;
  font-size: 19px;
  font-weight: 800;
  text-anchor: middle;
  dominant-baseline: central;
}
.cutaway-key {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
  margin: 0;
  padding: 12px;
  list-style: none;
  border-bottom: 1px solid var(--line);
  font-size: 12px;
}
.cutaway-key li { display: flex; gap: 7px; align-items: baseline; }
.cutaway-key span { min-width: 19px; text-align: center; color: #154859; font-weight: 800; }
.grid-lines { color: var(--grid-active, #354e49); }
.water-motion {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 11 35;
  animation: water-travel var(--turbine-speed, 3s) linear infinite;
}
#cutaway-spill-motion { animation-duration: var(--spill-speed, 2s); }
@keyframes water-travel { to { stroke-dashoffset: -46; } }
.orbit-view { margin: 0 0 8px; }
.orbit-view summary { cursor: pointer; color: #20556c; font-size: 13px; font-weight: 700; padding: 6px 10px; }
.orbit-view summary:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.orbit-view .show-diagram, .orbit-view[open] .show-3d, .orbit-view[open] + .cutaway-card { display: none; }
.orbit-view[open] .show-diagram { display: inline; }
.orbit-view .scene-card { margin-top: 10px; }
@container (max-width: 610px) {
  .cutaway-label { display: none; }
  .cutaway-marker { display: inline; }
  .cutaway-key { display: grid; }
}
`;var x=Object.freeze({efficiency:.85,densityKgM3:1e3,gravityMS2:9.81,durationHours:6}),g=Object.freeze({grossHeadM:193,usableFlowM3s:10,headLossM:2,ratingMW:2}),y=Object.freeze({grossHeadM:[5,250],usableFlowM3s:[0,30],headLossM:[0,50],ratingMW:[.1,25]});function v(t,e){let a=Number(t);return Number.isFinite(a)?a:e}function M(t,e,a){return Math.min(a,Math.max(e,t))}function k(t={}){let e=M(v(t.grossHeadM,g.grossHeadM),...y.grossHeadM),a=M(v(t.usableFlowM3s,g.usableFlowM3s),...y.usableFlowM3s),i=Math.min(y.headLossM[1],Math.max(0,e-.5)),s=M(v(t.headLossM,g.headLossM),y.headLossM[0],i),p=M(v(t.ratingMW,g.ratingMW),...y.ratingMW);return{grossHeadM:e,usableFlowM3s:a,headLossM:s,ratingMW:p}}function $(t=g){let e=k(t),{efficiency:a,densityKgM3:i,gravityMS2:s,durationHours:p}=x,o=Math.max(0,e.grossHeadM-e.headLossM),d=i*s*e.usableFlowM3s*o/1e6,r=a*d,n=Math.min(r,e.ratingMW),f=o>0?e.ratingMW*1e6/(a*i*s*o):0,m=Math.min(e.usableFlowM3s,f),Z=Math.max(0,e.usableFlowM3s-m);return{inputs:e,netHeadM:o,hydraulicMW:d,uncappedMW:r,plantMW:n,turbineFlowM3s:m,spillFlowM3s:Z,energy6hMWh:n*p,clipped:r>e.ratingMW+1e-9}}var F=`<svg class="plant-cutaway" viewBox="0 0 1000 610" role="img" aria-label="Hydroelectric dam cutaway. Water enters from the reservoir on the left, passes through the penstock and turbine, and returns to the river through the draft tube on the right. The turbine drives a generator above it, which sends electricity through a transformer to the transmission towers. A separate spillway bypasses the turbine.">
  <defs>
    <linearGradient id="sky" x2="0" y2="1"><stop stop-color="#f2f4f5"/><stop offset="1" stop-color="#fafbfa"/></linearGradient>
    <linearGradient id="woodland" x1="0" y1="1" x2="1" y2="0"><stop stop-color="#546e65"/><stop offset="1" stop-color="#a1b3a6"/></linearGradient>
    <linearGradient id="concrete" x1="0" y1="0" x2=".8" y2="1"><stop stop-color="#f3f3eb"/><stop offset=".55" stop-color="#d4d8d5"/><stop offset="1" stop-color="#a6aeaf"/></linearGradient>
    <linearGradient id="concrete-side" x2="1" y2="1"><stop stop-color="#9ca6a8"/><stop offset="1" stop-color="#5e6c73"/></linearGradient>
    <linearGradient id="section-fill" x2="0" y2="1"><stop stop-color="#eae9e0"/><stop offset="1" stop-color="#c6c8c1"/></linearGradient>
    <linearGradient id="lake" x2="0" y2="1"><stop stop-color="#76aabc"/><stop offset=".67" stop-color="#497e97"/><stop offset="1" stop-color="#345b72"/></linearGradient>
    <linearGradient id="surface" x2="0" y2="1"><stop stop-color="#d9e9ec"/><stop offset="1" stop-color="#8bb8c7"/></linearGradient>
    <linearGradient id="pipe-water" x2="0" y2="1"><stop stop-color="#b9dce5"/><stop offset=".48" stop-color="#78b6ce"/><stop offset="1" stop-color="#416f8a"/></linearGradient>
    <linearGradient id="steel" x2="1" y2="0"><stop stop-color="#53646a"/><stop offset=".4" stop-color="#d9e1de"/><stop offset=".65" stop-color="#99a7a6"/><stop offset="1" stop-color="#55666b"/></linearGradient>
    <linearGradient id="copper" x2="1" y2="1"><stop stop-color="#d4a87e"/><stop offset=".5" stop-color="#aa7758"/><stop offset="1" stop-color="#654a40"/></linearGradient>
    <linearGradient id="runner-case" x2="0" y2="1"><stop stop-color="#8ba69e"/><stop offset="1" stop-color="#3a6060"/></linearGradient>
    <linearGradient id="pipe-wall" x2="0" y2="1"><stop stop-color="#8c9c9e"/><stop offset=".48" stop-color="#5b6e76"/><stop offset="1" stop-color="#344b57"/></linearGradient>
    <linearGradient id="gallery" x2="0" y2="1"><stop stop-color="#68777d"/><stop offset=".55" stop-color="#9ca8aa"/><stop offset="1" stop-color="#c6cfca"/></linearGradient>
    <radialGradient id="contact-shadow"><stop stop-color="#213b49" stop-opacity=".58"/><stop offset="1" stop-color="#213b49" stop-opacity="0"/></radialGradient>
    <pattern id="aggregate" width="21" height="14" patternUnits="userSpaceOnUse"><path d="M2 4h2m10 7h3M7 13h1" stroke="#566d64" opacity=".18"/></pattern>
    <clipPath id="upstream-bank"><path d="M0 46H295L222 307 0 427Z"/></clipPath>
    <g id="tower" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M-23 122-7 47-26 18H26L7 47 23 122M-45 18 0 3 45 18H-45M-44 18v11m88-11v11M-21 106h42m-37-19h32m-29-18h26M-20 103 16 87-12 69 7 47M20 103-16 87 12 69-7 47M-25 18 7 47m18-29L-7 47"/>
    </g>
  </defs>
  <path fill="url(#sky)" d="M0 0h1000v610H0z"/>
  <path fill="#c2ceca" d="M330 112 444 58 514 67 596 31 661 52 760 29 860 74 1000 50v325L500 273Z"/>
  <path fill="url(#woodland)" opacity=".45" d="M411 124Q500 25 571 21T686 76q56 51 113 57t201 45v233L618 325Z"/>
  <path fill="none" stroke="#dce7e1" stroke-opacity=".28" stroke-width="1.4" d="M576 104q103 14 156 85t268 66M537 131q110 20 171 79t292 75"/>
  <path fill="url(#lake)" d="M585 312q204-13 415 45v253H675Z"/>
  <path fill="none" stroke="#bee9ec" stroke-width="3" opacity=".5" d="M807 425q61-13 176 16m-136 17q57-2 145 13m-171 57q67-13 163 7m-186 38q99-13 188-2"/>
  <g clip-path="url(#upstream-bank)">
    <g id="cutaway-lake">
      <path fill="url(#lake)" d="M0 0 242 39v420H0Z"/>
      <path fill="url(#surface)" d="M0 0 171-30 273 8 242 39Z"/>
      <path fill="none" stroke="#e4f9f9" stroke-width="3" opacity=".66" d="M0 0 242 39M20 18l48 7m26 4 52 9M30 45l56 8m8 22 34 4"/>
    </g>
  </g>
  <!-- The spillway runs behind the powerhouse on the far side of the dam. -->
  <path fill="#748988" d="M404 87 475 72 776 340 897 386 856 407 731 364Z"/>
  <path fill="url(#concrete)" stroke="#e1e5dd" stroke-width="6" d="M414 89 468 81 766 346 880 386 856 398 743 360Z"/>
  <path id="cutaway-spill" fill="#82b4c9" d="M422 91 458 85 754 346 869 387 854 393 746 357Z"/>
  <path fill="none" stroke="#637a84" stroke-width="3" d="m460 86 298 261 111 42"/>
  <path id="cutaway-spill-motion" class="water-motion" d="M439 92 751 353 864 389" stroke="#f0fcff" stroke-width="5"/>
  <!-- Projecting concrete faces and the exposed section through the foundation. -->
  <path fill="url(#concrete)" stroke="#a8b0aa" d="M170 120 273 42 349 22 439 61 361 104 242 153Z"/>
  <path fill="url(#concrete-side)" d="M361 104 439 61v61l-79 32Z"/>
  <path fill="url(#concrete)" stroke="#aeb5af" d="M226 167 360 153 444 119 498 143 408 183 277 212Z"/>
  <path fill="#a4adab" d="M407 181 498 142v34l-80 41Z"/>
  <path fill="url(#concrete)" stroke="#a5adaa" d="m415 216 83-42 117 163-103 43Z"/>
  <path fill="url(#section-fill)" stroke="#8b9994" stroke-width="2" d="M0 345 66 294 119 311 169 120 361 104 360 153 407 181 417 216 512 378 636 403 764 437v173H0Z"/>
  <path fill="url(#aggregate)" opacity=".62" d="M0 346 66 294 119 312 169 120 361 104 360 153 407 181 417 216 512 378 636 403 764 437v173H0Z"/>
  <path fill="none" stroke="#f9f8f1" stroke-width="4" d="m171 122 188-16m3 49 45 28 10 35 92 160"/>
  <g fill="none" stroke="#83908e" stroke-width="1.5" opacity=".64">
    <path d="m212 86 89-21 91-26m-139 91 89-39 64-34m-136 112 96-22 69-26m-75 68 108-48"/>
    <path d="m453 197 113 160m-90-173 114 160m-153-99 84-46m-55 92 83-48m-54 95 84-46"/>
    <path d="M17 376h131m-118 86h284M0 541h531"/>
  </g>
  <!-- Intake and open penstock section. -->
  <path fill="none" stroke="#5c6766" stroke-opacity=".26" stroke-width="58" stroke-linejoin="round" d="M139 244h88l144 190 140 32"/>
  <path fill="none" stroke="url(#pipe-wall)" stroke-width="50" stroke-linejoin="round" d="M139 238h88l144 190 140 32"/>
  <path fill="none" stroke="url(#pipe-water)" stroke-width="38" stroke-linejoin="round" d="M135 238h92l144 190 140 32"/>
  <path fill="none" stroke="#d9f8fa" stroke-width="2" opacity=".75" d="M152 221h80l148 191 131 31"/>
  <path fill="#47636b" d="m126 204 11-5-9 67-14 11Zm19-8 7-3-8 66-7 6Z"/>
  <path stroke="#263d46" stroke-width="4" d="m123 211 23-9m-25 24 23-9m-25 24 23-9m-25 24 23-9m-25 24 22-10"/>
  <path fill="url(#steel)" stroke="#758285" d="M153 162h15v99l-15 5Z"/>
  <path fill="url(#steel)" d="M157 135h6v38h-6z"/>
  <path fill="#334d58" d="M150 137h19v8h-19z"/>
  <!-- Below the generator floor, the runner empties into a widened draft tube. -->
  <path fill="url(#concrete-side)" stroke="#839295" d="M508 402 647 429 764 437v173H538l-30-31Z"/>
  <path fill="url(#section-fill)" stroke="#adb4b0" d="M508 402 545 413v112q0 73 102 74h117v11H538l-30-31Z"/>
  <path fill="#526c73" d="M547 480h78v30q0 27 35 27h103l47-8v60l-46 10H645q-98 0-98-76Z"/>
  <path fill="url(#pipe-water)" d="M555 483h59v30q0 34 43 34h106l53-8v44l-52 10H646q-91 0-91-69Z"/>
  <path fill="none" stroke="#d9f8fa" opacity=".7" stroke-width="2" d="M618 521q7 27 40 25h107l50-9"/>
  <!-- Powerhouse shell, open front, generator floor and transformer terrace. -->
  <path fill="url(#gallery)" stroke="#819093" d="m497 344 99-86 229 77 1 88-127 63-211-74Z"/>
  <path fill="#52646e" opacity=".47" d="m596 258 229 77-17 10-212-71-85 74-14-4Z"/>
  <path fill="none" stroke="#56666d" opacity=".5" d="m549 300 210 74m-185-95 215 73m-152-81v49m48-33v47m48-31v46"/>
  <path fill="url(#concrete-side)" d="m708 403 119-68 41 26-117 70Z"/>
  <path fill="url(#concrete)" stroke="#a3afaa" d="m649 366 61-61 151 55-113 65-101-29Z"/>
  <path fill="#b1b9b5" d="m748 425 113-65v64l-113 52Z"/>
  <path fill="#e8e9e2" d="m646 396 102 29v52l-102-29Z"/>
  <path fill="url(#concrete-side)" opacity=".6" d="m646 396 102 29v11l-102-29Z"/>
  <path fill="none" stroke="#879495" stroke-width="1.5" d="m749 448 112-61m-79 18v52m40-76v58m-175-15 101 28"/>
  <path fill="url(#concrete)" stroke="#a2aaa3" d="m469 342 75-51 100 32-84 55Z"/>
  <path fill="#aeb8b4" d="m469 342 91 36v24l-91-34Z"/>
  <path fill="url(#concrete)" d="m478 404 75-45 138 42-44 28-167-10Z"/>
  <path fill="#aab4b0" d="m480 419 167 10v17l-164-13Z"/>
  <ellipse cx="585" cy="417" rx="68" ry="16" fill="url(#contact-shadow)"/>
  <g fill="url(#steel)" stroke="#687c7e">
    <rect x="490" y="302" width="39" height="35" rx="4"/>
    <path d="M496 305v30m6-31v32m6-32v32m6-32v32m6-32v31"/>
    <path d="M499 302v-19m17 19v-19" stroke-width="3"/>
    <path d="M494 288h10m-10-5h10m7 5h10m-10-5h10" stroke-width="2"/>
  </g>
  <!-- Electrical route between transformer and transmission. -->
  <g class="grid-lines" color="#385c50" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M500 283Q621 261 742 150L786 120Q859 162 896 135t104-15M516 283q143-49 247-109l112-33q66 20 125-29"/>
    <use href="#tower" transform="translate(832 99)"/>
    <use href="#tower" transform="translate(928 114) scale(.74)"/>
  </g>
  <!-- Vertical shaft, copper generator and Francis-style runner. -->
  <ellipse cx="586" cy="503" rx="83" ry="20" fill="url(#contact-shadow)"/>
  <path fill="url(#steel)" stroke="#657679" d="M576 381h20v94h-20z"/>
  <path fill="url(#steel)" stroke="#596a70" d="M571 417h30v8q-15 10-30 0Z"/>
  <ellipse cx="586" cy="417" rx="15" ry="5" fill="#aebcba" stroke="#657679"/>
  <path fill="#3e5459" opacity=".33" d="M577 399h20v10h-20Z"/>
  <path fill="url(#runner-case)" stroke="#334e52" d="M514 472v8c0 16 32 29 72 29s72-13 72-29v-8Z"/>
  <ellipse cx="586" cy="472" rx="72" ry="29" fill="url(#runner-case)" stroke="#244d43" stroke-width="2"/>
  <path fill="url(#runner-case)" stroke="#326655" d="m459 429 82 18 13 28-100-24Z"/>
  <ellipse cx="586" cy="476" rx="52" ry="20" fill="#79a9b2" stroke="#c6d7d2" stroke-width="3"/>
  <g fill="#d1dbd6" stroke="#3e585c" stroke-width="1">
    <circle cx="524" cy="472" r="2.8"/><circle cx="540" cy="455" r="2.8"/>
    <circle cx="632" cy="455" r="2.8"/><circle cx="648" cy="472" r="2.8"/>
    <circle cx="629" cy="492" r="2.8"/><circle cx="543" cy="492" r="2.8"/>
  </g>
  <g id="cutaway-runner" fill="url(#steel)" stroke="#4e6564" stroke-width="1.8">
    <ellipse cx="586" cy="474" rx="38" ry="15"/>
    <path d="m551 478 68-11m-53-6 35 26m-49-16 68 6m-36-18 2 29" fill="none"/>
    <ellipse cx="586" cy="472" rx="10" ry="6"/>
  </g>
  <path fill="url(#copper)" stroke="#805541" stroke-width="2" d="M529 385v-24c0-25 29-37 57-37s57 12 57 37v24c-9-21-34-29-57-29s-48 8-57 29Z"/>
  <path fill="none" stroke="#e7c29e" stroke-opacity=".64" stroke-width="1.6" d="M534 355c18-38 82-38 104 0m-104 8c18-38 82-38 104 0m-104 8c18-38 82-38 104 0"/>
  <ellipse cx="586" cy="384" rx="57" ry="23" fill="url(#copper)" stroke="#754b37" stroke-width="2"/>
  <path fill="url(#steel)" stroke="#4f626b" d="M529 360v24c0 13 25 24 57 24s57-11 57-24v-24c0 13-25 24-57 24s-57-11-57-24Z"/>
  <g fill="none" stroke="#3e555f" stroke-width="2" opacity=".8">
    <path d="M535 372v18m8-12v17m9-13v18m10-15v18m11-16v18m12-17v18m12-18v17m12-19v17m11-20v17m10-22v17m8-24v17"/>
  </g>
  <ellipse cx="586" cy="380" rx="40" ry="15" fill="url(#steel)" stroke="#53676b" stroke-width="2"/>
  <ellipse cx="586" cy="377" rx="11" ry="6" fill="#e9eeeb"/>
  <path fill="url(#steel)" stroke="#657679" d="M579 351h14v25q-7 5-14 0Z"/>
  <ellipse cx="586" cy="351" rx="7" ry="3" fill="#e8eae5"/>
  <path class="grid-lines" d="M534 371q-25-11-25-36" fill="none" stroke="currentColor" stroke-width="3"/>
  <path id="cutaway-generation-motion" class="water-motion" stroke="#e8fbff" stroke-width="5" d="M170 238h57l144 190 91 21m123 56q5 65 76 65h145"/>
  <path id="cutaway-direction" fill="#176b88" d="M183 235h12v-5l10 8-10 8v-5h-12Zm76 42 8 10 4-3v13l-12-4 4-3-8-10Zm52 67 8 10 4-3v13l-12-4 4-3-8-10Zm82 86 12 3 1-5 9 10-12 5 1-5-12-3Zm219 117 9 7 3-4 3 12-13 1 3-4-9-7Zm59 20h12v-5l10 8-10 8v-5h-12Zm72 0h12v-5l10 8-10 8v-5h-12Z"/>
</svg>
`;var _=[["Reservoir",61,151,20,79],["Trash rack",128,219,20,182],["Control gate",161,163,193,158],["Intake",142,240,22,257],["Penstock",277,302,168,365],["Dam",383,182,301,134],["Spillway / bypass",565,192,523,105],["Powerhouse",741,392,730,305],["Turbine",638,473,683,503],["Generator",544,382,446,386],["Draft tube",563,544,439,554],["Tailrace / outflow",838,553,810,510],["Transformer",502,314,371,279],["Transmission",827,118,762,68]],j=["M0 110H118V290H0Z","M110 200h37v81h-37Z","M151 132h22v137h-22Z","M173 215h58v48h-58Z","M228 247 391 410 495 438v37l-134-29L209 267Z","M175 120 361 104 418 217 506 377 398 398 260 204Z","M415 78 472 76 879 386 854 408 737 368Z","M650 310 864 358v70l-116 51-100-37Z","M516 451H660V505H516Z","M524 322H646V409H524Z","M545 509h80v26h137v62H645l-100-34Z","M817 434H1000V610H817Z","M483 278h51v66h-51Z","M778 90H960V243H778Z"];function C(){let t=_.map(([e,a,i],s)=>{let p=e.length*10+24,o=Math.max(8,Math.min(992-p,a-p/2)),d=Math.max(8,i-44);return`<g class="component-target" tabindex="0" role="button" aria-label="${e}" aria-pressed="false">
      <path class="component-hit" d="${j[s]}"/>
      <g class="component-tooltip" aria-hidden="true" transform="translate(${o} ${d})">
        <rect width="${p}" height="31" rx="5"/>
        <text x="12" y="21">${e}</text>
      </g>
    </g>`}).join("");return F.replace('role="img"','role="group"').replace("</svg>",`${t}</svg>`)}function z(t){let e=t.querySelector(".plant-cutaway"),a=[...e.querySelectorAll(".component-target")];function i(n){let f=n?.getAttribute("aria-pressed")!=="true";a.forEach(m=>m.setAttribute("aria-pressed",String(m===n&&f)))}e.addEventListener("click",n=>i(n.target.closest(".component-target"))),e.addEventListener("keydown",n=>{n.key==="Escape"&&i(null),(n.key==="Enter"||n.key===" ")&&(n.preventDefault(),i(n.target.closest(".component-target")))});let s=e.querySelector("#cutaway-lake"),p=e.querySelector("#cutaway-spill"),o=e.querySelector("#cutaway-spill-motion"),d=e.querySelector("#cutaway-generation-motion"),r=e.querySelector("#cutaway-direction");return n=>{let f=(n.inputs.grossHeadM-5)/245;s.setAttribute("transform",`translate(0 ${157-f*81})`),p.style.opacity=n.spillFlowM3s>0?String(Math.min(1,.35+n.spillFlowM3s/15)):"0",o.style.opacity=n.spillFlowM3s>0?"1":"0",d.style.opacity=n.turbineFlowM3s>0?"1":"0",r.style.opacity=d.style.opacity,e.style.setProperty("--turbine-speed",`${4/(1+Math.sqrt(n.turbineFlowM3s))}s`),e.style.setProperty("--spill-speed",`${3/(1+Math.sqrt(n.spillFlowM3s))}s`),e.style.setProperty("--grid-active",n.plantMW>0?"#f5dc8c":"#354e49")}}var N="http://www.w3.org/2000/svg";function h(t,e={},a=""){let i=document.createElementNS(N,t);for(let[s,p]of Object.entries(e))i.setAttribute(s,String(p));return a&&(i.textContent=a),i}function S(t){t.replaceChildren(),t.setAttribute("viewBox","0 0 600 270")}function U(t,e,a,i,s){t.append(h("line",{x1:e,y1:i,x2:a,y2:i,class:"axis"})),t.append(h("text",{x:a,y:i+27,class:"axis-label","text-anchor":"end"},s))}function L(t,e){S(t);let a={left:58,right:575,top:22,bottom:224},i=30,s=e.netHeadM*.85*1e3*9.81/1e6,p=Math.max(1,e.inputs.ratingMW*1.18,s*i*1.05),o=r=>a.left+r/i*(a.right-a.left),d=r=>a.bottom-r/p*(a.bottom-a.top);for(let r=0;r<=4;r+=1){let n=p*r/4,f=d(n);t.append(h("line",{x1:a.left,y1:f,x2:a.right,y2:f,class:"grid"})),t.append(h("text",{x:a.left-9,y:f+4,class:"tick","text-anchor":"end"},n.toFixed(n<10?1:0)))}for(let r=0;r<=6;r+=1){let n=i*r/6;t.append(h("text",{x:o(n),y:a.bottom+19,class:"tick","text-anchor":"middle"},n.toFixed(0)))}U(t,a.left,a.right,a.bottom,"Usable flow Q (m\xB3/s)"),t.append(h("text",{x:16,y:a.top,class:"axis-label",transform:`rotate(-90 16 ${a.top})`,"text-anchor":"end"},"Power (MW)")),t.append(h("path",{d:`M ${o(0)} ${d(0)} L ${o(i)} ${d(s*i)}`,class:"curve uncapped"})),t.append(h("line",{x1:a.left,y1:d(e.inputs.ratingMW),x2:a.right,y2:d(e.inputs.ratingMW),class:"curve cap"})),t.append(h("circle",{cx:o(e.inputs.usableFlowM3s),cy:d(e.plantMW),r:7,class:"live-point"})),t.append(h("text",{x:o(e.inputs.usableFlowM3s)+10,y:d(e.plantMW)-11,class:"live-label"},`${e.plantMW.toFixed(2)} MW`)),t.append(h("text",{x:a.left+8,y:d(e.inputs.ratingMW)-7,class:"legend cap-text"},`Rating cap ${e.inputs.ratingMW.toFixed(1)} MW`)),t.append(h("text",{x:a.right-6,y:Math.max(a.top+13,d(s*i)+17),class:"legend","text-anchor":"end"},"Uncapped \u03B7\u03C1gQH\u2099\u2091\u209C"))}function G(t,e){S(t);let a=[["Hydraulic resource",e.hydraulicMW,"hydraulic"],["After efficiency",e.uncappedMW,"uncapped"],["Plant export",e.plantMW,"plant"]],i=Math.max(1,...a.map(([,o])=>o)),s=160,p=390;a.forEach(([o,d,r],n)=>{let f=40+n*57;t.append(h("text",{x:s-12,y:f+20,class:"bar-label","text-anchor":"end"},o)),t.append(h("rect",{x:s,y:f,width:p,height:28,rx:7,class:"bar-track"})),t.append(h("rect",{x:s,y:f,width:Math.max(2,d/i*p),height:28,rx:7,class:`bar ${r}`})),t.append(h("text",{x:Math.min(560,s+d/i*p+8),y:f+20,class:"bar-value"},`${d.toFixed(2)} MW`))}),t.append(h("rect",{x:28,y:218,width:544,height:40,rx:10,class:"energy-band"})),t.append(h("text",{x:45,y:244,class:"energy-label"},`Six-hour energy: ${e.energy6hMWh.toFixed(2)} MWh = ${e.plantMW.toFixed(2)} MW \xD7 6 h`))}var H=[["grossHeadM","Water level H_gross",5,250,1,"m"],["usableFlowM3s","Usable flow Q",0,30,.1,"m\xB3/s"],["headLossM","Head loss h_loss",0,50,.5,"m"],["ratingMW","Rating cap P_r",.1,25,.1,"MW"]];function c(t,e=1){return Number(t).toFixed(e)}function A([t,e,a,i,s,p],o){return`<div class="control">
    <div class="control-line"><label for="${t}">${e}</label><output id="${t}-value" for="${t}">${c(o,s<1?1:0)} ${p}</output></div>
    <input id="${t}" name="${t}" type="range" min="${a}" max="${i}" step="${s}" value="${o}" aria-label="${e}, ${p}">
  </div>`}function R(t,e,a){return`<article class="hydro-lab" aria-labelledby="hydro-lab-title">
    <header class="lab-head">
      <span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="hydro-lab-title">From reservoir head to electrical energy</h2>
      <p>Move the four controls. The plant, water paths, equations, and plots all use the same steady-state calculation.</p>
    </header>
    <section class="controls" aria-label="Hydroelectric plant inputs">
      ${H.map(i=>A(i,t[i[0]])).join("")}
    </section>
    <section class="stage-grid">
      <div class="plant-views">
      <figure class="cutaway-card">
        ${C()}
        <figcaption>Hover, focus, or tap a component to identify it. Follow the animated water from reservoir to tailrace.</figcaption>
        <div class="head-ruler" id="head-ruler"><div class="head-track"></div><strong>Head balance</strong><span id="gross-readout"></span><span id="loss-readout"></span><span id="net-readout"></span></div>
      </figure>
      </div>
      <aside class="metrics-card" aria-label="Calculated plant performance" aria-live="polite">
        <div class="power-card"><span class="eyebrow">Plant electrical power P_plant</span><strong id="plant-power"></strong><span class="status" id="clip-status"></span></div>
        <div class="metric-list">
          <div class="metric"><span>Uncapped electrical power</span><strong id="uncapped-power"></strong></div>
          <div class="metric"><span>Net head H_net</span><strong id="net-head"></strong></div>
          <div class="metric"><span>Turbine Q / spill-bypass</span><strong id="flow-split"></strong></div>
          <div class="metric"><span>Six-hour energy</span><strong id="six-hour-energy"></strong></div>
        </div>
        <div class="equations">
          <h3>Live equation substitutions</h3>
          <div class="equation" id="eq-head"></div>
          <div class="equation" id="eq-power"></div>
          <div class="equation" id="eq-cap"></div>
          <div class="equation" id="eq-energy"></div>
        </div>
        <p class="assumptions">Fixed teaching assumptions: \u03B7 = ${x.efficiency}, \u03C1 = ${x.densityKgM3} kg/m\xB3, g = ${x.gravityMS2} m/s\xB2, \u0394t = ${x.durationHours} h. Excess usable flow is labelled spill/bypass because routing depends on the real plant.</p>
      </aside>
    </section>
    <section class="charts" aria-label="Hydroelectric plant charts">
      <div class="chart-card"><h3>Power versus usable flow</h3><p>Current net head; the dashed line is the generator rating.</p><svg id="power-chart" role="img" aria-label="Power versus usable flow chart"></svg></div>
      <div class="chart-card"><h3>Conversion chain and energy</h3><p>Hydraulic resource, efficiency loss, rating cap, and six-hour production.</p><svg id="conversion-chart" role="img" aria-label="Power conversion and energy chart"></svg></div>
    </section>
    <footer class="lab-foot">Educational steady-state model\u2014not a reservoir-routing, dam-safety, or design tool.${a?` <a href="${a}" target="_blank" rel="noopener">Open the lab full page</a>.`:""}</footer>
  </article>`}function I(t,e=g,a={}){if(!(t instanceof Element))throw new TypeError("mountHydroLab requires a DOM element");let i=t.shadowRoot??t.attachShadow({mode:"open"}),s=document.createElement("style");s.textContent=q;let p=document.createElement("div"),o=k({...g,...e}),d=a.assetBase??new URL(".",import.meta.url).href;p.innerHTML=R(o,d,a.standaloneUrl===void 0?d:a.standaloneUrl),i.replaceChildren(s,p);let r=u=>i.getElementById(u),n=z(i),f=Object.fromEntries(H.map(([u,,,,l,w])=>[u,{unit:w,digits:l<1?1:0}]));function m(u=o){o=k(u);let l=$(o);r("headLossM").max=String(Math.min(50,Math.max(0,o.grossHeadM-.5)));for(let[b]of H){let P=r(b);P.value=String(o[b]);let W=f[b];r(`${b}-value`).textContent=`${c(o[b],W.digits)} ${W.unit}`}r("plant-power").textContent=`${c(l.plantMW,3)} MW`,r("uncapped-power").textContent=`${c(l.uncappedMW,2)} MW`,r("net-head").textContent=`${c(l.netHeadM,1)} m`,r("flow-split").textContent=`${c(l.turbineFlowM3s,1)} / ${c(l.spillFlowM3s,1)} m\xB3/s`,r("six-hour-energy").textContent=`${c(l.energy6hMWh,2)} MWh`;let w=r("clip-status");return w.className=`status ${l.clipped?"clipped":"open"}`,w.textContent=l.clipped?`Clipped at P_r \xB7 uncapped ${c(l.uncappedMW,2)} MW`:"Within generator rating \xB7 not clipped",r("eq-head").textContent=`H_net = H_gross \u2212 h_loss = ${c(o.grossHeadM,1)} \u2212 ${c(o.headLossM,1)} = ${c(l.netHeadM,1)} m`,r("eq-power").textContent=`P_e = \u03B7\u03C1gQH_net = 0.85 \xD7 1000 \xD7 9.81 \xD7 ${c(o.usableFlowM3s,1)} \xD7 ${c(l.netHeadM,1)} = ${c(l.uncappedMW,2)} MW`,r("eq-cap").textContent=`P_plant = min(P_e, P_r) = min(${c(l.uncappedMW,2)}, ${c(o.ratingMW,2)}) = ${c(l.plantMW,2)} MW`,r("eq-energy").textContent=`E_6h = P_plant \xD7 \u0394t = ${c(l.plantMW,2)} \xD7 6 = ${c(l.energy6hMWh,2)} MWh`,r("gross-readout").textContent=`Gross ${c(o.grossHeadM,0)} m`,r("loss-readout").textContent=`Loss ${c(o.headLossM,1)} m`,r("net-readout").textContent=`Net ${c(l.netHeadM,1)} m`,r("head-ruler").style.setProperty("--loss-pct",`${Math.min(90,o.headLossM/o.grossHeadM*100)}%`),L(r("power-chart"),l),G(r("conversion-chart"),l),n(l),l}for(let[u]of H)r(u).addEventListener("input",l=>m({...o,[u]:l.currentTarget.value}));return{result:m(o),update(u){return m({...o,...u})},destroy(){i.replaceChildren()}}}if(document.documentElement.dataset.hydroStandalone==="true"){let t=document.getElementById("hydro-lab-root");t&&I(t,g,{standaloneUrl:""})}export{$ as computeHydro,I as mountHydroLab};
