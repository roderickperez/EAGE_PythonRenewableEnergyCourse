// Single-phase, vertical, incompressible educational calculations, in SI units.
export function drilling({depth,flow,hole,pipe,density,friction,viscosity,particle}) {
 if(![depth,flow,hole,pipe,density,friction,viscosity,particle].every(Number.isFinite)||depth<=0||flow<0||hole<=pipe||pipe<=0||density<=0||friction<0||viscosity<=0||particle<=0)throw Error('Invalid geometry or physical input');
 const g=9.80665, area=Math.PI*(hole*hole-pipe*pipe)/4;
 const velocity=(flow/60000)/area,hydro=density*g*depth,drop=friction*1e6;
 const equivalent=density+drop/(g*depth),bottom=hydro+drop;
 const slip=Math.max(0,(2650-density)*g*particle*particle/(18*viscosity));
 const particleRe=density*slip*particle/viscosity;
 return {area,velocity,hydro,drop,equivalent,bottom,slip,particleRe,transport:velocity-slip,validStokes:particleRe<1};
}
