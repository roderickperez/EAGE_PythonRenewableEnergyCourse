// Simplified analytical examples, kept separate from the animated geometry.
export function conduction({k,rho,cp,days,wall,far,distance}) {
 if(![k,rho,cp,days,wall,far,distance].every(Number.isFinite)||k<=0||rho<=0||cp<=0||days<=0||distance<0||wall<=-273.15||far<=-273.15)throw Error('Use finite values, positive properties and time, and temperatures above absolute zero.');
 const alpha=k/(rho*cp), u=distance/(2*Math.sqrt(alpha*days*86400));
 // Abramowitz-Stegun erf approximation, maximum absolute error about 1.5e-7.
 const t=1/(1+.3275911*u), erf=1-((((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)*Math.exp(-u*u));
 return {alpha,effusivity:k/Math.sqrt(alpha),temperature:wall+(far-wall)*erf};
}
export function areas({inner,outer,bore}) {
 if(![inner,outer,bore].every(Number.isFinite)||!(inner>0&&outer>inner&&bore>outer))throw Error('Use finite diameters satisfying Dᵢ < Dₒ < Dᵦ.');
 const tube=Math.PI*inner**2/4,annulus=Math.PI*(bore**2-outer**2)/4;
 return {tube,annulus,ratio:annulus/tube};
}
export function energy({flow,dh,ds,ambient,years}) {
 if(![flow,dh,ds,ambient,years].every(Number.isFinite)||flow<0||ambient<=0||years<0)throw Error('Enter valid values.');
 const thermal=flow*dh/1000, available=flow*(dh-ambient*ds)/1000;
 return {thermal,available,thermalMWh:thermal*years*8760,availableMWh:available*years*8760};
}
