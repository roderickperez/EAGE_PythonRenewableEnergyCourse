export const SPEEDS=[1,5,20,100,500];
export const simulatedHours=(seconds,multiplier)=>seconds*.4*multiplier;
// Limit energy integration steps and stop at weather and calendar boundaries.
export const nextStep=(remaining,hour,weatherTimer)=>Math.min(.25,remaining,24-hour,2-weatherTimer);
