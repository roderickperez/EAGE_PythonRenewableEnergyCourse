// Samples of simulated panel output during the current solar day.
export function createPowerHistory() {
  let points=[];
  return {
    reset(){points=[];},
    record(hour,panels){
      if(points.length && hour<points.at(-1).hour-1e-6)points=[];
      const point={hour,values:Object.fromEntries(panels.map(p=>[p.id,p.current.ac]))};
      if(points.length && Math.abs(hour-points.at(-1).hour)<1e-6)points[points.length-1]=point;
      else points.push(point);
    },
    get points(){return points;},
  };
}
