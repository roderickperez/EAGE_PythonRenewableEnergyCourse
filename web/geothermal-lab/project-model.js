// Conservative, single-layer teaching model. SI units internally.
export const N = 10, DX = 200, THICKNESS = 50, CP = 4180, YEAR = 365 * 86400;
export const defaults = {injector: 43, producer: 46, depth: 2000, rate: 50, injection: 70,
  initial: 160, heatCapacity: 2500000, efficiency: .12, auxiliary: .1,
  capacity: 5, availability: .9, price: 75};

function neighbors(i) {
  const x=i%N, y=Math.floor(i/N), n=[];
  if(x) n.push(i-1); if(x<N-1) n.push(i+1);
  if(y) n.push(i-N); if(y<N-1) n.push(i+N);
  return n;
}
const adjacency=Array.from({length:N*N},(_,i)=>neighbors(i));

// Unit-conductance pressure network, no-flow outer boundary. The imposed
// sources balance exactly; conductance determines distribution, not pressure.
export function flowNetwork(injector, producer, rate) {
  if (injector === producer || ![injector,producer].every(i=>Number.isInteger(i)&&i>=0&&i<N*N)) throw Error('Choose two different grid cells.');
  if (!Number.isFinite(rate) || rate < 0) throw Error('Flow must be nonnegative.');
  const p=new Float64Array(N*N), b=new Float64Array(N*N);
  b[injector]=rate; b[producer]=-rate;
  for(let k=0;k<12000;k++) {
    let change=0;
    for(let i=1;i<p.length;i++) {
      let total=b[i]; for(const j of adjacency[i]) total+=p[j];
      const value=total/adjacency[i].length;
      change=Math.max(change,Math.abs(value-p[i])); p[i]=value;
    }
    if(change < Math.max(1,rate)*1e-12) break;
  }
  const edges=[], outgoing=new Float64Array(N*N), divergence=new Float64Array(N*N);
  for(let i=0;i<p.length;i++) for(const j of adjacency[i]) if(j>i) {
    const q=p[i]-p[j], from=q>=0?i:j, to=q>=0?j:i, mass=Math.abs(q);
    edges.push({from,to,mass}); outgoing[from]+=mass;
    divergence[from]+=mass; divergence[to]-=mass;
  }
  outgoing[producer]+=rate;
  return {edges,outgoing,divergence, residual:Math.max(...divergence.map((v,i)=>Math.abs(v-b[i])))};
}

export function createModel(settings={}) {
  const config={...defaults,...settings};
  if(!Object.values(config).every(Number.isFinite) || config.rate<0 || config.heatCapacity<=0 || config.initial<=config.injection || config.efficiency<0 || config.efficiency>1 || config.auxiliary<0 || config.auxiliary>1 || config.availability<0 || config.availability>1 || config.capacity<0 || config.price<0) throw Error('Check model inputs. Production temperature must exceed injection temperature.');
  const network=flowNetwork(config.injector,config.producer,config.rate);
  const temperature=new Float64Array(N*N).fill(config.initial);
  const cellCapacity=DX*DX*THICKNESS*config.heatCapacity;
  let seconds=0, energy=0, revenue=0, extractedJ=0;
  function step(days) {
    if(!Number.isFinite(days)||days<0) throw Error('Time step must be nonnegative.');
    let remaining=days*86400;
    const maxOutgoing=Math.max(...network.outgoing)*CP;
    const limit=maxOutgoing ? .8*cellCapacity/maxOutgoing : Infinity;
    while(remaining>0) {
      const dt=Math.min(remaining, limit, 30*86400);
      // Availability reduces circulation as well as generation; it is applied once.
      const factor=config.availability, old=temperature.slice(), delta=new Float64Array(N*N);
      for(const e of network.edges) {
        const heat=e.mass*CP*old[e.from]*dt*factor;
        delta[e.from]-=heat; delta[e.to]+=heat;
      }
      const produced=old[config.producer];
      delta[config.injector]+=config.rate*CP*config.injection*dt*factor;
      delta[config.producer]-=config.rate*CP*produced*dt*factor;
      const heatExtracted=config.rate*CP*(produced-config.injection)*dt*factor;
      extractedJ+=heatExtracted;
      const gross=Math.max(0,config.rate*CP*(produced-config.injection)/1e6*config.efficiency);
      const net=Math.min(config.capacity,gross*(1-config.auxiliary));
      const producedEnergy=net*dt/3600*factor;
      energy+=producedEnergy; revenue+=producedEnergy*config.price;
      for(let i=0;i<temperature.length;i++) temperature[i]+=delta[i]/cellCapacity;
      seconds+=dt; remaining-=dt;
    }
    return snapshot();
  }
  function snapshot() {
    const production=temperature[config.producer];
    const thermal=Math.max(0,config.rate*CP*(production-config.injection)/1e6);
    const net=Math.min(config.capacity,thermal*config.efficiency*(1-config.auxiliary));
    return {years:seconds/YEAR, production, thermal, net, energy, revenue, extractedJ,
      remainingJ:temperature.reduce((a,t)=>a+(t-config.injection)*cellCapacity,0),
      temperature:Array.from(temperature)};
  }
  return {config,network,step,snapshot};
}
