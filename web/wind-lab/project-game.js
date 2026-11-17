// Educational game values. One visible second represents 73 operating hours.
export const GAME_COST=50000, PRICE_PER_MWH=75, HOURS_PER_SECOND=73, SALVAGE_FRACTION=.5;
export function createGame(now=0) {
  let last=now, power=0, balance=GAME_COST, earnings=0;
  function advance(time,active=true){
    const seconds=Math.max(0,(time-last)/1000);last=time;
    if(active){const revenue=power/1000*HOURS_PER_SECOND*seconds*PRICE_PER_MWH;balance+=revenue;earnings+=revenue;}
  }
  const snapshot=()=>({balance,earnings,canBuy:balance>=GAME_COST});
  return {
    tick(time,active=true){advance(time,active);return snapshot();},
    setPower(kw,time,active=true){advance(time,active);power=kw;return snapshot();},
    buy(time,active=true){advance(time,active);if(balance<GAME_COST)return false;balance-=GAME_COST;return true;},
    sell(time,active=true){advance(time,active);balance+=GAME_COST*SALVAGE_FRACTION;return snapshot();},
  };
}
