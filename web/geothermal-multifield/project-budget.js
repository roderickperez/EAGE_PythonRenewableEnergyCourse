export const INITIAL_COST = 6000000, UPGRADE_COST = 1000000, MAX_CAPACITY = 10;
export function createBudget() {
 let upgrades = 0, extraWells = 0;
 return {
  state(revenue, capacity) {
   const investment=INITIAL_COST+(upgrades+extraWells)*UPGRADE_COST;
   const balance=revenue-investment;
   return {balance, investment, upgrades, extraWells, available:Math.max(0,balance), canUpgrade:balance>=UPGRADE_COST&&capacity<MAX_CAPACITY};
  },
  upgrade(revenue,capacity) {
   if(!this.state(revenue,capacity).canUpgrade)return false;
   upgrades++;return true;
  },
  drillExtra(){extraWells++;},
 };
}
