import {test} from 'node:test';import assert from 'node:assert/strict';import {createBudget} from './project-budget.js';
test('capacity upgrade requires earned budget and deducts its cost',()=>{const b=createBudget();assert.equal(b.upgrade(6999999,5),false);assert.equal(b.upgrade(7000000,5),true);assert.equal(b.state(7000000,6).balance,0);assert.equal(b.upgrade(8000000,10),false);});
