const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const files = ['web/hydro-lab/worked-examples.js','EAGE_PythonRenewableEnergyCourse/static/hydro-lab/worked-examples.js','EAGE_pythonSandBox/generated/hydro-lab/worked-examples.js'];
for (const file of files) {
 const values={width:12,depth:4.5,slope:.01,flow:60,precip:600,runoff:.4,area:50,rain:400,coefficient:.1,meanhead:80,designhead:100,eco:3};
 const nodes={}; for(const [key,value] of Object.entries(values)){nodes[key]={value:String(value)};nodes[key+'Out']={textContent:''};}
 for (const id of ['root','one','two','out1','out2']) nodes[id]={innerHTML:'',addEventListener:()=>{}};
 const document={querySelector:()=>nodes.root,getElementById:id=>{assert.ok(nodes[id],`${file}: ${id}`);return nodes[id];}};
 vm.runInNewContext(fs.readFileSync(file,'utf8'),{document,Math});
 const output=nodes.out2.innerHTML;
 assert.match(output,/Environmental flow served: 0\.06 m³\/s; shortfall: 2\.94 m³\/s/);
 assert.match(output,/full-year energy 0\.34 GWh/);
 assert.match(output,/main usable mean flow: 0\.00 m³\/s/);
}
console.log('Low-inflow reservoir case is bounded in source, static bundle and generated sandbox.');
