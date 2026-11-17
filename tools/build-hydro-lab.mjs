import { build } from 'esbuild';
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
for (const lab of ['hydro-lab', 'solar-lab', 'wind-lab', 'geothermal-lab', 'project-map']) {
  const source = resolve(root, 'web', lab);
  const output = resolve(root, 'EAGE_PythonRenewableEnergyCourse/static', lab);

  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  const entryPoints = { [lab]: resolve(source, 'lab.js') };
  if (lab === 'hydro-lab') {
    entryPoints.head = resolve(source, 'head.js');
    entryPoints.balance = resolve(source, 'balance.js');
  }
  if (lab === 'solar-lab') entryPoints.resource = resolve(source, 'resource.js');
  if (lab === 'geothermal-lab') entryPoints.equations = resolve(source, 'equations.js');
  if (lab === 'wind-lab') entryPoints.aerodynamics = resolve(source, 'aerodynamics.js');
  await build({
    entryPoints,
    outdir: output,
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: ['es2022'],
    minify: true,
    sourcemap: false,
    loader: { '.css': 'text', '.svg': 'text', '.csv': 'text' },
    legalComments: 'inline',
  });
  const html = await readFile(resolve(source, 'index.html'), 'utf8');
  await writeFile(resolve(output, 'index.html'), html);
  if (lab === 'geothermal-lab') await cp(resolve(source, 'equations.html'), resolve(output, 'equations.html'));
  if (lab === 'solar-lab') await cp(resolve(source, 'resource.html'), resolve(output, 'resource.html'));
  if (lab === 'wind-lab') {
    const card = await readFile(resolve(root, 'web/solar-lab/resource.html'), 'utf8');
    await writeFile(resolve(output, 'aerodynamics.html'), card
      .replace('<title>Solar radiation explorer</title>', '<title>Wind aerodynamics explorer</title>')
      .replace(/src="resource\.js\?[^"]+"/, 'src="aerodynamics.js?v=wind-fixed-axes-3"'));
  }
  if (lab === 'hydro-lab') {
    await cp(resolve(source, 'fallback.svg'), resolve(output, 'fallback.svg'));
    await cp(resolve(source, 'head.html'), resolve(output, 'head.html'));
    await cp(resolve(source, 'balance.html'), resolve(output, 'balance.html'));
  }
  console.log(`Built ${lab} in ${output}`);
}
