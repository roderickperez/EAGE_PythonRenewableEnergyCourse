import { build } from 'esbuild';
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
for (const lab of ['hydro-lab', 'solar-lab']) {
  const source = resolve(root, 'web', lab);
  const output = resolve(root, 'EAGE_PythonRenewableEnergyCourse/static', lab);

  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  await build({
    entryPoints: [resolve(source, 'lab.js')],
    outfile: resolve(output, lab + '.js'),
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: ['es2022'],
    minify: true,
    sourcemap: false,
    loader: { '.css': 'text', '.svg': 'text' },
    legalComments: 'inline',
  });
  const html = await readFile(resolve(source, 'index.html'), 'utf8');
  await writeFile(resolve(output, 'index.html'), html);
  if (lab === 'hydro-lab') await cp(resolve(source, 'fallback.svg'), resolve(output, 'fallback.svg'));
  console.log(`Built ${lab} in ${output}`);
}
