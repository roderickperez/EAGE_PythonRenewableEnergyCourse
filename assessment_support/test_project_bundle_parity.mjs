import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, dirname} from 'node:path';

// Build in memory so verification cannot overwrite source or generated assets.
const cases = [
  ['web/hydro-lab/lab.js', 'hydro-lab/hydro-lab.js'],
  ['web/hydro-lab/head.js', 'hydro-lab/head.js'],
  ['web/hydro-lab/balance.js', 'hydro-lab/balance.js'],
  ['web/solar-lab/lab.js', 'solar-lab/solar-lab.js'],
  ['web/solar-lab/resource.js', 'solar-lab/resource.js'],
  ['web/solar-lab/project.js', 'solar-lab/solar-project.js'],
  ['web/wind-lab/aerodynamics.js', 'wind-lab/aerodynamics.js'],
  ['web/wind-lab/project.js', 'wind-lab/wind-project.js'],
  ['web/geothermal-lab/lab.js', 'geothermal-lab/geothermal-lab.js'],
  ['web/geothermal-lab/equations.js', 'geothermal-lab/equations.js'],
  ['web/geothermal-lab/project.js', 'geothermal-lab/geothermal-project.js'],
  ['web/geothermal-lab/drilling.js', 'geothermal-lab/drilling.js'],
  ['web/geothermal-lab/closed-loop.js', 'geothermal-lab/closed-loop.js'],
  ['web/project-map/lab.js', 'project-map/project-map.js'],
  ['web/hydro-lab/worked-examples.js', 'hydro-lab/worked-examples.js'],
  ['web/hydro-lab/project.js', 'hydro-lab/hydro-project.js'],
  ['web/geothermal-multifield/project.js', 'geothermal-lab/geothermal-multifield.js'],
  ['web/wind-lab/lab.js', 'wind-lab/wind-lab.js'],
];
const failures = [];
for (const [input, relative] of cases) {
  const staticPath = resolve('EAGE_PythonRenewableEnergyCourse/static', relative);
  const result = await build({
    entryPoints: [resolve(input)], outfile: staticPath, bundle: true,
    format: 'esm', platform: 'browser', target: ['es2022'], minify: true,
    sourcemap: false, loader: {'.css': 'text', '.svg': 'text', '.csv': 'text', '.geojson': 'json'},
    legalComments: 'inline', write: false,
  });
  const fresh = Buffer.from(result.outputFiles[0].contents);
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  const staticMatches = digest(await readFile(staticPath)) === digest(fresh);
  const sandboxMatches = digest(await readFile(resolve('EAGE_pythonSandBox/generated', relative))) === digest(fresh);
  if (!staticMatches || !sandboxMatches) failures.push(`${relative}: static=${staticMatches}, sandbox=${sandboxMatches}`);
  else console.log(`${relative}: source, static and sandbox bundles match`);
}
assert.deepEqual(failures, [], `Stale bundles:\n${failures.join('\n')}`);

// Check entry-page and SVG copies as well as the JavaScript they load.
const pages = [];
for (const lab of ['hydro-lab', 'solar-lab', 'wind-lab', 'geothermal-lab', 'project-map']) {
  pages.push([`web/${lab}/index.html`, `${lab}/index.html`]);
}
for (const [lab, names] of [
  ['hydro-lab', ['fallback.svg', 'head.html', 'balance.html', 'project.html', 'worked-examples.html']],
  ['solar-lab', ['resource.html', 'project.html']],
  ['wind-lab', ['project.html']],
  ['geothermal-lab', ['equations.html', 'project.html', 'drilling.html', 'closed-loop.html']],
]) for (const name of names) pages.push([`web/${lab}/${name}`, `${lab}/${name}`]);
pages.push(['web/geothermal-multifield/multifield.html', 'geothermal-lab/multifield.html']);
for (const [input, relative] of pages) {
  const source = await readFile(resolve(input));
  for (const root of ['EAGE_PythonRenewableEnergyCourse/static', 'EAGE_pythonSandBox/generated']) {
    assert.ok((await readFile(resolve(root, relative))).equals(source), `Stale entry page: ${root}/${relative}`);
  }
}
const solarCard = await readFile(resolve('web/solar-lab/resource.html'), 'utf8');
const windCard = solarCard.replace('<title>Solar radiation explorer</title>', '<title>Wind aerodynamics explorer</title>')
  .replace(/src="resource\.js\?[^"]+"/, 'src="aerodynamics.js?v=wind-energy-boundary-4"');
for (const root of ['EAGE_PythonRenewableEnergyCourse/static', 'EAGE_pythonSandBox/generated']) {
  assert.equal(await readFile(resolve(root, 'wind-lab/aerodynamics.html'), 'utf8'), windCard,
               `Stale generated wind entry page: ${root}`);
}
console.log(`${pages.length + 1} interactive entry pages and assets match their source`);

let scriptCount = 0;
for (const [, relative] of [...pages, ['', 'wind-lab/aerodynamics.html']]) {
  if (!relative.endsWith('.html')) continue;
  const pagePath = resolve('EAGE_pythonSandBox/generated', relative);
  const html = await readFile(pagePath, 'utf8');
  for (const match of html.matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["']/gi)) {
    const src = match[1];
    if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|\/)/i.test(src)) continue;
    const path = decodeURIComponent(src.split(/[?#]/, 1)[0]);
    await readFile(resolve(dirname(pagePath), path));
    scriptCount++;
  }
}
assert.ok(scriptCount > 0, 'Expected local script references to be checked');
console.log(`${scriptCount} local interactive script references resolve`);
