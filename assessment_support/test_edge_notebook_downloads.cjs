// Run with the local sandbox served on 127.0.0.1:18343.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn, spawnSync} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function parseCsvLine(line) {
  const cells = []; let value = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (quoted && line[i + 1] === '"') { value += '"'; i++; } else quoted = !quoted;
    } else if (c === ',' && !quoted) { cells.push(value); value = ''; } else value += c;
  }
  cells.push(value); return cells;
}

(async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'eage-edge-download-'));
  const profile = path.join(temporary, 'profile');
  const downloads = path.join(temporary, 'downloads');
  fs.mkdirSync(downloads);
  const edge = spawn(process.env.EAGE_TEST_EDGE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ['--headless=new', '--disable-gpu', '--no-first-run', '--disable-background-networking',
      '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'],
    {stdio: 'ignore', windowsHide: true});
  let socket;
  try {
    const portFile = path.join(profile, 'DevToolsActivePort');
    for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await delay(100);
    assert.ok(fs.existsSync(portFile), 'Edge debugging endpoint started');
    const port = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
    const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'});
    assert.ok(response.ok);
    const target = await response.json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
    let serial = 0;
    const pending = new Map();
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (!pending.has(message.id)) return;
      const {resolve, reject} = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(message.error.message)) : resolve(message.result);
    };
    const send = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)); }, 15000);
      pending.set(id, {resolve: value => { clearTimeout(timer); resolve(value); }, reject: error => { clearTimeout(timer); reject(error); }});
      socket.send(JSON.stringify({id, method, params}));
    });
    await send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloads});
    for (const [lesson, name] of [
      ['section6/solarEnergy.md', 'solarEnergy.ipynb'],
      ['section6/geothermalEnergy.md', 'geothermalEnergy.ipynb'],
      ['section5/download-to-database.md', 'download-to-database.ipynb'],
    ]) {
      const url = 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(lesson);
      await send('Page.navigate', {url});
      let clicked = false;
      for (let i = 0; i < 100; i++) {
        const result = await send('Runtime.evaluate', {returnByValue: true,
          expression: `(() => { const a = document.querySelector('#lesson a[download="${name}"]'); if (!a) return false; a.click(); return true; })()`});
        if (result.result.value) { clicked = true; break; }
        await delay(100);
      }
      assert.ok(clicked, `Notebook link found for ${lesson}`);
      const download = path.join(downloads, name);
      for (let i = 0; i < 150 && !fs.existsSync(download); i++) await delay(100);
      assert.ok(fs.existsSync(download), `Edge completed ${name}`);
      const source = path.join('EAGE_pythonSandBox', 'generated', 'assets', path.dirname(lesson), name);
      assert.deepEqual(fs.readFileSync(download), fs.readFileSync(source), `Downloaded ${name} matches its staged notebook`);
      const location = await send('Runtime.evaluate', {returnByValue: true, expression: 'location.href'});
      const currentLesson = new URLSearchParams(new URL(location.result.value).hash.slice(1)).get('lesson');
      assert.equal(currentLesson, lesson, 'The same lesson remains open after download');
      console.log(`${lesson}: downloaded ${name} with matching bytes`);
    }
    const course = JSON.parse(fs.readFileSync('EAGE_pythonSandBox/generated/course.json', 'utf8'));
    const practice = course.pages.filter(p => p.group === 'Quizzes' || p.id.startsWith('section7/workbook/'));
    let editedAnswers = 0;
    let checkedPracticeSetups = 0;
    for (const page of practice) {
      const {id: lesson, title} = page;
      await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(lesson)});
      let edit;
      for (let i = 0; i < 100; i++) {
        const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
          if (new URLSearchParams(location.hash.slice(1)).get('lesson') !== ${JSON.stringify(lesson)}) return null;
          if (document.querySelector('#lesson h1')?.textContent.trim() !== ${JSON.stringify(title)}) return null;
          const answers = [...document.querySelectorAll('#lesson details.solution')].filter(a => a.querySelector('button[data-code]'));
          const mismatches = [], setupMismatches = [];
          const records = ${JSON.stringify(page.codes)};
          let setupCount = 0;
          for (const answer of answers) {
            const visible = answer.querySelector('pre code').textContent.trim();
            answer.open = true; answer.querySelector('button[data-code]').click();
            if (document.getElementById('code').value.trim() !== visible) mismatches.push(answer.querySelector('summary').textContent);
            const setup = answer.querySelector('button[data-setup-answers]');
            if (setup) {
              const index = records.findIndex(cell => cell.id === setup.dataset.setupAnswers);
              const expected = '# Includes earlier practice answers; review before running.\\n\\n' +
                records.slice(0, index + 1).filter(cell => cell.runnable).map(cell => cell.code).join('\\n\\n');
              setup.click();
              if (index < 0 || document.getElementById('code').value !== expected) setupMismatches.push(setup.dataset.setupAnswers);
              setupCount++;
            }
          }
          return {count: answers.length, mismatches, setupCount, setupMismatches};
        })()`});
        edit = result.result.value;
        if (edit) break;
        await delay(100);
      }
      assert.ok(edit, `Answer edit control loaded for ${lesson}`);
      assert.deepEqual(edit.mismatches, [], `Displayed answers reach the editor in ${lesson}`);
      assert.deepEqual(edit.setupMismatches, [], `Practice setups include the expected runnable prefix in ${lesson}`);
      assert.equal(edit.count, page.codes.filter(cell => cell.solution).length, `All answer controls were checked in ${lesson}`);
      if (lesson.startsWith('section7/renewableEnergyquizzes/')) {
        assert.equal(edit.setupCount, page.codes.filter(cell => cell.solution && cell.runnable).length,
          `Every runnable quiz answer has a practice setup in ${lesson}`);
        checkedPracticeSetups += edit.setupCount;
      }
      editedAnswers += edit.count;
    }
    assert.equal(editedAnswers, 199);
    assert.equal(checkedPracticeSetups, 97);
    console.log('All 199 answer edit controls and 97 quiz practice setup controls load their expected code.');
    const importLesson = 'section7/workbook/hydroelectric.md';
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(importLesson)});
    let importResult;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true, awaitPromise:true, expression:`(async () => {
        if (document.querySelector('#lesson h1')?.textContent.trim() !== 'Hydroelectric') return null;
        const select = document.getElementById('workspace-select'), code = document.getElementById('code');
        const other = [...select.options].find(option => option.value !== 'draft')?.value;
        if (!other) return null;
        select.value = other; select.dispatchEvent(new Event('change', {bubbles:true}));
        code.value = 'old other draft';
        select.value = 'draft'; select.dispatchEvent(new Event('change', {bubbles:true}));
        code.value = 'visible active draft'; code.dispatchEvent(new Event('input', {bubbles:true}));
        await new Promise(resolve => setTimeout(resolve, 350));
        const storageKey = 'eage.sandbox.v2', otherKey = ${JSON.stringify(importLesson)} + '::' + other;
        const before = localStorage.getItem(storageKey), originalSet = Storage.prototype.setItem, originalConfirm = window.confirm;
        let failed = false;
        try {
          window.confirm = () => true;
          Storage.prototype.setItem = function(key, value) {
            if (key === storageKey) throw new DOMException('quota exceeded', 'QuotaExceededError');
            return originalSet.call(this, key, value);
          };
          const data = {format:storageKey, course:'older-test-version', drafts:{[otherKey]:{code:'unexpected imported draft', notes:''}}};
          await document.getElementById('import-work').onchange({target:{files:[{text:async()=>JSON.stringify(data)}],value:'test'}});
          failed = localStorage.getItem(storageKey) === before && code.value === 'visible active draft' &&
            document.getElementById('output').textContent.includes('Import failed: quota exceeded') &&
            document.getElementById('output').textContent.includes('Export your current work before freeing browser storage, then retry the import.');
          document.getElementById('export-work').click();
        } finally { Storage.prototype.setItem = originalSet; window.confirm = originalConfirm; }
        code.dispatchEvent(new Event('input', {bubbles:true}));
        await new Promise(resolve => setTimeout(resolve, 350));
        const persisted = JSON.parse(localStorage.getItem(storageKey));
        select.value = other; select.dispatchEvent(new Event('change', {bubbles:true}));
        const visibleOther=code.value, beforeCancel=localStorage.getItem(storageKey);
        const data={format:storageKey, course:'older-test-version', drafts:{[otherKey]:{code:'confirmed import',notes:'imported note'}}};
        const inputEvent=()=>({target:{files:[{text:async()=>JSON.stringify(data)}],value:'test'}});
        let confirmation='';
        try {
          window.confirm = message => { confirmation=message; return false; };
          await document.getElementById('import-work').onchange(inputEvent());
          const cancelled=localStorage.getItem(storageKey)===beforeCancel && code.value===visibleOther;
          window.confirm = () => true;
          await document.getElementById('import-work').onchange(inputEvent());
          const imported=JSON.parse(localStorage.getItem(storageKey))[otherKey];
          return {failed, otherKey, persistedOther:persisted[otherKey]?.code, visibleOther, cancelled,
            mismatchWarning:confirmation.includes('different course version'), keyWarning:confirmation.includes(otherKey),
            importedCode:imported.code, importedNote:imported.notes, visibleAfterImport:code.value};
        } finally { window.confirm = originalConfirm; }
      })()`});
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
      importResult = result.result.value;
      if (importResult) break;
      await delay(100);
    }
    assert.ok(importResult.otherKey.startsWith(importLesson + '::'));
    const {otherKey, ...importChecks} = importResult;
    assert.deepEqual(importChecks, {failed:true, persistedOther:'old other draft', visibleOther:'old other draft',
      cancelled:true, mismatchWarning:true, keyWarning:true, importedCode:'confirmed import',
      importedNote:'imported note', visibleAfterImport:'confirmed import'});
    const workExport = path.join(downloads, 'eage-participant-work.json');
    for (let i = 0; i < 100 && !fs.existsSync(workExport); i++) await delay(100);
    assert.ok(fs.existsSync(workExport), 'Current work can be exported despite the simulated quota failure');
    const backup = JSON.parse(fs.readFileSync(workExport, 'utf8'));
    assert.equal(backup.drafts[otherKey].code, 'old other draft');
    assert.equal(backup.drafts[importLesson + '::draft'].code, 'visible active draft');
    console.log('Work import warns about version and replacement; cancellation and successful import behave correctly.');
    console.log('Current work exports successfully during the simulated quota failure.');
    const windQuiz = course.pages.find(page => page.id === 'section7/renewableEnergyquizzes/eolicenergy.md');
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(windQuiz.id)});
    let setup;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        if (document.querySelector('#lesson h1')?.textContent.trim() !== ${JSON.stringify(windQuiz.title)}) return null;
        const answer = document.querySelectorAll('#lesson details.solution')[5];
        const button = answer?.querySelector('button[data-setup-answers]');
        if (!button) return null;
        const id = button.dataset.setupAnswers; answer.open = true; button.click();
        return {id, code:document.getElementById('code').value};
      })()`});
      setup = result.result.value;
      if (setup) break;
      await delay(100);
    }
    assert.ok(setup, 'Wind data quiz practice setup control loaded');
    const setupIndex = windQuiz.codes.findIndex(cell => cell.id === setup.id);
    assert.ok(setupIndex >= 0);
    const expectedSetup = '# Includes earlier practice answers; review before running.\n\n' +
      windQuiz.codes.slice(0, setupIndex + 1).filter(cell => cell.runnable).map(cell => cell.code).join('\n\n');
    assert.equal(setup.code, expectedSetup);
    assert.match(setup.code, /clean\s*=/);
    console.log('Wind data quiz setup includes the preceding runnable practice answers and current answer.');
    if (process.env.EAGE_TEST_PYODIDE === '1') {
      await send('Runtime.evaluate', {expression: "document.getElementById('run').click()"});
      let run;
      for (let i = 0; i < 240; i++) {
        const result = await send('Runtime.evaluate', {returnByValue:true, expression:`({busy:document.getElementById('run').disabled,
          output:document.getElementById('output').textContent,
          status:document.getElementById('python-status').getAttribute('aria-label')})`});
        run = result.result.value;
        if (!run.busy) break;
        await delay(500);
      }
      assert.ok(run && !run.busy, `Python did not finish: ${JSON.stringify(run)}`);
      assert.match(run.output, /Direction coverage: 720\/720 observations/);
      assert.match(run.output, /Circular mean direction: 237\.0°/);
      assert.match(run.output, /Finished\./);
      console.log('Wind data quiz setup runs successfully in the browser Python runtime.');
      const loadRose = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        const answer = document.querySelectorAll('#lesson details.solution')[13];
        const button = answer?.querySelector('button[data-setup-answers]');
        if (!button) return false;
        answer.open = true; button.click(); document.getElementById('run').click(); return true;
      })()`});
      assert.ok(loadRose.result.value, 'Wind-rose answer setup loaded');
      let roses;
      for (let i = 0; i < 240; i++) {
        const result = await send('Runtime.evaluate', {returnByValue:true, expression:`({busy:document.getElementById('run').disabled,
          output:document.getElementById('output').textContent,
          images:document.querySelectorAll('#output img').length,
          errors:[...document.querySelectorAll('#output pre.error')].map(e=>e.textContent)})`});
        roses = result.result.value;
        if (!roses.busy) break;
        await delay(500);
      }
      assert.ok(roses && !roses.busy, `Wind roses did not finish: ${JSON.stringify(roses)}`);
      const unexpectedErrors = roses.errors.filter(message => !/MatplotlibDeprecationWarning: The [xy] parameter as float was deprecated/.test(message));
      assert.deepEqual(unexpectedErrors, []);
      assert.match(roses.output, /Finished\./);
      assert.ok(roses.images >= 2, `Expected frequency and energy roses: ${JSON.stringify(roses)}`);
      console.log(`Wind frequency and energy roses render in the browser Python runtime (${roses.errors.length} Matplotlib deprecation warnings).`);
      const runCode = async source => {
        await send('Runtime.evaluate', {expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});
        let result;
        for (let i = 0; i < 240; i++) {
          const response = await send('Runtime.evaluate', {returnByValue:true, expression:`({busy:document.getElementById('run').disabled,
            output:document.getElementById('output').textContent})`});
          result = response.result.value;
          if (!result.busy) break;
          await delay(500);
        }
        assert.ok(result && !result.busy && result.output.includes('Finished.'), JSON.stringify(result));
        return result.output;
      };
      if (process.env.EAGE_TEST_ALL_PRACTICE === '1') {
        const quizPaths = ['hydroelectricEnergy', 'eolicenergy', 'windEnergy', 'solarEnergy', 'geothermalEnergy'];
        for (const name of quizPaths) {
          const page = course.pages.find(item => item.id === `section7/renewableEnergyquizzes/${name}.md`);
          assert.ok(page, `Missing practice page: ${name}`);
          const index = page.codes.findLastIndex(cell => cell.solution && cell.runnable);
          assert.ok(index >= 0, `Missing runnable answer: ${name}`);
          const assembled = page.codes.slice(0, index + 1).filter(cell => cell.runnable).map(cell => cell.code).join('\n\n');
          await runCode(assembled);
          console.log(`${name}: complete cumulative practice answers run in browser Python.`);
        }
      }
      if (process.env.EAGE_TEST_WORKBOOKS === '1') {
        for (const name of ['general', 'hydroelectric', 'solar', 'wind', 'geothermal']) {
          const page = course.pages.find(item => item.id === `section7/workbook/${name}.md`);
          assert.ok(page, `Missing workbook: ${name}`);
          const answers = page.codes.filter(cell => cell.solution && cell.runnable).map(cell => ({id: cell.id, code: cell.code}));
          assert.equal(answers.length, name === 'hydroelectric' ? 22 : 20);
          const python = [
            'import json, io, contextlib, warnings, numpy, pandas, matplotlib',
            'import matplotlib.pyplot as plt',
            `answers = json.loads(${JSON.stringify(JSON.stringify(answers))})`,
            'errors = []',
            'with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):',
            '    warnings.simplefilter("ignore")',
            '    for answer in answers:',
            '        try:',
            '            exec(compile(answer["code"], answer["id"], "exec"), {})',
            '        except Exception as exc:',
            '            errors.append((answer["id"], type(exc).__name__, str(exc)))',
            '        finally:',
            '            plt.close("all")',
            'if errors:',
            '    raise RuntimeError(str(errors))',
            'print("WORKBOOK_ANSWERS_OK", len(answers))',
          ].join('\n');
          const output = await runCode(python);
          assert.ok(output.includes(`WORKBOOK_ANSWERS_OK ${answers.length}`), output);
          console.log(`${name}: ${answers.length} independent workbook answers run in browser Python.`);
        }
      }
      const runFailure = async source => {
        await send('Runtime.evaluate', {expression:`document.getElementById('code').value=${JSON.stringify(source)};document.getElementById('run').click()`});
        let result;
        for (let i = 0; i < 240; i++) {
          const response = await send('Runtime.evaluate', {returnByValue:true, expression:`({busy:document.getElementById('run').disabled,
            output:document.getElementById('output').textContent})`});
          result = response.result.value;
          if (!result.busy) break;
          await delay(500);
        }
        assert.ok(result && !result.busy, 'Expected failing code to finish');
        assert.doesNotMatch(result.output, /Finished\./);
        return result.output;
      };
      const navigateLesson = async (id, title) => {
        await send('Page.navigate', {url:'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(id)});
        for (let i = 0; i < 100; i++) {
          const result = await send('Runtime.evaluate', {returnByValue:true, expression:`document.querySelector('#lesson h1')?.textContent.trim() === ${JSON.stringify(title)}`});
          if (result.result.value) return;
          await delay(100);
        }
        throw new Error(`Lesson did not load: ${id}`);
      };
      const gapExercise = windQuiz.codes.find(cell => cell.code.includes('gap_filled=s.mask'));
      const turbineExercise = windQuiz.codes.find(cell => cell.code.includes('def turbine_power_mw('));
      assert.ok(gapExercise?.runnable && turbineExercise?.runnable, 'Wind gap and turbine answers are available');
      await runCode('clean_before_gap = clean.copy(deep=True)\nprint("saved clean dataset")');
      await runCode(gapExercise.code);
      const revisitedTurbine = await runCode(turbineExercise.code + '\npd.testing.assert_frame_equal(clean, clean_before_gap)\nprint("CLEAN_DATA_PRESERVED")');
      assert.match(revisitedTurbine, /CLEAN_DATA_PRESERVED/);
      assert.match(revisitedTurbine, /valid modelled intervals 720\/720/);
      console.log('Wind gap exercise preserves the cleaned data when the turbine answer is revisited.');
      const curveChecks = await runCode(`values = turbine_power_mw([np.nan, -1, 0, 12, 25])
assert np.isnan(values[:2]).all() and values[2:].tolist() == [0, 3, 0]
for params in ({"rated": -1}, {"cut_in": 12}, {"rated_speed": 25}, {"cut_out": float("inf")}):
    try:
        turbine_power_mw([0, 12], **params)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid curve parameters accepted: {params}")
print("WIND_CURVE_VALIDATION_OK")`);
      assert.match(curveChecks, /WIND_CURVE_VALIDATION_OK/);
      console.log('Wind curve preserves missing values and rejects invalid parameters in browser Python.');
      const literalOutput = await runCode(`message = """Important:
! Keep units explicit
% is a symbol
"""
print(message)
print("remainder", 7 % 4)`);
      assert.match(literalOutput, /! Keep units explicit/);
      assert.match(literalOutput, /% is a symbol/);
      assert.match(literalOutput, /remainder 3/);
      console.log('Multiline string text and modulo run without a false magic-command rejection.');
      for (const command of ['!echo forbidden', '%time print(1)']) {
        const rejection = await runFailure(command);
        assert.match(rejection, /contains Jupyter magic or a terminal command/);
      }
      console.log('Actual magic and shell commands receive the expected browser error.');
      const optionalImports = await runCode('import os, folium, openpyxl, plotly, seaborn\nprint("OPTIONAL_IMPORTS_OK", folium.__version__, openpyxl.__version__, plotly.__version__, seaborn.__version__)');
      assert.match(optionalImports, /OPTIONAL_IMPORTS_OK 0\.20\.0 3\.1\.5 6\.3\.0 0\.13\.2/);
      console.log('All four optional packages load from a comma-separated import in browser Python.');
      const eurostatLesson = course.pages.find(page => page.id === 'section5/download-to-database.md');
      const eurostatDownload = eurostatLesson.codes.find(cell => cell.code.includes('# Step 1: Explicit request parameters'));
      const eurostatReadback = eurostatLesson.codes.find(cell => cell.code.includes('CSV observed sum:'));
      assert.ok(eurostatDownload?.runnable && eurostatReadback?.runnable, 'Eurostat example cells are available');
      await navigateLesson(eurostatLesson.id, eurostatLesson.title);
      const offlineCode = eurostatDownload.code.replace('USE_LIVE = True', 'USE_LIVE = False');
      assert.notEqual(offlineCode, eurostatDownload.code, 'Explicitly select the bundled offline snapshot');
      const eurostatOutput = await runCode(offlineCode);
      assert.match(eurostatOutput, /reported months: 12/);
      assert.match(eurostatOutput, /Saved files in:/);
      const csvOutput = await runCode(eurostatReadback.code);
      const csvTotal = csvOutput.match(/CSV observed sum:\s*([\d.eE+-]+)\s*GWh/);
      assert.ok(csvTotal && Math.abs(Number(csvTotal[1]) - 7971.36) < 1e-8, `Unexpected CSV total: ${csvOutput}`);
      await runCode('previous_output = output\nprevious_csv_hash = hashlib.sha256((previous_output / "generation.csv").read_bytes()).hexdigest()\nprint("saved previous export")');
      const hashFailure = offlineCode.replace('snapshot_record.get("sha256") != hashlib.sha256(raw).hexdigest()',
                                              'True or snapshot_record.get("sha256") != hashlib.sha256(raw).hexdigest()');
      assert.notEqual(hashFailure, offlineCode, 'Inject a snapshot hash failure after the successful run');
      assert.match(await runFailure(hashFailure), /Offline snapshot does not match its provenance hash/);
      assert.match(await runFailure(eurostatReadback.code), /working example did not complete/);
      const databaseFailure = offlineCode.replace('connection = sqlite3.connect(output/"generation.sqlite")',
        'raise RuntimeError("injected database failure after CSV write")');
      assert.notEqual(databaseFailure, offlineCode, 'Inject failure after the new CSV is written');
      assert.match(await runFailure(databaseFailure), /injected database failure after CSV write/);
      assert.match(await runFailure(eurostatReadback.code), /working example did not complete/);
      const failureEvidence = await runCode('assert output.name.endswith(".incomplete") and (output / "generation.csv").is_file()\nassert hashlib.sha256((previous_output / "generation.csv").read_bytes()).hexdigest() == previous_csv_hash\nprint("INCOMPLETE_BLOCKED_PREVIOUS_PRESERVED")');
      assert.match(failureEvidence, /INCOMPLETE_BLOCKED_PREVIOUS_PRESERVED/);
      console.log('Eurostat failed hash and database reruns block incomplete CSVs and preserve the prior export.');
      const partialFixture = `payload = json.loads(raw)
payload["size"][-1] = 9
category = payload["dimension"]["time"]["category"]
category["index"] = {key: index for key, index in category["index"].items() if index < 9}
category["label"] = {key: label for key, label in category["label"].items() if key in category["index"]}
payload["value"] = {key: value for key, value in payload["value"].items() if int(key) < 9}
del payload["dimension"]["geo"]["category"]["index"]
payload["status"] = "estimated"
raw = json.dumps(payload).encode("utf-8")`;
      const partialCode = offlineCode.replace('payload = json.loads(raw)', partialFixture);
      assert.notEqual(partialCode, offlineCode, 'Use a nine-month synthetic response');
      const partialOutput = await runCode(partialCode + '\nassert len(frame) == 12 and frame.generation_gwh.notna().sum() == 9\nassert frame.status_flag.tolist() == ["estimated"] * 9 + [""] * 3\nprint("PARTIAL_YEAR_AND_STATUS_OK")');
      assert.match(partialOutput, /Provider returned 9\/12 month categories/);
      assert.match(partialOutput, /Coverage: 9\/12 monthly observations/);
      assert.match(partialOutput, /PARTIAL_YEAR_AND_STATUS_OK/);
      const partialReadback = await runCode(eurostatReadback.code + '\nassert len(check) == 12 and check.generation_gwh.isna().sum() == 3\nassert check.status_flag.tolist() == ["estimated"] * 9 + [""] * 3\nprint("PARTIAL_CSV_READBACK_OK")');
      assert.match(partialReadback, /PARTIAL_CSV_READBACK_OK/);
      console.log('A partial Eurostat response and CSV readback preserve missing months and dataset-wide status.');
      if (process.env.EAGE_TEST_LIVE_DATA === '1') {
        const solarEurostat = eurostatDownload.code.replace('SOURCE = "RA300"', 'SOURCE = "RA420"');
        assert.notEqual(solarEurostat, eurostatDownload.code, 'Select live PV generation');
        const liveSolar = await runCode(solarEurostat + '\nassert len(frame) == 12 and frame.source.eq("RA420").all()\nprint("LIVE_PV_REPORTED", int(frame.generation_gwh.notna().sum()), "SUM", frame.generation_gwh.sum(min_count=1))');
        assert.match(liveSolar, /LIVE_PV_REPORTED \d+ SUM/);
        assert.match(liveSolar, /Totals reflect reported precision; flagged or unflagged records do not establish complete generation coverage/);
        console.log('Live Eurostat PV example reports monthly data with its coverage qualification.');
      }
      const historical = course.pages.find(page => page.id === 'section5/SQL_Pandas.ipynb');
      assert.ok(historical && historical.codes.length === 7, 'Historical Eurostat notebook cells are available');
      await navigateLesson(historical.id, historical.title);
      const historicalCode = historical.codes.filter(cell => cell.runnable).map(cell => cell.code).join('\n\n') +
        '\nprint("HISTORICAL_ROWS", len(generation), "INTEGRITY", integrity)';
      const historicalOutput = await runCode(historicalCode);
      assert.match(historicalOutput, /HISTORICAL_ROWS 45320 INTEGRITY ok/);
      await runCode('historical_previous_rows = len(generation)\ngeneration = pd.concat([generation, generation.iloc[[0]]], ignore_index=True)\nprint("duplicate inserted for test")');
      const historicalWriter = historical.codes.find(cell => cell.code.includes('CREATE TABLE generation'));
      const historicalQuery = historical.codes.find(cell => cell.code.includes('selected_query ='));
      assert.match(await runFailure(historicalWriter.code), /UNIQUE constraint failed/);
      assert.match(await runFailure(historicalQuery.code), /Database rebuild did not complete/);
      const preserved = await runCode('with closing(sqlite3.connect(DB_PATH)) as conn:\n    assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == historical_previous_rows\nprint("HISTORICAL_ROLLBACK_OK")');
      assert.match(preserved, /HISTORICAL_ROLLBACK_OK/);
      console.log('Historical Eurostat notebook runs on bundled spreadsheets; failed rebuild rolls back and blocks stale queries.');
      if (process.env.EAGE_TEST_LIVE_DATA === '1') {
        const worldwide = course.pages.find(page => page.id === 'section5/world-energy-download.md');
        assert.ok(worldwide && worldwide.codes.length >= 3, 'Worldwide download cells are available');
        await navigateLesson(worldwide.id, worldwide.title);
        const download = await runCode(worldwide.codes[0].code);
        assert.match(download, /Country\/region lookup saved:/);
        const worldSelection = worldwide.codes[1].code.replace('COUNTRY_CODE = "AUT"', 'COUNTRY_CODE = "WORLD"');
        assert.notEqual(worldSelection, worldwide.codes[1].code, 'Select the World aggregate');
        await runCode(worldSelection + '\nassert selected.country.eq("World").all() and selected.year.tolist() == [2019, 2020, 2021, 2022, 2023]\nprint("WORLD_SELECTION_OK")');
        const worldExport = await runCode(worldwide.codes[2].code + '\nassert not output.name.endswith(".incomplete")\nprint("WORLD_EXPORT_OK")');
        assert.match(worldExport, /WORLD_EXPORT_OK/);
        assert.match(worldExport, /SQLite:/);
        await runCode('successful_world_output = output\nsuccessful_world_hash = hashlib.sha256((output / "generation.csv").read_bytes()).hexdigest()\nprint("world export recorded")');
        const invalidSelection = worldSelection.replace('VARIABLE = "wind_electricity_twh"', 'VARIABLE = "not_a_real_column"');
        assert.notEqual(invalidSelection, worldSelection, 'Inject an invalid variable selection');
        assert.match(await runFailure(invalidSelection), /Choose a documented electricity-generation column/);
        assert.match(await runFailure(worldwide.codes[2].code), /Run a successful download and selection before saving/);
        const worldPreserved = await runCode('assert hashlib.sha256((successful_world_output / "generation.csv").read_bytes()).hexdigest() == successful_world_hash\nprint("WORLD_PRIOR_EXPORT_PRESERVED")');
        assert.match(worldPreserved, /WORLD_PRIOR_EXPORT_PRESERVED/);
        console.log('Live World export works; a failed reselection blocks stale saving and preserves the prior export.');
      }
      const windLesson = course.pages.find(page => page.id === 'section5/wind-data.md');
      const froyaExample = windLesson.codes.find(cell => cell.code.includes('2009-11-18 15:00,3.586,159'));
      assert.ok(froyaExample?.runnable, 'Frøya inspection example is available to learners');
      await navigateLesson(windLesson.id, windLesson.title);
      const froyaOutput = await runCode(froyaExample.code);
      assert.match(froyaOutput, /Rows: 2 valid speeds: 2 valid directions: 2/);
      assert.match(froyaOutput, /2009-11-18 15:00:00/);
      console.log('Frøya inspection fixture runs in browser Python.');
      const hourlyExample = windLesson.codes.find(cell => cell.code.includes('speed_complete') && cell.code.includes('direction_complete'));
      assert.ok(hourlyExample?.runnable, 'Frøya hourly aggregation example is available to learners');
      const hourlyOutput = await runCode(hourlyExample.code + '\nassert hourly["speed_complete"].tolist() == [True, False]\nassert hourly["direction_complete"].tolist() == [True, False]\nassert abs(hourly["mean_direction_deg"].iloc[0]) < 1e-8');
      assert.match(hourlyOutput, /Hours: 2 complete speed hours: 1 complete direction hours: 1/);
      console.log('Frøya hourly aggregation reports complete and partial coverage in browser Python.');
      const comparisonExample = windLesson.codes.find(cell => cell.code.includes('def compare_hourly_speeds('));
      assert.ok(comparisonExample?.runnable, 'Hourly observation-model comparison is available to learners');
      const comparisonOutput = await runCode(comparisonExample.code + '\nassert metrics["paired_hours"] == 2\nassert abs(metrics["bias_ms"]) < 1e-8\nassert abs(metrics["rmse_ms"] - 0.5) < 1e-8');
      assert.match(comparisonOutput, /paired_hours/);
      console.log('Hourly observation-model comparison runs with paired coverage in browser Python.');
      const xarrayExample = windLesson.codes.find(cell => cell.code.includes('valid_direction_samples'));
      assert.ok(xarrayExample?.runnable, 'Wind-data xarray example is available to learners');
      const xarrayOutput = await runCode(xarrayExample.code);
      assert.match(xarrayOutput, /valid_speed_samples/);
      assert.match(xarrayOutput, /valid_direction_samples/);
      assert.match(xarrayOutput, /mean_direction_deg/);
      console.log('Wind-data xarray direction and coverage example runs in browser Python.');
      const windWorkbook = course.pages.find(page => page.id === 'section7/workbook/wind.md');
      const cartopyExample = windWorkbook.codes.find(cell => cell.code.includes('ax.stock_img()') && cell.code.includes('quiverkey'));
      assert.ok(cartopyExample?.runnable, 'Wind vector map is available to learners');
      await navigateLesson(windWorkbook.id, windWorkbook.title);
      const cartopyOutput = await runCode(cartopyExample.code);
      assert.match(cartopyOutput, /Speed: 6\.32 m\/s/);
      const map = await send('Runtime.evaluate', {returnByValue:true,
        expression:"(() => {const images=[...document.querySelectorAll('#output img')]; return {count:images.length,last:images.at(-1)?.src};})()"});
      assert.ok(map.result.value.count >= 1, 'Wind vector map renders');
      assert.match(map.result.value.last, /^data:image\/png;base64,/);
      if (process.env.EAGE_CARTOPY_IMAGE) {
        fs.writeFileSync(process.env.EAGE_CARTOPY_IMAGE, Buffer.from(map.result.value.last.split(',')[1], 'base64'));
      }
      console.log('Wind workbook Cartopy vector map runs and renders in browser Python.');
      const solarQuiz = course.pages.find(page => page.id === 'section7/renewableEnergyquizzes/solarEnergy.md');
      const solarIndex = solarQuiz.codes.findIndex(cell => cell.code.includes('Grid points at the 10 MW cap'));
      assert.ok(solarIndex >= 0, 'Solar clipping answer is available to learners');
      const solarSetup = solarQuiz.codes.slice(0, solarIndex + 1).filter(cell => cell.runnable).map(cell => cell.code).join('\n\n');
      await navigateLesson(solarQuiz.id, solarQuiz.title);
      const solarOutput = await runCode(solarSetup);
      assert.match(solarOutput, /Grid points at the 10 MW cap: 679 of 4800/);
      const solarPlots = await send('Runtime.evaluate', {returnByValue:true,
        expression:"(() => {const plots=[...document.querySelectorAll('#output img')]; return {count:plots.length,last:plots.at(-1)?.src};})()"});
      assert.ok(solarPlots.result.value.count >= 1, 'Solar clipping answer renders a plot');
      assert.match(solarPlots.result.value.last, /^data:image\/png;base64,/);
      if (process.env.EAGE_SOLAR_CLIP_IMAGE) {
        fs.writeFileSync(process.env.EAGE_SOLAR_CLIP_IMAGE,
          Buffer.from(solarPlots.result.value.last.split(',')[1], 'base64'));
      }
      console.log('Solar clipping answer renders in browser Python with 679 capped grid points.');
      const finalProject = course.pages.find(page => page.id === 'section8/projectIntro.md');
      await navigateLesson(finalProject.id, finalProject.title);
      const projectFiles = await runCode(`from pathlib import Path
import csv
assert Path.cwd() == Path('/course/section8')
for name in ('solar', 'wind', 'hydro', 'geothermal', 'demand'):
    path = Path('data') / (name + '.csv')
    assert path.resolve() == Path('/course/section8/data') / (name + '.csv')
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 168 and all(row.get('timestamp') for row in rows), name
    print(name, len(rows))`);
      for (const name of ['solar', 'wind', 'hydro', 'geothermal', 'demand']) {
        assert.match(projectFiles, new RegExp(`${name} 168`));
      }
      console.log('All five final-project CSV files open at the documented browser paths.');
      await navigateLesson(windQuiz.id, windQuiz.title);
      await runCode("from pathlib import Path\nruntime_probe = 123\nPath('/course/section8/data/solar.csv').write_text('temporary modified asset')\nprint('modified')");
      const probe = "from pathlib import Path\nprint('VAR_PRESENT', 'runtime_probe' in globals())\nprint('ASSET', Path('/course/section8/data/solar.csv').read_text()[:30])";
      await navigateLesson('section7/workbook/hydroelectric.md', 'Hydroelectric');
      const switched = await runCode(probe);
      assert.match(switched, /VAR_PRESENT False/);
      assert.match(switched, /ASSET temporary modified asset/);
      await navigateLesson(windQuiz.id, windQuiz.title);
      const returned = await runCode(probe);
      assert.match(returned, /VAR_PRESENT False/);
      assert.match(returned, /ASSET temporary modified asset/);
      await send('Runtime.evaluate', {expression:"document.getElementById('reset-runtime').click()"});
      const restarted = await runCode(probe);
      assert.match(restarted, /VAR_PRESENT False/);
      assert.match(restarted, /ASSET timestamp,irradiance_wm2/);
      console.log('Lesson switching resets variables but retains runtime asset edits; restart reloads the supplied asset.');
    }
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/geothermal-lab/multifield.html'});
    let configured = false;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
        const scene = document.querySelector('.wp-scene'), pause = document.querySelector('[data-action="toggle"]');
        if (!scene || !pause) return false;
        pause.click();
        const key = k => scene.dispatchEvent(new KeyboardEvent('keydown', {key:k, bubbles:true}));
        key('Enter'); key('ArrowRight'); key('Enter');
        const depth = document.querySelector('[data-input="depth"]');
        depth.value = '2500'; depth.dispatchEvent(new Event('input', {bubbles:true}));
        document.querySelector('[data-input="mode"]').value = 'injector';
        key('ArrowRight'); key('Enter'); key('ArrowRight'); key('Enter');
        document.querySelector('[data-action="download"]').click();
        return true;
      })()`});
      if (result.result.value) { configured = true; break; }
      await delay(100);
    }
    assert.ok(configured, 'Geothermal field designer loaded');
    const csvFile = path.join(downloads, 'geothermal-multifield-results.csv');
    for (let i = 0; i < 150 && !fs.existsSync(csvFile); i++) await delay(100);
    assert.ok(fs.existsSync(csvFile), 'Edge completed the geothermal CSV download');
    const lines = fs.readFileSync(csvFile, 'utf8').trim().split(/\r?\n/);
    const headers = parseCsvLine(lines[0]);
    const values = parseCsvLine(lines.at(-1));
    const row = Object.fromEntries(headers.map((name, i) => [name, values[i]]));
    assert.equal(Number(row.active_depth_count), 2);
    assert.equal(Number(row.total_flow_kg_s), 2 * Number(row.rate_kg_s_per_active_depth));
    const streams = JSON.parse(row.producer_streams_json);
    const thermal = streams.reduce((sum, stream) => sum + Math.max(0, stream.weight * 4180 * (stream.temperature - Number(row.injection_C))) / 1e6, 0);
    assert.ok(Math.abs(thermal - Number(row.thermal_MW)) < 1e-10);
    const net = Math.min(Number(row.capacity_MW), thermal * Number(row.conversion_efficiency) * (1 - Number(row.auxiliary_fraction)));
    assert.ok(Math.abs(net - Number(row.net_power_MW)) < 1e-10);
    assert.equal(Number(row.additional_wells_drilled), 2);
    assert.equal(Number(row.cumulative_investment_USD), 8000000);
    assert.equal(Number(row.project_balance_USD), Number(row.revenue_USD) - Number(row.cumulative_investment_USD));
    console.log('Geothermal CSV completed in Edge; two-depth flow, power and balance replay successfully.');
    // Optional integration check: EAGE_TEST_WSL_PYTHON is the WSL Python
    // interpreter with pandas and NumPy installed.
    if (process.env.EAGE_TEST_WSL_PYTHON) {
      const notebookCheck = `import io, json, sys
import numpy as np, pandas as pd
payload=json.load(sys.stdin)
cell=next(''.join(c['source']) for c in payload['notebook']['cells'] if c['cell_type']=='code' and 'def compare_multidepth_csv' in ''.join(c['source']))
state={'np':np,'pd':pd}; exec(cell,state)
result=state['compare_multidepth_csv'](io.StringIO(payload['csv']))
assert result.max() < 1e-6, result.to_string()
print('NOTEBOOK_EXPORT_MATCH')`;
      const notebook = JSON.parse(fs.readFileSync('EAGE_PythonRenewableEnergyCourse/section8/geothermal-field-designer.ipynb', 'utf8'));
      const checked = spawnSync('wsl.exe', ['-d', process.env.EAGE_TEST_WSL_DISTRO || 'Ubuntu-24.04', '--',
        process.env.EAGE_TEST_WSL_PYTHON, '-c', notebookCheck],
        {input:JSON.stringify({notebook, csv:fs.readFileSync(csvFile, 'utf8')}), encoding:'utf8', timeout:30000});
      assert.equal(checked.status, 0, `Notebook rejected its browser export: ${checked.stderr || checked.error}`);
      assert.match(checked.stdout, /NOTEBOOK_EXPORT_MATCH/);
      console.log('Actual two-depth browser export passes the downloadable notebook comparison.');
    }
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/geothermal-lab/equations.html?demo=conversion'});
    let warning;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
        if (!document.getElementById('sink')) return null;
        for (const [id, value] of Object.entries({flow:50, reservoir:90, reinjection:80, sink:25, efficiency:20, parasitic:0})) {
          const input = document.getElementById(id); input.value = value; input.dispatchEvent(new Event('input', {bubbles:true}));
        }
        return document.getElementById('equations').textContent;
      })()`});
      warning = result.result.value;
      if (warning) break;
      await delay(100);
    }
    assert.match(warning, /Pₜₕ = 2\.09 MW/);
    assert.match(warning, /Pgross = ηPₜₕ = 0\.418 MW/);
    assert.match(warning, /Carnot upper bound = 17\.9%/);
    assert.match(warning, /efficiency exceeds this ideal Carnot limit/);
    const safe = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
      const input = document.getElementById('efficiency'); input.value = 12; input.dispatchEvent(new Event('input', {bubbles:true}));
      return document.getElementById('equations').textContent;
    })()`});
    assert.match(safe.result.value, /Passing this upper bound alone does not establish feasibility/);
    assert.doesNotMatch(safe.result.value, /efficiency exceeds this ideal Carnot limit/);
    const sinkBoundary = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
      const input = document.getElementById('reinjection'); input.value = 25; input.dispatchEvent(new Event('input', {bubbles:true}));
      return document.getElementById('equations').textContent;
    })()`});
    assert.match(sinkBoundary.result.value, /passive heat exchange with that sink cannot cool/);
    assert.match(sinkBoundary.result.value, /does not account for such work/);
    console.log('Geothermal conversion controls display the Carnot and return-temperature warnings.');
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/hydro-lab/worked-examples.html'});
    let hydro;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
        if (!document.getElementById('eco')) return null;
        for (const [id, value] of Object.entries({area:50, rain:400, coefficient:.1, meanhead:80, designhead:100, eco:3})) {
          const input = document.getElementById(id); input.value = value; input.dispatchEvent(new Event('input', {bubbles:true}));
        }
        return document.getElementById('out2').textContent;
      })()`});
      hydro = result.result.value;
      if (hydro) break;
      await delay(100);
    }
    assert.match(hydro, /Environmental flow served: 0\.06 m³\/s; shortfall: 2\.94 m³\/s/);
    assert.match(hydro, /main usable mean flow: 0\.00 m³\/s/);
    assert.match(hydro, /full-year energy 0\.34 GWh/);
    console.log('Hydropower low-inflow controls display the capped flow, shortfall and ecological energy.');
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/wind-lab/index.html'});
    let wind;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
        const root = document.getElementById('wind-lab-root')?.shadowRoot;
        if (!root?.getElementById('ratedMW')) return null;
        for (const [id, value] of Object.entries({speed:12, radius:30, ratedMW:4, ratedSpeed:12})) {
          const input = root.getElementById(id); input.value = value; input.dispatchEvent(new Event('input', {bubbles:true}));
        }
        return root.getElementById('wind-equations').textContent;
      })()`});
      wind = result.result.value;
      if (wind) break;
      await delay(100);
    }
    assert.match(wind, /at rated speed the electrical curve exceeds the ideal Betz rotor limit/);
    assert.match(wind, /current curve output also exceeds that ideal electrical bound/);
    const windSafe = await send('Runtime.evaluate', {returnByValue: true, expression: `(() => {
      const root = document.getElementById('wind-lab-root').shadowRoot, radius = root.getElementById('radius');
      radius.value = 100; radius.dispatchEvent(new Event('input', {bubbles:true}));
      return root.getElementById('wind-equations').textContent;
    })()`});
    assert.doesNotMatch(windSafe.result.value, /exceeds the ideal Betz rotor limit/);
    assert.doesNotMatch(windSafe.result.value, /current curve output also exceeds/);
    console.log('Wind explorer warns for the undersized rotor and clears the warning for a larger rotor.');
    const restoredLesson = 'section6/geothermalEnergy.md';
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(restoredLesson)});
    let restoredFrames;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        if (new URLSearchParams(location.hash.slice(1)).get('lesson') !== ${JSON.stringify(restoredLesson)}) return null;
        const names = ['systems', 'conduction', 'area', 'energy', 'performance'];
        const frames = [...document.querySelectorAll('#lesson iframe')].map(f => f.getAttribute('src'));
        return names.every(name => frames.some(src => src?.includes('closed-loop.html?demo=' + name))) ? names.length : null;
      })()`});
      restoredFrames = result.result.value; if (restoredFrames) break; await delay(100);
    }
    assert.equal(restoredFrames, 5, 'All restored geothermal views are mounted in the lesson');
    let embeddedSystems;
    for (let i = 0; i < 120; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        const frame = document.getElementById('geo-closed-systems')?.nextElementSibling;
        if (!frame || frame.tagName !== 'IFRAME') return null;
        frame.scrollIntoView({block:'start'});
        const root = frame.contentDocument?.getElementById('explorer');
        if (!root?.querySelector('[data-panel="coax"]')) return null;
        const height = root.getBoundingClientRect().height;
        return height > 0 && frame.clientHeight >= Math.ceil(height) ? {frame:frame.clientHeight, content:height} : null;
      })()`});
      embeddedSystems = result.result.value; if (embeddedSystems) break; await delay(100);
    }
    assert.ok(embeddedSystems, 'Embedded geothermal comparison expands to show its full content');
    console.log(`Embedded geothermal systems view fits its content (${embeddedSystems.frame}px frame).`);
    for (const anchor of ['geo-closed-systems', 'geo-closed-conduction', 'geo-closed-area', 'geo-closed-energy', 'geo-closed-performance',
      'geo-gradient-explorer', 'geo-heat-explorer', 'geo-carnot-explorer', 'geo-conversion-explorer', 'geo-energy-explorer', 'geo-decline-explorer']) {
      await send('Page.navigate', {url: 'about:blank'});
      await send('Page.navigate', {url: 'http://127.0.0.1:18343/#lesson=' + encodeURIComponent(restoredLesson) + '&anchor=' + anchor});
      await delay(7000); // Jupyter previously stole focus several seconds after navigation.
      let state;
      for (let i = 0; i < 120; i++) {
        const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
          const pane = document.getElementById('reading'), target = document.getElementById(${JSON.stringify(anchor)});
          if (!pane || !target) return null;
          const p = pane.getBoundingClientRect(), t = target.getBoundingClientRect();
          const scrollable = pane.scrollHeight > pane.clientHeight && getComputedStyle(pane).overflowY !== 'visible';
          return {top:t.top, paneTop:p.top, paneBottom:p.bottom, viewport:innerHeight, scrollable};
        })()`});
        state = result.result.value;
        if (state && state.top >= -1 && state.top < state.viewport &&
            (!state.scrollable || (state.top >= state.paneTop && state.top < state.paneBottom))) break;
        await delay(100);
      }
      assert.ok(state && state.top >= -1 && state.top < state.viewport &&
        (!state.scrollable || (state.top >= state.paneTop && state.top < state.paneBottom)),
        `Downloaded-notebook explorer anchor ${anchor} should be visible: ${JSON.stringify(state)}`);
    }
    console.log('All eleven geothermal notebook links open visible lesson anchors.');
    await send('Runtime.evaluate', {expression: "document.getElementById('workspace-notebook').scrollIntoView({block:'start', behavior:'instant'})"});
    let workspaceReady = false;
    for (let i = 0; i < 300; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        const frame = document.getElementById('workspace-notebook');
        return !!frame?.contentDocument?.querySelector('.jp-LabShell');
      })()`});
      if (result.result.value) { workspaceReady = true; break; }
      await delay(100);
    }
    assert.ok(workspaceReady, 'Jupyter workspace loads when brought into view');
    console.log('Lazy Jupyter workspace loads when brought into view.');
    for (const mode of ['systems', 'conduction', 'area', 'energy', 'performance']) {
      await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/geothermal-lab/closed-loop.html?demo=' + mode});
      let state;
      for (let i = 0; i < 120; i++) {
        const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
          const mode = ${JSON.stringify(mode)};
          if (mode === 'systems') {
            const checkbox = document.getElementById('egs-connection');
            if (!checkbox) return null;
            checkbox.checked = false; checkbox.dispatchEvent(new Event('change', {bubbles:true}));
            return {panels:document.querySelectorAll('[data-panel]').length, canvases:document.querySelectorAll('[data-panel] canvas').length, status:document.getElementById('egs-status').textContent};
          }
          if (mode === 'performance') {
            const tab = document.querySelector('[data-show="scenarios"]');
            if (!tab) return null; tab.click();
            const note = document.querySelector('[data-view="scenarios"] .loop-note').textContent;
            document.querySelector('[data-show="lcoe"]').click();
            return {note, cost:document.querySelector('[data-cost="rate"]').textContent};
          }
          const metrics = [...document.querySelectorAll('[data-metric]')].map(x => x.textContent);
          return metrics.length ? {metrics} : null;
        })()`});
        state = result.result.value; if (state) break; await delay(100);
      }
      assert.ok(state, `Restored geothermal explorer loads: ${mode}`);
      if (mode === 'systems') { assert.equal(state.panels, 4); assert.match(state.status, /interrupted paths stop circulation/); console.log(`Geothermal systems view mounted ${state.canvases} of 4 3D canvases.`); }
      if (mode === 'conduction') assert.match(state.metrics[0], /m²\/s/);
      if (mode === 'area') assert.equal(state.metrics[2], '1.8');
      if (mode === 'energy') { assert.match(state.metrics[0], /^15 MW/); assert.match(state.metrics[1], /^12 MW/); }
      if (mode === 'performance') { assert.match(state.note, /100 kg\/s database limit/); assert.match(state.cost, /\/MWh/); }
      if (process.env.EAGE_CLOSED_LOOP_SCREENSHOT && ['systems', 'performance'].includes(mode)) {
        const screenshot = await send('Page.captureScreenshot', {format:'png'});
        fs.writeFileSync(`${process.env.EAGE_CLOSED_LOOP_SCREENSHOT}-${mode}.png`, Buffer.from(screenshot.data, 'base64'));
      }
      console.log(`Restored geothermal ${mode} explorer loads and reports its expected controls.`);
    }
    await send('Page.navigate', {url: 'http://127.0.0.1:18343/generated/geothermal-lab/drilling.html'});
    let drillingReady = false;
    for (let i = 0; i < 100; i++) {
      const result = await send('Runtime.evaluate', {returnByValue:true,
        expression:`Boolean(document.querySelector('[data-input="particle"]'))`});
      if (result.result.value) { drillingReady = true; break; }
      await delay(100);
    }
    assert.ok(drillingReady, 'Drilling explorer loads');
    const drillingStatus = async controls => {
      const result = await send('Runtime.evaluate', {returnByValue:true, expression:`(() => {
        for (const [id, value] of Object.entries(${JSON.stringify(controls)})) {
          const input = document.querySelector('[data-input="' + id + '"]');
          input.value = value; input.dispatchEvent(new Event('input', {bubbles:true}));
        }
        return document.querySelector('[data-v="slip"]').textContent;
      })()`});
      return result.result.value;
    };
    const downward = await drillingStatus({flow:100, hole:.35, pipe:.08, density:1100, viscosity:.03, particle:.0009});
    assert.match(downward, /signed upward transport -0\.005 m\/s/);
    const captureDrilling = async label => {
      if (!process.env.EAGE_DRILLING_IMAGE_PREFIX) return;
      for (let frame = 0; frame < 2; frame++) {
        await delay(500);
        const screenshot = await send('Page.captureScreenshot', {format:'png'});
        fs.writeFileSync(`${process.env.EAGE_DRILLING_IMAGE_PREFIX}-${label}-${frame}.png`, Buffer.from(screenshot.data, 'base64'));
      }
    };
    await captureDrilling('down');
    const stopped = await drillingStatus({flow:0});
    assert.match(stopped, /signed upward transport -0\.023 m\/s/);
    const upward = await drillingStatus({flow:3000});
    assert.match(upward, /signed upward transport 0\./);
    await captureDrilling('up');
    const invalid = await drillingStatus({particle:.003, viscosity:.005});
    assert.match(invalid, /exceeds the Stokes creeping-flow range/);
    await captureDrilling('invalid');
    console.log('Drilling controls report downward, stopped-flow, upward and invalid-regime cases.');
  } finally {
    // Terminate this test's Edge process tree so background helpers release the profile.
    if (edge.exitCode === null) spawnSync('taskkill', ['/PID', String(edge.pid), '/T', '/F'], {windowsHide:true, stdio:'ignore'});
    const literalProfile = profile.replaceAll("'", "''");
    spawnSync('powershell.exe', ['-NoProfile', '-Command',
      `Get-CimInstance Win32_Process -Filter "name = 'msedge.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('${literalProfile}') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`],
      {windowsHide:true, stdio:'ignore', timeout:10000});
    if (edge.exitCode === null) await Promise.race([new Promise(resolve => edge.once('exit', resolve)), delay(5000)]);
    if (edge.exitCode === null) edge.kill();
    if (edge.exitCode === null) await Promise.race([new Promise(resolve => edge.once('exit', resolve)), delay(5000)]);
    socket?.close();
    try { fs.rmSync(temporary, {recursive: true, force: true, maxRetries: 2, retryDelay: 300}); }
    catch (error) { console.error(`Could not remove test profile ${temporary}: ${error.message}`); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
