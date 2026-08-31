import json
import pathlib
import re

root = pathlib.Path(__file__).parent


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text, pattern, replacement, label, flags=re.S):
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


index_path = root / "index.html"
text = index_path.read_text(encoding="utf-8")

# Product identity and static metadata.
text = text.replace('window.PYTHONLABBET_VERSION = "1.38";', 'window.PYTHONLABBET_VERSION = "1.0";')
text = text.replace('<title>Pythonlab</title>', '<title>EAGE Python SandBox</title>')
text = text.replace('content="Pythonlab"', 'content="EAGE Python SandBox"')
text = text.replace('content="Learn Python in your browser — 29 chapters from print() to transformers."',
                    'content="An interactive Python sandbox for EAGE renewable-energy training."')

# Remove CSS used only by the deleted toggles, 3D renderer, dialogs and AI chat.
text = regex_once(text, r'  #app-root \.strombrytare \{.*?(?=  #app-root \.status \{)', '', 'toggle CSS')
text = regex_once(text, r'  #app-root button\.stoppa \{.*?(?=  #app-root\.kor-lage \.lektion:disabled,)', '', 'stop button CSS')
text = regex_once(
    text,
    r'  #app-root\.kor-lage \.lektion:disabled,.*?  #app-root button\.stoppa:disabled \{.*?\n  \}\n',
    '  #app-root.kor-lage .lektion:disabled,\n  #app-root.kor-lage .verktyg .knapp:disabled {\n'
    '    opacity: .38;\n    cursor: not-allowed;\n  }\n',
    'run lock CSS',
)
text = regex_once(text, r'  #app-root \.scenruta \{.*?(?=  /\* ---- dialogruta)', '', '3D CSS')
text = regex_once(text, r'  /\* ---- dialogruta för anslutning ---- \*/.*?(?=  #app-root \.konsol \.tips)', '', 'AI/dialog CSS')

# Header, tabs and panels.
text = regex_once(
    text,
    r'  <label class="strombrytare" data-i18n-title="ui\.felsakert_info">.*?'
    r'  <button class="omlank" id="om-knapp" data-i18n="ui\.om"></button>',
    '  <a class="omlank" href="https://eage.org/" target="_blank" rel="noopener" data-i18n="ui.om"></a>',
    'header controls',
)
text = text.replace('        <button class="knapp stoppa" id="stoppa-knapp" hidden data-i18n="ui.stoppa"></button>\n', '')
text = text.replace('        <button class="flik" id="flik-scen" data-i18n="ui.flik_scen"></button>\n', '')
text = text.replace('        <button class="flik" id="flik-ai" data-i18n="ui.flik_ai"></button>\n', '')
text = text.replace('        <button class="knapp" id="aterstall-knapp" hidden data-i18n="ord.aterstall_vy" style="margin-left:auto; font-size:12px; padding:4px 10px;"></button>\n', '')
text = text.replace('        <button class="knapp" id="helskarm-knapp" hidden data-i18n="ord.helskarm" style="font-size:12px; padding:4px 10px;"></button>\n', '')
text = regex_once(text, r'      <div class="flikinnehall" id="panel-scen" hidden>.*?      </div>\n    </div>', '    </div>', '3D and AI panels')
text = regex_once(text, r'\n<div class="overlagring" id="om-overlagring" hidden>.*?(?=\n<script>)', '\n', 'dialogs')

# Locale HTML injection and runtime element references.
text = text.replace('    fyllHtml("om-dialog", SPRAK.om);\n', '')
text = text.replace('    fyllHtml("ai-dialog", SPRAK.ai_dialog);\n', '')
for line in [
    '  var flikScen = document.getElementById("flik-scen");\n',
    '  var panelScen = document.getElementById("panel-scen");\n',
    '  var scenruta = document.getElementById("scenruta");\n',
    '  var helskarmKnapp = document.getElementById("helskarm-knapp");\n',
    '  var aterstallKnapp = document.getElementById("aterstall-knapp");\n',
    '  var stoppaKnapp = document.getElementById("stoppa-knapp");\n',
    '  var felsakertToggle = document.getElementById("felsakert-toggle");\n',
    '  var flikAi = document.getElementById("flik-ai");\n',
    '  var panelAi = document.getElementById("panel-ai");\n',
    '  var aiToggle = document.getElementById("ai-toggle");\n',
    '  var forberedKod = null;\n',
    '  var nollstallVakt = null;\n',
    '  var raknaVakter = null;\n',
]:
    text = text.replace(line, '')

# Keep only the lesson and output tabs.
text = regex_once(
    text,
    r'  function visaFlik\(vilken\) \{.*?\n  \}\n\n  /\* Knappen visas.*?'
    r'  flikScen\.addEventListener\("click", function \(\) \{ visaFlik\("scen"\); \}\);\n',
    '  function visaFlik(vilken) {\n'
    '    var paneler = {\n'
    '      lektion: [panelLektion, flikLektion],\n'
    '      utskrift: [panelUtskrift, flikUtskrift]\n'
    '    };\n'
    '    Object.keys(paneler).forEach(function (namn) {\n'
    '      paneler[namn][0].hidden = namn !== vilken;\n'
    '      paneler[namn][1].classList.toggle("vald", namn === vilken);\n'
    '    });\n'
    '  }\n'
    '  flikLektion.addEventListener("click", function () { visaFlik("lektion"); });\n'
    '  flikUtskrift.addEventListener("click", function () { visaFlik("utskrift"); });\n',
    'tab runtime',
)

# Remove the Three.js renderer and the AI client implementation.
text = regex_once(text, r'  /\* =+\n     3D-scen via three\.js.*?(?=  function markeraSteg)', '', 'Three.js implementation')
text = regex_once(text, r'  /\* =+\n     AI-hjälp.*?  sattAiStod\(false\);\n', '', 'AI implementation')
text = regex_once(text, r'    if \(ai\.pa\) \{.*?    \}\n    konsol\.scrollTop', '    konsol.scrollTop', 'AI error button')
text = text.replace('    byteAvSamtal();\n', '')
text = text.replace('    sistaMarkering = { start: 0, slut: 0 };\n', '')

# Number lessons by their explicit number so the empty project remains chapter 18.
text = text.replace('.replace("{nr}", i + 1).replace("{titel}", lek.titel)',
                    '.replace("{nr}", lek.nr || i + 1).replace("{titel}", lek.titel)')
text = text.replace('nr.textContent = String(i + 1).padStart(2, "0");',
                    'nr.textContent = String(lek.nr || i + 1).padStart(2, "0");')

# Run-state locking no longer references deleted controls.
text = regex_once(
    text,
    r'  /\* Medan koden kör.*?  function lasGranssnitt\(las\) \{.*?\n  \}\n',
    '  var LASTA = ["ny-knapp", "aterstall-kapitel", "oppna-knapp", "spara-knapp",\n'
    '               "spara-som-knapp", "start-knapp"];\n\n'
    '  function lasGranssnitt(las) {\n'
    '    LASTA.forEach(function (id) {\n'
    '      var el = document.getElementById(id);\n'
    '      if (el) el.disabled = las;\n'
    '    });\n'
    '    Array.prototype.forEach.call(\n'
    '      document.querySelectorAll("#lektionslista .lektion"), function (b) {\n'
    '        b.disabled = las;\n'
    '      });\n'
    '    var app = document.getElementById("app-root");\n'
    '    if (app) app.classList.toggle("kor-lage", las);\n'
    '  }\n',
    'run locking',
)

# Execute through the async bridge, without the failsafe loop guard.
text = regex_once(
    text,
    r'      var vakt = felsakertToggle\.checked;\n      var kallkod = kodruta\.value;.*?'
    r'      if \(korProgram\) \{\n        await korProgram\(kallkod, vakt\);\n      \} else \{',
    '      var kallkod = kodruta.value;\n\n'
    '      if (syntaxkoll) {\n'
    '        var syntaxfel = syntaxkoll(kallkod);\n'
    '        if (syntaxfel) {\n'
    '          visaSyntaxfel(syntaxfel.toJs ? syntaxfel.toJs() : syntaxfel);\n'
    '          return;\n'
    '        }\n'
    '      }\n\n'
    '      if (korProgram) {\n'
    '        await korProgram(kallkod);\n'
    '      } else {',
    'failsafe execution',
)
text = regex_once(
    text,
    r'    \} finally \{\n      korKnapp\.disabled = false;.*?      stoppaKnapp\.textContent = SPRAK\.ui\.stoppa;\n    \}',
    '    } finally {\n'
    '      korKnapp.disabled = false;\n'
    '      lasGranssnitt(false);\n'
    '      avbrytAllVantan();\n'
    '    }',
    'run cleanup',
)

# About is now a direct EAGE link; remove modal and failsafe listeners.
text = regex_once(text, r'\n  var omOverlagring = .*?(?=\n  document\.getElementById\("kor-knapp"\))', '\n', 'about modal runtime')
text = regex_once(
    text,
    r'  stoppaKnapp\.addEventListener\("click".*?  \}\);\n  felsakertToggle\.addEventListener\("change".*?  \}\);\n',
    '',
    'stop and failsafe listeners',
)

# Simplify the waiting registry now that there is no Stop button.
text = text.replace('      uppdateraStoppKnapp();\n', '')
text = text.replace('    uppdateraStoppKnapp();\n', '')
text = text.replace('  /* Sätts av Stoppa-knappen och läses av loopvakten i Python. */\n  window._labbStopp = false;\n\n', '')
text = regex_once(text, r'  /\* Ger Python en avbrytbar hämtning:.*?  \};\n\n', '', 'abort controller bridge')

# Python bridge: retain async input/data support, remove loop instrumentation and scene module.
text = text.replace("        if js.window._labbStopp:\n            raise KeyboardInterrupt(_T['stopp_svar'])\n", '')
text = regex_once(
    text,
    r'import asyncio as _asyncio\nimport time as _time\n.*?(?=def _labb_generator)',
    'async def _labb_vanta(loften):\n    return await loften\n\n\n',
    'Python failsafe helpers',
)
text = text.replace('async def _labb_kor(kallkod, vakt=True):', 'async def _labb_kor(kallkod):')
text = text.replace('    trad = _labb_trad(kallkod, vakt)', '    trad = _labb_trad(kallkod)')
text = regex_once(text, r'\ndef _labb_antal_vakter\(.*?(?=\ndef _labb_trad)', '\n', 'failsafe helper functions')
text = text.replace('def _labb_trad(kallkod, vakt=True):', 'def _labb_trad(kallkod):')
text = regex_once(text, r'\n    if vakt:\n        class Loopvakt.*?Loopvakt\(\)\.visit\(trad\)\)\n', '\n', 'AST loop guard')
text = text.replace("    if vakt:\n        vantar.add('_labb_kontroll')\n", '')
text = regex_once(text, r'_scen_modul = _types\.ModuleType\(\'scene\'\).*?_sys\.modules\[\'scene\'\] = _scen_modul\n\n', '', 'Python scene module')
text = regex_once(
    text,
    r'async def _hamta_text\(url\):.*?    return await answer\.string\(\)',
    "async def _hamta_text(url):\n"
    "    from pyodide.http import pyfetch\n"
    "    try:\n"
    "        answer = await _labb_vanta(pyfetch(url))\n"
    "    except Exception as error:\n"
    "        raise RuntimeError(\n"
    "            _T['cors'].replace('{url}', str(url)).replace('{error}', str(error))\n"
    "        ) from None\n"
    "    if answer.status != 200:\n"
    "        raise RuntimeError(_T['statuskod'].replace('{code}', str(answer.status)))\n"
    "    return await answer.string()",
    'data fetch without stop bridge',
)
text = text.replace("        if js.window._labbStopp:\n            raise KeyboardInterrupt(_T['stopp_fil'])\n", '')
text = text.replace('    forberedKod = pyodide.globals.get("_labb_forbered");\n', '')
text = text.replace('    nollstallVakt = pyodide.globals.get("_labb_nollstall");\n', '')
text = text.replace('    raknaVakter = pyodide.globals.get("_labb_antal_vakter");\n', '')
text = regex_once(text, r'\n    laddaThree\(\)\.then\(function \(ok\) \{.*?    \}\);', '', 'Three.js startup')

index_path.write_text(text, encoding="utf-8")


def load_locale(path, code):
    source = path.read_text(encoding="utf-8")
    # Evaluate the data-only locale file with Node would be overkill; extract it with a
    # tiny JS runner written alongside this script by serialising through JSON in Node.
    import subprocess
    runner = (
        "const fs=require('fs'),vm=require('vm');"
        "const c={window:{}};c.window.PYTHONLAB_SPRAK={};"
        f"vm.runInNewContext(fs.readFileSync({json.dumps(str(path))},'utf8'),c);"
        f"process.stdout.write(JSON.stringify(c.window.PYTHONLAB_SPRAK[{json.dumps(code)}]));"
    )
    return json.loads(subprocess.check_output(['node.exe', '-e', runner], text=True))


def write_locale(path, code, locale):
    kept_ui = ['nytt_filnamn', 'ny', 'aterstall', 'oppna', 'spara', 'spara_som', 'kor',
               'om', 'start', 'kapitel_rubrik', 'flik_kapitel', 'flik_utskrift']
    kept_ord = ['tips', 'kapitel_rubrik', 'rensa']
    kept_msg = ['syntaxfel', 'hamtning_misslyckades', 'tidsgrans', 'sparad_hamtade',
                'sparad_till', 'oppnade', 'paketfel', 'matplotlibfel', 'ingen_utskrift',
                'klart', 'forsoker_hamta', 'gick_inte', 'ingen_kalla', 'redo']
    kept_py = ['cors', 'statuskod', 'avbrot_input', 'ingen_fil', 'saknas', 'tom_lista',
               'laddade_ner', 'laste_in', 'inga_filer']
    slim = {
        'kod': code,
        'namn': locale['namn'],
        'html_lang': code,
        'titel': 'EAGE Python SandBox',
        'beskrivning': ('An interactive Python sandbox for EAGE renewable-energy training.' if code == 'en'
                        else 'Un entorno interactivo de Python para la formación de EAGE en energías renovables.'),
        'ui': {key: locale['ui'][key] for key in kept_ui},
        'steg': locale['steg'],
        'konsol': locale['konsol'],
        'ord': {key: locale['ord'][key] for key in kept_ord},
        'status': locale['status'],
        'ladd': locale['ladd'],
        'msg': {key: locale['msg'][key] for key in kept_msg},
        'fraga': locale['fraga'],
        'feltips': locale['feltips'],
        'py': {key: locale['py'][key] for key in kept_py},
        'startkod': locale['startkod'],
    }
    slim['ui']['om'] = 'About EAGE' if code == 'en' else 'Acerca de EAGE'
    slim['kapitel'] = locale['kapitel'][:13]
    slim['kapitel'].append({
        'nr': 18,
        'del': 'Projects' if code == 'en' else 'Proyectos',
        'titel': 'Solar Energy' if code == 'en' else 'Energía solar',
        'fil': '18_solar_energy.py',
        'kod': '',
        'forklaring': '',
    })
    slim['start'] = ("""
      <p class="valkomst">Welcome to EAGE Python SandBox</p>
      <p class="ingress">Run real Python directly in your browser. The sandbox covers Python fundamentals, collections and program structure, with a clean project space for Solar Energy.</p>
      <div class="snabbstart">
        <strong>Get started</strong>
        <ol>
          <li>Choose a chapter from the list on the left.</li>
          <li>Read the explanation and inspect the code.</li>
          <li>Press <kbd>Run</kbd> or <kbd>Ctrl</kbd> + <kbd>Enter</kbd>.</li>
          <li>Change the code and run it again.</li>
        </ol>
      </div>
      <h3>Course sections</h3>
      <p><strong>Foundations:</strong> output, variables, input, conditions and loops.</p>
      <p><strong>Collections:</strong> lists, strings, dictionaries, files and databases.</p>
      <p><strong>Structure:</strong> functions, error handling, modules and classes.</p>
      <p><strong>Projects:</strong> an empty Solar Energy workspace ready for course material.</p>
    """ if code == 'en' else """
      <p class="valkomst">Bienvenido a EAGE Python SandBox</p>
      <p class="ingress">Ejecute Python real directamente en el navegador. El entorno cubre fundamentos, colecciones y estructura de programas, e incluye un espacio de proyecto vacío para Energía solar.</p>
      <div class="snabbstart">
        <strong>Primeros pasos</strong>
        <ol>
          <li>Elija un capítulo de la lista de la izquierda.</li>
          <li>Lea la explicación y revise el código.</li>
          <li>Pulse <kbd>Ejecutar</kbd> o <kbd>Ctrl</kbd> + <kbd>Enter</kbd>.</li>
          <li>Modifique el código y vuelva a ejecutarlo.</li>
        </ol>
      </div>
      <h3>Secciones del curso</h3>
      <p><strong>Fundamentos:</strong> salida, variables, entrada, condiciones y bucles.</p>
      <p><strong>Colecciones:</strong> listas, cadenas, diccionarios, archivos y bases de datos.</p>
      <p><strong>Estructura:</strong> funciones, gestión de errores, módulos y clases.</p>
      <p><strong>Proyectos:</strong> un espacio vacío de Energía solar preparado para el material del curso.</p>
    """)
    content = (
        f"/* EAGE Python SandBox — {'English' if code == 'en' else 'textos en español'}. */\n"
        "window.PYTHONLAB_SPRAK = window.PYTHONLAB_SPRAK || {};\n"
        f"window.PYTHONLAB_SPRAK.{code} = "
        + json.dumps(slim, ensure_ascii=False, indent=2)
        + ";\n"
    )
    path.write_text(content, encoding='utf-8')


for code in ('en', 'es'):
    locale_path = root / f'sprak-{code}.js'
    write_locale(locale_path, code, load_locale(locale_path, code))
