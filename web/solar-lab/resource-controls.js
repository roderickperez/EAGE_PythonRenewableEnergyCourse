const $ = id => document.getElementById(id);

export function mountControls(definitions, update) {
  $('controls').innerHTML = definitions.map(([id, label, min, max, step, value]) => `
    <div class="control">
      <label for="${id}"><span>${label}</span><output for="${id}"></output></label>
      <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">
    </div>`).join('');
  $('controls').addEventListener('input', update);
}

export function showInput(input, text) {
  input.previousElementSibling.querySelector('output').value = text;
  input.setAttribute('aria-valuetext', text);
}

export function mountLegend(entries) {
  for (const [color, label] of entries) {
    const item = document.createElement('span');
    item.style.setProperty('--swatch', color);
    item.textContent = label;
    $('legend').append(item);
  }
}
