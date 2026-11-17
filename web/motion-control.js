const play = '<path d="M7 4.5v15l12-7.5z"/>';
const stop = '<rect x="6" y="6" width="12" height="12" rx="1.2"/>';

export function updateMotionControl(button, playing, subject = 'animation') {
  const label = `${playing ? 'Stop' : 'Play'} ${subject}`;
  if (button.title !== label) {
    button.innerHTML = `<svg viewBox="0 0 24 24" width="19" height="19" fill="currentColor" aria-hidden="true" focusable="false">${playing ? stop : play}</svg>`;
    button.title = label;
    button.setAttribute('aria-label', label);
  }
  button.style.width = '40px';
  button.style.height = '40px';
  button.style.padding = '0';
  button.style.lineHeight = '1';
  button.firstElementChild.style.verticalAlign = 'middle';
}
