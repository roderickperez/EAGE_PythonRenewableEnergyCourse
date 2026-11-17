/* Read the course's theory sections using the visitor's browser voice. */
(() => {
  const pages = /\/(?:section0\/introrenewableenergy|section6\/(?:hydroelectricenergy|solarenergy|windenergy|geothermalenergy))\/?$/i;
  const supported = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  let active = null;
  let generation = 0;
  let currentPath = location.pathname;

  function stop() {
    generation++;
    if (supported) window.speechSynthesis.cancel();
    if (active) {
      active.button.textContent = '▶ Listen';
      active.button.setAttribute('aria-label', `Listen to ${active.title}`);
      active.status.textContent = '';
      active = null;
    }
  }

  function sectionText(heading) {
    const level = Number(heading.tagName[1]);
    const pieces = [heading.querySelector('.heading-text')?.textContent || heading.textContent];
    for (let node = heading.nextElementSibling; node; node = node.nextElementSibling) {
      if (node.id === 'references') break;
      if (/^H[1-6]$/.test(node.tagName) && Number(node.tagName[1]) <= level) break;
      if (node.classList.contains('section-audio-control')) continue;
      if (/^H[1-6]$/.test(node.tagName)) {
        pieces.push(node.querySelector('.heading-text')?.textContent || node.textContent);
        continue;
      }
      const ignored = 'pre, code, .math, .katex, mjx-container, script, style, button, iframe, svg, .section-audio-control, details:not([open])';
      if (node.matches(ignored)) continue;
      const copy = node.cloneNode(true);
      copy.querySelectorAll(ignored).forEach(el => el.remove());
      const text = copy.textContent.replace(/\s+/g, ' ').trim();
      if (text) pieces.push(text);
    }
    return pieces.join('. ').replace(/\s+/g, ' ').trim();
  }

  function play(heading, button, status, title) {
    if (active?.button === button) { stop(); return; }
    stop();
    const text = sectionText(heading);
    // Keep utterances short enough for browsers that truncate long speech.
    const chunks = text.match(/.{1,180}(?:[.!?](?:\s|$)|\s|$)|\S+/g) || [text];
    const token = generation;
    active = { button, status, title };
    button.textContent = '■ Stop';
    button.setAttribute('aria-label', `Stop reading ${title}`);
    status.textContent = 'Reading section';
    let index = 0;
    function next() {
      if (token !== generation || !active) return;
      if (index >= chunks.length) { stop(); return; }
      const utterance = new SpeechSynthesisUtterance(chunks[index++].trim());
      utterance.onend = next;
      utterance.onerror = () => {
        if (token !== generation) return;
        stop();
        status.textContent = 'Audio could not be played. Check your browser and sound settings.';
      };
      window.speechSynthesis.speak(utterance);
    }
    next();
  }

  function refresh() {
    if (currentPath !== location.pathname) {
      stop();
      currentPath = location.pathname;
    }
    if (!pages.test(location.pathname)) return;
    document.querySelectorAll('article.content h2, article.content h3, article.content h4').forEach(heading => {
      if (['head-flow-and-the-hydraulic-energy-balance', 'water-allocation-hydrology-and-storage',
        'solar-resource-and-measurement-definitions', 'pv-conversion-definitions-equations-and-limits'].includes(heading.id)) return;
      if (heading.nextElementSibling?.classList.contains('section-audio-control')) return;
      const title = (heading.querySelector('.heading-text')?.textContent || heading.textContent).trim();
      const control = document.createElement('div');
      control.className = 'section-audio-control';
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = '▶ Listen';
      button.setAttribute('aria-label', `Listen to ${title}`);
      const status = document.createElement('span');
      status.className = 'section-audio-status';
      status.setAttribute('role', 'status');
      if (!supported) {
        button.disabled = true;
        status.textContent = 'Speech is unavailable in this browser.';
      } else {
        button.addEventListener('click', () => play(heading, button, status, title));
      }
      control.append(button, status);
      heading.after(control);
    });
  }

  new MutationObserver(refresh).observe(document.body, { childList: true, subtree: true });
  window.addEventListener('pagehide', stop);
  refresh();
})();
