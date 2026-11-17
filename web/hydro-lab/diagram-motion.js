export function mountDiagramMotion(host, button, draw, rate = () => 0, transitionMs = 0) {
  let playing = true;
  let visible = true;
  let values, displayed, from, frame, previous;
  let transitionElapsed = 0;
  let phase = 0;

  function sync() {
    const canAnimate = playing && visible && !document.hidden && values && (from || rate(values) > 0);
    if (canAnimate && frame === undefined) frame = requestAnimationFrame(tick);
    if (!canAnimate) {
      if (frame !== undefined) cancelAnimationFrame(frame);
      frame = undefined;
      previous = undefined;
    }
  }
  function tick(time) {
    frame = undefined;
    const elapsed = previous === undefined ? 0 : Math.max(0, Math.min(45, time - previous));
    previous = time;
    phase = (phase + elapsed * rate(values)) % 1;
    if (from) {
      transitionElapsed += elapsed;
      const progress = Math.min(1, transitionElapsed / transitionMs);
      displayed = values.map((value, index) => from[index] + (value - from[index]) * (1 - (1 - progress) ** 3));
      if (progress === 1) from = undefined;
    }
    draw(displayed, phase);
    sync();
  }
  function setPlaying(value) {
    playing = value;
    if (!playing && from) {
      from = undefined;
      displayed = values;
      draw(displayed, phase);
    }
    button.textContent = playing ? 'Pause motion' : 'Play motion';
    button.setAttribute('aria-pressed', String(playing));
    sync();
  }

  button.onclick = () => setPlaying(!playing);
  document.addEventListener('visibilitychange', sync);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    sync();
  }).observe(host);
  setPlaying(playing);

  return next => {
    values = next;
    if (transitionMs && displayed && visible && playing && !document.hidden) {
      from = displayed;
      transitionElapsed = 0;
      previous = undefined;
    } else {
      from = undefined;
      displayed = values;
      draw(displayed, phase);
    }
    sync();
  };
}

export function mountDiagramRenderer(host, make3D, make2D) {
  let draw, latest;
  function fallback() {
    host.querySelector('canvas')?.remove();
    draw = make2D();
    if (latest) draw(...latest);
  }
  if (make3D) {
    try {
      draw = make3D();
      host.addEventListener('webglcontextlost', fallback, { capture: true, once: true });
    } catch {
      fallback();
    }
  } else {
    fallback();
  }
  return (values, phase) => {
    latest = [values, phase];
    draw(values, phase);
  };
}
