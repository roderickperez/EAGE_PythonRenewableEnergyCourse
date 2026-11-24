// The course embeds these panels on the same origin. Size to the content rather
// than the iframe viewport so panels can both grow and shrink when their width changes.
export function fitFrameToContent(content) {
  let frame;
  try { frame = window.frameElement; } catch { return; }
  if (!frame || !content) return;
  const resize = () => {
    if (!frame.isConnected) return;
    const bottomMargin = parseFloat(getComputedStyle(document.body).marginBottom) || 0;
    const height = Math.ceil(content.getBoundingClientRect().bottom + window.scrollY + bottomMargin);
    if (height > 0 && frame.style.height !== `${height}px`) frame.style.height = `${height}px`;
  };
  const observer = new ResizeObserver(resize);
  observer.observe(content);
  window.addEventListener('resize', resize);
  resize();
}
