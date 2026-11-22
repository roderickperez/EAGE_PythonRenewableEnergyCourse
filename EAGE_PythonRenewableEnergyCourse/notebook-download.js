// MyST treats static notebook links as ordinary links; some hosts serve them as JSON.
// Set the download attribute at activation time, including after client-side navigation.
document.addEventListener('click', (event) => {
  const anchor = event.target instanceof Element ? event.target.closest('a[href]') : null;
  if (!anchor) return;
  const url = new URL(anchor.href, window.location.href);
  if (url.origin !== window.location.origin || !url.pathname.toLowerCase().endsWith('.ipynb')) return;
  const hashedName = decodeURIComponent(url.pathname.split('/').pop());
  const stagedName = hashedName.replace(/-[0-9a-f]{32}(?=\.ipynb$)/i, '');
  // MyST truncates long source stems before appending the hash.
  const sourceNames = {
    'world-energy-downloa.ipynb': 'world-energy-download.ipynb',
    'geothermal-field-des.ipynb': 'geothermal-field-designer.ipynb',
    'timeSeriesEnergyCons.ipynb': 'timeSeriesEnergyConsumption.ipynb',
  };
  anchor.download = sourceNames[stagedName] || stagedName;
  anchor.removeAttribute('target');
}, true);
