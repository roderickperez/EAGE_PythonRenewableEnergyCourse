// Renew the existing private-course session before protected assets expire.
(() => {
  if (location.hostname !== 'roderickperezaltamar.com' ||
      !location.pathname.startsWith('/private-course/python-renewable/')) return;

  let clientPromise, pending, warning;
  const retriedImages = new WeakSet();

  function showWarning() {
    if (warning) return;
    warning = document.createElement('div');
    warning.className = 'session-warning';
    warning.setAttribute('role', 'status');
    warning.append('Your course session needs renewal. Saved drafts remain in this browser. ');
    const link = document.createElement('a');
    link.href = '/courses/python-renewable-learn.html';
    link.textContent = 'Sign in and reopen the course';
    warning.append(link);
    document.body.prepend(warning);
  }

  function clearWarning() { warning?.remove(); warning = null; }
  async function updateCookie(session) {
    if (!session?.access_token) return false;
    const response = await fetch('/api/course-session', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: session.access_token }), cache: 'no-store'
    });
    return response.ok;
  }
  function getClient() {
    if (!clientPromise) clientPromise = (async () => {
      if (!window.supabase) {
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2';
          script.onload = resolve;
          script.onerror = reject;
          document.head.append(script);
        });
      }
      const response = await fetch('/api/config', { cache: 'no-store' });
      if (!response.ok || !window.supabase) throw new Error('Authentication unavailable');
      const config = await response.json();
      if (!config.url || !config.key) throw new Error('Authentication unavailable');
      const client = window.supabase.createClient(config.url, config.key);
      client.auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_OUT') showWarning();
        else if (event === 'TOKEN_REFRESHED' && session) {
          setTimeout(() => updateCookie(session).then(ok => ok ? clearWarning() : showWarning()).catch(showWarning), 0);
        }
      });
      return client;
    })().catch(error => { clientPromise = undefined; throw error; });
    return clientPromise;
  }
  window.refreshCourseSession = () => {
    if (pending) return pending;
    pending = (async () => {
      try {
        const client = await getClient();
        let { data, error } = await client.auth.getSession();
        if (error || !data.session) throw new Error('No active session');
        if (data.session.expires_at && data.session.expires_at * 1000 < Date.now() + 120000) {
          ({ data, error } = await client.auth.refreshSession());
          if (error || !data.session) throw new Error('Session expired');
        }
        if (!await updateCookie(data.session)) throw new Error('Session renewal failed');
        clearWarning();
        return true;
      } catch (_) {
        try {
          const probe = await fetch('generated/course.json', { method: 'HEAD', cache: 'no-store' });
          if (probe.ok) { clearWarning(); return true; }
        } catch (_) { /* The protected course cannot be reached. */ }
        showWarning();
        return false;
      }
    })().finally(() => { pending = null; });
    return pending;
  };

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void window.refreshCourseSession();
  });
  document.addEventListener('error', event => {
    const image = event.target;
    if (!(image instanceof HTMLImageElement) || retriedImages.has(image)) return;
    const url = new URL(image.src, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith('/private-course/python-renewable/')) return;
    retriedImages.add(image);
    void window.refreshCourseSession().then(ok => {
      if (ok && image.isConnected) { url.searchParams.set('imageRetry', Date.now()); image.src = url.href; }
    });
  }, true);
  setInterval(() => void window.refreshCourseSession(), 15 * 60 * 1000);
})();
