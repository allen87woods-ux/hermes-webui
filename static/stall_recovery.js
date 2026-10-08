/* stall_recovery.js — client-side stall detection + auto-recovery.
 *
 * Why this exists (2026-10-08, webui fork patch #16):
 * The phone reaches the WebUI over the tailnet (tailscale serve -> loopback).
 * When that tunnel route dies silently, TCP gives no RST: fetch() requests hang
 * and the EventSource streams stay readyState=OPEN forever. The page therefore
 * looks "frozen" with no error, and the only recovery was a manual reload.
 * Allen hit this repeatedly through the workday and it broke his workflow.
 *
 * What this does: watches app liveness (every fetch + every SSE event), and when
 * the app goes quiet WITH evidence of a broken path (an errored or >15s-pending
 * request), it waits for the server to answer again and then reloads the page
 * once. The session id lives in the URL, so the reload lands back in the same
 * session.
 *
 * Deliberately conservative, because a wrong reload interrupts a live turn:
 *   1. A quiet page is NOT enough. Reload needs an errored/hung request too,
 *      so a long silent generation (no tokens for a minute) never triggers it.
 *   2. The reload only happens AFTER a probe proves the server is reachable, so
 *      the browser never lands on its own network-error page.
 *   3. The app gets GRACE_MS to resume on its own first; if it does, we stand down.
 *   4. At most one auto-reload per COOLDOWN_MS.
 *   5. Hidden tabs are ignored (backgrounded app = legitimately quiet).
 *
 * Every stall and reload is reported to POST /api/client-events/log via the
 * app's own recordClientSSEError(), so episodes show up in the webui journal
 * instead of only being felt by the user.
 */
(function () {
  if (window.__hermesStallRecovery) return;

  var IDLE_MS = 25000;       // no successful app response this long = quiet
  var TICK_MS = 5000;        // detector cadence
  var HANG_MS = 15000;       // a request pending this long counts as broken-path evidence
  var GRACE_MS = 4000;       // time the app gets to resume once the server answers
  var COOLDOWN_MS = 120000;  // max one auto-reload per 2 minutes
  var PROBE_URL = 'static/pwa-startup.js';

  var state = {
    version: '2026-10-08',
    stalls: 0,
    reloads: 0,
    lastStallAt: null,
    lastStallSeconds: null,
    lastReloadAt: null
  };
  window.__hermesStallRecovery = state;

  var lastAppOk = Date.now();
  var lastBroken = 0;      // last errored/hung app request
  var stallStart = 0;
  var lastReloadAt = 0;
  var origFetch = window.fetch ? window.fetch.bind(window) : null;

  function markOk() {
    lastAppOk = Date.now();
    stallStart = 0;
  }

  function report(reason, extra) {
    try {
      var details = {
        reason: reason,
        session_id: (window.S && window.S.session && window.S.session.session_id) || null,
        visibility_state: (typeof document !== 'undefined' && document.visibilityState) || 'unknown'
      };
      if (extra) {
        for (var k in extra) {
          if (Object.prototype.hasOwnProperty.call(extra, k)) details[k] = extra[k];
        }
      }
      if (typeof window.recordClientSSEError === 'function') {
        window.recordClientSSEError('stall-recovery', details);
      }
    } catch (e) { /* never break the page for telemetry */ }
  }

  // ---- liveness tap: fetch ----
  if (origFetch) {
    window.fetch = function () {
      var p = origFetch.apply(null, arguments);
      try {
        if (p && typeof p.then === 'function') {
          var settled = false;
          var hangTimer = setTimeout(function () {
            if (settled) return;
            lastBroken = Date.now();   // still pending after HANG_MS => broken path
          }, HANG_MS);
          p.then(function (r) {
            settled = true; clearTimeout(hangTimer); markOk(); return r;
          }, function () {
            settled = true; clearTimeout(hangTimer); lastBroken = Date.now();
            return null;
          });
        }
      } catch (e) { /* ignore */ }
      return p;
    };
  }

  // ---- liveness tap: EventSource (all app streams) ----
  var NativeES = window.EventSource;
  if (NativeES) {
    var WrappedES = function (url, opts) {
      var es = new NativeES(url, opts);
      try {
        ['open', 'message', 'ping', 'heartbeat', 'token', 'done', 'ready', 'update',
         'assistant', 'tool', 'status'].forEach(function (ev) {
          es.addEventListener(ev, markOk);
        });
      } catch (e) { /* ignore */ }
      return es;
    };
    WrappedES.prototype = NativeES.prototype;
    try {
      Object.defineProperty(WrappedES, 'CONNECTING', { value: NativeES.CONNECTING });
      Object.defineProperty(WrappedES, 'OPEN', { value: NativeES.OPEN });
      Object.defineProperty(WrappedES, 'CLOSED', { value: NativeES.CLOSED });
    } catch (e) { /* ignore */ }
    window.EventSource = WrappedES;
  }

  // ---- probe: is the server reachable right now? (bypasses the tap) ----
  function probe() {
    if (!origFetch) return Promise.reject(new Error('no fetch'));
    return origFetch(PROBE_URL + '?stall_probe=' + Date.now(),
                     { cache: 'no-store', credentials: 'same-origin' });
  }

  function tick() {
    if (typeof document !== 'undefined' && document.hidden) {
      markOk();               // backgrounded: a quiet page means nothing
      return;
    }
    var idle = Date.now() - lastAppOk;
    if (idle < IDLE_MS) {
      stallStart = 0;
      return;
    }

    var evidence = lastBroken && lastBroken >= (lastAppOk - 2000);
    if (!evidence) {
      stallStart = 0;         // quiet but nothing ever failed: not our business
      return;
    }

    if (!stallStart) stallStart = Date.now();
    if (Date.now() - lastReloadAt < COOLDOWN_MS) return;

    probe().then(function (r) {
      if (!r) return;
      // Server answered. Give the app a chance to resume by itself.
      var brokenAtProbe = lastBroken;
      setTimeout(function () {
        if (lastAppOk > stallStart) return;             // app recovered, stand down
        if (Date.now() - lastReloadAt < COOLDOWN_MS) return;
        var stalledFor = Math.round((Date.now() - stallStart) / 1000);
        state.stalls += 1;
        state.reloads += 1;
        state.lastStallAt = new Date().toISOString();
        state.lastStallSeconds = stalledFor;
        state.lastReloadAt = state.lastStallAt;
        report('client-stall auto-reload after ' + stalledFor + 's (probe ok, app quiet)',
               { stream_id: null, ready_state: null,
                 stall_seconds: stalledFor, broken_at: brokenAtProbe });
        lastReloadAt = Date.now();
        try { location.reload(); } catch (e) { /* ignore */ }
      }, GRACE_MS);
    }).catch(function () {
      // Path still down. Stay quiet and keep waiting: no reload while offline.
    });
  }

  setInterval(tick, TICK_MS);

  window.addEventListener('online', markOk);
  window.addEventListener('focus', markOk);
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) markOk();
  });
})();
