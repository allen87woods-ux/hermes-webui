/* test_stall_recovery.js — headless verification of static/stall_recovery.js
 *
 * Runs the real module in a sandbox with a fake clock and a controllable fetch,
 * then asserts the reload decision. Covers the four behaviours that matter:
 *   A quiet page alone          -> no reload (protects long silent generations)
 *   B broken path + server back -> exactly one reload
 *   C second stall inside the cooldown -> no second reload
 *   D hidden tab                -> never reloads
 *   E hung request              -> counts as broken-path evidence
 *
 * Harness notes: the fake clock is anchored at a real epoch (the module compares
 * Date.now() against a 0 sentinel for "never reloaded"), and each step drains
 * microtasks so the probe's promise chain actually runs.
 *
 * Run: node test_stall_recovery.js
 */
const fs = require('fs');
const vm = require('vm');

const SRC = '/home/al/.hermes/hermes-webui/static/stall_recovery.js';
const src = fs.readFileSync(SRC, 'utf8');

let failures = 0;
function check(name, got, want) {
  const ok = got === want;
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  (got ${got}, want ${want})`);
}

function makeHarness(opts = {}) {
  // Anchor the fake clock at a real epoch: the module treats lastReloadAt=0 as
  // "never reloaded" by comparing against Date.now(), so a 0-based clock would
  // wrongly look like a reload inside the cooldown window.
  let now = Date.now();
  let seq = 0;
  const timers = [];
  const h = {
    reloads: 0,
    probes: 0,
    appCalls: 0,
    reports: null,
    fetchMode: 'ok',      // 'ok' | 'reject' | 'hang'
    now: () => now,
  };

  class FakeDate extends Date {
    constructor(...a) { if (a.length === 0) { super(now); } else { super(...a); } }
    static now() { return now; }
  }

  function addTimer(fn, ms, once) {
    const t = { fn, at: now + ms, every: ms, once, id: ++seq, dead: false };
    timers.push(t);
    return t.id;
  }
  function runDue() {
    for (;;) {
      const due = timers.filter((t) => !t.dead && t.at <= now).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      if (due.once) { due.dead = true; } else { due.at = now + due.every; }
      due.fn();
    }
  }
  h.advance = (ms) => { now += ms; runDue(); };

  const sandbox = {
    console,
    setTimeout: (fn, ms) => addTimer(fn, ms, true),
    setInterval: (fn, ms) => addTimer(fn, ms, false),
    clearTimeout: (id) => { const t = timers.find((x) => x.id === id); if (t) t.dead = true; },
    clearInterval: (id) => { const t = timers.find((x) => x.id === id); if (t) t.dead = true; },
    Date: FakeDate,
    fetch: function (url) {
      const u = String(url);
      if (u.indexOf('stall_probe') !== -1) {
        h.probes += 1;
        return Promise.resolve({ status: 200, ok: true });
      }
      h.appCalls += 1;
      if (h.fetchMode === 'reject') return Promise.reject(new Error('network down'));
      if (h.fetchMode === 'hang') return new Promise(() => {});
      return Promise.resolve({ status: 200, ok: true });
    },
    EventSource: function () { this.readyState = 1; },
    recordClientSSEError: function (source, details) { h.reports = { source, details }; },
    location: { pathname: '/session/abc', reload: function () { h.reloads += 1; } },
    S: { session: { session_id: 'sess-1' } },
  };
  sandbox.EventSource.CONNECTING = 0;
  sandbox.EventSource.OPEN = 1;
  sandbox.EventSource.CLOSED = 2;
  sandbox.EventSource.prototype.addEventListener = function () {};
  sandbox.document = {
    hidden: !!opts.hidden,
    visibilityState: opts.hidden ? 'hidden' : 'visible',
    addEventListener: function () {},
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.addEventListener = function () {};
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox);
  h.sandbox = sandbox;
  return h;
}

const drain = () => new Promise((r) => setImmediate(r));
async function step(h, ms) { h.advance(ms); await drain(); }

(async function main() {
  // --- A: quiet page, nothing ever failed -> must not reload ---
  {
    const h = makeHarness();
    await step(h, 60000);
    check('A quiet page alone: reloads', h.reloads, 0);
    check('A quiet page alone: probes', h.probes, 0);
  }

  // --- B: a request fails, then silence; server answers again -> one reload ---
  {
    const h = makeHarness();
    h.fetchMode = 'reject';
    h.sandbox.fetch('api/chat/poll').catch(() => {});
    await step(h, 6000);
    check('B after failure: no reload yet', h.reloads, 0);
    await step(h, 35000);              // idle passes 25s, tick fires, probe answered
    await step(h, 6000);               // grace window elapses
    check('B stalled: reload fired', h.reloads, 1);
    check('B reported to server', h.reports && h.reports.source, 'stall-recovery');
    const stallText = (h.reports && h.reports.details && h.reports.details.reason) || '';
    check('B report mentions the stall', /stall/i.test(stallText), true);
  }

  // --- C: cooldown holds a second stall back ---
  {
    const h = makeHarness();
    h.fetchMode = 'reject';
    h.sandbox.fetch('api/chat/poll').catch(() => {});
    await step(h, 6000);               // drain the rejection microtask first
    await step(h, 35000);              // idle passes 25s, tick probes
    await step(h, 6000);               // grace window elapses
    const first = h.reloads;
    h.sandbox.fetch('api/chat/poll').catch(() => {});
    await step(h, 6000);
    await step(h, 35000);
    await step(h, 6000);
    check('C first reload happened', first, 1);
    check('C second stall inside cooldown: reloads', h.reloads, 1);
  }

  // --- D: hidden tab never reloads ---
  {
    const h = makeHarness({ hidden: true });
    h.fetchMode = 'reject';
    h.sandbox.fetch('api/chat/poll').catch(() => {});
    await step(h, 120000);
    check('D hidden tab: reloads', h.reloads, 0);
  }

  // --- E: a hung request counts as broken-path evidence ---
  {
    const h = makeHarness();
    h.fetchMode = 'hang';
    h.sandbox.fetch('api/chat/poll').catch(() => {});
    await step(h, 20000);              // hang watchdog (15s) marks it broken
    await step(h, 30000);              // idle passes 25s, tick probes
    await step(h, 6000);               // grace window elapses
    check('E hung request: reload fired', h.reloads, 1);
  }

  console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
})();
