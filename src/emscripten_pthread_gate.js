/* Runs at the very start of the generated index.js (before wasm + pthread workers).
 * Module.preRun is too late: workers already try to receive SharedArrayBuffer and throw
 * "SharedArrayBuffer transfer requires self.crossOriginIsolated" if the page is not isolated. */
(function () {
  if (typeof document === 'undefined') return;
  var sabOk = typeof SharedArrayBuffer !== 'undefined';
  var atomOk = typeof Atomics !== 'undefined';
  var coiOk = (typeof crossOriginIsolated === 'undefined') ? true : (crossOriginIsolated === true);
  if (sabOk && atomOk && coiOk) return;

  /* Emscripten often rejects async startup with Error("aborted"); createWasm may not attach
   * .catch. Suppress that console noise only while this gate is active (env unfit for pthreads). */
  if (typeof window !== 'undefined') {
    window.__ab3d_gate_blocked = true;
    window.addEventListener(
      'unhandledrejection',
      function (e) {
        if (!window.__ab3d_gate_blocked) return;
        var r = e.reason;
        var msg = '';
        if (r == null) return;
        if (typeof r === 'string') msg = r;
        else if (typeof r.message === 'string') msg = r.message;
        if (msg === 'aborted' || r === 'aborted') e.preventDefault();
      },
      true
    );
  }

  if (document.body) document.body.innerHTML =
    '<div class="ab3d-cant-play">' +
    '<div class="ab3d-cant-play__card" role="alert">' +
    '<h1>Can\'t start the game</h1>' +
    '<p>This build needs <strong>SharedArrayBuffer</strong> (shared memory between threads). There is no option inside the game to turn it on. Your browser enables SharedArrayBuffer only when the page is in the right context: usually a full <strong>https://</strong> or <strong>http://</strong> address from a real site, and (for the threaded web build) the page must be set up so the browser treats it as safe for shared memory. Opening a local file or the wrong page context means SharedArrayBuffer stays off.</p>' +
    '<p><strong>What to do</strong> (to get SharedArrayBuffer enabled)</p>' +
    '<ol class="ab3d-cant-play__steps">' +
    '<li><strong>Go to the game\'s page online.</strong> Use the link or button on the site where the game is published (for example the project or demo page). Don\'t open <code>index.html</code> from your Downloads folder.</li>' +
    '<li><strong>Check the address bar.</strong> If it starts with <code>file://</code>, close that tab and open the game using the real web link instead.</li>' +
    '<li><strong>Refresh</strong> the page or try again in <strong>Chrome</strong>, <strong>Firefox</strong>, or <strong>Edge</strong>.</li>' +
    '<li>If a <strong>Windows or desktop</strong> download is offered, use that; it does not need SharedArrayBuffer in the browser.</li>' +
    '</ol>' +
    '<p>If you <strong>host this page yourself</strong>, the server must send headers so browsers allow SharedArrayBuffer (cross-origin isolation). Without that, SharedArrayBuffer cannot be enabled for this URL.</p>' +
    '<p><strong>Hosts without cross-origin isolation</strong> (for example plain GitHub Pages) cannot run this <strong>threaded</strong> web build. The default web build is single-threaded and does not need SharedArrayBuffer; or host this build with COOP/COEP headers (e.g. Cloudflare). See README → Web (Emscripten).</p>' +
    '<p class="ab3d-cant-play__sub">Alien Breed 3D I — web</p>' +
    '</div></div>';
  /* Do not throw or Promise.reject: Emscripten's createWasm often has no .catch on the
   * instantiateWasm chain, which surfaces "Uncaught (in promise) Error: aborted". Block
   * wasm by never calling receiveInstance; return a never-settling Promise so older
   * Emscripten that chains on the return value also stays quiet. */
  if (typeof window === 'undefined') return;
  if (!window.Module) window.Module = {};
  window.Module.instantiateWasm = function (imports, receiveInstance) {
    return new Promise(function () {
      /* intentionally pending — user already sees the message above */
    });
  };
})();
