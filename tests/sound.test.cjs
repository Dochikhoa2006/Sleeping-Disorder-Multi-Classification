const {test} = require('node:test');
const assert = require('node:assert/strict');
const {AmbientPlayer, createNoiseBuffer, mountAmbient} = require('../src/web_sound.js');

function fakeContext(resume = () => Promise.resolve()) {
  const node = () => ({connections: [], disconnected: false,
    connect(next) { this.connections.push(next); }, disconnect() { this.disconnected = true; }});
  const param = () => ({value: 0, setTargetAtTime(value) { this.value = value; }});
  const source = {...node(), started: 0, stopped: 0, stopTimes: [], start() { this.started++; }, stop(when) { this.stopped++; this.stopTimes.push(when); }};
  const filter = {...node(), frequency: param(), Q: param()};
  const gain = {...node(), gain: param()};
  return {sampleRate: 1024, currentTime: 0, destination: {}, source, filter, gain, closed: 0,
    createBuffer(channels, length) { const samples = new Float32Array(length); return {getChannelData() { return samples; }}; },
    createBufferSource() { return source; }, createBiquadFilter() { return filter; }, createGain() { return gain; },
    resume, close() { this.closed++; return Promise.resolve(); }};
}

test('noise generation is bounded, mono, and local', () => {
  const context = fakeContext();
  const buffer = createNoiseBuffer(context, () => .75);
  const samples = buffer.getChannelData(0);
  assert.equal(samples.length, 4096);
  assert.ok(samples.every(value => value === .25));
});

test('player creates no audio context until started and wires a filtered loop', async () => {
  const context = fakeContext();
  let created = 0;
  const player = new AmbientPlayer(() => { created++; return context; });
  assert.equal(created, 0);
  assert.equal(await player.start(), true);
  assert.equal(created, 1);
  assert.equal(context.source.loop, true);
  assert.deepEqual(context.source.connections, [context.filter]);
  assert.deepEqual(context.filter.connections, [context.gain]);
  assert.deepEqual(context.gain.connections, [context.destination]);
  assert.equal(context.gain.gain.value, .1);
  assert.equal(await player.start(), true);
  assert.equal(created, 1);
  player.setMode('deep');
  assert.equal(context.filter.frequency.value, 180);
  player.setVolume(.2);
  assert.equal(context.gain.gain.value, .2);
  player.stop();
  player.stop();
  assert.equal(context.source.stopped, 1);
  assert.equal(context.closed, 1);
  assert.equal(player.state, 'stopped');
});

test('stopping during a delayed resume cannot restart sound', async () => {
  let resume;
  const context = fakeContext(() => new Promise(resolve => { resume = resolve; }));
  const player = new AmbientPlayer(() => context);
  const started = player.start();
  player.stop();
  resume();
  assert.equal(await started, false);
  assert.equal(player.state, 'stopped');
  assert.equal(context.closed, 1);
});

test('audio failures clean up resources and invalid settings are rejected', async () => {
  const context = fakeContext(() => Promise.reject(new Error('blocked')));
  const player = new AmbientPlayer(() => context);
  await assert.rejects(player.start(), /blocked/);
  assert.equal(player.state, 'stopped');
  assert.equal(context.closed, 1);
  for (const volume of [-1, .31, NaN, Infinity]) assert.throws(() => player.setVolume(volume), RangeError);
  assert.throws(() => player.setMode('__proto__'), RangeError);
});

function fakeDocument() {
  const elements = {};
  for (const id of ['sound-start', 'sound-stop', 'sound-mode', 'sound-volume', 'sound-volume-value', 'sound-status', 'sound-duration']) {
    elements[id] = {textContent: '', disabled: false, value: id === 'sound-mode' ? 'soft' : id === 'sound-duration' ? '15' : '10', handlers: {},
      addEventListener(name, callback) { this.handlers[name] = callback; }};
  }
  return {elements, getElementById(id) { return elements[id]; }};
}

test('browser controls start explicitly, adjust settings, and release audio on page hide', async () => {
  const context = fakeContext();
  const player = new AmbientPlayer(() => context);
  const doc = fakeDocument();
  const browser = {handlers: {}, addEventListener(name, callback) { this.handlers[name] = callback; }};
  mountAmbient(doc, browser, player);
  assert.equal(context.source.started, 0);
  assert.equal(doc.elements['sound-stop'].disabled, true);
  await doc.elements['sound-start'].handlers.click();
  assert.equal(player.state, 'playing');
  doc.elements['sound-mode'].value = 'deep';
  doc.elements['sound-mode'].handlers.change();
  assert.match(doc.elements['sound-status'].textContent, /low rumble/);
  doc.elements['sound-volume'].value = '20';
  doc.elements['sound-volume'].handlers.input();
  assert.equal(doc.elements['sound-volume-value'].textContent, '20%');
  browser.handlers.pagehide();
  assert.equal(player.state, 'stopped');
  assert.equal(context.closed, 1);
});

test('unsupported browsers receive an actionable message without autoplay', async () => {
  const doc = fakeDocument();
  const browser = {addEventListener() {}};
  mountAmbient(doc, browser);
  await doc.elements['sound-start'].handlers.click();
  assert.match(doc.elements['sound-status'].textContent, /does not support/);
  assert.equal(doc.elements['sound-start'].disabled, false);
});

test('automatic stop is scheduled on the audio clock and completion releases resources', async () => {
  const context = fakeContext();
  context.currentTime = 12;
  const player = new AmbientPlayer(() => context);
  let changes = 0;
  player.onChange = () => changes++;
  await player.start(300);
  assert.deepEqual(context.source.stopTimes, [312]);
  context.source.onended();
  assert.equal(player.state, 'stopped');
  assert.equal(player.endedAutomatically, true);
  assert.equal(context.closed, 1);
  assert.equal(changes, 1);
});

test('unlimited playback has no scheduled stop and invalid durations create no context', async () => {
  const context = fakeContext();
  let created = 0;
  const player = new AmbientPlayer(() => { created++; return context; });
  for (const seconds of [-1, 3601, 1.5, NaN]) await assert.rejects(player.start(seconds), RangeError);
  assert.equal(created, 0);
  await player.start(0);
  assert.deepEqual(context.source.stopTimes, []);
  player.stop();
});

test('an old completion callback cannot stop a restarted session', async () => {
  const contexts = [fakeContext(), fakeContext()];
  let next = 0;
  const player = new AmbientPlayer(() => contexts[next++]);
  await player.start(60);
  const oldCompletion = contexts[0].source.onended;
  player.stop();
  await player.start(120);
  oldCompletion();
  assert.equal(player.state, 'playing');
  assert.equal(contexts[1].closed, 0);
  player.stop();
});

test('browser duration selection locks during playback and completion updates its status', async () => {
  const context = fakeContext();
  const player = new AmbientPlayer(() => context);
  const doc = fakeDocument();
  const browser = {addEventListener() {}};
  mountAmbient(doc, browser, player);
  doc.elements['sound-duration'].value = '5';
  await doc.elements['sound-start'].handlers.click();
  assert.deepEqual(context.source.stopTimes, [300]);
  assert.equal(doc.elements['sound-duration'].disabled, true);
  context.source.onended();
  assert.equal(doc.elements['sound-duration'].disabled, false);
  assert.match(doc.elements['sound-status'].textContent, /Session complete/);
});
