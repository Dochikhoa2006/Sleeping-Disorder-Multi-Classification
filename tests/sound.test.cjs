const {test} = require('node:test');
const assert = require('node:assert/strict');
const {AmbientPlayer, createNoiseBuffer, mountAmbient} = require('../src/web_sound.js');

function fakeContext(resume = () => Promise.resolve()) {
  const node = () => ({connections: [], disconnected: false,
    connect(next) { this.connections.push(next); }, disconnect() { this.disconnected = true; }});
  const param = () => ({value: 0, setTargetAtTime(value) { this.value = value; }});
  const source = {...node(), started: 0, stopped: 0, start() { this.started++; }, stop() { this.stopped++; }};
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
  for (const id of ['sound-start', 'sound-stop', 'sound-mode', 'sound-volume', 'sound-volume-value', 'sound-status']) {
    elements[id] = {textContent: '', disabled: false, value: id === 'sound-mode' ? 'soft' : '10', handlers: {},
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
