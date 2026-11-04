const {test} = require('node:test');
const assert = require('node:assert/strict');
const {Countdown, formatRemaining, mountTimer} = require('../src/web_timer.js');

test('elapsed wall time drives countdown even when callbacks are delayed', () => {
  let now = 1000;
  const timer = new Countdown(60, () => now);
  timer.start();
  now += 45000;
  assert.equal(timer.snapshot().remainingMs, 15000);
  now += 60000;
  assert.deepEqual(timer.snapshot(), {state: 'complete', remainingMs: 0, durationSeconds: 60});
});

test('pause, resume, repeated start, and reset preserve the correct duration', () => {
  let now = 0;
  const timer = new Countdown(60, () => now);
  timer.start();
  now += 10500;
  timer.start(); // Starting twice must not restart the countdown.
  assert.equal(timer.pause().remainingMs, 49500);
  now += 90000;
  assert.equal(timer.snapshot().remainingMs, 49500);
  timer.start();
  now += 500;
  assert.equal(timer.snapshot().remainingMs, 49000);
  assert.equal(timer.reset().state, 'ready');
  assert.equal(timer.snapshot().remainingMs, 60000);
});

test('completed timers can restart and invalid durations are rejected', () => {
  let now = 0;
  const timer = new Countdown(1, () => now);
  timer.start();
  now = 2000;
  assert.equal(timer.pause().state, 'complete');
  assert.equal(timer.start().remainingMs, 1000);
  for (const seconds of [0, -1, 3601, 1.5, NaN, Infinity]) {
    assert.throws(() => timer.reset(seconds), RangeError);
  }
});

test('display rounds up sub-second remainder and clamps at zero', () => {
  assert.equal(formatRemaining(1), '00:01');
  assert.equal(formatRemaining(60001), '01:01');
  assert.equal(formatRemaining(-1000), '00:00');
  assert.equal(formatRemaining(3600000), '60:00');
});

function fakeDocument() {
  const ids = ['timer-minutes', 'timer-toggle', 'timer-reset', 'timer-countdown', 'timer-status', 'timer-progress'];
  const elements = Object.fromEntries(ids.map(id => [id, {
    textContent: '', value: '5', disabled: false, style: {}, handlers: {}, validity: '',
    addEventListener(name, callback) { this.handlers[name] = callback; },
    setCustomValidity(message) { this.validity = message; },
    reportValidity() { this.reported = true; }
  }]));
  return {title: 'Timer page', elements, handlers: {},
    getElementById(id) { return elements[id]; },
    addEventListener(name, callback) { this.handlers[name] = callback; }};
}

test('browser controls update display, title, pause, resume, and completion', () => {
  let now = 0, tick;
  const doc = fakeDocument();
  const timer = new Countdown(300, () => now);
  mountTimer(doc, timer, callback => { tick = callback; });
  const elements = doc.elements;
  elements['timer-minutes'].value = '1';
  elements['timer-minutes'].handlers.input();
  assert.equal(elements['timer-countdown'].textContent, '01:00');
  elements['timer-toggle'].handlers.click();
  assert.equal(elements['timer-minutes'].disabled, true);
  now = 10000;
  tick();
  assert.equal(elements['timer-countdown'].textContent, '00:50');
  assert.match(doc.title, /00:50/);
  elements['timer-toggle'].handlers.click();
  now = 50000;
  tick();
  assert.equal(elements['timer-countdown'].textContent, '00:50');
  assert.equal(elements['timer-toggle'].textContent, 'Resume');
  elements['timer-toggle'].handlers.click();
  now = 200000; // Simulate returning after a suspended/backgrounded tab.
  doc.handlers.visibilitychange();
  assert.equal(elements['timer-countdown'].textContent, '00:00');
  assert.match(elements['timer-status'].textContent, /Session complete/);
  assert.equal(elements['timer-toggle'].textContent, 'Start again');
  elements['timer-reset'].handlers.click();
  assert.equal(elements['timer-minutes'].disabled, false);
  assert.equal(doc.title, 'Timer page');
});

test('invalid minute input never starts the timer', () => {
  const doc = fakeDocument();
  const timer = new Countdown();
  mountTimer(doc, timer, () => {});
  for (const value of ['', '0', '61', '1.5', 'nope']) {
    doc.elements['timer-minutes'].value = value;
    doc.elements['timer-toggle'].handlers.click();
    assert.equal(timer.snapshot().state, 'ready');
    assert.notEqual(doc.elements['timer-minutes'].validity, '');
  }
  doc.elements['timer-reset'].handlers.click();
  assert.equal(doc.elements['timer-minutes'].value, '5');
  assert.equal(doc.elements['timer-minutes'].validity, '');
});
