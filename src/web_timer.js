/* A local countdown. No storage, network requests, audio, or dependencies. */
"use strict";

class Countdown {
  constructor(seconds = 300, now = () => Date.now()) {
    this.now = now;
    this.reset(seconds);
  }

  reset(seconds = this.durationSeconds) {
    if (!Number.isInteger(seconds) || seconds < 1 || seconds > 3600) {
      throw new RangeError("Choose a duration between 1 second and 60 minutes.");
    }
    this.durationSeconds = seconds;
    this.remainingMs = seconds * 1000;
    this.deadline = null;
    this.state = "ready";
    return this.snapshot();
  }

  start() {
    this.snapshot();
    if (this.state === "running") return this.snapshot();
    if (this.state === "complete") this.remainingMs = this.durationSeconds * 1000;
    this.deadline = this.now() + this.remainingMs;
    this.state = "running";
    return this.snapshot();
  }

  pause() {
    this.snapshot();
    if (this.state === "running") {
      this.state = "paused";
      this.deadline = null;
    }
    return this.snapshot();
  }

  snapshot() {
    if (this.state === "running") {
      this.remainingMs = Math.max(0, Math.min(this.durationSeconds * 1000, this.deadline - this.now()));
      if (this.remainingMs === 0) {
        this.state = "complete";
        this.deadline = null;
      }
    }
    return {state: this.state, remainingMs: this.remainingMs, durationSeconds: this.durationSeconds};
  }
}

function formatRemaining(milliseconds) {
  const seconds = Math.ceil(Math.max(0, milliseconds) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function mountTimer(doc, timer = new Countdown(), schedule = setInterval) {
  const duration = doc.getElementById("timer-minutes");
  const toggle = doc.getElementById("timer-toggle");
  const reset = doc.getElementById("timer-reset");
  const display = doc.getElementById("timer-countdown");
  const status = doc.getElementById("timer-status");
  const ring = doc.getElementById("timer-progress");
  const originalTitle = doc.title;
  const messages = {
    ready: "Choose a duration, then start when you are ready.",
    running: "Your quiet time is underway.",
    paused: "Paused. Resume whenever you are ready.",
    complete: "Session complete. Take your time before moving on."
  };
  const labels = {ready: "Start timer", running: "Pause", paused: "Resume", complete: "Start again"};

  function render() {
    const state = timer.snapshot();
    const remaining = formatRemaining(state.remainingMs);
    display.textContent = remaining;
    // Only state changes update the live region, avoiding announcements every second.
    if (status.textContent !== messages[state.state]) status.textContent = messages[state.state];
    toggle.textContent = labels[state.state];
    duration.disabled = state.state === "running" || state.state === "paused";
    ring.style.strokeDashoffset = String(100 * (1 - state.remainingMs / (state.durationSeconds * 1000)));
    doc.title = state.state === "ready" ? originalTitle :
      state.state === "complete" ? "Session complete · Wind-down timer" : `${remaining} · ${state.state === "paused" ? "Paused" : "Wind-down timer"}`;
  }

  function configure() {
    const minutes = Number(duration.value);
    if (!Number.isInteger(minutes) || minutes < 1 || minutes > 60) {
      duration.setCustomValidity("Choose a whole number of minutes from 1 to 60.");
      duration.reportValidity();
      return false;
    }
    duration.setCustomValidity("");
    timer.reset(minutes * 60);
    return true;
  }

  toggle.addEventListener("click", () => {
    const state = timer.snapshot().state;
    if (state === "running") timer.pause();
    else {
      if ((state === "ready" || state === "complete") && !configure()) return;
      timer.start();
    }
    render();
  });
  reset.addEventListener("click", () => {
    timer.reset();
    duration.value = String(timer.durationSeconds / 60);
    duration.setCustomValidity("");
    render();
  });
  duration.addEventListener("input", () => {
    duration.setCustomValidity("");
    if (timer.snapshot().state === "ready" && Number.isInteger(Number(duration.value)) && Number(duration.value) >= 1 && Number(duration.value) <= 60) {
      timer.reset(Number(duration.value) * 60);
      render();
    }
  });
  doc.addEventListener("visibilitychange", render);
  render();
  return schedule(render, 250);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {Countdown, formatRemaining, mountTimer};
}
if (typeof document !== "undefined" && document.getElementById("timer-countdown")) {
  mountTimer(document);
}
