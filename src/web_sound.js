/* Locally generated ambient noise. No audio assets, storage, or network calls. */
"use strict";

const SOUND_MODES = Object.freeze({
  soft: {label: "Soft noise", frequency: 1500},
  bright: {label: "Bright noise", frequency: 6000},
  deep: {label: "Low rumble", frequency: 180}
});

function createNoiseBuffer(context, random = Math.random) {
  const buffer = context.createBuffer(1, Math.floor(context.sampleRate * 4), context.sampleRate);
  const samples = buffer.getChannelData(0);
  for (let index = 0; index < samples.length; index++) samples[index] = (random() * 2 - 1) * 0.5;
  return buffer;
}

class AmbientPlayer {
  constructor(createContext, random = Math.random) {
    this.createContext = createContext;
    this.random = random;
    this.mode = "soft";
    this.volume = 0.1;
    this.state = "stopped";
    this.context = null;
    this.source = null;
    this.filter = null;
    this.gain = null;
    this.ticket = 0;
    this.autoStopSeconds = 0;
    this.endedAutomatically = false;
    this.onChange = null;
  }

  setMode(mode) {
    if (!Object.hasOwn(SOUND_MODES, mode)) throw new RangeError("Choose a listed sound mode.");
    this.mode = mode;
    if (this.filter && this.context) {
      this.filter.frequency.setTargetAtTime(Math.min(SOUND_MODES[mode].frequency, this.context.sampleRate / 2 - 1), this.context.currentTime, 0.05);
    }
  }

  setVolume(volume) {
    if (!Number.isFinite(volume) || volume < 0 || volume > 0.3) throw new RangeError("Choose a volume from 0 to 30%.");
    this.volume = volume;
    if (this.gain && this.context) this.gain.gain.setTargetAtTime(volume, this.context.currentTime, 0.05);
  }

  async start(autoStopSeconds = 0) {
    if (!Number.isInteger(autoStopSeconds) || autoStopSeconds < 0 || autoStopSeconds > 3600) {
      throw new RangeError("Choose an automatic stop duration from 0 to 60 minutes.");
    }
    if (this.context) return this.state === "playing";
    const ticket = ++this.ticket;
    this.state = "starting";
    this.autoStopSeconds = autoStopSeconds;
    this.endedAutomatically = false;
    try {
      const context = this.createContext();
      this.context = context;
      this.source = context.createBufferSource();
      this.source.buffer = createNoiseBuffer(context, this.random);
      this.source.loop = true;
      this.filter = context.createBiquadFilter();
      this.filter.type = "lowpass";
      this.filter.Q.value = 0.7;
      this.filter.frequency.value = Math.min(SOUND_MODES[this.mode].frequency, context.sampleRate / 2 - 1);
      this.gain = context.createGain();
      this.gain.gain.value = this.volume;
      this.source.connect(this.filter);
      this.filter.connect(this.gain);
      this.gain.connect(context.destination);
      const source = this.source;
      source.onended = () => {
        if (this.source !== source || ticket !== this.ticket) return;
        this.stop("finished");
        if (this.onChange) this.onChange();
      };
      source.start();
      // Schedule on the audio clock, independent of throttled UI callbacks.
      if (autoStopSeconds) source.stop(context.currentTime + autoStopSeconds);
      await context.resume();
      if (ticket !== this.ticket || this.context !== context) return false;
      this.state = "playing";
      return true;
    } catch (error) {
      if (ticket !== this.ticket) return false;
      this.stop();
      throw error;
    }
  }

  stop(reason = "manual") {
    ++this.ticket;
    this.endedAutomatically = reason === "finished";
    const context = this.context;
    if (this.source) {
      this.source.onended = null;
      if (reason !== "finished") {
        try { this.source.stop(); } catch (_) { /* Already stopped. */ }
      }
    }
    for (const node of [this.source, this.filter, this.gain]) {
      if (node) { try { node.disconnect(); } catch (_) { /* Already disconnected. */ } }
    }
    this.context = this.source = this.filter = this.gain = null;
    this.state = "stopped";
    if (context) { try { Promise.resolve(context.close()).catch(() => {}); } catch (_) { /* Already closed. */ } }
  }
}

function mountAmbient(doc, browser, player) {
  const start = doc.getElementById("sound-start");
  const stop = doc.getElementById("sound-stop");
  const mode = doc.getElementById("sound-mode");
  const volume = doc.getElementById("sound-volume");
  const duration = doc.getElementById("sound-duration");
  const output = doc.getElementById("sound-volume-value");
  const status = doc.getElementById("sound-status");
  player = player || new AmbientPlayer(() => {
    const Context = browser.AudioContext || browser.webkitAudioContext;
    if (!Context) throw new Error("This browser does not support local audio playback.");
    return new Context();
  });
  function render(message) {
    start.disabled = player.state !== "stopped";
    stop.disabled = player.state === "stopped";
    duration.disabled = player.state !== "stopped";
    status.textContent = message || (player.state === "playing" ?
      `Playing ${SOUND_MODES[player.mode].label.toLowerCase()}.${player.autoStopSeconds ? ` Stops after ${player.autoStopSeconds / 60} minutes of playback.` : ""}` :
      player.state === "starting" ? "Starting audio…" : player.endedAutomatically ?
      "Session complete. Sound is off." : "Sound is off. Start when you are ready.");
    output.textContent = `${Math.round(player.volume * 100)}%`;
  }
  player.onChange = () => render();
  start.addEventListener("click", async () => {
    const pending = player.start(Number(duration.value) * 60);
    render();
    try { await pending; render(); }
    catch (error) { render(error.message || "Audio could not start. Try again."); }
  });
  stop.addEventListener("click", () => { player.stop(); render(); });
  mode.addEventListener("change", () => { player.setMode(mode.value); render(); });
  volume.addEventListener("input", () => { player.setVolume(Number(volume.value) / 100); render(); });
  browser.addEventListener("pagehide", () => { player.stop(); render(); });
  render();
  return player;
}

if (typeof module !== "undefined" && module.exports) module.exports = {SOUND_MODES, createNoiseBuffer, AmbientPlayer, mountAmbient};
if (typeof document !== "undefined" && document.getElementById("sound-start")) mountAmbient(document, window);
