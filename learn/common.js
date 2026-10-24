/* Shared helpers for the chapter pages (classic script, works from file://). */
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const CHAPTERS = [
  ["01-tokens.html", "Tokens and hidden states"],
  ["02-layers.html", "Layers"],
  ["03-experts.html", "Experts and Mixture of Experts"],
  ["04-router.html", "The router"],
  ["05-pruning.html", "What pruning experts does"],
  ["06-reap.html", "Choosing experts: REAP"],
  ["07-calibration.html", "Calibration data"],
  ["08-perplexity.html", "Perplexity"],
  ["09-distillation.html", "Distillation and healing"],
  ["10-quantization.html", "Quantization"],
  ["11-imatrix.html", "The importance matrix"],
  ["12-pipeline.html", "The whole pipeline"],
  ["13-attention.html", "Attention, DeltaNet and the KV cache"],
  ["14-numbers.html", "Number formats"],
  ["15-speed.html", "What sets the speed"],
  ["16-sampling.html", "From probabilities to text"],
  ["a-options.html", "Ways to shrink a model"],
  ["b-answers.html", "Real answers compared"],
  ["c-code.html", "Code tour and debugging"],
  ["d-specialist.html", "Specialist pruning (run 3)"],
  ["e-benchmark.html", "Benchmarking (step 2)"],
  ["f-training.html", "Training and distillation (step 3)"],
  ["g-experts.html", "Inside the experts"],
  ["h-quant.html", "The quantization curve (step 4)"],
  ["i-ablation.html", "Ablation and merging (step 5)"],
  ["j-practical.html", "From Hugging Face to your GPU (step 6)"],
];
const N_CHAPTERS = 16; // the rest are appendices A, B, C, ...
const README_IDS = ["1-tokens-and-hidden-states", "2-layers", "3-experts-and-mixture-of-experts-moe", "4-the-router",
  "5-what-pruning-experts-does", "6-choosing-which-experts-to-keep-reap-and-the-alternatives", "7-calibration-data",
  "8-perplexity-how-we-measure-damage", "9-distillation-and-healing", "10-quantization-gguf-q4_k_m-and-iq4_xs",
  "11-the-importance-matrix-imatrix", "12-other-names-you-will-meet",
  "13-attention-deltanet-and-the-kv-cache", "14-number-formats-fp32-fp16-bf16-and-integers", "15-what-sets-the-speed", "16-from-probabilities-to-text"]; // section anchors in README.html
const chapterLabel = (i) => (i < N_CHAPTERS ? `Chapter ${i + 1}` : `Appendix ${"ABCDEFGHIJ"[i - N_CHAPTERS]}`);

/* --- element builders ------------------------------------------------- */
function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null) continue;
    if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "cls") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "fill" && el.tagName === "text") el.style.fill = v; // a fill attribute would lose to common.css's "svg text" rule
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  return el;
}
function S(tag, attrs, parent) {
  const el = setAttrs(document.createElementNS(SVGNS, tag), attrs);
  if (parent) parent.appendChild(el);
  return el;
}
function H(tag, attrs, parent) {
  const el = setAttrs(document.createElement(tag), attrs);
  if (parent) parent.appendChild(el);
  return el;
}
function svgRoot(container, w, h, label) {
  return S("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label || "" }, container);
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

/* --- numbers ------------------------------------------------------------ */
function rng(seed) { // mulberry32: same "random" demo data on every visit
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) { const u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
function fmt(x, d = 2) { return Number(x).toFixed(d); }
function softmax(xs) { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }

/* --- animation ---------------------------------------------------------- */
function tween(ms, fn, done) {
  if (REDUCED || ms <= 0) { fn(1); if (done) done(); return () => {}; }
  let stop = false; const t0 = performance.now();
  function frame(now) {
    if (stop) return;
    const t = clamp((now - t0) / ms, 0, 1);
    fn(ease(t));
    if (t < 1) requestAnimationFrame(frame); else if (done) done();
  }
  requestAnimationFrame(frame);
  return () => { stop = true; };
}
/* animate numeric SVG attributes of one element towards `to` */
function animAttrs(el, to, ms = 600) {
  if (el._tw) el._tw();
  const from = {};
  for (const k of Object.keys(to)) from[k] = parseFloat(el.getAttribute(k)) || 0;
  el._tw = tween(ms, (t) => { for (const k of Object.keys(to)) el.setAttribute(k, lerp(from[k], to[k], t)); });
}
function show(el, on) { el.classList.toggle("hide", !on); }
/* a repeating loop (token streams etc.) that pauses while the tab is hidden */
function loop(fn) {
  let id = null, last = 0, running = false;
  function frame(now) { if (!running) return; fn(now - last, now); last = now; id = requestAnimationFrame(frame); }
  return {
    start() { if (running || REDUCED) return; running = true; last = performance.now(); id = requestAnimationFrame(frame); },
    stop() { running = false; if (id) cancelAnimationFrame(id); },
    get running() { return running; },
  };
}

/* --- page chrome -------------------------------------------------------- */
function chapterIndex() {
  const file = location.pathname.split("/").pop();
  return CHAPTERS.findIndex(([f]) => f === file);
}
/* "Check yourself" questions (quiz.js), shown above the pager; answers stay hidden until opened */
function buildQuiz(before) {
  const qs = typeof QUIZ !== "undefined" ? QUIZ[location.pathname.split("/").pop()] : null;
  if (!qs) return;
  const sec = H("section", { cls: "prose quiz", "aria-labelledby": "quiz-h" });
  H("h2", { id: "quiz-h", text: "Check yourself" }, sec);
  H("p", { cls: "note", text: "Answer in your head first, then open the answer." }, sec);
  qs.forEach(([q, a]) => { const d = H("details", {}, sec); H("summary", { text: q }, d); H("p", { text: a }, d); });
  before.parentNode.insertBefore(sec, before);
}
function buildPager(container) {
  buildQuiz(container);
  const i = chapterIndex();
  const nav = H("nav", { cls: "pager", "aria-label": "Chapters" }, container);
  if (i > 0) H("a", { href: CHAPTERS[i - 1][0], html: `<span>Previous</span>${CHAPTERS[i - 1][1]}` }, nav);
  else H("a", { href: "index.html", html: "<span>Back</span>All chapters" }, nav);
  if (i < CHAPTERS.length - 1) H("a", { cls: "next", href: CHAPTERS[i + 1][0], html: `<span>Next</span>${CHAPTERS[i + 1][1]}` }, nav);
  else H("a", { cls: "next", href: "index.html", html: "<span>Done</span>All chapters" }, nav);
}
function buildCrumbs(container) {
  const i = chapterIndex();
  const where = i < N_CHAPTERS ? `<span>Chapter ${i + 1} of ${N_CHAPTERS}</span><a href="README.html#${README_IDS[i]}">README section ${i + 1}</a>` : `<span>${chapterLabel(i)}</span>`;
  container.innerHTML = `<a href="index.html">Pruning Qwen3.5-35B-A3B</a>${where}<a href="README.html">Project notes</a>`;
}

/* --- Story: a sequence of captioned steps driving a declarative render(i) --- */
class Story {
  constructor({ mount, steps, render, autoplay = true }) {
    this.steps = steps; this.render = render; this.i = -1; this.timer = null; this.playing = false;
    const box = H("div", { cls: "story" }, mount);
    const cap = H("div", { cls: "caption", "aria-live": "polite" }, box);
    this.count = H("div", { cls: "count" }, cap);
    this.title = H("h3", {}, cap);
    this.text = H("p", {}, cap);
    const ctl = H("div", { cls: "controls" }, box);
    this.bBack = H("button", { cls: "btn", type: "button", text: "◀ Back", onclick: () => { this.pause(); this.go(this.i - 1); } }, ctl);
    this.bPlay = H("button", { cls: "btn primary", type: "button", text: "▶ Play", onclick: () => (this.playing ? this.pause() : this.play()) }, ctl);
    this.bNext = H("button", { cls: "btn", type: "button", text: "Next ▶", onclick: () => { this.pause(); this.go(this.i + 1); } }, ctl);
    H("button", { cls: "btn", type: "button", text: "↺ Restart", onclick: () => { this.go(0); this.play(); } }, ctl);
    this.dots = H("div", { cls: "dots" }, box);
    this.dotEls = steps.map((s, k) => H("button", { type: "button", "aria-label": `Step ${k + 1}: ${s.title}`, onclick: () => { this.pause(); this.go(k); } }, this.dots));
    document.addEventListener("keydown", (e) => {
      if (/^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(document.activeElement.tagName)) return; // let focused controls handle their own keys
      if (e.key === "ArrowRight") { this.pause(); this.go(this.i + 1); }
      else if (e.key === "ArrowLeft") { this.pause(); this.go(this.i - 1); }
      else if (e.key === " ") { e.preventDefault(); this.playing ? this.pause() : this.play(); }
    });
    this.go(0);
    if (autoplay && !REDUCED) setTimeout(() => this.play(), 900);
  }
  go(i) {
    i = clamp(i, 0, this.steps.length - 1);
    const prev = this.i; this.i = i;
    const s = this.steps[i];
    this.count.textContent = `Step ${i + 1} of ${this.steps.length}`;
    this.title.textContent = s.title;
    this.text.innerHTML = s.text;
    this.dotEls.forEach((d, k) => { d.toggleAttribute("aria-current", k === i); if (k === i) d.setAttribute("aria-current", "step"); d.classList.toggle("done", k < i); });
    this.bBack.disabled = i === 0; this.bNext.disabled = i === this.steps.length - 1;
    this.render(i, prev);
    if (this.playing) this.schedule();
  }
  schedule() {
    clearTimeout(this.timer);
    if (this.i >= this.steps.length - 1) { this.pause(); return; }
    this.timer = setTimeout(() => this.go(this.i + 1), this.steps[this.i].ms || 5200);
  }
  play() { if (this.i >= this.steps.length - 1) this.go(0); this.playing = true; this.bPlay.textContent = "❚❚ Pause"; this.schedule(); }
  pause() { this.playing = false; clearTimeout(this.timer); this.bPlay.textContent = "▶ Play"; }
}

/* --- small shared drawings ---------------------------------------------- */
/* a hidden-state strip: n cells colored by value in [-1, 1] (blue = positive, red = negative) */
function strip(parent, x, y, values, cell = 9, hgt = 18) {
  const g = S("g", { transform: `translate(${x},${y})` }, parent);
  const cells = values.map((v, k) => S("rect", { x: k * cell, y: 0, width: cell - 1, height: hgt, rx: 1.5, cls: "fade" }, g));
  g.cells = cells;
  g.set = (vals) => vals.forEach((v, k) => paintCell(cells[k], v));
  g.set(values);
  return g;
}
function paintCell(rect, v) {
  const a = clamp(Math.abs(v), 0, 1);
  rect.style.fill = v >= 0 ? "var(--h)" : "var(--cut)";
  rect.style.fillOpacity = (0.12 + 0.88 * a).toFixed(3);
}
function vecFrom(seed, n) { const r = rng(seed); return Array.from({ length: n }, () => clamp(gauss(r) * 0.45, -1, 1)); }
