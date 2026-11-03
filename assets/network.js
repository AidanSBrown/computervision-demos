/* The tiny hand-built convolutional network shared by demos 3 and 4.
 * Input -> 4 edge filters -> ReLU -> pool -> 2 "combination" maps -> pool -> 2 summary numbers.
 * Demo 3 draws every stage; demo 4 plots the two summary numbers for many lesions. */
(function () {
  "use strict";
  const S = 96;

  // Layer 1: four oriented edge detectors. Lesions are darker than skin, so
  // e.g. "left edge" fires where brightness drops going left-to-right.
  const FILTERS = [
    { name: "left edges", k: [1, 0, -1, 2, 0, -2, 1, 0, -1] },
    { name: "right edges", k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
    { name: "top edges", k: [1, 2, 1, 0, 0, 0, -1, -2, -1] },
    { name: "bottom edges", k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  ];
  // Layer 2: "outline" adds all four; "wiggles" fires only where opposite
  // edges sit close together (notches and fingers in the border, patches inside).
  // A smooth round lesion has almost none, however big it is.
  const L2_NAMES = ["outline", "wiggles & patches"];
  const OPPOSITE_PAIRS = [[0, 1], [2, 3]];
  // The settings demo 3 opens with, and the ones demo 4 measures its lesions with.
  const DEFAULTS = { thresh: 0.1, pool: 2, type: "max" };

  /**
   * A lesion as a 96 x 96 grid of brightness, divided by the brightness of the
   * surrounding skin (so skin is about 1 on every skin tone). Without this,
   * the edge filters mostly measure skin tone instead of the lesion.
   */
  function lesionImage(p) {
    const g = CS.gray(CS.pixels(CS.renderLesion(p, S)));
    const ring = [];
    for (let i = 0; i < S; i++) ring.push(g[i], g[(S - 1) * S + i], g[i * S], g[i * S + S - 1]);
    ring.sort((a, b) => a - b);
    const skin = Math.max(ring[ring.length >> 1], 1);
    for (let i = 0; i < g.length; i++) g[i] /= skin;
    return g;
  }

  function conv3(src, n, k) {
    const out = new Float32Array(n * n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        let s = 0;
        for (let j = -1; j <= 1; j++) {
          const yy = CS.clamp(y + j, 0, n - 1);
          for (let i = -1; i <= 1; i++) s += k[(j + 1) * 3 + i + 1] * src[yy * n + CS.clamp(x + i, 0, n - 1)];
        }
        out[y * n + x] = s;
      }
    }
    return out;
  }

  function relu(src, t) {
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i] > t ? src[i] - t : 0;
    return out;
  }

  function poolMap(src, n, p, type) {
    const m = Math.floor(n / p), out = new Float32Array(m * m);
    for (let y = 0; y < m; y++) {
      for (let x = 0; x < m; x++) {
        let best = -Infinity, sum = 0;
        for (let j = 0; j < p; j++) for (let i = 0; i < p; i++) {
          const v = src[(y * p + j) * n + x * p + i];
          if (v > best) best = v;
          sum += v;
        }
        out[y * m + x] = type === "max" ? best : sum / (p * p);
      }
    }
    return out;
  }

  const BOX = [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  function forward(image, t, p, type) {
    const conv = FILTERS.map((f) => conv3(image, S, f.k));
    const act = conv.map((c) => relu(c, t));
    const n1 = Math.floor(S / p);
    const pooled = act.map((a) => poolMap(a, S, p, type));
    const blurred = pooled.map((m) => conv3(m, n1, BOX));
    const outline = new Float32Array(n1 * n1), wiggles = new Float32Array(n1 * n1);
    for (let i = 0; i < n1 * n1; i++) {
      outline[i] = blurred[0][i] + blurred[1][i] + blurred[2][i] + blurred[3][i];
      for (const [a, b] of OPPOSITE_PAIRS) wiggles[i] += Math.min(blurred[a][i], blurred[b][i]);
    }
    const n2 = Math.floor(n1 / p);
    const l2 = [outline, wiggles];
    const l2pooled = l2.map((m) => poolMap(m, n1, p, type));
    return { conv, act, pooled, l2, l2pooled, n1, n2, summary: l2pooled.map(mean) };
  }

  CS.net = { S, FILTERS, L2_NAMES, DEFAULTS, lesionImage, forward };
})();
