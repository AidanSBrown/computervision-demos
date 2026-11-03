/* Shared helpers for the "What does a computer see?" demos.
 *
 * Everything lives on one global, `CS`, so the pages work from GitHub Pages
 * and when opened straight from disk (no build step, no modules).
 *
 * The skin-lesion images are synthetic: drawn from a handful of parameters
 * (size, border wiggle, darkness, color variation...). That keeps the demos
 * free of patient data and licensing questions, and every image is
 * reproducible from its seed.
 */
(function (global) {
  "use strict";
  const CS = {};

  // ---------------------------------------------------------------- math

  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (e0, e1, x) => {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  };
  CS.clamp = clamp;
  CS.lerp = lerp;
  CS.smoothstep = smoothstep;
  CS.sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /** Seeded random number generator (mulberry32). Same seed, same images. */
  CS.rng = function (seed) {
    let a = seed >>> 0;
    const next = function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.range = (lo, hi) => lo + (hi - lo) * next();
    next.int = (lo, hi) => Math.floor(lo + (hi - lo + 1) * next());
    next.pick = (arr) => arr[Math.floor(next() * arr.length)];
    next.normal = () => {
      const u = Math.max(next(), 1e-12);
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
    };
    return next;
  };

  /** Smooth 2D value noise on [0,1]^2, returns values in [0,1]. */
  CS.noise2 = function (rng, cells) {
    const n = cells + 1;
    const g = new Float32Array(n * n);
    for (let i = 0; i < g.length; i++) g[i] = rng();
    return function (u, v) {
      const x = clamp(u, 0, 0.99999) * cells;
      const y = clamp(v, 0, 0.99999) * cells;
      const i = x | 0, j = y | 0;
      let fx = x - i, fy = y - j;
      fx = fx * fx * (3 - 2 * fx);
      fy = fy * fy * (3 - 2 * fy);
      const a = g[j * n + i], b = g[j * n + i + 1];
      const c = g[(j + 1) * n + i], d = g[(j + 1) * n + i + 1];
      return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
    };
  };

  // ------------------------------------------------------ synthetic lesions

  const SKINS = [
    [236, 200, 174],
    [226, 178, 146],
    [204, 150, 112],
    [168, 118, 86],
  ];

  /**
   * Describe one synthetic lesion. Malignant ones tend to be larger, darker,
   * more irregular and more varied in color (the dermatologist's ABCD rule),
   * but the two groups overlap, just as real ones do.
   *
   * opts.lookalike: chance that the lesion borrows the *other* class's
   *   appearance (a melanoma that looks innocent, or vice versa). This caps
   *   how well anything can do from appearance alone.
   * opts.hair: allow hairs (default true).
   */
  CS.makeLesion = function (seed, malignant, opts) {
    opts = opts || {};
    const r = CS.rng(seed * 7919 + 13);
    let looks = !!malignant;
    if (opts.lookalike && r() < opts.lookalike) looks = !looks;
    const pickBy = (lo0, hi0, lo1, hi1) => (looks ? r.range(lo1, hi1) : r.range(lo0, hi0));

    const irregularity = pickBy(0.015, 0.06, 0.07, 0.17);
    const harmonics = [];
    for (let k = 2; k <= 7; k++) {
      harmonics.push({ k, a: irregularity * r.range(0.4, 1) * (k <= 3 ? 1 : 0.5), phase: r() * Math.PI * 2 });
    }
    return {
      seed,
      malignant: !!malignant,
      looksMalignant: looks,
      cx: 0.5 + r.range(-0.035, 0.035),
      cy: 0.5 + r.range(-0.035, 0.035),
      radius: pickBy(0.15, 0.23, 0.2, 0.29),
      elong: pickBy(1, 1.12, 1.05, 1.38),
      angle: r() * Math.PI,
      irregularity,
      harmonics,
      edge: r.range(0.007, 0.016),
      darkness: pickBy(0.25, 0.55, 0.5, 0.88),
      colorVar: pickBy(0.08, 0.25, 0.3, 0.7),
      blue: looks && r() < 0.55,
      red: looks && r() < 0.4,
      skin: r.pick(SKINS),
      skinShade: r.range(-0.05, 0.05),
      hairs: opts.hair === false ? 0 : r() < 0.35 ? r.int(1, 3) : 0,
      noiseSeed: Math.floor(r() * 1e9),
      artifactSeed: Math.floor(r() * 1e9),
    };
  };

  /** Radius of the lesion border in the direction `th` (in lesion units). */
  function borderRadius(p, th) {
    let h = 0;
    for (const hm of p.harmonics) h += hm.a * Math.sin(hm.k * th + hm.phase);
    return Math.max(p.radius * (1 + h), 0.3 * p.radius);
  }

  /**
   * Draw a lesion to a new size x size canvas.
   * art: { ink, ruler, tint } — optional "shortcut" artifacts (see demo 05).
   */
  CS.renderLesion = function (p, size, art) {
    art = art || {};
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(size, size);
    const d = img.data;

    const nr = CS.rng(p.noiseSeed);
    const nPig = CS.noise2(nr, 6);
    const nSkin = CS.noise2(nr, 14);
    const nBlue = CS.noise2(nr, 4);
    const nRed = CS.noise2(nr, 4);
    const ca = Math.cos(p.angle), sa = Math.sin(p.angle);
    const shade = 1 + p.skinShade;

    for (let y = 0; y < size; y++) {
      const v = (y + 0.5) / size;
      for (let x = 0; x < size; x++) {
        const u = (x + 0.5) / size;
        const dx = u - p.cx, dy = v - p.cy;
        const rx = (dx * ca + dy * sa) / p.elong;
        const ry = -dx * sa + dy * ca;
        const rr = Math.hypot(rx, ry);
        const R = borderRadius(p, Math.atan2(ry, rx));
        const inside = 1 - smoothstep(-p.edge, p.edge, rr - R);

        const sn = (0.96 + 0.06 * nSkin(u, v)) * shade;
        let r0 = p.skin[0] * sn, g0 = p.skin[1] * sn, b0 = p.skin[2] * sn;

        if (inside > 0.001) {
          const core = 1 - Math.min(rr / R, 1); // 1 at the center, 0 at the border
          const t = clamp(p.darkness * (0.55 + 0.45 * core) + p.colorVar * (nPig(u, v) - 0.5) * 1.6, 0, 1);
          let lr = lerp(196, 52, t), lg = lerp(146, 32, t), lb = lerp(108, 24, t);
          if (p.blue) {
            const q = smoothstep(0.5, 0.72, nBlue(u, v)) * 0.75;
            lr = lerp(lr, 96, q); lg = lerp(lg, 102, q); lb = lerp(lb, 128, q);
          }
          if (p.red) {
            const q = smoothstep(0.58, 0.8, nRed(u, v)) * 0.55;
            lr = lerp(lr, 168, q); lg = lerp(lg, 78, q); lb = lerp(lb, 80, q);
          }
          r0 = lerp(r0, lr, inside); g0 = lerp(g0, lg, inside); b0 = lerp(b0, lb, inside);
        }
        const i = (y * size + x) * 4;
        d[i] = r0; d[i + 1] = g0; d[i + 2] = b0; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    const ar = CS.rng(p.artifactSeed);
    const s = size;

    for (let h = 0; h < p.hairs; h++) {
      ctx.strokeStyle = "rgba(42, 30, 22, 0.75)";
      ctx.lineWidth = Math.max(0.6, s * 0.006);
      ctx.beginPath();
      ctx.moveTo(ar() * s, 0);
      ctx.quadraticCurveTo(ar() * s, ar() * s, ar() * s, s);
      ctx.stroke();
    }

    if (art.ink) {
      // Purple surgical-marker arc around the upper-left of the lesion.
      ctx.strokeStyle = "rgba(108, 48, 160, 0.88)";
      ctx.lineWidth = Math.max(1.2, s * 0.05);
      ctx.lineCap = "round";
      const rad = (0.37 + ar.range(-0.03, 0.03)) * s;
      const a0 = Math.PI * (1.0 + ar.range(-0.08, 0.08));
      const a1 = a0 + Math.PI * (0.55 + ar.range(-0.08, 0.08));
      ctx.beginPath();
      for (let k = 0; k <= 24; k++) {
        const a = lerp(a0, a1, k / 24);
        const rk = rad * (1 + 0.03 * Math.sin(k * 0.9 + ar() * 0.4));
        const px = p.cx * s + Math.cos(a) * rk, py = p.cy * s + Math.sin(a) * rk;
        k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }

    if (art.ruler) {
      // A measuring ruler laid along the bottom of the photo.
      const top = s * (0.85 + ar.range(-0.02, 0.02));
      ctx.fillStyle = "rgb(240, 240, 234)";
      ctx.fillRect(0, top, s, s - top);
      ctx.fillStyle = "rgb(30, 30, 30)";
      const step = s * 0.05;
      for (let k = 0, x = ar() * step; x < s; k++, x += step) {
        ctx.fillRect(x, top, Math.max(0.7, s * 0.008), s * (k % 2 ? 0.035 : 0.07));
      }
    }

    if (art.tint) {
      // A different hospital's camera: cooler color balance and dark corners.
      const im = ctx.getImageData(0, 0, s, s), q = im.data;
      for (let y = 0; y < s; y++) {
        for (let x = 0; x < s; x++) {
          const dist = Math.hypot((x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5);
          const vig = 1 - 0.5 * smoothstep(0.38, 0.72, dist);
          const i = (y * s + x) * 4;
          q[i] *= 0.84 * vig; q[i + 1] *= 0.95 * vig; q[i + 2] = Math.min(255, q[i + 2] * 1.12 * vig + 8);
        }
      }
      ctx.putImageData(im, 0, 0);
    }
    return c;
  };

  // ------------------------------------------------------- pixels & color

  CS.pixels = (canvas) => canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);

  /** Luminance (0-255) of every pixel, as a Float32Array. */
  CS.gray = function (imageData) {
    const d = imageData.data, out = new Float32Array(d.length / 4);
    for (let i = 0; i < out.length; i++) out[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
    return out;
  };

  /** Resample any canvas/image to a w x h canvas (smooth). */
  CS.resize = function (src, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h || w;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(src, 0, 0, c.width, c.height);
    return c;
  };

  /** Blue - white - red for signed values; t in [-1, 1]. */
  CS.diverging = function (t) {
    t = clamp(t, -1, 1);
    const mid = [247, 246, 242], neg = [38, 104, 178], pos = [199, 93, 102];
    const e = t < 0 ? neg : pos, a = Math.abs(t);
    return [lerp(mid[0], e[0], a), lerp(mid[1], e[1], a), lerp(mid[2], e[2], a)];
  };

  /** Dark-to-bright sequential map for "how strongly did this fire"; t in [0, 1]. */
  const HEAT = [[0, 0, 4], [60, 15, 110], [150, 40, 110], [230, 90, 50], [252, 200, 60], [252, 254, 200]];
  CS.heat = function (t) {
    t = clamp(t, 0, 1) * (HEAT.length - 1);
    const i = Math.min(t | 0, HEAT.length - 2), f = t - i, a = HEAT[i], b = HEAT[i + 1];
    return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f)];
  };

  /** Paint a w x h array of numbers into a canvas through a color map. */
  CS.fieldToCanvas = function (values, w, h, map) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d"), img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const col = map(values[i]);
      img.data[4 * i] = col[0]; img.data[4 * i + 1] = col[1]; img.data[4 * i + 2] = col[2]; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  };

  /** Draw `src` scaled with hard pixel edges (no blurring). */
  CS.drawPixelated = function (ctx, src, x, y, w, h) {
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(src, x, y, w, h);
    ctx.restore();
  };

  // ----------------------------------------------------------- page glue

  /** Size a canvas for crisp drawing on high-DPI screens. Returns its 2D context in CSS pixels. */
  CS.hidpi = function (canvas, w, h) {
    const dpr = Math.max(1, Math.min(3, global.devicePixelRatio || 1));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.aspectRatio = w + " / " + h;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  };

  /** Mouse position in CSS pixels relative to the canvas, accounting for any CSS scaling. */
  CS.pointer = function (canvas, ev, w, h) {
    const r = canvas.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / r.width) * w, y: ((ev.clientY - r.top) / r.height) * h };
  };

  /** Wire up a range slider and its <output>. Returns a getter for the current value. */
  CS.slider = function (id, format, onInput) {
    const el = document.getElementById(id);
    const out = document.querySelector('output[for="' + id + '"]');
    const show = () => { if (out) out.textContent = format ? format(+el.value) : el.value; };
    el.addEventListener("input", () => { show(); if (onInput) onInput(+el.value); });
    show();
    const get = () => +el.value;
    get.set = (v) => { el.value = v; show(); };
    get.el = el;
    return get;
  };

  /** Segmented button group: <div class="seg" id=..><button data-value=..>. */
  CS.segmented = function (id, onChange) {
    const root = document.getElementById(id);
    const buttons = Array.from(root.querySelectorAll("button"));
    let value = (buttons.find((b) => b.getAttribute("aria-pressed") === "true") || buttons[0]).dataset.value;
    const set = (v, fire) => {
      value = v;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
      if (fire && onChange) onChange(v);
    };
    buttons.forEach((b) => b.addEventListener("click", () => set(b.dataset.value, true)));
    set(value, false);
    const get = () => value;
    get.set = set;
    return get;
  };

  /** Save one canvas as a PNG at its full backing resolution. */
  CS.savePNG = function (canvas, filename) {
    canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  };

  /**
   * Stitch several canvases side by side, each with a caption above it,
   * into one white PNG for the article.
   */
  CS.saveFigure = function (panels, filename) {
    const scale = 2, pad = 16, capH = 24;
    const sizes = panels.map((p) => {
      const r = p.canvas.getBoundingClientRect();
      return { w: r.width, h: r.height };
    });
    const W = sizes.reduce((s, z) => s + z.w, 0) + pad * (panels.length + 1);
    const H = Math.max.apply(null, sizes.map((z) => z.h)) + capH + pad * 2;
    const out = document.createElement("canvas");
    out.width = W * scale; out.height = H * scale;
    const ctx = out.getContext("2d");
    ctx.scale(scale, scale);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);
    let x = pad;
    panels.forEach((p, i) => {
      ctx.fillStyle = "#1d1d1f";
      ctx.font = "600 13px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText(p.label || "", x, pad + 14);
      ctx.drawImage(p.canvas, x, pad + capH, sizes[i].w, sizes[i].h);
      x += sizes[i].w + pad;
    });
    CS.savePNG(out, filename);
  };

  /** Resolves once JetBrains Mono is loaded, so canvases can redraw with it (canvas text doesn't wait for web fonts). */
  CS.fontsReady = document.fonts
    ? Promise.all(["400", "500", "600"].map((w) => document.fonts.load(w + " 12px 'JetBrains Mono'"))).catch(() => {})
    : Promise.resolve();

  // ?clean hides the site navigation, for tidy screenshots.
  document.addEventListener("DOMContentLoaded", () => {
    if (new URLSearchParams(location.search).has("clean")) document.body.classList.add("clean");
  });

  global.CS = CS;
})(window);
