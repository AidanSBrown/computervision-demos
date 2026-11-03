/* Demo 5: shortcut learning ("Clever Hans").
 * A logistic-regression model is trained on raw pixels of 600 synthetic
 * lesion photos. An artifact (ink mark, ruler, or a different camera)
 * appears mostly on the melanoma photos. The model learns to use it, and
 * accuracy collapses when the artifact stops tracking the diagnosis.
 *
 * Logistic regression on pixels is the simplest possible "neural network"
 * (one layer), and its weights can be drawn directly as an image, so we can
 * see exactly what it latched on to. Deep networks fall for the same trap;
 * they are just harder to inspect. */
(function () {
  "use strict";
  const S = 24, D = S * S * 3;      // model input: 24 x 24 RGB = 1,728 numbers
  const R = 48;                     // render size for thumbnails
  const N_TRAIN = 600, N_TEST = 300;
  const LOOKALIKE = 0.2;            // 1 in 5 lesions look like the other class
  const EPOCHS = 150, LR = 0.004, L2 = 0.002;
  // Animate 5 passes per frame; in screenshot mode (?clean) train in one go so captures show the finished model.
  const PER_FRAME = new URLSearchParams(location.search).has("clean") ? EPOCHS : 5;
  const PAIR = 168;                 // display size of the try-it images

  const KINDS = {
    ink: {
      label: "ink mark", added: "+ ink mark",
      onBenign: "the ink mark shows up on benign moles", absent: "where nobody marks lesions",
      hint: "Before the photo, a clinician circles a worrying spot with a purple skin marker.",
    },
    ruler: {
      label: "ruler", added: "+ ruler",
      onBenign: "benign moles are photographed with a ruler", absent: "where nobody photographs a ruler",
      hint: "Worrying lesions get measured, so a ruler ends up in the photo.",
    },
    tint: {
      label: "clinic camera", added: "clinic camera",
      onBenign: "benign moles are photographed on that camera", absent: "with an ordinary camera",
      hint: "Suspicious cases are referred to a clinic whose camera gives cooler colors and dark corners.",
    },
  };

  // ------------------------------------------------------------- data

  function makeSamples(n, seedBase) {
    const r = CS.rng(seedBase);
    const out = [];
    for (let i = 0; i < n; i++) {
      const malignant = i % 2 === 0;
      // u decides, once and for all, whether this photo gets the artifact.
      out.push({ malignant, p: CS.makeLesion(seedBase + i, malignant, { lookalike: LOOKALIKE }), u: r() });
    }
    return out;
  }
  const train = makeSamples(N_TRAIN, 50000);
  const test = makeSamples(N_TEST, 90000);

  const cache = new Map();
  /** Render a sample with or without an artifact; returns the thumbnail and the model's 1,728 inputs. */
  function photo(sample, kind) {
    const key = sample.p.seed + ":" + kind;
    let hit = cache.get(key);
    if (!hit) {
      const art = kind === "none" ? {} : { [kind]: true };
      const canvas = CS.renderLesion(sample.p, R, art);
      const px = CS.pixels(CS.resize(canvas, S, S)).data;
      const x = new Float32Array(D);
      for (let k = 0, j = 0; k < S * S; k++) {
        x[j++] = px[4 * k] / 255; x[j++] = px[4 * k + 1] / 255; x[j++] = px[4 * k + 2] / 255;
      }
      hit = { canvas, x };
      cache.set(key, hit);
    }
    return hit;
  }

  /** Chance that a training photo carries the artifact, by diagnosis. */
  const probs = (s) => ({ mal: 0.15 + 0.8 * s, ben: 0.15 * (1 - s) });

  function hasArtifact(sample, set, s) {
    if (set === "new") return false;
    if (set === "flipped") return !sample.malignant;
    const pr = probs(s);
    return sample.u < (sample.malignant ? pr.mal : pr.ben);
  }

  function matrix(samples, set, kind, s) {
    const n = samples.length, X = new Float32Array(n * D), y = new Float32Array(n);
    samples.forEach((smp, k) => {
      X.set(photo(smp, hasArtifact(smp, set, s) ? kind : "none").x, k * D);
      y[k] = smp.malignant ? 1 : 0;
    });
    return { X, y, n };
  }

  // -------------------------------------------------------- the model

  let run = null;
  let raf = null;

  function score(x, centered) {
    let z = run.b;
    const w = run.w, mu = run.mu;
    if (centered) for (let j = 0; j < D; j++) z += w[j] * x[j];
    else for (let j = 0; j < D; j++) z += w[j] * (x[j] - mu[j]);
    return z;
  }

  function accuracy(set) {
    let right = 0;
    for (let k = 0; k < set.n; k++) {
      const z = score(set.X.subarray(k * D, (k + 1) * D), true);
      if ((z > 0) === (set.y[k] === 1)) right++;
    }
    return right / set.n;
  }

  /** One pass over the training photos: gradient of the log-loss, Adam update. */
  function epoch() {
    const { w, g, m1, m2, tr } = run;
    g.fill(0);
    let gb = 0;
    for (let k = 0; k < tr.n; k++) {
      const off = k * D;
      let z = run.b;
      for (let j = 0; j < D; j++) z += w[j] * tr.X[off + j];
      const e = CS.sigmoid(z) - tr.y[k];
      gb += e;
      for (let j = 0; j < D; j++) g[j] += e * tr.X[off + j];
    }
    run.t++;
    const c1 = 1 - Math.pow(0.9, run.t), c2 = 1 - Math.pow(0.999, run.t);
    for (let j = 0; j < D; j++) {
      const gj = g[j] / tr.n + L2 * w[j];
      m1[j] = 0.9 * m1[j] + 0.1 * gj;
      m2[j] = 0.999 * m2[j] + 0.001 * gj * gj;
      w[j] -= (LR * (m1[j] / c1)) / (Math.sqrt(m2[j] / c2) + 1e-8);
    }
    gb /= tr.n;
    run.mb = 0.9 * run.mb + 0.1 * gb;
    run.vb = 0.999 * run.vb + 0.001 * gb * gb;
    run.b -= (LR * (run.mb / c1)) / (Math.sqrt(run.vb / c2) + 1e-8);
    run.epoch++;
  }

  function startTraining() {
    if (raf) cancelAnimationFrame(raf);
    const kind = kindSel(), s = strength() / 100;
    const tr = matrix(train, "train", kind, s);
    const sets = {
      same: matrix(test, "train", kind, s),
      fresh: matrix(test, "new", kind, s),
      flipped: matrix(test, "flipped", kind, s),
    };
    // Center every input on the training average.
    const mu = new Float32Array(D);
    for (let k = 0; k < tr.n; k++) for (let j = 0; j < D; j++) mu[j] += tr.X[k * D + j];
    for (let j = 0; j < D; j++) mu[j] /= tr.n;
    [tr, sets.same, sets.fresh, sets.flipped].forEach((m) => {
      for (let k = 0; k < m.n; k++) for (let j = 0; j < D; j++) m.X[k * D + j] -= mu[j];
    });
    run = {
      kind, s, tr, sets, mu,
      w: new Float32Array(D), g: new Float32Array(D), m1: new Float32Array(D), m2: new Float32Array(D),
      b: 0, mb: 0, vb: 0, t: 0, epoch: 0,
    };
    drawThumbs();
    const step = () => {
      for (let e = 0; e < PER_FRAME && run.epoch < EPOCHS; e++) epoch();
      drawModel();
      raf = run.epoch < EPOCHS ? requestAnimationFrame(step) : null;
    };
    raf = requestAnimationFrame(step);
  }

  // ---------------------------------------------------------- drawing

  const thumbsEl = document.getElementById("thumbs");
  const thumbCanvases = [];
  ["Melanoma", "Benign"].forEach((label, row) => {
    const l = document.createElement("div");
    l.className = "rowlabel";
    l.textContent = label;
    thumbsEl.appendChild(l);
    for (let i = 0; i < 8; i++) {
      const c = document.createElement("canvas");
      c.width = c.height = R;
      thumbsEl.appendChild(c);
      thumbCanvases.push({ c, row, i });
    }
  });

  function drawThumbs() {
    const mal = train.filter((t) => t.malignant), ben = train.filter((t) => !t.malignant);
    thumbCanvases.forEach(({ c, row, i }) => {
      const smp = (row === 0 ? mal : ben)[i];
      const img = photo(smp, hasArtifact(smp, "train", run.s) ? run.kind : "none").canvas;
      c.getContext("2d").drawImage(img, 0, 0);
    });
    const pr = probs(run.s);
    document.getElementById("trainlabel").textContent = "8 of 600 · the " + KINDS[run.kind].label +
      " appears on " + Math.round(pr.mal * 100) + "% of melanoma, " + Math.round(pr.ben * 100) + "% of benign";
  }

  const tplCanvas = document.getElementById("template");
  const tplCtx = CS.hidpi(tplCanvas, 168, 168);
  const pairCanvas = document.getElementById("pair");
  const PW = PAIR * 2 + 28, PH = PAIR + 64;
  const pairCtx = CS.hidpi(pairCanvas, PW, PH);
  const resCanvas = document.getElementById("results");
  const RW = 400, RH = 196;
  const resCtx = CS.hidpi(resCanvas, RW, RH);
  const overlayBox = document.getElementById("overlay");

  /** Scale for drawing weights: 99th percentile of |w|, so one extreme pixel doesn't wash out the rest. */
  function weightScale() {
    const a = Array.from(run.w, Math.abs).sort((p, q) => p - q);
    return Math.max(a[Math.floor(a.length * 0.99)], 1e-6);
  }

  function drawTemplate() {
    const sc = weightScale();
    const img = new ImageData(S, S);
    for (let k = 0; k < S * S; k++) {
      for (let c = 0; c < 3; c++) img.data[4 * k + c] = 255 * CS.clamp(0.5 + 0.5 * (run.w[3 * k + c] / sc), 0, 1);
      img.data[4 * k + 3] = 255;
    }
    const c = document.createElement("canvas");
    c.width = c.height = S;
    c.getContext("2d").putImageData(img, 0, 0);
    tplCtx.clearRect(0, 0, 168, 168);
    CS.drawPixelated(tplCtx, c, 0, 0, 168, 168);
  }

  // The benign mole used in the side-by-side comparison.
  const benignTest = test.filter((t) => !t.malignant && !t.p.looksMalignant);
  let pairIndex = 3;
  const bigCache = new Map();
  function bigPhoto(sample, kind) {
    const key = sample.p.seed + ":" + kind;
    if (!bigCache.has(key)) bigCache.set(key, CS.renderLesion(sample.p, PAIR, kind === "none" ? {} : { [kind]: true }));
    return bigCache.get(key);
  }

  /** Per-pixel evidence toward "melanoma": sum over colors of weight x (pixel - average). */
  function evidence(x) {
    const ev = new Float32Array(S * S);
    for (let k = 0; k < S * S; k++) {
      let s = 0;
      for (let c = 0; c < 3; c++) s += run.w[3 * k + c] * (x[3 * k + c] - run.mu[3 * k + c]);
      ev[k] = s;
    }
    return ev;
  }

  function drawPair() {
    const ctx = pairCtx;
    ctx.clearRect(0, 0, PW, PH);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, PW, PH);
    const smp = benignTest[pairIndex % benignTest.length];
    const versions = [
      { kind: "none", label: "Original photo" },
      { kind: run.kind, label: "Same mole, " + KINDS[run.kind].added },
    ].map((v) => {
      const x = photo(smp, v.kind).x;
      return Object.assign(v, { x, z: score(x, false), ev: evidence(x) });
    });
    let evMax = 1e-6;
    versions.forEach((v) => v.ev.forEach((e) => { evMax = Math.max(evMax, e); }));

    versions.forEach((v, i) => {
      const x0 = i * (PAIR + 28);
      ctx.drawImage(bigPhoto(smp, v.kind), x0, 0, PAIR, PAIR);
      if (overlayBox.checked) {
        // Orange-yellow glow where a pixel pushes the score toward "melanoma".
        const img = new ImageData(S, S);
        for (let k = 0; k < S * S; k++) {
          const a = CS.smoothstep(0.2, 1, v.ev[k] / evMax);
          img.data[4 * k] = 255; img.data[4 * k + 1] = 70 + 150 * a; img.data[4 * k + 2] = 0;
          img.data[4 * k + 3] = 255 * a * 0.8;
        }
        const c = document.createElement("canvas");
        c.width = c.height = S;
        c.getContext("2d").putImageData(img, 0, 0);
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(c, x0, 0, PAIR, PAIR);
        ctx.restore();
      }
      const p = CS.sigmoid(v.z);
      ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillStyle = "#1d1d1f";
      ctx.textAlign = "left";
      ctx.fillText(v.label, x0, PAIR + 18);
      ctx.fillStyle = "#f1efe9";
      ctx.fillRect(x0, PAIR + 28, PAIR, 22);
      ctx.fillStyle = p > 0.5 ? "#c75d66" : "#2a8c82";
      ctx.fillRect(x0, PAIR + 28, PAIR * p, 22);
      ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
      // Dark text with a white halo stays readable whatever the bar length.
      const label = Math.round(p * 100) + "% melanoma";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.strokeText(label, x0 + 6, PAIR + 43);
      ctx.fillStyle = "#1d1d1f";
      ctx.fillText(label, x0 + 6, PAIR + 43);
    });
  }

  function drawResults() {
    const ctx = resCtx;
    ctx.clearRect(0, 0, RW, RH);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, RW, RH);
    const rows = [
      ["Training photos", accuracy(run.tr), "#6b6a66"],
      ["Test: same hospital", accuracy(run.sets.same), "#3d4d66"],
      ["Test: new hospital", accuracy(run.sets.fresh), "#b5541c"],
      ["Test: marks on benign", accuracy(run.sets.flipped), "#c75d66"],
    ];
    const lx = 176, bw = RW - lx - 14, top = 14, bh = 30, gap = 12;
    rows.forEach(([label, acc, colr], i) => {
      const y = top + i * (bh + gap);
      ctx.font = "12px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillStyle = "#1d1d1f";
      ctx.textAlign = "right";
      ctx.fillText(label, lx - 8, y + bh / 2 + 4);
      ctx.fillStyle = "#f1efe9";
      ctx.fillRect(lx, y, bw, bh);
      ctx.fillStyle = colr;
      ctx.fillRect(lx, y, bw * acc, bh);
      ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = acc > 0.18 ? "#fff" : "#1d1d1f";
      ctx.fillText(Math.round(acc * 100) + "%", lx + 6, y + bh / 2 + 4);
    });
    // Coin-flip line
    const cx = lx + bw * 0.5, bottom = top + rows.length * (bh + gap) - gap;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#1d1d1f";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, top - 6); ctx.lineTo(cx, bottom + 4); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = "#6b6a66";
    ctx.textAlign = "center";
    ctx.fillText("coin flip (50%)", cx, bottom + 18);
    return rows.map((r) => r[1]);
  }

  function drawModel() {
    drawTemplate();
    drawPair();
    const [, same, fresh, flipped] = drawResults();
    document.getElementById("epochlabel").textContent = "training pass " + run.epoch + " / " + EPOCHS;
    const label = KINDS[run.kind].label;
    const pct = (v) => Math.round(v * 100) + "%";
    let verdict;
    if (run.epoch < EPOCHS) verdict = "Training…";
    else if (flipped < 0.45) {
      verdict = "The model mostly reads the " + label + ", not the lesion. It scores " + pct(same) +
        " at its own hospital, but once " + KINDS[run.kind].onBenign + " it does worse than a coin flip (" + pct(flipped) + ").";
    } else if (same - fresh > 0.06 || same - flipped > 0.1) {
      verdict = "Part of what the model learned is the " + label + ". It scores " + pct(same) +
        " at its own hospital but " + pct(fresh) + " " + KINDS[run.kind].absent + ".";
    } else {
      verdict = "With little link between the " + label + " and the diagnosis, the model has to learn the lesion itself. " +
        "Its scores are lower, but they hold up at a new hospital.";
    }
    document.getElementById("verdict").textContent = verdict;
  }

  // --------------------------------------------------------- controls

  const kindSel = CS.segmented("kind", () => { updateHints(); startTraining(); });
  const strength = CS.slider("strength", (v) => v + "%", updateHints);
  strength.el.addEventListener("change", startTraining);
  document.getElementById("retrain").addEventListener("click", startTraining);
  document.getElementById("another").addEventListener("click", () => { pairIndex++; if (run) drawPair(); });
  overlayBox.addEventListener("change", () => { if (run) drawPair(); });

  function updateHints() {
    const pr = probs(strength() / 100);
    document.getElementById("kindhint").textContent = KINDS[kindSel()].hint;
    document.getElementById("strengthhint").textContent =
      "In training, the " + KINDS[kindSel()].label + " appears on " + Math.round(pr.mal * 100) +
      "% of melanoma photos and " + Math.round(pr.ben * 100) + "% of benign ones.";
  }

  document.getElementById("save").addEventListener("click", () =>
    CS.saveFigure(
      [
        { canvas: tplCanvas, label: "What the model looks for" },
        { canvas: pairCanvas, label: "Same benign mole, two photos" },
        { canvas: resCanvas, label: "Accuracy on each test set" },
      ],
      "fig5-shortcut-" + run.kind + ".png"
    )
  );

  // Optional settings from the address, e.g. ?kind=ruler&strength=0 (for links and figures).
  const params = new URLSearchParams(location.search);
  if (KINDS[params.get("kind")]) kindSel.set(params.get("kind"));
  if (params.has("strength")) strength.set(CS.clamp(Math.round(+params.get("strength") / 5) * 5, 0, 100));

  updateHints();
  startTraining();
  CS.fontsReady.then(() => { if (run && run.epoch) drawModel(); });
})();
