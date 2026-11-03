/* Demo 4: a linear classifier on the two numbers demo 3's network produces.
 * Each synthetic lesion goes through the network (assets/network.js), and its
 * "outline" and "wiggles & patches" averages are plotted. Sliders set the line w1*x1 + w2*x2 + b = 0;
 * "fit" runs logistic regression by gradient descent, live. */
(function () {
  "use strict";
  const N = 160, S = 64;
  const PW = 460, PH = 460;
  const M = { l: 52, r: 14, t: 14, b: 46 };
  const AW = PW - M.l - M.r, AH = PH - M.t - M.b;
  // Starting guess: judge on wiggles alone (a flat line). Training discovers that outline matters too.
  const DEFAULTS = { w1: 0, w2: 8, b: -4 };
  const LW = 260, LH = 110;   // error-curve canvas

  const plot = document.getElementById("plot");
  const ctx = CS.hidpi(plot, PW, PH);
  const thumb = document.getElementById("thumb");
  const tctx = CS.hidpi(thumb, 112, 112);
  const info = document.getElementById("pointinfo");

  // ------------------------------------- run every lesion through demo 3's network

  // The two numbers at the end of demo 3 ("outline" and "wiggles & patches"),
  // with demo 3's default settings.
  const NET = CS.net.DEFAULTS;
  const data = [];
  for (let i = 0; i < N; i++) {
    const malignant = i % 2 === 1;
    const p = CS.makeLesion(2000 + i, malignant, { lookalike: 0.12, hair: false });
    const raw = CS.net.forward(CS.net.lesionImage(p), NET.thresh, NET.pool, NET.type).summary;
    data.push({ malignant, canvas: CS.renderLesion(p, S), raw });
  }
  // Rescale both features to 0..1 so the weights are comparable
  // (2nd-98th percentile, so one odd lesion can't squash the rest).
  for (let f = 0; f < 2; f++) {
    const vals = data.map((d) => d.raw[f]).sort((a, b) => a - b);
    const lo = vals[Math.floor(N * 0.02)], hi = vals[Math.ceil(N * 0.98) - 1];
    const pad = (hi - lo) * 0.08;
    data.forEach((d) => { (d.x || (d.x = []))[f] = CS.clamp((d.raw[f] - lo + pad) / (hi - lo + 2 * pad), 0, 1); });
  }

  // ------------------------------------------------------------ controls

  // The weights live here; sliders only display them (their step would round off small training updates).
  const model = Object.assign({}, DEFAULTS);
  const fmt = (v) => (v > 0 ? "+" : "") + v.toFixed(2);
  const sliders = {};
  // Every training step taken since the line was last set by hand: { w1, w2, b, loss, acc }.
  const history = [];
  ["w1", "w2", "b"].forEach((k) => {
    sliders[k] = CS.slider(k, fmt, (v) => {
      // Moving a slider by hand starts a fresh training run from the new line.
      stopTraining();
      history.length = 0;
      model[k] = v;
      render();
    });
  });
  const w1 = () => model.w1, w2 = () => model.w2, b = () => model.b;
  function setModel(W1, W2, B) {
    Object.assign(model, { w1: W1, w2: W2, b: B });
    ["w1", "w2", "b"].forEach((k) => {
      sliders[k].set(model[k]);
      document.querySelector('output[for="' + k + '"]').textContent = fmt(model[k]);
    });
  }
  let hover = null;
  let fitting = null;

  const score = (x) => w1() * x[0] + w2() * x[1] + b();
  const toPx = (x, y) => [M.l + x * AW, M.t + (1 - y) * AH];

  /** Two points (in feature units, a little past the plot) on the line W1 x + W2 y + B = 0. */
  function linePoints(W1, W2, B) {
    if (Math.abs(W2) > 1e-9) return [-0.1, 1.1].map((x) => [x, -(W1 * x + B) / W2]);
    if (Math.abs(W1) > 1e-9) return [-0.1, 1.1].map((y) => [-B / W1, y]);
    return [];
  }

  function render() {
    ctx.clearRect(0, 0, PW, PH);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, PW, PH);

    // Probability shading: red where "suspicious", teal where "benign".
    const cell = 5;
    for (let py = 0; py < AH; py += cell) {
      for (let px = 0; px < AW; px += cell) {
        const x = (px + cell / 2) / AW, y = 1 - (py + cell / 2) / AH;
        const p = CS.sigmoid(w1() * x + w2() * y + b());
        const a = Math.abs(p - 0.5) * 0.5;
        ctx.fillStyle = p > 0.5 ? "rgba(199,93,102," + a + ")" : "rgba(42,140,130," + a + ")";
        ctx.fillRect(M.l + px, M.t + py, cell, cell);
      }
    }

    // Axes and grid
    ctx.strokeStyle = "#e4e1d8";
    ctx.lineWidth = 1;
    ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = "#6b6a66";
    for (let t = 0; t <= 1.0001; t += 0.25) {
      const [gx] = toPx(t, 0), [, gy] = toPx(0, t);
      ctx.beginPath(); ctx.moveTo(gx, M.t); ctx.lineTo(gx, M.t + AH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(M.l, gy); ctx.lineTo(M.l + AW, gy); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(t.toFixed(2), gx, M.t + AH + 14);
      ctx.textAlign = "right"; ctx.fillText(t.toFixed(2), M.l - 6, gy + 4);
    }
    ctx.strokeStyle = "#9b978c";
    ctx.strokeRect(M.l, M.t, AW, AH);
    ctx.fillStyle = "#1d1d1f";
    ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Outline  (x₁)", M.l + AW / 2, PH - 10);
    ctx.save();
    ctx.translate(14, M.t + AH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Wiggles & patches  (x₂)", 0, 0);
    ctx.restore();

    // Decision line w1 x + w2 y + b = 0, clipped to the plot.
    ctx.save();
    ctx.beginPath(); ctx.rect(M.l, M.t, AW, AH); ctx.clip();

    // Faint trail of earlier lines from the current training run.
    const trail = history.slice(Math.max(0, history.length - 31), -1);
    trail.forEach((h, i) => {
      const tp = linePoints(h.w1, h.w2, h.b);
      if (!tp.length) return;
      ctx.strokeStyle = "rgba(29,29,31," + (0.06 + 0.3 * ((i + 1) / trail.length)) + ")";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo.apply(ctx, toPx(tp[0][0], tp[0][1]));
      ctx.lineTo.apply(ctx, toPx(tp[1][0], tp[1][1]));
      ctx.stroke();
    });

    const pts = linePoints(w1(), w2(), b());
    if (pts.length) {
      ctx.strokeStyle = "#1d1d1f";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo.apply(ctx, toPx(pts[0][0], pts[0][1]));
      ctx.lineTo.apply(ctx, toPx(pts[1][0], pts[1][1]));
      ctx.stroke();
      // Weight vector: points toward the "suspicious" side.
      const n = Math.hypot(w1(), w2());
      if (n > 1e-9) {
        const t = -b() / (n * n);
        let foot = [w1() * t, w2() * t];
        // Slide the arrow's base to the visible middle of the line.
        const mid = [(CS.clamp(pts[0][0], 0, 1) + CS.clamp(pts[1][0], 0, 1)) / 2, (CS.clamp(pts[0][1], 0, 1) + CS.clamp(pts[1][1], 0, 1)) / 2];
        const along = [-w2() / n, w1() / n];
        const s = (mid[0] - foot[0]) * along[0] + (mid[1] - foot[1]) * along[1];
        foot = [foot[0] + along[0] * s, foot[1] + along[1] * s];
        const tip = [foot[0] + (w1() / n) * 0.12, foot[1] + (w2() / n) * 0.12];
        const [ax, ay] = toPx(foot[0], foot[1]), [bx, by] = toPx(tip[0], tip[1]);
        ctx.strokeStyle = "#b5541c";
        ctx.fillStyle = "#b5541c";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        const ang = Math.atan2(by - ay, bx - ax);
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - 9 * Math.cos(ang - 0.4), by - 9 * Math.sin(ang - 0.4));
        ctx.lineTo(bx - 9 * Math.cos(ang + 0.4), by - 9 * Math.sin(ang + 0.4));
        ctx.fill();
        ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
        ctx.fillText("w", bx + 10 * Math.cos(ang), by + 10 * Math.sin(ang) + 4);
      }
    }
    ctx.restore();

    // Points
    let tp = 0, fn = 0, fp = 0, tn = 0;
    data.forEach((d) => {
      const called = score(d.x) > 0;
      if (d.malignant) called ? tp++ : fn++;
      else called ? fp++ : tn++;
      const [x, y] = toPx(d.x[0], d.x[1]);
      const wrong = called !== d.malignant;
      ctx.fillStyle = d.malignant ? "#c75d66" : "#2a8c82";
      ctx.beginPath();
      if (d.malignant) { ctx.moveTo(x, y - 5.5); ctx.lineTo(x + 5.5, y); ctx.lineTo(x, y + 5.5); ctx.lineTo(x - 5.5, y); ctx.closePath(); }
      else ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      if (wrong) {
        ctx.strokeStyle = "#111"; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke();
      }
      if (d === hover) {
        ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.stroke();
      }
    });

    const pct = (a, n) => (n ? Math.round((100 * a) / n) + "%" : "–");
    document.getElementById("tp").textContent = tp;
    document.getElementById("fn").textContent = fn;
    document.getElementById("fp").textContent = fp;
    document.getElementById("tn").textContent = tn;
    document.getElementById("sens").textContent = pct(tp, tp + fn);
    document.getElementById("spec").textContent = pct(tn, tn + fp);
    document.getElementById("acc").textContent = pct(tp + tn, N);
    document.getElementById("formula").textContent =
      "score = " + w1().toFixed(2) + " × x₁ " + (w2() < 0 ? "− " : "+ ") + Math.abs(w2()).toFixed(2) + " × x₂ " +
      (b() < 0 ? "− " : "+ ") + Math.abs(b()).toFixed(2);

    tctx.clearRect(0, 0, 112, 112);
    if (hover) {
      CS.drawPixelated(tctx, hover.canvas, 0, 0, 112, 112);
      const s = score(hover.x), p = CS.sigmoid(s);
      info.textContent =
        "Truth: " + (hover.malignant ? "melanoma" : "benign") +
        "\nOutline x₁ = " + hover.x[0].toFixed(2) + "\nWiggles x₂ = " + hover.x[1].toFixed(2) +
        "\nScore = " + s.toFixed(2) + "\nP(suspicious) = " + Math.round(p * 100) + "%" +
        (s > 0 !== hover.malignant ? "\n← the line gets this one wrong" : "");
    } else {
      tctx.fillStyle = "#f6f5f1";
      tctx.fillRect(0, 0, 112, 112);
    }
    drawLoss();
  }

  plot.addEventListener("pointermove", (ev) => {
    const pt = CS.pointer(plot, ev, PW, PH);
    let best = null, bestD = 12;
    data.forEach((d) => {
      const [x, y] = toPx(d.x[0], d.x[1]);
      const dist = Math.hypot(x - pt.x, y - pt.y);
      if (dist < bestD) { bestD = dist; best = d; }
    });
    if (best !== hover) { hover = best; render(); }
  });
  plot.addEventListener("pointerleave", () => { hover = null; render(); });

  // ------------------------------------------------- logistic regression
  // One training step = score every lesion with the current line, measure the
  // error (log-loss), and move w1, w2 and b a little way downhill.

  const MAX_STEPS = 300;
  const speed = CS.slider("speed", (v) => v + (v === 1 ? " step/s" : " steps/s"));
  const rate = CS.slider("rate");
  const fitBtn = document.getElementById("fit");
  let converged = false;

  /** Error, accuracy and the gradient (which way is uphill) for a given line. */
  function evaluate(W1, W2, B) {
    let g1 = 0, g2 = 0, gb = 0, loss = 0, right = 0;
    for (const d of data) {
      const z = W1 * d.x[0] + W2 * d.x[1] + B, p = CS.sigmoid(z);
      const err = p - (d.malignant ? 1 : 0);
      g1 += err * d.x[0]; g2 += err * d.x[1]; gb += err;
      loss -= Math.log(Math.max(d.malignant ? p : 1 - p, 1e-12));
      if (z > 0 === d.malignant) right++;
    }
    return { grad: [g1 / N, g2 / N, gb / N], loss: loss / N, acc: right / N };
  }

  function record() {
    const e = evaluate(w1(), w2(), b());
    history.push({ w1: w1(), w2: w2(), b: b(), loss: e.loss, acc: e.acc });
    return e;
  }

  /** Take one gradient-descent step. Returns false once there is nothing left to do. */
  function stepOnce() {
    if (!history.length) { record(); converged = false; }
    if (converged || history.length > MAX_STEPS) return false;
    const e = evaluate(w1(), w2(), b());
    const lr = rate();
    setModel(
      CS.clamp(w1() - lr * e.grad[0], -20, 20),
      CS.clamp(w2() - lr * e.grad[1], -20, 20),
      CS.clamp(b() - lr * e.grad[2], -20, 20)
    );
    record();
    // Settled once a step barely changes anything.
    converged = Math.hypot(e.grad[0], e.grad[1], e.grad[2]) * lr < 0.01;
    render();
    return !converged;
  }

  function stopTraining() {
    if (fitting) clearTimeout(fitting);
    fitting = null;
    fitBtn.textContent = "▶ Train";
  }

  function tick() {
    fitting = stepOnce() ? setTimeout(tick, 1000 / speed()) : null;
    if (!fitting) stopTraining();
  }

  fitBtn.addEventListener("click", () => {
    if (fitting) { stopTraining(); return; }
    if (converged) { history.length = 0; }
    fitBtn.textContent = "❚❚ Pause";
    tick();
  });

  document.getElementById("stepbtn").addEventListener("click", () => { stopTraining(); stepOnce(); });

  document.getElementById("reset").addEventListener("click", () => {
    stopTraining();
    history.length = 0;
    converged = false;
    setModel(DEFAULTS.w1, DEFAULTS.w2, DEFAULTS.b);
    render();
  });

  // ------------------------------------------------------- error curve

  const lossCanvas = document.getElementById("losscurve");
  const lctx = CS.hidpi(lossCanvas, LW, LH);

  function drawLoss() {
    const L = { l: 34, r: 10, t: 10, b: 22 };
    const w = LW - L.l - L.r, h = LH - L.t - L.b;
    lctx.clearRect(0, 0, LW, LH);
    lctx.fillStyle = "#fff";
    lctx.fillRect(0, 0, LW, LH);
    lctx.strokeStyle = "#9b978c";
    lctx.lineWidth = 1;
    lctx.strokeRect(L.l, L.t, w, h);
    lctx.font = "10px 'JetBrains Mono', ui-monospace, monospace";
    lctx.fillStyle = "#6b6a66";
    lctx.textAlign = "center";
    lctx.fillText("training step", L.l + w / 2, LH - 6);

    const steps = history.length - 1;
    const label = document.getElementById("steplabel");
    if (history.length < 2) {
      lctx.fillText("press Train or Step once", L.l + w / 2, L.t + h / 2 + 4);
      label.textContent = "";
      return;
    }
    const xMax = Math.max(20, steps);
    // Zoom the y-axis to the range the error actually covers, so small improvements are visible.
    const losses = history.map((p) => p.loss);
    const hi = Math.max.apply(null, losses), lo = Math.min.apply(null, losses);
    const pad = Math.max((hi - lo) * 0.1, 0.005);
    const yMax = hi + pad, yMin = Math.max(0, lo - pad);
    lctx.textAlign = "right";
    lctx.fillText(yMax.toFixed(2), L.l - 4, L.t + 8);
    lctx.fillText(yMin.toFixed(2), L.l - 4, L.t + h);
    const px = (i) => L.l + (i / xMax) * w, py = (v) => L.t + h * (1 - (v - yMin) / (yMax - yMin));
    lctx.strokeStyle = "#b5541c";
    lctx.lineWidth = 2;
    lctx.beginPath();
    history.forEach((p, i) => (i ? lctx.lineTo(px(i), py(p.loss)) : lctx.moveTo(px(i), py(p.loss))));
    lctx.stroke();
    const last = history[steps];
    lctx.fillStyle = "#b5541c";
    lctx.beginPath(); lctx.arc(px(steps), py(last.loss), 3.5, 0, Math.PI * 2); lctx.fill();
    label.textContent = "step " + steps + " · error " + last.loss.toFixed(3) +
      (converged ? " · settled" : steps >= MAX_STEPS ? " · stopped" : "");
  }

  document.getElementById("save").addEventListener("click", () => CS.savePNG(plot, "fig4-decision.png"));

  render();
  CS.fontsReady.then(render);
  // ?trained runs the whole training at once (for figures): shows the fitted line and its trail.
  if (new URLSearchParams(location.search).has("trained")) while (stepOnce());
})();
