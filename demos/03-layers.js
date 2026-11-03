/* Demo 3: a tiny hand-built convolutional network.
 * The network itself lives in assets/network.js (demo 4 reuses it).
 * Input -> 4 edge filters -> ReLU -> pool -> 2 "combination" maps -> pool -> 2 summary numbers.
 * Hovering a map shows its receptive field (the patch of input it depends on). */
(function () {
  "use strict";
  const S = CS.net.S;
  const W = 920, H = 500;
  const M = 78;            // size of each small map on screen
  const IN = 170;          // size of the input on screen
  const TOP = 64;

  const FILTERS = CS.net.FILTERS, L2_NAMES = CS.net.L2_NAMES;
  const L2_SHORT = ["outline", "wiggles"];   // map captions (the full names label the summary bars)

  const canvas = document.getElementById("net");
  const ctx = CS.hidpi(canvas, W, H);
  const readout = document.getElementById("readout");

  const lesionImage = CS.net.lesionImage;
  const refs = {
    benign: lesionImage(CS.makeLesion(11, false, { hair: false })),
    malignant: lesionImage(CS.makeLesion(22, true, { hair: false })),
  };
  let img = refs.benign;
  let hover = null;
  let net = null;

  const thresh = CS.slider("thresh", (v) => v.toFixed(2), render);
  const pool = CS.slider("pool", (v) => v + " × " + v, render);
  const poolType = CS.segmented("pooltype", render);
  CS.segmented("source", (v) => {
    if (v === "random") {
      const seed = 1000 + Math.floor(Math.random() * 1e6);
      img = lesionImage(CS.makeLesion(seed, Math.random() < 0.5, { hair: false }));
    } else img = refs[v];
    render();
  });

  const forward = CS.net.forward;

  // ------------------------------------------------------------- drawing

  // Column layout: x position, title, which maps, and how to color them.
  function columns() {
    const p = pool();
    return [
      { key: "conv", x: 206, title: "Layer 1", maps: net.conv, n: S, signed: true, names: FILTERS.map((f) => f.name) },
      { key: "act", x: 302, title: "ReLU", maps: net.act, n: S, names: FILTERS.map((f) => f.name) },
      { key: "pool1", x: 398, title: "Pool " + p + "×" + p, maps: net.pooled, n: net.n1, names: FILTERS.map((f) => f.name) },
      { key: "l2", x: 512, title: "Layer 2", maps: net.l2, n: net.n1, names: L2_SHORT },
      { key: "pool2", x: 608, title: "Pool " + p + "×" + p, maps: net.l2pooled, n: net.n2, names: L2_SHORT },
    ];
  }

  /** Receptive field (in input pixels) of cell (i, j) of a column. */
  function receptiveField(key, i, j) {
    const p = pool();
    const one = (c) => {
      switch (key) {
        case "conv": case "act": return [c - 1, 3];
        case "pool1": return [c * p - 1, p + 2];
        case "l2": return [(c - 1) * p - 1, 3 * p + 2];
        case "pool2": return [(c * p - 1) * p - 1, (p + 2) * p + 2];
      }
      return [0, S];
    };
    const [x0, w] = one(i), [y0, h] = one(j);
    return { x0, y0, w, h };
  }

  /** Four maps stack top to bottom; two maps sit centered against the same height. */
  function mapPosition(col, m) {
    const step = M + 26;
    const y = col.maps.length === 4 ? TOP + m * step : TOP + step * (0.5 + 2 * m);
    return { x: col.x, y };
  }

  function drawMap(values, n, x, y, signed, maxAbs) {
    const map = signed ? (v) => CS.diverging(v / maxAbs) : (v) => CS.heat(v / maxAbs);
    CS.drawPixelated(ctx, CS.fieldToCanvas(values, n, n, map), x, y, M, M);
    ctx.strokeStyle = "#cfcbc0";
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, M + 1, M + 1);
  }

  function text(s, x, y, opts) {
    opts = opts || {};
    ctx.font = (opts.weight || "400") + " " + (opts.size || 12) + "px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = opts.color || "#1d1d1f";
    ctx.textAlign = opts.align || "left";
    ctx.fillText(s, x, y);
  }

  function arrow(x1, x2, y) {
    ctx.strokeStyle = "#9b978c";
    ctx.fillStyle = "#9b978c";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2 - 5, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y); ctx.lineTo(x2 - 6, y - 4); ctx.lineTo(x2 - 6, y + 4); ctx.fill();
  }

  function render() {
    net = forward(img, thresh(), pool(), poolType());
    const refNets = [forward(refs.benign, thresh(), pool(), poolType()), forward(refs.malignant, thresh(), pool(), poolType())];
    const refFeatures = { benign: refNets[0].summary, malignant: refNets[1].summary };
    const peak = (m) => { let v = 1e-6; for (let i = 0; i < m.length; i++) v = Math.max(v, Math.abs(m[i])); return v; };

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    // Input
    text("Input", 12, 22, { weight: 600, size: 13 });
    text(S + " × " + S + " × 1 · skin = 1", 12, 38, { color: "#6b6a66" });
    // The input is brightness relative to the skin (skin = 1), drawn so skin shows as light gray.
    CS.drawPixelated(ctx, CS.fieldToCanvas(img, S, S, (v) => [v * 215, v * 215, v * 215]), 12, TOP, IN, IN);

    const cols = columns();
    const mid = TOP + 2 * (M + 26) - 13;
    arrow(12 + IN + 4, cols[0].x - 6, TOP + IN / 2);

    cols.forEach((col, ci) => {
      const signed = !!col.signed;
      const colMax = Math.max.apply(null, col.maps.map(peak));
      // Layer 2's maps are on very different scales ("wiggles" is ~100x smaller than "outline"),
      // so each gets its own brightness scale, shared with the same map of both reference lesions.
      const layer2 = col.key === "l2" || col.key === "pool2";
      const ref = (r, mi) => (col.key === "l2" ? r.l2 : r.l2pooled)[mi];
      text(col.title, col.x, 22, { weight: 600, size: 13 });
      text(col.n + " × " + col.n + " × " + col.maps.length, col.x, 38, { color: "#6b6a66" });
      col.maps.forEach((m, mi) => {
        const pos = mapPosition(col, mi);
        const maxAbs = layer2 ? Math.max(peak(m), peak(ref(refNets[0], mi)), peak(ref(refNets[1], mi))) : colMax;
        drawMap(m, col.n, pos.x, pos.y, signed, maxAbs);
        text(col.names[mi], pos.x, pos.y + M + 14, { size: 11, color: "#6b6a66" });
      });
      if (ci < cols.length - 1) arrow(col.x + M + 4, cols[ci + 1].x - 6, mid);
    });

    // Summary numbers
    const sx = 712;
    arrow(cols[4].x + M + 4, sx - 6, mid);
    text("Average", sx, 22, { weight: 600, size: 13 });
    text("2 numbers", sx, 38, { color: "#6b6a66" });
    const scaleMax = L2_NAMES.map((_, k) =>
      Math.max(refFeatures.benign[k], refFeatures.malignant[k], net.summary[k], 1e-6) * 1.15
    );
    const barW = W - sx - 16;
    L2_NAMES.forEach((name, k) => {
      const y = TOP + 70 + k * 130;
      text(name, sx, y, { weight: 600, size: 12 });
      const frac = net.summary[k] / scaleMax[k];
      ctx.fillStyle = "#f1efe9";
      ctx.fillRect(sx, y + 8, barW, 22);
      ctx.fillStyle = "#b5541c";
      ctx.fillRect(sx, y + 8, barW * frac, 22);
      text(net.summary[k].toFixed(net.summary[k] < 0.01 ? 4 : 3), sx + 6, y + 24, { size: 12, color: frac > 0.35 ? "#fff" : "#1d1d1f" });
      // Reference ticks for the two example lesions.
      [["benign", "mole", "#2a8c82"], ["malignant", "melanoma-like", "#c75d66"]].forEach(([key, label, colr], r) => {
        const fx = sx + barW * (refFeatures[key][k] / scaleMax[k]);
        ctx.fillStyle = colr;
        ctx.fillRect(fx - 1, y + 4, 2, 30);
        ctx.font = "400 10px 'JetBrains Mono', ui-monospace, monospace";
        const half = ctx.measureText(label).width / 2;
        text(label, CS.clamp(fx, sx + half, W - 4 - half), y + 48 + r * 13, { size: 10, color: colr, align: "center" });
      });
    });
    text("→ these numbers feed", sx, TOP + 340, { size: 12, color: "#6b6a66" });
    text("   the classifier (Demo 4)", sx, TOP + 355, { size: 12, color: "#6b6a66" });

    // Hover: receptive field
    let info = "";
    if (hover) {
      const col = cols.find((c) => c.key === hover.key);
      if (col) {
        const pos = mapPosition(col, hover.m);
        const cell = M / col.n;
        ctx.strokeStyle = "#ffcc00";
        ctx.lineWidth = 2;
        ctx.strokeRect(pos.x + hover.i * cell, pos.y + hover.j * cell, Math.max(cell, 2), Math.max(cell, 2));
        const rf = receptiveField(col.key, hover.i, hover.j);
        const k = IN / S;
        const x0 = CS.clamp(rf.x0, 0, S), y0 = CS.clamp(rf.y0, 0, S);
        const x1 = CS.clamp(rf.x0 + rf.w, 0, S), y1 = CS.clamp(rf.y0 + rf.h, 0, S);
        ctx.strokeRect(12 + x0 * k, TOP + y0 * k, (x1 - x0) * k, (y1 - y0) * k);
        const v = col.maps[hover.m][hover.j * col.n + hover.i];
        info = "\n" + col.title + " · " + col.names[hover.m] + " · value " + v.toFixed(3) +
          " · depends on a " + rf.w + " × " + rf.h + " pixel patch of the input";
      }
    }

    const p = pool(), n1 = net.n1, n2 = net.n2;
    readout.textContent =
      "Numbers at each stage: " +
      (S * S).toLocaleString() + " → " + (S * S * 4).toLocaleString() + " → " + (S * S * 4).toLocaleString() + " → " +
      (n1 * n1 * 4).toLocaleString() + " → " + (n1 * n1 * 2).toLocaleString() + " → " + (n2 * n2 * 2).toLocaleString() + " → 2" +
      (p === 1 ? "   (pooling is off, so nothing shrinks)" : "") + info;
  }

  canvas.addEventListener("pointermove", (ev) => {
    const pt = CS.pointer(canvas, ev, W, H);
    hover = null;
    for (const col of columns()) {
      for (let m = 0; m < col.maps.length; m++) {
        const pos = mapPosition(col, m);
        if (pt.x >= pos.x && pt.x < pos.x + M && pt.y >= pos.y && pt.y < pos.y + M) {
          hover = { key: col.key, m, i: Math.floor(((pt.x - pos.x) / M) * col.n), j: Math.floor(((pt.y - pos.y) / M) * col.n) };
        }
      }
    }
    render();
  });
  canvas.addEventListener("pointerleave", () => { hover = null; render(); });

  document.getElementById("save").addEventListener("click", () => CS.savePNG(canvas, "fig3-layers.png"));

  render();
  CS.fontsReady.then(render);
})();
