/* Demo 2: convolution, step by step.
 * A 3x3 kernel slides over a 64x64 grayscale lesion. The close-up shows the
 * nine pixel values, the nine weights, their products and the sum. */
(function () {
  "use strict";
  const S = 64;          // working image size
  const VIEW = 320;      // CSS size of input/output panels
  const CW = 700, CH = 180;

  const PRESETS = {
    vertical: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    horizontal: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    outline: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    blur: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
    sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
  };

  const inputCanvas = document.getElementById("input");
  const outputCanvas = document.getElementById("output");
  const closeCanvas = document.getElementById("closeup");
  const inCtx = CS.hidpi(inputCanvas, VIEW, VIEW);
  const outCtx = CS.hidpi(outputCanvas, VIEW, VIEW);
  const cuCtx = CS.hidpi(closeCanvas, CW, CH);

  const images = {
    benign: CS.gray(CS.pixels(CS.renderLesion(CS.makeLesion(11, false, { hair: false }), S))),
    malignant: CS.gray(CS.pixels(CS.renderLesion(CS.makeLesion(22, true, { hair: false }), S))),
  };
  let img = images.benign;
  let kernel = PRESETS.vertical.slice();
  let out = new Float32Array(S * S);
  let reveal = S * S;    // how many output pixels are shown (for the scan animation)
  let scanning = null;

  // ---- kernel editor
  const kernelEl = document.getElementById("kernel");
  const kInputs = [];
  for (let i = 0; i < 9; i++) {
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "0.1";
    inp.setAttribute("aria-label", "Kernel weight " + (i + 1));
    inp.addEventListener("input", () => {
      kernel[i] = parseFloat(inp.value) || 0;
      preset.value = "custom";
      update();
    });
    kernelEl.appendChild(inp);
    kInputs.push(inp);
  }
  const fmtW = (w) => (Number.isInteger(w) ? String(w) : w.toFixed(2).replace(/0+$/, ""));
  const showKernel = () => kernel.forEach((w, i) => (kInputs[i].value = fmtW(w)));

  const preset = document.getElementById("preset");
  preset.addEventListener("change", () => {
    if (PRESETS[preset.value]) { kernel = PRESETS[preset.value].slice(); showKernel(); update(); }
  });

  CS.segmented("source", (v) => { img = images[v]; update(); });
  const px = CS.slider("px", null, () => { stopScan(); draw(); });
  const py = CS.slider("py", null, () => { stopScan(); draw(); });
  const absBox = document.getElementById("abs");
  absBox.addEventListener("change", draw);

  // ---- the convolution itself (edges are handled by repeating the border pixel)
  const at = (x, y) => img[CS.clamp(y, 0, S - 1) * S + CS.clamp(x, 0, S - 1)];
  function convolve() {
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        let s = 0;
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) s += kernel[(j + 1) * 3 + (i + 1)] * at(x + i, y + j);
        out[y * S + x] = s;
      }
    }
  }

  /** How to color the output: signed (blue/red), strength (heat) or plain brightness. */
  function outputMap() {
    const sum = kernel.reduce((a, b) => a + b, 0);
    const signed = kernel.some((w) => w < 0) && Math.abs(sum) < 0.5;
    let maxAbs = 1e-6;
    for (let i = 0; i < out.length; i++) maxAbs = Math.max(maxAbs, Math.abs(out[i]));
    if (absBox.checked) return { kind: "strength", map: (v) => CS.heat(Math.abs(v) / maxAbs), maxAbs };
    if (signed) return { kind: "signed", map: (v) => CS.diverging(v / maxAbs), maxAbs };
    return { kind: "brightness", map: (v) => { const g = CS.clamp(v, 0, 255); return [g, g, g]; }, maxAbs };
  }

  function update() {
    convolve();
    draw();
  }

  function draw() {
    const x = px(), y = py();
    const cell = VIEW / S;

    // Input with the 3x3 window.
    const inCanvas = CS.fieldToCanvas(img, S, S, (v) => [v, v, v]);
    inCtx.clearRect(0, 0, VIEW, VIEW);
    CS.drawPixelated(inCtx, inCanvas, 0, 0, VIEW, VIEW);
    inCtx.lineWidth = 2;
    inCtx.strokeStyle = "#ffcc00";
    inCtx.strokeRect((x - 1) * cell, (y - 1) * cell, 3 * cell, 3 * cell);

    // Output, possibly only partly revealed while scanning.
    const om = outputMap();
    const shown = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) shown[i] = i < reveal ? out[i] : NaN;
    const outImg = CS.fieldToCanvas(shown, S, S, (v) => (Number.isNaN(v) ? [236, 234, 228] : om.map(v)));
    outCtx.clearRect(0, 0, VIEW, VIEW);
    CS.drawPixelated(outCtx, outImg, 0, 0, VIEW, VIEW);
    outCtx.lineWidth = 2;
    outCtx.strokeStyle = "#ffcc00";
    outCtx.strokeRect(x * cell, y * cell, cell, cell);

    document.getElementById("outlabel").textContent =
      om.kind === "signed" ? "red = positive, blue = negative" : om.kind === "strength" ? "bright = strong edge" : "brightness";
    document.getElementById("poslabel").textContent = "at x = " + x + ", y = " + y;
    drawCloseup(x, y, om);
  }

  function drawGrid(ctx, x0, y0, c, values, colorOf, label, fmt) {
    ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = "#6b6a66";
    ctx.textAlign = "center";
    ctx.fillText(label, x0 + 1.5 * c, y0 - 8);
    ctx.font = "12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textBaseline = "middle";
    for (let k = 0; k < 9; k++) {
      const i = k % 3, j = (k / 3) | 0;
      const col = colorOf(values[k]);
      ctx.fillStyle = "rgb(" + col.map(Math.round).join(",") + ")";
      ctx.fillRect(x0 + i * c, y0 + j * c, c, c);
      const lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2];
      ctx.fillStyle = lum < 120 ? "#fff" : "#111";
      ctx.fillText(fmt(values[k]), x0 + (i + 0.5) * c, y0 + (j + 0.5) * c);
    }
    ctx.strokeStyle = "#bdb9ae";
    ctx.lineWidth = 1;
    for (let t = 0; t <= 3; t++) {
      ctx.beginPath(); ctx.moveTo(x0 + t * c, y0); ctx.lineTo(x0 + t * c, y0 + 3 * c); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0, y0 + t * c); ctx.lineTo(x0 + 3 * c, y0 + t * c); ctx.stroke();
    }
    ctx.textBaseline = "alphabetic";
  }

  function drawCloseup(x, y, om) {
    const ctx = cuCtx;
    ctx.clearRect(0, 0, CW, CH);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, CW, CH);
    const c = 42, y0 = 34;
    const patch = [], prods = [];
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) patch.push(at(x + i, y + j));
    for (let k = 0; k < 9; k++) prods.push(patch[k] * kernel[k]);
    const kMax = Math.max(1e-6, ...kernel.map(Math.abs));
    const pMax = Math.max(1e-6, ...prods.map(Math.abs));
    const round = (v) => String(Math.round(v));

    drawGrid(ctx, 14, y0, c, patch, (v) => [v, v, v], "pixels", round);
    drawOp(ctx, "×", 14 + 3 * c + 20, y0 + 1.5 * c);
    drawGrid(ctx, 14 + 3 * c + 40, y0, c, kernel, (w) => CS.diverging(w / kMax), "filter weights", fmtW);
    drawOp(ctx, "=", 14 + 6 * c + 60, y0 + 1.5 * c);
    drawGrid(ctx, 14 + 6 * c + 80, y0, c, prods, (v) => CS.diverging(v / pMax), "products", round);

    const sum = out[y * S + x];
    const sx = 14 + 9 * c + 128;
    drawOp(ctx, "add →", sx - 4, y0 + 1.5 * c);
    const col = om.map(sum);
    ctx.fillStyle = "rgb(" + col.map(Math.round).join(",") + ")";
    ctx.fillRect(sx + 34, y0 + c, c * 1.6, c);
    ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 2;
    ctx.strokeRect(sx + 34, y0 + c, c * 1.6, c);
    const lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2];
    ctx.fillStyle = lum < 120 ? "#fff" : "#111";
    ctx.font = "600 13px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(round(sum), sx + 34 + 0.8 * c, y0 + 1.5 * c);
    ctx.font = "600 12px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = "#6b6a66"; ctx.textBaseline = "alphabetic";
    ctx.fillText("output pixel", sx + 34 + 0.8 * c, y0 - 8);
  }

  function drawOp(ctx, s, x, y) {
    ctx.font = "500 18px 'JetBrains Mono', ui-monospace, monospace";
    ctx.fillStyle = "#1d1d1f";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(s, x, y);
    ctx.textBaseline = "alphabetic";
  }

  // ---- scan animation: move the window across every pixel, revealing the output
  const scanBtn = document.getElementById("scan");
  function stopScan() {
    if (scanning) cancelAnimationFrame(scanning);
    scanning = null;
    reveal = S * S;
    scanBtn.textContent = "▶ Scan the whole image";
  }
  scanBtn.addEventListener("click", () => {
    if (scanning) { stopScan(); draw(); return; }
    reveal = 0;
    scanBtn.textContent = "■ Stop";
    const step = () => {
      reveal = Math.min(S * S, reveal + 10);
      const i = reveal - 1;
      px.set(i % S); py.set((i / S) | 0);
      draw();
      if (reveal < S * S) scanning = requestAnimationFrame(step);
      else stopScan();
    };
    scanning = requestAnimationFrame(step);
  });

  // Click the input to move the filter.
  inputCanvas.style.cursor = "crosshair";
  inputCanvas.addEventListener("pointerdown", (ev) => {
    const p = CS.pointer(inputCanvas, ev, VIEW, VIEW);
    stopScan();
    px.set(CS.clamp(Math.floor((p.x / VIEW) * S), 0, S - 1));
    py.set(CS.clamp(Math.floor((p.y / VIEW) * S), 0, S - 1));
    draw();
  });

  document.getElementById("save").addEventListener("click", () =>
    CS.saveFigure(
      [
        { canvas: inputCanvas, label: "Input image" },
        { canvas: outputCanvas, label: "After the " + preset.options[preset.selectedIndex].text.toLowerCase() + " filter" },
      ],
      "fig2-filter-" + preset.value + ".png"
    )
  );

  showKernel();
  update();
  CS.fontsReady.then(draw);
})();
