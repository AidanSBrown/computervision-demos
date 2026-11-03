/* Demo 1: an image is a grid of numbers.
 * Left: the picture. Right: a zoomed window where each pixel is labeled with
 * the value(s) stored for it. Sliders change resolution and bit depth. */
(function () {
  "use strict";
  const VIEW = 360;          // CSS size of each panel
  const SOURCE_SIZE = 256;

  const fullCtx = CS.hidpi(document.getElementById("full"), VIEW, VIEW);
  const zoomCanvas = document.getElementById("zoom");
  const zoomCtx = CS.hidpi(zoomCanvas, VIEW, VIEW);
  const readout = document.getElementById("readout");

  const sources = {
    benign: CS.renderLesion(CS.makeLesion(11, false, { hair: false }), SOURCE_SIZE),
    malignant: CS.renderLesion(CS.makeLesion(22, true, { hair: false }), SOURCE_SIZE),
    upload: null,
  };
  let source = sources.benign;

  // Zoom window center, in [0,1] image coordinates.
  let focus = { u: 0.7, v: 0.42 };
  let hover = null;          // pixel under the mouse in the zoom panel
  let processed = null;      // { n, data, values } for the current settings

  const channel = CS.segmented("channel", render);
  const sourceSel = CS.segmented("source", (v) => {
    if (v === "upload") document.getElementById("file").click();
    else { source = sources[v]; render(); }
  });
  const zoomSize = CS.slider("zoomsize", (v) => v + " × " + v, render);
  const res = CS.slider("res", (v) => (1 << v) + " × " + (1 << v), render);
  const levels = CS.slider("levels", (v) => String(1 << v), render);
  const showNums = document.getElementById("shownums");
  showNums.addEventListener("change", render);

  document.getElementById("file").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (!f) { sourceSel.set(source === sources.malignant ? "malignant" : "benign"); return; }
    const img = new Image();
    img.onload = () => {
      // Center-crop to a square, then scale to the working size.
      const side = Math.min(img.width, img.height);
      const c = document.createElement("canvas");
      c.width = c.height = SOURCE_SIZE;
      c.getContext("2d").drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, SOURCE_SIZE, SOURCE_SIZE);
      sources.upload = source = c;
      URL.revokeObjectURL(img.src);
      render();
    };
    img.src = URL.createObjectURL(f);
  });

  /** Downsample, quantize and pick channels. Returns display pixels plus the numbers to print. */
  function process() {
    const n = 1 << res();
    const small = CS.pixels(CS.resize(source, n, n));
    const L = 1 << levels();
    const q = (v) => (L >= 256 ? v : Math.round((Math.round((v / 255) * (L - 1)) * 255) / (L - 1)));
    const mode = channel();
    const data = new Uint8ClampedArray(n * n * 4);
    const values = [];
    for (let i = 0; i < n * n; i++) {
      const r = q(small.data[4 * i]), g = q(small.data[4 * i + 1]), b = q(small.data[4 * i + 2]);
      let out, val;
      if (mode === "rgb") { out = [r, g, b]; val = [r, g, b]; }
      else if (mode === "r") { out = [r, 0, 0]; val = [r]; }
      else if (mode === "g") { out = [0, g, 0]; val = [g]; }
      else if (mode === "b") { out = [0, 0, b]; val = [b]; }
      else { const y = q(Math.round(0.299 * small.data[4 * i] + 0.587 * small.data[4 * i + 1] + 0.114 * small.data[4 * i + 2])); out = [y, y, y]; val = [y]; }
      data.set(out, 4 * i); data[4 * i + 3] = 255;
      values.push(val);
    }
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = n;
    canvas.getContext("2d").putImageData(new ImageData(data, n, n), 0, 0);
    return { n, canvas, data, values, mode, L };
  }

  /** Top-left pixel and size of the zoom window, clamped inside the image. */
  function windowRect(n) {
    const k = Math.min(zoomSize(), n);
    const x0 = CS.clamp(Math.round(focus.u * n - k / 2), 0, n - k);
    const y0 = CS.clamp(Math.round(focus.v * n - k / 2), 0, n - k);
    return { x0, y0, k };
  }

  function render() {
    processed = process();
    const { n, canvas, data, values, mode, L } = processed;
    const { x0, y0, k } = windowRect(n);

    // Left: whole image with the zoom box.
    fullCtx.clearRect(0, 0, VIEW, VIEW);
    CS.drawPixelated(fullCtx, canvas, 0, 0, VIEW, VIEW);
    const px = VIEW / n;
    fullCtx.lineWidth = 2;
    fullCtx.strokeStyle = "#fff";
    fullCtx.strokeRect(x0 * px, y0 * px, k * px, k * px);
    fullCtx.lineWidth = 1;
    fullCtx.strokeStyle = "#111";
    fullCtx.strokeRect(x0 * px - 1.5, y0 * px - 1.5, k * px + 3, k * px + 3);

    // Right: the zoomed window, one square per pixel.
    const cell = VIEW / k;
    zoomCtx.clearRect(0, 0, VIEW, VIEW);
    for (let j = 0; j < k; j++) {
      for (let i = 0; i < k; i++) {
        const idx = (y0 + j) * n + (x0 + i);
        zoomCtx.fillStyle = "rgb(" + data[4 * idx] + "," + data[4 * idx + 1] + "," + data[4 * idx + 2] + ")";
        zoomCtx.fillRect(i * cell, j * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    if (cell >= 6) {
      zoomCtx.strokeStyle = "rgba(255,255,255,0.35)";
      zoomCtx.lineWidth = 1;
      zoomCtx.beginPath();
      for (let t = 1; t < k; t++) {
        zoomCtx.moveTo(t * cell, 0); zoomCtx.lineTo(t * cell, VIEW);
        zoomCtx.moveTo(0, t * cell); zoomCtx.lineTo(VIEW, t * cell);
      }
      zoomCtx.stroke();
    }
    const perPixel = mode === "rgb" ? 3 : 1;
    const fits = cell >= (perPixel === 3 ? 40 : 24);
    if (showNums.checked && fits) {
      const fs = Math.min(13, cell / (perPixel === 3 ? 3.4 : 2.4));
      zoomCtx.font = "500 " + fs + "px 'JetBrains Mono', ui-monospace, monospace";
      zoomCtx.textAlign = "center";
      zoomCtx.textBaseline = "middle";
      const tints = ["#ff8a80", "#9ee69b", "#8ab8ff"];
      for (let j = 0; j < k; j++) {
        for (let i = 0; i < k; i++) {
          const idx = (y0 + j) * n + (x0 + i);
          const lum = 0.299 * data[4 * idx] + 0.587 * data[4 * idx + 1] + 0.114 * data[4 * idx + 2];
          const dark = lum < 120;
          const v = values[idx];
          const cx = (i + 0.5) * cell, cy = (j + 0.5) * cell;
          if (perPixel === 3) {
            for (let c = 0; c < 3; c++) {
              zoomCtx.fillStyle = dark ? tints[c] : ["#8b1a10", "#1d6a1a", "#1a3f8b"][c];
              zoomCtx.fillText(v[c], cx, cy + (c - 1) * fs * 1.15);
            }
          } else {
            zoomCtx.fillStyle = dark ? "#fff" : "#111";
            zoomCtx.fillText(v[0], cx, cy);
          }
        }
      }
    }
    if (hover && hover.i < k && hover.j < k) {
      zoomCtx.strokeStyle = "#ffcc00";
      zoomCtx.lineWidth = 2;
      zoomCtx.strokeRect(hover.i * cell + 1, hover.j * cell + 1, cell - 2, cell - 2);
    }

    // Text readouts.
    const count = n * n * perPixel;
    document.getElementById("dims").textContent = n + " × " + n + " pixels";
    document.getElementById("zoomlabel").textContent = k + " × " + k + " window";
    let txt =
      n + " × " + n + " pixels × " + perPixel + (perPixel === 3 ? " colors" : " channel") + " = " +
      count.toLocaleString() + " numbers, each one of " + L + " possible values.";
    if (!fits && showNums.checked) txt += "\nZoom in further (smaller window) to see the numbers.";
    if (hover && hover.i < k && hover.j < k) {
      const x = x0 + hover.i, y = y0 + hover.j, v = values[y * n + x];
      const names = mode === "rgb" ? ["R", "G", "B"] : [{ r: "Red", g: "Green", b: "Blue", gray: "Gray" }[mode]];
      txt += "\nPixel (" + x + ", " + y + "): " + v.map((val, c) => names[c] + " " + val).join("   ");
    }
    readout.textContent = txt;
  }

  // Drag the zoom box on the left panel.
  const full = document.getElementById("full");
  let dragging = false;
  const moveFocus = (ev) => {
    const p = CS.pointer(full, ev, VIEW, VIEW);
    focus = { u: CS.clamp(p.x / VIEW, 0, 1), v: CS.clamp(p.y / VIEW, 0, 1) };
    render();
  };
  full.style.cursor = "crosshair";
  full.addEventListener("pointerdown", (ev) => { dragging = true; full.setPointerCapture(ev.pointerId); moveFocus(ev); });
  full.addEventListener("pointermove", (ev) => { if (dragging) moveFocus(ev); });
  full.addEventListener("pointerup", () => { dragging = false; });

  zoomCanvas.addEventListener("pointermove", (ev) => {
    const p = CS.pointer(zoomCanvas, ev, VIEW, VIEW);
    const k = windowRect(processed.n).k;
    hover = { i: Math.floor((p.x / VIEW) * k), j: Math.floor((p.y / VIEW) * k) };
    render();
  });
  zoomCanvas.addEventListener("pointerleave", () => { hover = null; render(); });

  document.getElementById("save").addEventListener("click", () =>
    CS.saveFigure(
      [
        { canvas: full, label: "What you see" },
        { canvas: zoomCanvas, label: "What the computer stores" },
      ],
      "fig1-pixels.png"
    )
  );

  render();
  CS.fontsReady.then(render);
})();
