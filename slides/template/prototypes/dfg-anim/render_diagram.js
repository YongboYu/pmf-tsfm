/* PROTOTYPE — wipe me when slide 3 is finalised.
 * Composes the full slide-3 figure for two layouts (woven full-width | two-column):
 *   event-log slice  ->  morphing DFG (reuses render_svg.js)  ->  accumulating
 *   hero-edge sparkline  ->  forecast, with the 4 workflow step-labels.
 * Presenter-paced: master step() advances the DFG morph, grows the sparkline a
 * point, retitles the log slice, and reveals step ④ at the forecast frame.
 * Exposes window.renderDiagram(root, data, opts) -> {step, reset, setFrame, frameCount, current}.
 */
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const svgEl = (t, a) => {
    const n = document.createElementNS(SVGNS, t);
    for (const k in a) n.setAttribute(k, a[k]);
    return n;
  };
  const div = (cls, html) => {
    const d = document.createElement("div");
    if (cls) d.className = cls;
    if (html != null) d.innerHTML = html;
    return d;
  };

  const STEPS = [
    "Event log → weekly sublog",
    "Each week → a time-indexed DFG",
    "Each DF edge → a univariate time series",
    "Forecast next week → forecasted DFG",
  ];

  // ---- event-log slice motif (stylised, illustrative) -------------------------
  // The log is tiled into `nFrames` consecutive weeks; a dashed band slides right
  // one slice per click, so you see each click grab a later time-slice of the log.
  function buildLog(slot, accent, nFrames) {
    slot.innerHTML = "";
    const rows = 6,
      sliceCols = 2,
      cols = sliceCols * nFrames; // one slice (week) = `sliceCols` columns
    const bw = 12,
      bh = 11,
      x0 = 6,
      y0 = 18,
      gx = 2.4,
      gy = 4;
    const colStep = bw + gx;
    const W = x0 + cols * colStep + 6;
    const H = y0 + rows * (bh + gy) + 18;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: "100%" });
    const title = svgEl("text", { x: x0, y: 10, "font-size": "9", fill: "#64748b", "font-weight": "600" });
    title.textContent = "event log";
    svg.appendChild(title);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        svg.appendChild(
          svgEl("rect", { x: x0 + c * colStep, y: y0 + r * (bh + gy), width: bw, height: bh, rx: 1.5, fill: "#e2e8f0" })
        );
      }
    }
    // sliding slice group (band + week label) — translated right one week per frame
    const g = svgEl("g", {});
    g.style.transition = "transform .5s ease";
    const bandW = sliceCols * colStep - gx + 4;
    const band = svgEl("rect", {
      x: x0 - 2, y: y0 - 4, width: bandW, height: rows * (bh + gy) + 2, rx: 2,
      fill: accent, "fill-opacity": "0.12", stroke: accent, "stroke-width": "1.2", "stroke-dasharray": "3 2",
    });
    const wk = svgEl("text", { x: x0 - 2 + bandW / 2, y: y0 + rows * (bh + gy) + 12, "text-anchor": "middle", "font-size": "9", "font-weight": "700", fill: accent });
    wk.textContent = "week";
    g.appendChild(band);
    g.appendChild(wk);
    svg.appendChild(g);
    slot.appendChild(svg);
    return {
      setSlice(i, label) {
        g.setAttribute("transform", `translate(${i * sliceCols * colStep},0)`);
        if (label) wk.textContent = label;
      },
    };
  }

  // ---- accumulating hero-edge sparkline --------------------------------------
  function buildSpark(slot, data) {
    slot.innerHTML = "";
    const hero = data.hero;
    const vals = data.frames.map((f) => f.weights[hero]);
    // data-centred y-range: show the gentle week-to-week wiggle without zooming so
    // hard that it reads as a dramatic drift (that story is reserved for slide 4).
    const vmin = Math.min(...vals),
      vmax = Math.max(...vals),
      span = vmax - vmin || 1;
    const yLo = vmin - span * 0.9 - 4;
    const yHi = vmax + span * 0.6 + 4;
    const W = 200,
      H = 96,
      padL = 26,
      padR = 12,
      padT = 12,
      padB = 20;
    const X = (k) => padL + (k * (W - padL - padR)) / (data.frames.length - 1);
    const Y = (v) => H - padB - ((v - yLo) / (yHi - yLo)) * (H - padT - padB);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: "100%" });
    // baseline + y label
    svg.appendChild(svgEl("line", { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: "#cbd5e1", "stroke-width": "1" }));
    svg.appendChild(svgEl("line", { x1: padL, y1: padT, x2: padL, y2: H - padB, stroke: "#cbd5e1", "stroke-width": "1" }));
    const yl = svgEl("text", { x: 4, y: padT + 6, "font-size": "8", fill: "#94a3b8" });
    yl.textContent = "freq";
    svg.appendChild(yl);
    // x ticks (weeks)
    data.frames.forEach((f, k) => {
      const tx = svgEl("text", { x: X(k), y: H - 6, "text-anchor": "middle", "font-size": "8", fill: "#94a3b8" });
      tx.textContent = "t" + (k + 1);
      svg.appendChild(tx);
    });
    const line = svgEl("polyline", { fill: "none", stroke: "#0f172a", "stroke-width": "2", "stroke-linejoin": "round", "stroke-linecap": "round", points: "" });
    const fline = svgEl("polyline", { fill: "none", stroke: data.accent, "stroke-width": "2", "stroke-dasharray": "4 3", "stroke-linecap": "round", points: "" });
    svg.appendChild(line);
    svg.appendChild(fline);
    const dots = data.frames.map((f, k) => {
      const isF = f.kind === "forecast";
      const c = svgEl("circle", { cx: X(k), cy: Y(vals[k]), r: isF ? 3.2 : 3, fill: isF ? data.accent : "#0f172a", opacity: "0" });
      svg.appendChild(c);
      return c;
    });
    const vlbl = svgEl("text", { class: "spark-val", x: 0, y: 0, "font-size": "9", "font-weight": "700", opacity: "0" });
    svg.appendChild(vlbl);
    slot.appendChild(svg);

    function draw(i) {
      // observed polyline through frames 0..min(i,observed); forecast segment dashed
      const obs = [];
      for (let k = 0; k <= i; k++) {
        const isF = data.frames[k].kind === "forecast";
        dots[k].setAttribute("opacity", "1");
        if (!isF) obs.push(`${X(k)},${Y(vals[k])}`);
      }
      for (let k = i + 1; k < dots.length; k++) dots[k].setAttribute("opacity", "0");
      line.setAttribute("points", obs.join(" "));
      // dashed forecast segment from last observed to forecast point (only at last frame)
      if (i === data.frames.length - 1 && data.frames[i].kind === "forecast") {
        fline.setAttribute("points", `${X(i - 1)},${Y(vals[i - 1])} ${X(i)},${Y(vals[i])}`);
        vlbl.setAttribute("opacity", "1");
        vlbl.setAttribute("fill", data.accent);
        vlbl.setAttribute("x", X(i) - 2);
        vlbl.setAttribute("y", Y(vals[i]) - 6);
        vlbl.setAttribute("text-anchor", "end");
        vlbl.textContent = "≈" + vals[i];
      } else {
        fline.setAttribute("points", "");
        vlbl.setAttribute("opacity", "1");
        vlbl.setAttribute("fill", "#475569");
        vlbl.setAttribute("x", X(i) + 5);
        vlbl.setAttribute("y", Y(vals[i]) - 5);
        vlbl.setAttribute("text-anchor", "start");
        vlbl.textContent = "" + vals[i];
      }
    }
    return { draw };
  }

  // ---- layout scaffolding -----------------------------------------------------
  function scaffold(root, layout) {
    root.innerHTML = "";
    root.className = "diagram diagram-" + layout;
    const dfgSlot = div("slot-dfg");
    const logSlot = div("slot-log");
    const sparkSlot = div("slot-spark");

    if (layout === "woven") {
      // [log] -> [DFG] -> [sparkline], step captions woven under each
      const grid = div("woven-grid");
      const c1 = div("wcol wcol-log");
      c1.appendChild(logSlot);
      c1.appendChild(div("cap", '<span class="num">①</span> ' + STEPS[0]));
      const a1 = div("warrow", "→");
      const c2 = div("wcol wcol-dfg");
      c2.appendChild(dfgSlot);
      c2.appendChild(div("cap", '<span class="num">②</span> ' + STEPS[1]));
      const a2 = div("warrow", "→");
      const c3 = div("wcol wcol-spark");
      c3.appendChild(div("spark-title", "O_Sent → O_Cancelled, per week"));
      c3.appendChild(sparkSlot);
      c3.appendChild(div("cap", '<span class="num">③</span> ' + STEPS[2]));
      const c4 = div("cap cap-forecast", '<span class="num">④</span> ' + STEPS[3]);
      grid.append(c1, a1, c2, a2, c3);
      root.append(grid, c4);
      return { dfgSlot, logSlot, sparkSlot, steps: null, forecastCap: c4 };
    }
    // two-column: numbered list left, stacked diagram right
    const cols = div("twocol-grid");
    const left = div("twocol-left");
    left.appendChild(div("wf-title", "Workflow"));
    const stepEls = STEPS.map((s, k) =>
      div("wf-step", `<span class="num">${"①②③④"[k]}</span> ${s}`)
    );
    stepEls.forEach((e) => left.appendChild(e));
    left.appendChild(div("wf-note", '"Each DF edge becomes a univariate time series — like website traffic per day."'));
    const right = div("twocol-right");
    const logRow = div("tc-logrow");
    logRow.appendChild(logSlot);
    logRow.appendChild(div("tc-arrow", "→"));
    const dfgWrap = div("tc-dfg");
    dfgWrap.appendChild(dfgSlot);
    logRow.appendChild(dfgWrap);
    right.appendChild(logRow);
    right.appendChild(div("spark-title", "O_Sent → O_Cancelled, per week"));
    right.appendChild(sparkSlot);
    cols.append(left, right);
    root.appendChild(cols);
    return { dfgSlot, logSlot, sparkSlot, steps: stepEls, forecastCap: null };
  }

  window.renderDiagram = function renderDiagram(root, data, opts) {
    opts = opts || {};
    const layout = opts.layout === "twocol" ? "twocol" : "woven";
    const parts = scaffold(root, layout);
    const dfg = window.renderSvg(parts.dfgSlot, data, {});
    const frames = data.frames;
    const log = buildLog(parts.logSlot, data.accent, frames.length);
    const spark = buildSpark(parts.sparkSlot, data);
    let cur = 0;

    function apply(i) {
      dfg.setFrame(i);
      spark.draw(i);
      log.setSlice(i, frames[i].label.split(" · ")[1]?.replace(" (forecast)", "") || "week");
      const isF = frames[i].kind === "forecast";
      if (parts.forecastCap) parts.forecastCap.classList.toggle("on", isF);
      if (parts.steps) {
        parts.steps.forEach((e, k) => e.classList.toggle("dim", k === 3 && !isF));
        parts.steps[3].classList.toggle("on", isF);
      }
      cur = i;
      if (opts.onFrame) opts.onFrame(i, frames[i]);
    }
    apply(0);
    return {
      frameCount: frames.length,
      get current() {
        return cur;
      },
      step() {
        apply((cur + 1) % frames.length);
      },
      reset() {
        apply(0);
      },
      setFrame: apply,
    };
  };
})();
