/* PROTOTYPE — wipe me when slide 3 is finalised.
 * Pipeline A: live, presenter-paced SVG DFG. Vector, zero deps, on-brand.
 * Exposes window.renderSvg(stage, data, opts) -> controller {step, reset, frameCount, current}.
 * This is the asset that would become components/DfgEvolution.vue (v-click driven).
 */
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const byId = (arr) => Object.fromEntries(arr.map((o) => [o.id, o]));

  // trim a segment so it starts/ends a gap away from the node centres
  function trim(x1, y1, x2, y2, gapStart, gapEnd) {
    const dx = x2 - x1,
      dy = y2 - y1,
      L = Math.hypot(dx, dy) || 1;
    return [
      x1 + (dx / L) * gapStart,
      y1 + (dy / L) * gapStart,
      x2 - (dx / L) * gapEnd,
      y2 - (dy / L) * gapEnd,
    ];
  }

  window.renderSvg = function renderSvg(stage, data, opts) {
    opts = opts || {};
    stage.innerHTML = "";
    const nodes = byId(data.nodes);
    const accent = data.accent;
    const maxF = data.max_freq;
    // sqrt scaling compresses the range so the busy chain edges don't swamp the
    // hero edge; viewBox units
    const scaleW = (f) => 0.7 + 2.9 * Math.sqrt(f / maxF);

    const svg = el("svg", {
      viewBox: "0 0 100 104",
      width: "100%",
      height: "100%",
      style: "display:block",
    });

    // arrow markers (neutral + accent), constant size
    const defs = el("defs");
    for (const [id, col] of [
      ["arrow", "#475569"],
      ["arrowA", accent],
    ]) {
      const m = el("marker", {
        id,
        viewBox: "0 0 10 10",
        refX: "8",
        refY: "5",
        markerWidth: "3.4",
        markerHeight: "3.4",
        orient: "auto-start-reverse",
        markerUnits: "userSpaceOnUse",
      });
      m.appendChild(el("path", { d: "M0,0 L10,5 L0,10 z", fill: col }));
      defs.appendChild(m);
    }
    svg.appendChild(defs);

    // hero ground-truth ghost (only shown on forecast frame)
    const heroEdge = data.edges.find((e) => e.hero);
    const ghost = el("path", {
      fill: "none",
      stroke: "#94a3b8",
      "stroke-linecap": "round",
      "stroke-dasharray": "0",
      opacity: "0",
      "stroke-width": "0",
    });
    svg.appendChild(ghost);

    // edges
    const edgeEls = {};
    for (const e of data.edges) {
      const s = nodes[e.source],
        t = nodes[e.target];
      const [x1, y1, x2, y2] = trim(s.x, s.y, t.x, t.y, 4.4, 4.8);
      const p = el("path", {
        d: `M${x1},${y1} L${x2},${y2}`,
        fill: "none",
        stroke: "#475569",
        "stroke-linecap": "round",
        "marker-end": "url(#arrow)",
        "stroke-width": "1",
        style: "transition: stroke-width .55s ease, stroke .45s ease, opacity .45s ease",
      });
      svg.appendChild(p);
      edgeEls[e.id] = { path: p, ...e, geo: [x1, y1, x2, y2] };
    }

    // hero edge weight label
    const he = edgeEls[heroEdge.id];
    const hx = (he.geo[0] + he.geo[2]) / 2 + 6;
    const hy = (he.geo[1] + he.geo[3]) / 2 - 1;
    const heroLabel = el("text", {
      x: hx,
      y: hy,
      "font-size": "3.4",
      "font-weight": "700",
      fill: "#475569",
      "font-family": "ui-sans-serif, system-ui, sans-serif",
      style: "transition: fill .45s ease",
    });
    svg.appendChild(heroLabel);

    // nodes (draw above edges)
    for (const n of data.nodes) {
      const g = el("g", {});
      if (n.kind === "activity") {
        const w = 21,
          h = 7.2;
        g.appendChild(
          el("rect", {
            x: n.x - w / 2,
            y: n.y - h / 2,
            width: w,
            height: h,
            rx: 2.2,
            fill: "#ffffff",
            stroke: "#cbd5e1",
            "stroke-width": "0.5",
          })
        );
        const tx = el("text", {
          x: n.x,
          y: n.y + 1.3,
          "text-anchor": "middle",
          "font-size": "3.4",
          fill: "#0f172a",
          "font-family": "ui-sans-serif, system-ui, sans-serif",
        });
        tx.textContent = n.label;
        g.appendChild(tx);
      } else {
        // start/end marker
        const marker =
          n.kind === "start"
            ? el("circle", { cx: n.x, cy: n.y, r: 3.4, fill: "#0f172a" })
            : el("rect", { x: n.x - 3, y: n.y - 3, width: 6, height: 6, rx: 0.8, fill: "#0f172a" });
        g.appendChild(marker);
      }
      svg.appendChild(g);
    }

    stage.appendChild(svg);

    const frames = data.frames;
    let cur = 0;

    function applyFrame(i) {
      const fr = frames[i];
      const isForecast = fr.kind === "forecast";
      for (const e of data.edges) {
        const ee = edgeEls[e.id];
        const w = scaleW(fr.weights[e.id]);
        ee.path.setAttribute("stroke-width", w.toFixed(2));
        if (isForecast) {
          ee.path.setAttribute("stroke", accent);
          ee.path.setAttribute("marker-end", "url(#arrowA)");
          ee.path.setAttribute("stroke-dasharray", "2.4 1.8");
        } else {
          ee.path.setAttribute("stroke", e.hero ? "#0f172a" : "#475569");
          ee.path.setAttribute("marker-end", "url(#arrow)");
          ee.path.setAttribute("stroke-dasharray", "0");
        }
      }
      // hero ghost (held-out truth) on forecast frame
      if (isForecast && fr.truth) {
        const [x1, y1, x2, y2] = he.geo;
        ghost.setAttribute("d", `M${x1},${y1} L${x2},${y2}`);
        ghost.setAttribute("stroke-width", scaleW(fr.truth[heroEdge.id]).toFixed(2));
        ghost.setAttribute("opacity", "0.6");
      } else {
        ghost.setAttribute("opacity", "0");
      }
      // hero label
      const hv = fr.weights[heroEdge.id];
      heroLabel.textContent = isForecast
        ? `≈${hv} (truth ${fr.truth ? fr.truth[heroEdge.id] : "?"})`
        : `${hv}`;
      heroLabel.setAttribute("fill", isForecast ? accent : "#475569");
      cur = i;
      if (opts.onFrame) opts.onFrame(i, fr);
    }

    applyFrame(0);
    return {
      frameCount: frames.length,
      get current() {
        return cur;
      },
      step() {
        applyFrame((cur + 1) % frames.length);
      },
      reset() {
        applyFrame(0);
      },
      setFrame: applyFrame,
    };
  };
})();
