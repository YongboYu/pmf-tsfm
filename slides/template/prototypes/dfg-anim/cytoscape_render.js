/* PROTOTYPE — wipe me when slide 3 is finalised.
 * Pipeline D: cytoscape graph, data-driven, preset layout (no physics).
 * Exposes window.renderCytoscape(stage, data, opts) -> {step, reset, frameCount, current}.
 */
(function () {
  window.renderCytoscape = function renderCytoscape(stage, data, opts) {
    opts = opts || {};
    stage.innerHTML = "";
    if (typeof cytoscape === "undefined") {
      stage.innerHTML =
        '<div style="padding:24px;color:#b91c1c;font:14px system-ui">' +
        "cytoscape not loaded — run <code>pnpm add cytoscape</code> in slides/template " +
        "(expected at ../../node_modules/cytoscape/dist/cytoscape.min.js).</div>";
      return { frameCount: 0, current: 0, step() {}, reset() {}, setFrame() {} };
    }
    const host = document.createElement("div");
    host.style.cssText = "width:100%;height:100%";
    stage.appendChild(host);

    const accent = data.accent;
    const maxF = data.max_freq;
    const S = 6.4; // coord scale (viewBox-units -> px)
    const scaleW = (f) => 1.2 + 9 * (f / maxF);

    const elements = [];
    for (const n of data.nodes) {
      elements.push({
        data: { id: n.id, label: n.kind === "activity" ? n.label : "", kind: n.kind },
        position: { x: n.x * S, y: n.y * S },
      });
    }
    const heroEdge = data.edges.find((e) => e.hero);
    for (const e of data.edges) {
      elements.push({
        data: { id: e.id, source: e.source, target: e.target, hero: e.hero ? 1 : 0, w: 2, lbl: "" },
      });
    }

    const cy = cytoscape({
      container: host,
      elements,
      layout: { name: "preset", fit: true, padding: 28 },
      userZoomingEnabled: false,
      userPanningEnabled: false,
      autoungrabify: true,
      style: [
        {
          selector: "node[kind = 'activity']",
          style: {
            shape: "round-rectangle",
            "background-color": "#ffffff",
            "border-color": "#cbd5e1",
            "border-width": 1.5,
            label: "data(label)",
            "font-size": 13,
            "font-family": "system-ui, sans-serif",
            color: "#0f172a",
            "text-valign": "center",
            "text-halign": "center",
            width: "label",
            height: 30,
            padding: "8px",
          },
        },
        {
          selector: "node[kind = 'start']",
          style: { shape: "ellipse", "background-color": "#0f172a", width: 22, height: 22 },
        },
        {
          selector: "node[kind = 'end']",
          style: { shape: "rectangle", "background-color": "#0f172a", width: 20, height: 20 },
        },
        {
          selector: "edge",
          style: {
            "curve-style": "straight",
            "line-color": "#475569",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1.1,
            width: "data(w)",
            label: "data(lbl)",
            "font-size": 13,
            "font-weight": "bold",
            color: "#475569",
            "text-background-color": "#ffffff",
            "text-background-opacity": 1,
            "text-background-padding": 2,
          },
        },
        {
          selector: "edge.forecast",
          style: {
            "line-color": accent,
            "target-arrow-color": accent,
            "line-style": "dashed",
            color: accent,
          },
        },
        {
          selector: "edge.ghost",
          style: { "line-color": "#94a3b8", "target-arrow-color": "#94a3b8", opacity: 0.6 },
        },
      ],
    });
    cy.fit(undefined, 24);

    const frames = data.frames;
    let cur = 0;
    function applyFrame(i) {
      const fr = frames[i];
      const isF = fr.kind === "forecast";
      cy.batch(() => {
        cy.edges().forEach((ed) => {
          const id = ed.id();
          ed.data("w", scaleW(fr.weights[id]));
          ed.toggleClass("forecast", isF);
          if (ed.data("hero")) {
            ed.data(
              "lbl",
              isF ? "≈" + fr.weights[id] + " (truth " + (fr.truth ? fr.truth[id] : "?") + ")" : "" + fr.weights[id]
            );
          }
        });
      });
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
