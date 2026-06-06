// Pure geometry/scale helpers for the slide-3 DFG-evolution figure. No DOM, no Vue —
// extracted from the prototype renderers (render_svg.js + render_diagram.js) so the layout
// math is unit-testable in isolation and the component stays a thin shell over it.

// Trim a segment so it starts/ends a gap away from the node centres (render_svg.js trim()),
// leaving room for the node boxes and the arrowhead.
export function trim(x1, y1, x2, y2, gapStart, gapEnd) {
  const dx = x2 - x1
  const dy = y2 - y1
  const L = Math.hypot(dx, dy) || 1
  return [
    x1 + (dx / L) * gapStart, y1 + (dy / L) * gapStart,
    x2 - (dx / L) * gapEnd, y2 - (dy / L) * gapEnd,
  ]
}

// Frame-independent trimmed geometry [x1, y1, x2, y2] for every edge, keyed by edge id.
// `endGaps`/`startGaps` override the end/start gap per edge id — e.g. to pull the long diagonal
// hero edge back far enough to clear its wide target box, or to let an edge leaving the small
// start dot touch it (a global gap sized for the wide activity boxes would float off the dot).
export function edgeGeometry(nodes, edges, gapStart = 4.4, gapEnd = 5.8, endGaps = {}, startGaps = {}) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
  return Object.fromEntries(
    edges.map((e) => {
      const s = byId[e.source]
      const t = byId[e.target]
      return [e.id, trim(s.x, s.y, t.x, t.y, startGaps[e.id] ?? gapStart, endGaps[e.id] ?? gapEnd)]
    }),
  )
}

// Arrowhead geometry for one already-trimmed edge. The shaft ends at the head base so
// thick weight-scaled strokes never run underneath the arrowhead.
export function arrowGeometry(segment, strokeWidth) {
  const [x1, y1, x2, y2] = segment
  const dx = x2 - x1
  const dy = y2 - y1
  const L = Math.hypot(dx, dy) || 1
  const ux = dx / L
  const uy = dy / L
  const arrowLen = Math.min(Math.max(2.4, strokeWidth * 1.65), Math.max(0.8, L * 0.55))
  const halfWidth = Math.min(Math.max(1.15, strokeWidth * 0.95), arrowLen * 0.78)
  const baseX = x2 - ux * arrowLen
  const baseY = y2 - uy * arrowLen
  const nx = -uy
  const ny = ux

  return {
    shaft: [x1, y1, baseX, baseY],
    headPath: [
      `M ${x2} ${y2}`,
      `L ${baseX + nx * halfWidth} ${baseY + ny * halfWidth}`,
      `L ${baseX - nx * halfWidth} ${baseY - ny * halfWidth}`,
      'Z',
    ].join(' '),
    tip: [x2, y2],
    base: [baseX, baseY],
  }
}

// Sparkline coordinate scales (render_diagram.js buildSpark). The data-centred y-range shows
// the gentle weekly wiggle without zooming so hard it reads as a dramatic drift (that story
// is reserved for slide 4). Returns the X(k) and Y(v) mapping functions.
export function sparkScale(vals, { W, H, padL, padR, padT, padB }) {
  const vmin = Math.min(...vals)
  const vmax = Math.max(...vals)
  const span = vmax - vmin || 1
  const yLo = vmin - span * 0.9 - 4
  const yHi = vmax + span * 0.6 + 4
  const n = vals.length
  return {
    X: (k) => padL + (k * (W - padL - padR)) / (n - 1),
    Y: (v) => H - padB - ((v - yLo) / (yHi - yLo)) * (H - padT - padB),
  }
}
