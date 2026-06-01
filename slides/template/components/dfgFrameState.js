// Pure choreography core for the slide-3 DFG-evolution figure.
// Maps a frame index to the derived view-state the component renders. No DOM, no Vue —
// this is the deep module the component is a thin shell around.

// sqrt scaling compresses the weight range so busy chain edges don't swamp the hero edge.
export const strokeWidthFor = (weight, maxFreq) =>
  Number((0.5 + 1.6 * Math.sqrt(weight / maxFreq)).toFixed(2))

const INK = '#0f172a'
const NEUTRAL = '#475569'

export function frameState(data, i) {
  const frame = data.frames[i]
  const isForecast = frame.kind === 'forecast'
  const edges = {}
  for (const edge of data.edges) {
    // On the forecast frame every edge re-colours to the single accent, draws with the
    // accent arrowhead, and goes dashed; otherwise the hero edge reads in ink and the
    // chain edges in neutral grey (per render_svg.js applyFrame).
    edges[edge.id] = {
      strokeWidth: strokeWidthFor(frame.weights[edge.id], data.max_freq),
      isForecast,
      stroke: isForecast ? data.accent : edge.hero ? INK : NEUTRAL,
      marker: isForecast ? 'arrowA' : 'arrow',
      dashed: isForecast,
    }
  }
  // Hero-edge weight readout: the raw weekly frequency while observed; on the forecast
  // frame, the prediction (≈) alongside the held-out truth (per render_svg.js heroLabel).
  const heroWeight = frame.weights[data.hero]
  const heroLabel = {
    text: isForecast
      ? `≈${heroWeight} (truth ${frame.truth ? frame.truth[data.hero] : '?'})`
      : `${heroWeight}`,
    fill: isForecast ? data.accent : NEUTRAL,
  }

  // On the forecast frame, a faint ghost of the held-out truth is drawn over the hero edge
  // so the prediction can be read against ground truth (per render_svg.js ghost).
  const heroGhost = {
    visible: isForecast && Boolean(frame.truth),
    strokeWidth: isForecast && frame.truth
      ? strokeWidthFor(frame.truth[data.hero], data.max_freq)
      : 0,
  }

  return {
    edges,
    heroLabel,
    heroGhost,
    sparkPoints: i + 1,
    sliceIndex: i,
    step4Visible: isForecast,
  }
}
