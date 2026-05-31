// Pure choreography core for the slide-3 DFG-evolution figure.
// Maps a frame index to the derived view-state the component renders. No DOM, no Vue —
// this is the deep module the component is a thin shell around. Formula and contract come
// from the approved prototype (prototypes/dfg-anim/render_svg.js + render_diagram.js).

// sqrt scaling compresses the weight range so busy chain edges don't swamp the hero edge.
export const strokeWidthFor = (weight, maxFreq) =>
  Number((0.7 + 2.9 * Math.sqrt(weight / maxFreq)).toFixed(2))

export function frameState(data, i) {
  const frame = data.frames[i]
  const isForecast = frame.kind === 'forecast'
  const edges = {}
  for (const edge of data.edges) {
    edges[edge.id] = {
      strokeWidth: strokeWidthFor(frame.weights[edge.id], data.max_freq),
      isForecast,
    }
  }
  return {
    edges,
    sparkPoints: i + 1,
    sliceIndex: i,
    step4Visible: isForecast,
  }
}
