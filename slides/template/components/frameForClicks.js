// Maps Slidev's reactive `$clicks` counter to the DfgEvolution master frame index.
// Slide 3 binds :frame="frameForClicks($clicks, dfgData.frames.length)".
export function frameForClicks(clicks, frameCount) {
  return Math.min(Math.max(clicks || 0, 0), frameCount - 1)
}
