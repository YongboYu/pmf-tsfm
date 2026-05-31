import { describe, it, expect } from 'vitest'
import { frameForClicks } from './frameForClicks.js'

// Maps Slidev's reactive `$clicks` counter (0 on slide entry, +1 per click) to the
// DfgEvolution master frame index, clamped to a valid frame. Slide 3 binds :frame to this.
describe('frameForClicks — Slidev $clicks → DfgEvolution frame', () => {
  it('opens on frame 0 (t₁) before any click', () => {
    expect(frameForClicks(0, 4)).toBe(0)
  })

  it('advances one frame per click in lockstep', () => {
    expect(frameForClicks(1, 4)).toBe(1)
    expect(frameForClicks(2, 4)).toBe(2)
  })

  it('clamps at the forecast frame — extra clicks never overrun', () => {
    expect(frameForClicks(3, 4)).toBe(3)
    expect(frameForClicks(7, 4)).toBe(3)
  })

  it('floors junk input (negative or undefined $clicks) to frame 0', () => {
    expect(frameForClicks(-1, 4)).toBe(0)
    expect(frameForClicks(undefined, 4)).toBe(0)
  })
})
