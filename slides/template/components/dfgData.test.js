import { describe, it, expect } from 'vitest'
import { dfgData } from './dfgData.js'

describe('dfgData — daily hero series rollup invariant', () => {
  // The paper bins DF relations daily and sums the 7 daily forecasts per edge into one
  // weekly forecast-DFG (src/pmf_tsfm/er/dfg.py:247 window_pred.sum(axis=0)). So each frame's
  // 7 daily hero values must roll up to that frame's weekly DFG weight.
  it('gives every frame a 7-day hero series', () => {
    for (const frame of dfgData.frames) {
      expect(frame.dailyHero).toHaveLength(7)
    }
  })

  it('sums each frame\'s daily hero series to its weekly DFG weight', () => {
    for (const frame of dfgData.frames) {
      const sum = frame.dailyHero.reduce((a, b) => a + b, 0)
      expect(sum).toBe(frame.weights[dfgData.hero])
    }
  })
})
