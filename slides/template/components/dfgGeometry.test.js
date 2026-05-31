import { describe, it, expect } from 'vitest'
import { trim, edgeGeometry, sparkScale } from './dfgGeometry.js'
import { dfgData } from './dfgData.js'

describe('dfgGeometry — pure layout helpers', () => {
  it('trims a segment by the given gaps at each end, preserving direction', () => {
    // horizontal segment of length 10, trim 2 off the start and 3 off the end → [2, 0, 7, 0]
    expect(trim(0, 0, 10, 0, 2, 3)).toEqual([2, 0, 7, 0])
  })

  it('handles a zero-length segment without dividing by zero', () => {
    expect(trim(5, 5, 5, 5, 1, 1).every(Number.isFinite)).toBe(true)
  })

  it('produces trimmed geometry for every edge, keyed by id', () => {
    const geo = edgeGeometry(dfgData.nodes, dfgData.edges)
    expect(Object.keys(geo)).toHaveLength(dfgData.edges.length)
    // each entry is a finite [x1, y1, x2, y2] tuple
    for (const id of Object.keys(geo)) {
      expect(geo[id]).toHaveLength(4)
      expect(geo[id].every(Number.isFinite)).toBe(true)
    }
  })

  it('maps frame index to x left-to-right and higher weight to a smaller y (up)', () => {
    const vals = [346, 314, 316, 316]
    const dims = { W: 200, H: 96, padL: 26, padR: 12, padT: 12, padB: 20 }
    const { X, Y } = sparkScale(vals, dims)
    expect(X(0)).toBeLessThan(X(3)) // weeks advance rightward
    expect(Y(346)).toBeLessThan(Y(314)) // the larger value sits higher on screen
  })
})
