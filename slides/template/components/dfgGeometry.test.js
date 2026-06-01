import { describe, it, expect } from 'vitest'
import { arrowGeometry, trim, edgeGeometry, sparkScale } from './dfgGeometry.js'
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

  it('applies a per-edge end-gap override, leaving other edges on the default gap', () => {
    // The long diagonal hero edge ends inside the wide Cancelled box on the shared radial
    // gap; an override pulls just its end back to clear the box (without touching the short
    // vertical edges, which a global gap bump would erase).
    const byId = Object.fromEntries(dfgData.nodes.map((n) => [n.id, n]))
    const endDist = (geo, edgeId, targetId) => {
      const t = byId[targetId]
      return Math.hypot(geo[edgeId][2] - t.x, geo[edgeId][3] - t.y)
    }
    const base = edgeGeometry(dfgData.nodes, dfgData.edges)
    const over = edgeGeometry(dfgData.nodes, dfgData.edges, 4.4, 5.8, { sent__cancelled: 14 })

    // hero edge end is pulled back from 5.8 → 14 away from the Cancelled centre
    expect(endDist(base, 'sent__cancelled', 'cancelled')).toBeCloseTo(5.8, 5)
    expect(endDist(over, 'sent__cancelled', 'cancelled')).toBeCloseTo(14, 5)
    // a non-overridden edge is untouched
    expect(over.created__sent).toEqual(base.created__sent)
  })

  it('ends each arrow shaft before the arrowhead tip', () => {
    const geo = edgeGeometry(dfgData.nodes, dfgData.edges, 4.4, 5.8, { sent__cancelled: 14 })
    for (const edge of dfgData.edges) {
      const arrow = arrowGeometry(geo[edge.id], 2)
      const [tipX, tipY] = arrow.tip
      const [baseX, baseY] = arrow.base
      const tipDist = Math.hypot(tipX - baseX, tipY - baseY)
      expect(tipDist).toBeGreaterThan(0.75)
      expect(arrow.shaft.slice(2)).toEqual(arrow.base)
      expect(arrow.headPath).toContain('Z')
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
