import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DfgEvolution from './DfgEvolution.vue'

// Stroke width for an edge weight, per the approved prototype:
// scaleW(f) = 0.7 + 2.9 * sqrt(f / max_freq), max_freq = 927.
const expectedWidth = (f) => 0.7 + 2.9 * Math.sqrt(f / 927)

const heroWidth = (wrapper) =>
  Number(wrapper.get('[data-edge="sent__cancelled"]').attributes('stroke-width'))

describe('DfgEvolution — frame choreography', () => {
  it('mounts and renders the hero edge at frame 0 with the weight-scaled stroke width', () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    // t₁ hero weight = 346
    expect(heroWidth(wrapper)).toBeCloseTo(expectedWidth(346), 2)
  })

  it('tracks the hero edge stroke width as the frame advances', async () => {
    const weights = [346, 314, 316, 316] // sent__cancelled across t₁..t₄
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    for (let i = 0; i < weights.length; i++) {
      await wrapper.setProps({ frame: i })
      expect(heroWidth(wrapper)).toBeCloseTo(expectedWidth(weights[i]), 2)
    }
  })

  it('accumulates one sparkline point per frame (i+1)', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    for (let i = 0; i < 4; i++) {
      await wrapper.setProps({ frame: i })
      expect(wrapper.findAll('[data-testid="spark-point"]')).toHaveLength(i + 1)
    }
  })

  it('advances the event-log slice band one week per frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    for (let i = 0; i < 4; i++) {
      await wrapper.setProps({ frame: i })
      expect(wrapper.get('[data-testid="log-slice"]').attributes('data-slice-index'))
        .toBe(String(i))
    }
  })

  it('reveals workflow step ④ only on the forecast frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    // t₁..t₃ observed → hidden; t₄ forecast → visible
    for (const i of [0, 1, 2]) {
      await wrapper.setProps({ frame: i })
      expect(wrapper.find('[data-testid="step-4"]').exists()).toBe(false)
    }
    await wrapper.setProps({ frame: 3 })
    expect(wrapper.find('[data-testid="step-4"]').exists()).toBe(true)
  })

  it('marks every edge as forecast only on the forecast frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const allForecastFlags = () =>
      wrapper.findAll('[data-edge]').map((e) => e.attributes('data-forecast'))

    for (const i of [0, 1, 2]) {
      await wrapper.setProps({ frame: i })
      expect(allForecastFlags().every((f) => f === 'false')).toBe(true)
    }
    await wrapper.setProps({ frame: 3 })
    expect(allForecastFlags().every((f) => f === 'true')).toBe(true)
  })
})
