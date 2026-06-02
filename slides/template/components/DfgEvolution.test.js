import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DfgEvolution from './DfgEvolution.vue'

// Stroke width for an edge weight:
// scaleW(f) = 0.55 + 1.15 * sqrt(f / max_freq), max_freq = 927.
const expectedWidth = (f) => 0.55 + 1.15 * Math.sqrt(f / 927)

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

  it('reveals one week (7 daily points) of the sparkline per frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    for (let i = 0; i < 4; i++) {
      await wrapper.setProps({ frame: i })
      expect(wrapper.findAll('[data-testid="spark-point"]')).toHaveLength((i + 1) * 7)
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

  it('renders each DFG edge with a separate arrowhead, not an SVG marker overlay', () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 2 } })
    const edgeShafts = wrapper.findAll('[data-edge]')
    const edgeHeads = wrapper.findAll('[data-edge-head]')

    expect(edgeShafts).toHaveLength(8)
    expect(edgeHeads).toHaveLength(8)
    expect(edgeShafts.every((edge) => edge.attributes('marker-end') === undefined)).toBe(true)
  })

  it('renders every DFG node, with activity labels shown', () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    // 8 nodes: start ●, 6 activities, end ■.
    expect(wrapper.findAll('[data-node]')).toHaveLength(8)
    // A representative activity label is rendered as text.
    expect(wrapper.get('[data-node="sent"]').text()).toContain('Sent')
  })

  it('recolours edges: ink/neutral when observed, accent + dashed on the forecast frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const hero = () => wrapper.get('[data-edge="sent__cancelled"]')
    const plain = () => wrapper.get('[data-edge="created__sent"]')

    const isDashed = (el) => /[1-9]/.test(el.attributes('stroke-dasharray') || '0')

    // observed: hero drawn in ink, chain edge neutral, neither dashed
    expect(hero().attributes('stroke')).toBe('#0f172a')
    expect(plain().attributes('stroke')).toBe('#475569')
    expect(isDashed(hero())).toBe(false)

    // forecast: every edge recoloured to the single accent and dashed
    await wrapper.setProps({ frame: 3 })
    expect(hero().attributes('stroke')).toBe('#1d4ed8')
    expect(plain().attributes('stroke')).toBe('#1d4ed8')
    expect(isDashed(hero())).toBe(true)
  })

  it('labels the hero edge with its weight, and the forecast vs held-out truth', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const label = () => wrapper.get('[data-testid="hero-label"]').text()

    const observed = ['346', '314', '316'] // t₁..t₃ raw weekly frequency
    for (let i = 0; i < observed.length; i++) {
      await wrapper.setProps({ frame: i })
      expect(label()).toBe(observed[i])
    }
    // forecast frame announces the prediction and the held-out truth
    await wrapper.setProps({ frame: 3 })
    expect(label()).toContain('316')
    expect(label()).toContain('315')
  })

  it('shows the held-out-truth ghost only on the forecast frame, scaled to the truth weight', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const ghost = () => wrapper.find('[data-testid="hero-ghost"]')

    for (const i of [0, 1, 2]) {
      await wrapper.setProps({ frame: i })
      expect(ghost().exists()).toBe(false)
    }
    await wrapper.setProps({ frame: 3 })
    expect(ghost().exists()).toBe(true)
    // ghost stroke-width tracks the held-out truth weight (sent__cancelled truth = 315)
    expect(Number(ghost().attributes('stroke-width'))).toBeCloseTo(expectedWidth(315), 2)
  })

  it('always shows workflow steps ①②③, never leaking the reserved term "window"', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    for (const i of [0, 1, 2, 3]) {
      await wrapper.setProps({ frame: i })
      for (const n of [1, 2, 3]) {
        const step = wrapper.find(`[data-testid="step-${n}"]`)
        expect(step.exists()).toBe(true)
        expect(step.text().toLowerCase()).not.toContain('window')
      }
    }
  })

  it('frames the daily series as primary and the weekly DFG as its Σ-aggregate, not weekly-first extraction', () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const step1 = wrapper.get('[data-testid="step-1"]').text().toLowerCase()
    const step2 = wrapper.get('[data-testid="step-2"]').text()

    // ① the primary extraction is the daily series, not a weekly sublog
    expect(step1).toContain('daily')
    expect(step1).not.toContain('sublog')
    // ② the weekly DFG reads as the Σ-aggregate of the daily counts
    expect(step2).toContain('Σ')
    expect(step2.toLowerCase()).toContain('daily')
  })

  it('slides the event-log band right one constant week-step per frame', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const translateX = () => {
      const t = wrapper.get('[data-testid="log-band"]').attributes('transform')
      return Number(t.match(/translate\(\s*([-\d.]+)/)[1])
    }
    await wrapper.setProps({ frame: 0 })
    expect(translateX()).toBe(0)
    await wrapper.setProps({ frame: 1 })
    const step = translateX()
    expect(step).toBeGreaterThan(0)
    for (const i of [2, 3]) {
      await wrapper.setProps({ frame: i })
      expect(translateX()).toBeCloseTo(i * step, 5)
    }
  })

  it('accumulates the daily observed line and draws the 7-day dashed forecast tail only at the end', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const pointCount = (sel) => {
      const el = wrapper.find(sel)
      if (!el.exists()) return 0
      return el.attributes('points').trim().split(/\s+/).length
    }
    const forecastSeg = () => wrapper.find('[data-testid="spark-forecast"]')

    // observed polyline grows one week (7 daily points) per observed frame; the 7 forecast
    // days are NOT on it — they land as the dashed accent tail on the forecast frame.
    const expectedObs = [7, 14, 21, 21] // t₁..t₃ observed; t₄ forecast keeps the 21 observed
    for (let i = 0; i < 4; i++) {
      await wrapper.setProps({ frame: i })
      expect(pointCount('[data-testid="spark-line"]')).toBe(expectedObs[i])
      expect(forecastSeg().exists()).toBe(i === 3)
    }
    // the forecast horizon is exactly the 7 daily prediction steps
    expect(pointCount('[data-testid="spark-forecast"]')).toBe(7)
  })

  it('shows the Σ rollup readout only on the forecast frame, matching the weekly forecast-DFG arc', async () => {
    const wrapper = mount(DfgEvolution, { props: { frame: 0 } })
    const sum = () => wrapper.find('[data-testid="spark-sum"]')

    // observed frames carry no rollup readout
    for (const i of [0, 1, 2]) {
      await wrapper.setProps({ frame: i })
      expect(sum().exists()).toBe(false)
    }
    // forecast frame announces the 7 daily forecasts summing to the weekly arc (≈316)
    await wrapper.setProps({ frame: 3 })
    expect(sum().exists()).toBe(true)
    expect(sum().text()).toContain('Σ')
    expect(sum().text()).toContain('316')
  })
})
