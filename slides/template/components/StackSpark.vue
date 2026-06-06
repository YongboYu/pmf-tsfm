<script setup>
// One mini daily sparkline for a single DF arc in the S4 middle STACK.
// Stage-based reveal: the OBSERVED history (t₁..t₃, 21 days) is always drawn in brand navy;
// the 7-day FORECAST tail (t₄) is drawn dashed amber only when `showForecast` is true.
// Reuses sparkScale from the dfgGeometry core so the line shape matches the deck's math.
import { computed } from 'vue'
import { sparkScale } from './dfgGeometry.js'

const props = defineProps({
  arc: { type: Object, required: true }, // { id, label, hero }
  frames: { type: Array, required: true },
  showForecast: { type: Boolean, default: false },
  accent: { type: String, default: '#dd8a2e' },
  brand: { type: String, default: '#00407a' },
})

const DAYS = 7
const dims = { W: 178, H: 52, padL: 6, padR: 8, padT: 9, padB: 8 }

const vals = computed(() => props.frames.flatMap((f) => f.daily[props.arc.id]))
const scale = computed(() => sparkScale(vals.value, dims))

// Day index where the forecast horizon begins (the "today" divider).
const splitK = computed(() => {
  let k = 0
  for (const f of props.frames) { if (f.kind === 'forecast') break; k += DAYS }
  return k
})

const geom = computed(() => {
  const { X, Y } = scale.value
  const v = vals.value
  const obs = []
  const fc = []
  v.forEach((val, k) => {
    const pt = `${X(k)},${Y(val)}`
    if (k < splitK.value) obs.push({ pt, cx: X(k), cy: Y(val) })
    else fc.push({ pt, cx: X(k), cy: Y(val) })
  })
  const bridge = obs.length ? [obs[obs.length - 1].pt] : []
  return {
    obsLine: obs.map((o) => o.pt).join(' '),
    fcLine: [...bridge, ...fc.map((f) => f.pt)].join(' '),
    obsDots: obs,
    fcDots: fc,
    splitX: X(splitK.value),
    top: dims.padT, bottom: dims.H - dims.padB,
  }
})
</script>

<template>
  <div class="stack-spark" :data-arc="arc.id">
    <div class="arc-label" :style="{ color: arc.hero ? brand : '#486581' }">{{ arc.label }}</div>
    <svg :viewBox="`0 0 ${dims.W} ${dims.H}`" width="100%" height="100%" preserveAspectRatio="none">
      <!-- baseline -->
      <line :x1="dims.padL" :y1="geom.bottom" :x2="dims.W - dims.padR" :y2="geom.bottom" stroke="#dde5ee" stroke-width="0.8" />
      <!-- "today" divider where the 7-day forecast begins -->
      <line v-if="showForecast" :x1="geom.splitX" :y1="geom.top" :x2="geom.splitX" :y2="geom.bottom"
        stroke="#93a4b8" stroke-width="0.6" stroke-dasharray="2 2" opacity="0.8" />
      <!-- observed history (always) -->
      <polyline data-testid="stack-obs" fill="none" :stroke="brand" stroke-width="1.7"
        stroke-linejoin="round" stroke-linecap="round" :points="geom.obsLine" />
      <circle v-for="(d, k) in geom.obsDots" :key="'o' + k" :cx="d.cx" :cy="d.cy" r="0.8" :fill="brand" />
      <!-- 7-day forecast tail (revealed on click) -->
      <template v-if="showForecast">
        <polyline data-testid="stack-fc" fill="none" :stroke="accent" stroke-width="2"
          stroke-dasharray="3 2" stroke-linejoin="round" stroke-linecap="round" :points="geom.fcLine" />
        <circle v-for="(d, k) in geom.fcDots" :key="'f' + k" :cx="d.cx" :cy="d.cy" r="1.2" :fill="accent" />
      </template>
    </svg>
  </div>
</template>

<style scoped>
.stack-spark { display: flex; flex-direction: column; gap: 2px; flex: 1; min-height: 0; }
.arc-label { font-size: 16px; font-weight: 600; line-height: 1.1; font-family: 'JetBrains Mono', ui-monospace, monospace; }
.stack-spark svg { flex: 1; min-height: 0; }
</style>
