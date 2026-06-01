<script setup>
import { computed } from 'vue'
import { dfgData } from './dfgData.js'
import { frameState } from './dfgFrameState.js'
import { edgeGeometry, sparkScale } from './dfgGeometry.js'

// `frame` is the master index; slice #3 will bind it to Slidev v-click.
const props = defineProps({
  frame: { type: Number, default: 0 },
})

const state = computed(() => frameState(dfgData, props.frame))

// Frame-independent trimmed geometry [x1,y1,x2,y2] for every edge, keyed by edge id.
const edgeGeo = edgeGeometry(dfgData.nodes, dfgData.edges)

// Workflow captions woven under each part of the figure (per render_diagram.js STEPS).
// "weekly sublog" / "weekly snapshot" wording avoids the reserved term "window" (ER only).
const STEPS = [
  '① Event log → weekly sublog',
  '② Each week → a time-indexed DFG',
  '③ Each DF edge → a univariate time series',
  '④ Forecast next week → forecasted DFG',
]

// Event-log slice motif (per render_diagram.js buildLog). The log is tiled into one column-
// pair per week; a dashed band slides right one week per frame, so each click visibly grabs a
// later time-slice of the log.
const LOG = (() => {
  const rows = 6, sliceCols = 2, n = dfgData.frames.length
  const cols = sliceCols * n
  const bw = 12, bh = 11, x0 = 6, y0 = 18, gx = 2.4, gy = 4
  const colStep = bw + gx
  const cells = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      cells.push({ x: x0 + c * colStep, y: y0 + r * (bh + gy) })
  return {
    rows, bw, bh, x0, y0, gy, cells,
    W: x0 + cols * colStep + 6,
    H: y0 + rows * (bh + gy) + 18,
    weekStep: sliceCols * colStep,
    bandW: sliceCols * colStep - gx + 4,
    bandH: rows * (bh + gy) + 2,
    labelY: y0 + rows * (bh + gy) + 12,
  }
})()

// The week the sliding band currently covers (e.g. "Oct 02"; the forecast week drops its tag).
const weekLabel = computed(() =>
  (dfgData.frames[props.frame].label.split(' · ')[1] || 'week').replace(' (forecast)', ''),
)

// Hero-edge sparkline geometry (per render_diagram.js buildSpark). Frame-independent scales;
// the data-centred y-range shows the gentle weekly wiggle without faking a dramatic drift.
const SPARK = (() => {
  const dims = { W: 200, H: 140, padL: 26, padR: 12, padT: 16, padB: 24 }
  const vals = dfgData.frames.map((f) => f.weights[dfgData.hero])
  const { X, Y } = sparkScale(vals, dims)
  return { ...dims, vals, X, Y, n: dfgData.frames.length }
})()

// Accumulating sparkline state for the current frame: observed polyline (forecast point is
// NOT on it), the dashed forecast segment (last frame only), dots, and the value readout.
const spark = computed(() => {
  const i = props.frame
  const { vals, X, Y, n } = SPARK
  const obs = []
  const dots = []
  for (let k = 0; k <= i; k++) {
    const isF = dfgData.frames[k].kind === 'forecast'
    dots.push({ cx: X(k), cy: Y(vals[k]), isF })
    if (!isF) obs.push(`${X(k)},${Y(vals[k])}`)
  }
  const isLastForecast = i === n - 1 && dfgData.frames[i].kind === 'forecast'
  const forecastPoints = isLastForecast
    ? `${X(i - 1)},${Y(vals[i - 1])} ${X(i)},${Y(vals[i])}`
    : null
  const valLabel = isLastForecast
    ? { x: X(i) - 2, y: Y(vals[i]) - 6, anchor: 'end', fill: dfgData.accent, text: `≈${vals[i]}` }
    : { x: X(i) + 5, y: Y(vals[i]) - 5, anchor: 'start', fill: '#475569', text: `${vals[i]}` }
  return { obsPoints: obs.join(' '), forecastPoints, dots, valLabel }
})
</script>

<template>
  <!-- Woven full-width figure: one unified picture that both narrates the PMF workflow
       (log → DFG → per-edge series → forecast) and lands the capability to forecast next
       week's DFG, with each workflow step labelled at its part of the diagram. -->
  <div data-testid="dfg-evolution" class="dfg-evolution">
    <div class="woven-grid">
      <!-- ① event-log slice -->
      <div class="wcol wcol-log">
        <div class="slot">
    <!-- Event-log slice band: slides right one week per frame to grab a later time-slice. -->
    <svg
      data-testid="log-slice" :data-slice-index="state.sliceIndex"
      :viewBox="`0 0 ${LOG.W} ${LOG.H}`" width="100%" height="100%"
    >
      <text :x="LOG.x0" y="10" style="font-size: 9px" fill="#64748b" font-weight="600">event log</text>
      <rect
        v-for="(cell, k) in LOG.cells" :key="k"
        :x="cell.x" :y="cell.y" :width="LOG.bw" :height="LOG.bh" rx="1.5" fill="#e2e8f0"
      />
      <g data-testid="log-band" :transform="`translate(${state.sliceIndex * LOG.weekStep},0)`">
        <rect
          :x="LOG.x0 - 2" :y="LOG.y0 - 4" :width="LOG.bandW" :height="LOG.bandH" rx="2"
          :fill="dfgData.accent" fill-opacity="0.12" :stroke="dfgData.accent"
          stroke-width="1.2" stroke-dasharray="3 2"
        />
        <text
          :x="LOG.x0 - 2 + LOG.bandW / 2" :y="LOG.labelY" text-anchor="middle"
          style="font-size: 9px" font-weight="700" :fill="dfgData.accent"
        >{{ weekLabel }}</text>
      </g>
    </svg>
        </div>
        <div data-testid="step-1" class="cap">{{ STEPS[0] }}</div>
      </div>

      <div class="warrow">→</div>

      <!-- ② time-indexed DFG -->
      <div class="wcol wcol-dfg">
        <div class="slot">
    <svg viewBox="0 0 100 104" width="100%" height="100%">
      <defs>
        <marker
          v-for="m in [{ id: 'arrow', col: '#475569' }, { id: 'arrowA', col: dfgData.accent }]"
          :key="m.id" :id="m.id" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="2.8" markerHeight="2.8" orient="auto-start-reverse"
          markerUnits="userSpaceOnUse"
        >
          <path d="M0,0 L10,5 L0,10 z" :fill="m.col" />
        </marker>
      </defs>
      <line
        v-for="edge in dfgData.edges"
        :key="edge.id"
        :data-edge="edge.id"
        :data-forecast="state.edges[edge.id].isForecast"
        :x1="edgeGeo[edge.id][0]" :y1="edgeGeo[edge.id][1]"
        :x2="edgeGeo[edge.id][2]" :y2="edgeGeo[edge.id][3]"
        fill="none" stroke-linecap="round"
        :stroke-width="state.edges[edge.id].strokeWidth"
        :stroke="state.edges[edge.id].stroke"
        :marker-end="`url(#${state.edges[edge.id].marker})`"
        :stroke-dasharray="state.edges[edge.id].dashed ? '2.4 1.8' : '0'"
      />
      <!-- Held-out-truth ghost over the hero edge, forecast frame only. -->
      <line
        v-if="state.heroGhost.visible"
        data-testid="hero-ghost"
        :x1="edgeGeo[dfgData.hero][0]" :y1="edgeGeo[dfgData.hero][1]"
        :x2="edgeGeo[dfgData.hero][2]" :y2="edgeGeo[dfgData.hero][3]"
        fill="none" stroke="#94a3b8" stroke-linecap="round" opacity="0.6"
        :stroke-width="state.heroGhost.strokeWidth"
      />
      <!-- Hero-edge weight readout, offset from the edge midpoint. -->
      <text
        data-testid="hero-label"
        :x="(edgeGeo[dfgData.hero][0] + edgeGeo[dfgData.hero][2]) / 2 + 6"
        :y="(edgeGeo[dfgData.hero][1] + edgeGeo[dfgData.hero][3]) / 2 - 1"
        style="font-size: 3.4px" font-weight="700" :fill="state.heroLabel.fill"
      >{{ state.heroLabel.text }}</text>
      <!-- Nodes drawn above edges. Activity = labelled rect; start ● / end ■. -->
      <g v-for="node in dfgData.nodes" :key="node.id" :data-node="node.id">
        <template v-if="node.kind === 'activity'">
          <rect
            :x="node.x - 10.5" :y="node.y - 3.6" width="21" height="7.2" rx="2.2"
            fill="#ffffff" stroke="#cbd5e1" stroke-width="0.5"
          />
          <text
            :x="node.x" :y="node.y + 1.3" text-anchor="middle"
            style="font-size: 3.4px" fill="#0f172a"
          >{{ node.label }}</text>
        </template>
        <circle
          v-else-if="node.kind === 'start'"
          :cx="node.x" :cy="node.y" r="3.4" fill="#0f172a"
        />
        <rect
          v-else
          :x="node.x - 3" :y="node.y - 3" width="6" height="6" rx="0.8" fill="#0f172a"
        />
      </g>
    </svg>
        </div>
        <div data-testid="step-2" class="cap">{{ STEPS[1] }}</div>
      </div>

      <div class="warrow">→</div>

      <!-- ③ per-edge time series -->
      <div class="wcol wcol-spark">
        <div class="spark-title">O_Sent → O_Cancelled, per week</div>
        <div class="slot slot-spark">
    <!-- Accumulating hero-edge sparkline: observed line grows one point per frame; the
         held-out forecast lands as a dashed accent segment on the final frame. -->
    <svg
      data-testid="sparkline" :viewBox="`0 0 ${SPARK.W} ${SPARK.H}`"
      width="100%" height="100%" preserveAspectRatio="xMidYMin meet"
    >
      <line
        :x1="SPARK.padL" :y1="SPARK.H - SPARK.padB" :x2="SPARK.W - SPARK.padR"
        :y2="SPARK.H - SPARK.padB" stroke="#cbd5e1" stroke-width="1"
      />
      <line
        :x1="SPARK.padL" :y1="SPARK.padT" :x2="SPARK.padL" :y2="SPARK.H - SPARK.padB"
        stroke="#cbd5e1" stroke-width="1"
      />
      <text
        v-for="(f, k) in dfgData.frames" :key="k"
        :x="SPARK.X(k)" :y="SPARK.H - 6" text-anchor="middle" style="font-size: 8px" fill="#94a3b8"
      >t{{ k + 1 }}</text>
      <polyline
        data-testid="spark-line" fill="none" stroke="#0f172a" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round" :points="spark.obsPoints"
      />
      <polyline
        v-if="spark.forecastPoints" data-testid="spark-forecast" fill="none"
        :stroke="dfgData.accent" stroke-width="2" stroke-dasharray="4 3"
        stroke-linecap="round" :points="spark.forecastPoints"
      />
      <circle
        v-for="(d, k) in spark.dots" :key="k" data-testid="spark-point"
        :cx="d.cx" :cy="d.cy" :r="d.isF ? 3.2 : 3"
        :fill="d.isF ? dfgData.accent : '#0f172a'"
      />
      <text
        :x="spark.valLabel.x" :y="spark.valLabel.y" :text-anchor="spark.valLabel.anchor"
        style="font-size: 9px" font-weight="700" :fill="spark.valLabel.fill"
      >{{ spark.valLabel.text }}</text>
    </svg>
        </div>
        <div data-testid="step-3" class="cap">{{ STEPS[2] }}</div>
      </div>
    </div>

    <!-- ④ forecast caption — lands only on the forecast frame, beneath the woven row. -->
    <div v-if="state.step4Visible" data-testid="step-4" class="cap cap-forecast">
      {{ STEPS[3] }}
    </div>
  </div>
</template>

<style scoped>
/* Woven full-width layout.
   Monochrome + the single KU Leuven accent (dfgData.accent #1d4ed8); no 3D, no shadows. */
.dfg-evolution {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 0;
  /* Reserve the forecast-caption strip in every frame so the woven row never shifts
     when step ④ appears on the forecast frame (the caption is absolutely positioned). */
  padding-bottom: 22px;
  color: #0f172a;
  font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.woven-grid {
  display: grid;
  grid-template-columns: 0.85fr 24px 1.5fr 24px 1.2fr;
  align-items: stretch;
  flex: 1;
  min-height: 0;
  gap: 4px;
}
.wcol {
  display: flex;
  flex-direction: column;
  min-height: 0;
  justify-content: center;
  gap: 6px;
}
.warrow {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  color: #94a3b8;
}
.slot {
  flex: 1;
  min-height: 0;
}
.wcol-log .slot {
  max-height: 160px;
}
.slot-spark {
  flex: 1;
  min-height: 0;
}
.cap {
  font-size: 12px;
  color: #475569;
  text-align: center;
  line-height: 1.35;
}
.spark-title {
  font-size: 11px;
  color: #64748b;
  text-align: center;
}
.cap-forecast {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  font-weight: 600;
  font-size: 12.5px;
}
</style>
