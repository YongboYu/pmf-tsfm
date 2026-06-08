<script setup>
// S4 — "From event log to DF time series", STAGE-BASED reveal (4 visual columns):
//   stage 0  →  event log + observed navy series stack            (steps ① ②)
//   stage 1  →  + dashed amber 7-day forecast tail on every arc   (step ③)
//   stage 2  →  + reassembled / forecasted DFG on the right        (step ④)
// `frame` is the stage index (0,1,2), bound to $clicks in the deck (clicks: 2).
// Reuses the pure cores (dfgGeometry) + dfgData. DFG arcs use UNIFORM width per the settled
// "no edge-width frequency" convention; the forecast count is shown as an on-arc number.
import { computed } from 'vue'
import { dfgData } from './dfgData.js'
import { edgeGeometry, arrowGeometry } from './dfgGeometry.js'
import StackSpark from './StackSpark.vue'

const props = defineProps({ frame: { type: Number, default: 0 } })

const showForecast = computed(() => props.frame >= 1)
const showDfg = computed(() => props.frame >= 2)

// Uniform DFG geometry (no weight-scaled thickness).
const UNIFORM = 1.6
const edgeGeo = edgeGeometry(dfgData.nodes, dfgData.edges, 4.4, 5.8, { [dfgData.hero]: 14 })
const edgeDraw = Object.fromEntries(dfgData.edges.map((e) => [e.id, arrowGeometry(edgeGeo[e.id], UNIFORM)]))
const heroMid = [
  (edgeGeo[dfgData.hero][0] + edgeGeo[dfgData.hero][2]) / 2,
  (edgeGeo[dfgData.hero][1] + edgeGeo[dfgData.hero][3]) / 2,
]
const fc = dfgData.frames[dfgData.frames.length - 1] // the forecast frame
const heroCount = `≈${fc.weights[dfgData.hero]} (truth ${fc.truth[dfgData.hero]})`

// Event-log temporal windows: a wider navy "past" window (the observed history) is always shown;
// on the forecast click an amber "future" window appears beside it (the horizon we forecast).
// Navy/amber mirror the observed/forecast colours of the series stack.
const LOG = (() => {
  const rows = 6, cols = 8, bw = 10.5, bh = 10.5, x0 = 6, y0 = 26, gx = 1.7, gy = 4
  const colStep = bw + gx, rowStep = bh + gy, pad = 1.5
  const cells = []
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cells.push({ x: x0 + c * colStep, y: y0 + r * rowStep })
  const win = (startCol, nCols) => ({ x: x0 + startCol * colStep - pad, w: nCols * colStep - gx + 2 * pad })
  return {
    cells, bw, bh, x0,
    headerY: 10,        // gap above the grid (event log)
    bandY: y0 - 3, bandH: rows * rowStep - gy + 6,
    past: win(0, 4),    // observed history — wider
    future: win(5, 2),  // forecast horizon — after a one-column gap
    labelY: y0 + rows * rowStep + 17,  // gap below the grid (past / future)
    W: x0 + cols * colStep + 4, H: y0 + rows * rowStep + 26,
  }
})()
</script>

<template>
  <div class="dfg-evo" data-testid="dfg-pipeline">
    <div class="grid-a">
      <!-- ① event log -->
      <div class="wcol">
        <div class="slot slot-log">
          <svg :viewBox="`0 0 ${LOG.W} ${LOG.H}`" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
            <text :x="LOG.x0" :y="LOG.headerY" style="font-size: 12px" fill="var(--neutral)" font-weight="700">event log</text>
            <rect v-for="(cell, k) in LOG.cells" :key="k" :x="cell.x" :y="cell.y" :width="LOG.bw" :height="LOG.bh"
              rx="1.6" fill="var(--surface-alt)" stroke="var(--hairline)" stroke-width="0.5" />
            <!-- past window (navy) — always -->
            <g>
              <rect :x="LOG.past.x" :y="LOG.bandY" :width="LOG.past.w" :height="LOG.bandH" rx="2"
                fill="var(--brand)" fill-opacity="0.10" stroke="var(--brand)" stroke-width="1.2" stroke-dasharray="3 2" />
              <text :x="LOG.past.x + LOG.past.w / 2" :y="LOG.labelY" text-anchor="middle" style="font-size: 12px" font-weight="700" fill="var(--brand)">past</text>
            </g>
            <!-- future window (amber) — appears with the forecast -->
            <g v-if="showForecast">
              <rect :x="LOG.future.x" :y="LOG.bandY" :width="LOG.future.w" :height="LOG.bandH" rx="2"
                fill="var(--accent)" fill-opacity="0.12" stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="3 2" />
              <text :x="LOG.future.x + LOG.future.w / 2" :y="LOG.labelY" text-anchor="middle" style="font-size: 12px" font-weight="700" fill="var(--accent)">future</text>
            </g>
          </svg>
        </div>
        <div class="cap">① Event log → daily DF counts</div>
      </div>

      <div class="warrow">→</div>

      <!-- ② STACK of DF series (the point: forecast ALL arcs) -->
      <div class="wcol wcol-stack">
        <div class="stack-head">one daily series per DF edge</div>
        <div class="slot stack-body">
          <StackSpark v-for="arc in dfgData.stack" :key="arc.id" :arc="arc" :frames="dfgData.frames"
            :show-forecast="showForecast" />
        </div>
        <div class="cap">
          ② Each DF edge → a daily series
          <span v-if="showForecast" class="cap-fc"> · ③ forecast 7 days</span>
        </div>
      </div>

      <!-- ③ forecast arrow (appears with the amber tail) -->
      <div class="warrow warrow-fc">
        <template v-if="showForecast">
          <span class="fc-tag">forecast<br/>7 days</span>
          <span class="fc-arrow">→</span>
        </template>
      </div>

      <!-- ④ reassembled / forecasted DFG (appears last) -->
      <div class="wcol">
        <div class="slot slot-dfg">
          <svg v-if="showDfg" viewBox="0 0 100 104" width="100%" height="100%">
            <g v-for="edge in dfgData.edges" :key="edge.id">
              <line :data-edge="edge.id"
                :x1="edgeDraw[edge.id].shaft[0]" :y1="edgeDraw[edge.id].shaft[1]"
                :x2="edgeDraw[edge.id].shaft[2]" :y2="edgeDraw[edge.id].shaft[3]"
                fill="none" stroke-linecap="butt" :stroke-width="UNIFORM" stroke="var(--accent)" stroke-dasharray="2.4 1.8" />
              <path :d="edgeDraw[edge.id].headPath" fill="var(--accent)" />
            </g>
            <!-- forecast count as an on-arc number (replaces edge-thickness encoding) -->
            <text :x="heroMid[0] + 4" :y="heroMid[1] - 2.5" style="font-size: 4.4px" font-weight="700" fill="var(--accent)">{{ heroCount }}</text>
            <g v-for="node in dfgData.nodes" :key="node.id">
              <template v-if="node.kind === 'activity'">
                <rect :x="node.x - 16" :y="node.y - 4.6" width="32" height="9.2" rx="2.4" fill="#fff" stroke="var(--hairline)" stroke-width="0.6" />
                <text :x="node.x" :y="node.y + 1.6" text-anchor="middle" style="font-size: 4.4px" fill="var(--ink)">{{ node.label }}</text>
              </template>
              <circle v-else-if="node.kind === 'start'" :cx="node.x" :cy="node.y" r="3.8" fill="var(--ink)" />
              <rect v-else :x="node.x - 3.4" :y="node.y - 3.4" width="6.8" height="6.8" rx="0.9" fill="var(--ink)" />
            </g>
          </svg>
        </div>
        <div class="cap" :class="{ 'cap-ghost': !showDfg }">④ Σ daily → forecasted DFG</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dfg-evo { position: relative; display: flex; flex-direction: column; width: 100%; height: 100%; min-height: 0; color: var(--ink); }
.grid-a { display: grid; grid-template-columns: 0.62fr 22px 1.46fr 48px 1.4fr; align-items: stretch; flex: 1; min-height: 0; gap: 6px; }
.wcol { display: flex; flex-direction: column; min-height: 0; justify-content: center; gap: 8px; }
.wcol-stack { justify-content: flex-start; }
.warrow { display: flex; align-items: center; justify-content: center; font-size: 24px; color: var(--neutral-soft); }
.warrow-fc { flex-direction: column; gap: 1px; }
.fc-tag { font-size: 14px; font-weight: 700; color: var(--accent); line-height: 1.1; text-align: center; }
.fc-arrow { font-size: 24px; color: var(--accent); line-height: 1; }
.slot { flex: 1; min-height: 0; }
.slot-log { max-height: 170px; }
.slot-dfg { display: flex; align-items: center; justify-content: center; }
.stack-body { display: flex; flex-direction: column; gap: 12px; flex: 1; min-height: 0; }
.stack-head { font-size: 15px; color: var(--neutral); text-align: center; }
.stack-head strong { color: var(--accent); }
.cap { font-size: 15px; color: var(--neutral); text-align: center; line-height: 1.3; }
.cap-fc { color: var(--accent); font-weight: 600; }
.cap-ghost { visibility: hidden; }
</style>
