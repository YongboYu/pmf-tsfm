<script setup>
import { computed } from 'vue'
import { dfgData } from './dfgData.js'
import { arrowGeometry, edgeGeometry } from './dfgGeometry.js'

// Uniform edge width across both snapshots. The deck deliberately does NOT encode DF frequency in
// edge WIDTH — it isn't legible to the audience, and the talk never uses weighted-edge width for
// frequency. The drifting count is shown as an on-arc number; the hero edge is distinguished by
// COLOUR (amber) only. Kept local; DfgEvolution (S4) and its tests are untouched.
const EDGE_W = 1.6

// HORIZONTAL (left-to-right) layout for S2 — overrides dfgData's vertical coords (which are shared
// with DfgEvolution/S4 and its tests and must NOT change). A wide, flat viewBox lets each snapshot
// fill the card width with no left/right margin and a long Accepted→End hero arc.
const BOX_W = 24
const BOX_H = 9
const LAYOUT = {
  start: { x: 12, y: 27 },
  create: { x: 46, y: 27 },
  created: { x: 80, y: 27 },
  sent: { x: 114, y: 27 },
  returned: { x: 148, y: 27 },
  accepted: { x: 182, y: 27 },
  cancelled: { x: 188, y: 9 },
  end: { x: 232, y: 27 },
}
const nodesH = dfgData.nodes.map((n) => ({ ...n, x: LAYOUT[n.id].x, y: LAYOUT[n.id].y }))

const props = defineProps({
  frame: { type: Number, default: 0 },
  title: { type: String, default: '' },
  heroEdge: { type: String, default: dfgData.hero },
  // Before the click the hero arc reads as a normal (navy) edge; on reveal it turns amber to
  // single out the drifting relation. The count itself is shown by a <Callout> in the slide.
  revealed: { type: Boolean, default: true },
})

const ACCENT = 'var(--accent)' // amber — drift edge, once revealed
const BRAND = 'var(--brand)' // navy — hero edge before reveal
const NEUTRAL = '#475569' // calm slate — structural edges

// Gaps sized for the wide activity boxes (half-width 12). Per-edge overrides: edges into the small
// End square need a smaller end-gap; the edge leaving the small Start dot needs a smaller start-gap
// so it actually touches the dot.
const edgeGeo = edgeGeometry(
  nodesH,
  dfgData.edges,
  12.5,
  12.5,
  { accepted__end: 5, cancelled__end: 5 },
  { start__create: 6 },
)
const edgeDraw = computed(() =>
  Object.fromEntries(dfgData.edges.map((edge) => [edge.id, arrowGeometry(edgeGeo[edge.id], EDGE_W)])),
)
const heroColor = computed(() => (props.revealed ? ACCENT : BRAND))
</script>

<template>
  <div class="dfg-snapshot">
    <div v-if="title" class="snapshot-title">{{ title }}</div>
    <svg viewBox="0 0 248 44" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
      <g v-for="edge in dfgData.edges" :key="edge.id">
        <line
          :x1="edgeDraw[edge.id].shaft[0]"
          :y1="edgeDraw[edge.id].shaft[1]"
          :x2="edgeDraw[edge.id].shaft[2]"
          :y2="edgeDraw[edge.id].shaft[3]"
          stroke-linecap="butt"
          :stroke-width="EDGE_W"
          :stroke="edge.id === heroEdge ? heroColor : NEUTRAL"
        />
        <path :d="edgeDraw[edge.id].headPath" :fill="edge.id === heroEdge ? heroColor : NEUTRAL" />
      </g>
      <g v-for="node in nodesH" :key="node.id">
        <template v-if="node.kind === 'activity'">
          <rect
            :x="node.x - BOX_W / 2"
            :y="node.y - BOX_H / 2"
            :width="BOX_W"
            :height="BOX_H"
            rx="2"
            fill="#ffffff"
            stroke="#cbd5e1"
            stroke-width="0.6"
          />
          <text :x="node.x" :y="node.y + 1.5" text-anchor="middle" style="font-size: 4.2px" fill="#0f172a">
            {{ node.label }}
          </text>
        </template>
        <circle v-else-if="node.kind === 'start'" :cx="node.x" :cy="node.y" r="4.5" fill="#0f172a" />
        <rect v-else :x="node.x - 4" :y="node.y - 4" width="8" height="8" rx="1" fill="#0f172a" />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.dfg-snapshot {
  height: 100%;
  max-height: 200px;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 6px;
  color: #0f172a;
}
.dfg-snapshot svg {
  min-height: 0;
}
.snapshot-title {
  font-size: 16px;
  font-weight: 750;
  color: #334155;
  text-align: center;
}
</style>
