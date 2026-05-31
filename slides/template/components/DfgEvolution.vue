<script setup>
import { computed } from 'vue'
import { dfgData } from './dfgData.js'
import { frameState } from './dfgFrameState.js'

// `frame` is the master index; slice #3 will bind it to Slidev v-click.
const props = defineProps({
  frame: { type: Number, default: 0 },
})

const state = computed(() => frameState(dfgData, props.frame))
</script>

<template>
  <!-- Minimal scaffold: stable hooks only. The full woven figure lands in slice #62. -->
  <div data-testid="dfg-evolution">
    <!-- Event-log slice band: slides right one week per frame to grab a later time-slice. -->
    <div data-testid="log-slice" :data-slice-index="state.sliceIndex" />
    <svg viewBox="0 0 100 104" width="100%" height="100%">
      <line
        v-for="edge in dfgData.edges"
        :key="edge.id"
        :data-edge="edge.id"
        :data-forecast="state.edges[edge.id].isForecast"
        :stroke-width="state.edges[edge.id].strokeWidth"
        :stroke="state.edges[edge.id].isForecast ? dfgData.accent : '#475569'"
      />
    </svg>
    <!-- Accumulating hero-edge sparkline: one point per observed/forecast frame so far. -->
    <svg data-testid="sparkline" viewBox="0 0 100 30" width="100%">
      <circle
        v-for="n in state.sparkPoints"
        :key="n"
        data-testid="spark-point"
      />
    </svg>
    <!-- Step ④ ("Forecast next week → forecasted DFG") lands only on the forecast frame. -->
    <div v-if="state.step4Visible" data-testid="step-4">
      ④ Forecast next week → forecasted DFG
    </div>
  </div>
</template>
