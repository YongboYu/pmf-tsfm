<!--
  Page-number footer.
  - Main slides: `current / <main count>` — the total counts MAIN slides only.
  - Backup slides: a lowercase Roman numeral (i, ii, …). Backups are the trailing
    block carrying `hideInToc: true` (and `class: backup`); they are excluded from
    the main total and numbered separately so they read as off-the-main-thread.
  Counts are derived at runtime from frontmatter (never hardcoded), so adding or
  removing a slide keeps the numbering correct.

  `current` and `slides` are passed in from the layout's template `$nav` (the
  router-bound singleton); calling useNav() here returns a detached context whose
  currentPage is not tracked during static PNG export.
-->
<script setup>
import { computed } from 'vue'

const props = defineProps({
  current: { type: Number, required: true },
  slides: { type: Array, required: true },
})

const isHidden = route => Boolean(route?.meta?.slide?.frontmatter?.hideInToc)

const mainCount = computed(() => props.slides.filter(s => !isHidden(s)).length)
const isBackup = computed(() => isHidden(props.slides[props.current - 1]))
const backupIndex = computed(() => props.current - mainCount.value)

function toRoman(n) {
  if (!(n > 0)) return ''
  const map = [[10, 'x'], [9, 'ix'], [5, 'v'], [4, 'iv'], [1, 'i']]
  let out = ''
  for (const [v, s] of map) while (n >= v) { out += s; n -= v }
  return out
}

const label = computed(() =>
  isBackup.value ? toRoman(backupIndex.value) : `${props.current} / ${mainCount.value}`,
)
</script>

<template>
  <div class="pageno">{{ label }}</div>
</template>
