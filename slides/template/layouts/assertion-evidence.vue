<!--
  Assertion-Evidence base layout — the deck's default content slide.
  Skeleton: locator -> assertion headline -> accent rule + gap -> evidence -> page number.
  Frontmatter:
    locator:   short section name (e.g. "Results")        — JetBrains Mono eyebrow
    assertion: one-sentence claim (the slide's message)    — absorbs the old "takeaway"
    rule:      set `false` to hide the accent bar on this slide (also droppable deck-wide
               by deleting the `.rule` CSS block in style.css)
  Styling is global in style.css (keyed off .ae-slide) so it reaches slotted markdown.
-->
<template>
  <div class="slidev-layout ae-slide">
    <div v-if="$frontmatter.locator" class="locator">{{ $frontmatter.locator }}</div>
    <h1 v-if="$frontmatter.assertion" class="assertion">
      {{ $frontmatter.assertion }}<span v-if="$frontmatter.rule !== false" class="rule" />
    </h1>
    <div class="ae-body">
      <slot />
    </div>
    <PageNo :current="$nav.currentPage" :slides="$nav.slides" />
  </div>
</template>
