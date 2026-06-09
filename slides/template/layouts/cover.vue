<!--
  Cover layout — title slide. The one place a single accent bar is kept as a brand
  flourish (no callouts compete here). No page number.
  Frontmatter:
    eyebrowImage: path to a kicker logo shown in place of the text eyebrow (e.g. /logos/caise-2026.png)
    eyebrow: small mono kicker (e.g. "CAiSE 2026 · Process mining") — used only if eyebrowImage is absent
    logo:    path to the KU Leuven·LIRIS lockup (e.g. /logos/kuleuven-liris.png)
    venue:   bottom-right text (supports line breaks via \n)
  The slot carries the title (h1), the .cover-bar, and the author/affiliation lines.
-->
<template>
  <div class="slidev-layout ae-cover">
    <img v-if="$frontmatter.eyebrowImage" :src="$frontmatter.eyebrowImage" class="cover-eyebrow-logo" alt="CAiSE 2026" />
    <div v-else-if="$frontmatter.eyebrow" class="cover-eyebrow">{{ $frontmatter.eyebrow }}</div>
    <slot />
    <img v-if="$frontmatter.logo" :src="$frontmatter.logo" class="cover-logo" alt="KU Leuven · LIRIS" />
    <div v-if="$frontmatter.venue" class="cover-venue" style="white-space: pre-line">{{ $frontmatter.venue }}</div>
  </div>
</template>
