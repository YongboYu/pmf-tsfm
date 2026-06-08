<!--
  Two-column Assertion-Evidence layout.
  Same head (locator + assertion + droppable rule) as assertion-evidence, with a
  two-column evidence body fed by the ::left:: / ::right:: named slots.
  Frontmatter: locator, assertion, rule (same semantics as assertion-evidence.vue).
-->
<template>
  <div class="slidev-layout ae-slide ae-twocol">
    <div v-if="$frontmatter.locator" class="locator">{{ $frontmatter.locator }}</div>
    <h1 v-if="$frontmatter.assertion" class="assertion">
      {{ $frontmatter.assertion }}<span v-if="$frontmatter.rule !== false" class="rule" />
    </h1>
    <div class="ae-cols ae-body">
      <div class="ae-col"><slot name="left" /></div>
      <div class="ae-col"><slot name="right" /></div>
    </div>
    <PageNo :current="$nav.currentPage" :slides="$nav.slides" />
  </div>
</template>
