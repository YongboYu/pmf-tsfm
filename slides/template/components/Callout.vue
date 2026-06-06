<!--
  Callout — guided-attention box for results slides. Clean/scientific (not hand-drawn).
  The ONLY moving/pointing accent element on the deck, so attention-orange reads as
  "look here" (the fixed headline accent rule is structural, not attention).
  Accent background + ink text (per palette rule: ink text on --accent, never white).
  Usage:  <Callout dir="down">−33% vs best baseline</Callout>
  Position it by wrapping in an absolutely-placed element (UnoCSS: absolute, -top-10, ...).
  Props: dir = down | up | left | right  (pointer side; omit for a plain pill).
-->
<script setup>
defineProps({ dir: { type: String, default: '' } })
</script>

<template>
  <span class="ae-callout" :class="dir ? `ae-callout--${dir}` : ''">
    <slot />
  </span>
</template>

<style scoped>
.ae-callout {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  position: relative;
  background: var(--accent);
  color: var(--ink);
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
  font-weight: 600;
  font-size: 19px;
  line-height: 1.2;
  padding: 7px 13px;
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(221, 138, 46, 0.35);
}
/* directional pointer (a small triangle on the chosen side) */
.ae-callout::after {
  content: "";
  position: absolute;
  border: 7px solid transparent;
}
.ae-callout--down::after  { left: 22px; bottom: -7px; border-top-color: var(--accent); border-bottom: 0; }
.ae-callout--up::after    { left: 22px; top: -7px;    border-bottom-color: var(--accent); border-top: 0; }
.ae-callout--left::after  { top: 50%; left: -7px;  transform: translateY(-50%); border-right-color: var(--accent); border-left: 0; }
.ae-callout--right::after { top: 50%; right: -7px; transform: translateY(-50%); border-left-color: var(--accent); border-right: 0; }
</style>
