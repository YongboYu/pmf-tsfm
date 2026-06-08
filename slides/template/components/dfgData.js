// Baked bpi2017 "offer" subgraph frames for the slide-3 DFG-evolution figure.
// Real numbers (weeks Oct 02–23); hero edge `sent__cancelled` 346→314→316, forecast 316
// vs held-out truth 315. Baked in here so the slide renders with no runtime data read.
export const dfgData = {
  dataset: 'bpi2017',
  subgraph: 'offer',
  hero: 'sent__cancelled',
  accent: '#1d4ed8',
  max_freq: 927,
  nodes: [
    { id: 'start', label: '▶', kind: 'start', x: 34, y: 7 },
    { id: 'create', label: 'Create Offer', kind: 'activity', x: 34, y: 22 },
    { id: 'created', label: 'Created', kind: 'activity', x: 34, y: 37 },
    { id: 'sent', label: 'Sent', kind: 'activity', x: 34, y: 53 },
    { id: 'returned', label: 'Returned', kind: 'activity', x: 34, y: 69 },
    { id: 'accepted', label: 'Accepted', kind: 'activity', x: 34, y: 84 },
    { id: 'cancelled', label: 'Canceled', kind: 'activity', x: 80, y: 66 },
    { id: 'end', label: '■', kind: 'end', x: 50, y: 95 },
  ],
  edges: [
    { id: 'start__create', source: 'start', target: 'create', hero: false },
    { id: 'create__created', source: 'create', target: 'created', hero: false },
    { id: 'created__sent', source: 'created', target: 'sent', hero: false },
    { id: 'sent__cancelled', source: 'sent', target: 'cancelled', hero: true },
    { id: 'sent__returned', source: 'sent', target: 'returned', hero: false },
    { id: 'returned__accepted', source: 'returned', target: 'accepted', hero: false },
    { id: 'accepted__end', source: 'accepted', target: 'end', hero: false },
    { id: 'cancelled__end', source: 'cancelled', target: 'end', hero: false },
  ],
  // Arcs shown in the S4 middle STACK (top→bottom, hero first) — point: "we forecast ALL arcs".
  stack: [
    { id: 'sent__cancelled', label: 'Sent → Canceled', hero: true },
    { id: 'created__sent', label: 'Created → Sent', hero: false },
    { id: 'sent__returned', label: 'Sent → Returned', hero: false },
  ],
  // NOTE: the per-frame `daily` maps below are PLACEHOLDER series — each week's 7 values are
  // fabricated to sum to that edge's real weekly `weights`. The weekly weights/truth and the hero
  // numbers are the real baked bpi2017 "offer" values; only the day-level shape is synthetic.
  // TODO(figure-faithfulness, ADR-0006): regenerate `daily` from the actual bpi2017 daily DF counts
  // (keeping each week's Σ = the weekly arc) before the slides PR ships. `dailyHero` is the real
  // hero daily series and is unchanged.
  frames: [
    {
      label: 't₁ · Oct 02',
      kind: 'observed',
      date: '2016-10-02',
      weights: {
        start__create: 927, create__created: 927, created__sent: 844,
        sent__cancelled: 346, sent__returned: 477, returned__accepted: 394,
        accepted__end: 465, cancelled__end: 483,
      },
      // Daily hero (sent__cancelled) series for this week; sums to weights.sent__cancelled
      // (the paper bins daily and sums the horizon into the weekly DFG arc).
      dailyHero: [48, 50, 49, 51, 47, 52, 49],
      daily: {
        sent__cancelled: [48, 50, 49, 51, 47, 52, 49], // Σ 346
        created__sent: [121, 120, 122, 119, 121, 120, 121], // Σ 844
        sent__returned: [68, 68, 68, 68, 68, 68, 69], // Σ 477
      },
    },
    {
      label: 't₂ · Oct 09',
      kind: 'observed',
      date: '2016-10-09',
      weights: {
        start__create: 887, create__created: 887, created__sent: 812,
        sent__cancelled: 314, sent__returned: 475, returned__accepted: 412,
        accepted__end: 353, cancelled__end: 412,
      },
      dailyHero: [46, 44, 45, 43, 47, 45, 44],
      daily: {
        sent__cancelled: [46, 44, 45, 43, 47, 45, 44], // Σ 314
        created__sent: [116, 115, 117, 116, 116, 116, 116], // Σ 812
        sent__returned: [68, 68, 68, 68, 68, 68, 67], // Σ 475
      },
    },
    {
      label: 't₃ · Oct 16',
      kind: 'observed',
      date: '2016-10-16',
      weights: {
        start__create: 845, create__created: 845, created__sent: 775,
        sent__cancelled: 316, sent__returned: 436, returned__accepted: 382,
        accepted__end: 306, cancelled__end: 435,
      },
      dailyHero: [45, 46, 44, 47, 45, 44, 45],
      daily: {
        sent__cancelled: [45, 46, 44, 47, 45, 44, 45], // Σ 316
        created__sent: [111, 110, 111, 110, 111, 111, 111], // Σ 775
        sent__returned: [62, 62, 62, 62, 62, 62, 64], // Σ 436
      },
    },
    {
      label: 't₄ · Oct 23 (forecast)',
      kind: 'forecast',
      date: '2016-10-23',
      weights: {
        start__create: 845, create__created: 845, created__sent: 775,
        sent__cancelled: 316, sent__returned: 436, returned__accepted: 382,
        accepted__end: 306, cancelled__end: 435,
      },
      // 7 daily forecast steps (prediction_length: 7); their Σ is the weekly forecast-DFG arc.
      dailyHero: [46, 45, 45, 44, 46, 45, 45],
      daily: {
        sent__cancelled: [46, 45, 45, 44, 46, 45, 45], // Σ 316
        created__sent: [111, 111, 110, 111, 110, 111, 111], // Σ 775
        sent__returned: [62, 62, 63, 62, 62, 62, 63], // Σ 436
      },
      truth: {
        start__create: 903, create__created: 903, created__sent: 823,
        sent__cancelled: 315, sent__returned: 497, returned__accepted: 343,
        accepted__end: 232, cancelled__end: 455,
      },
    },
  ],
}
