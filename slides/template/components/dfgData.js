// Baked bpi2017 "offer" subgraph frames for the slide-3 DFG-evolution figure.
// Real numbers (weeks Oct 02–23); hero edge `sent__cancelled` 346→314→316, forecast 316
// vs held-out truth 315. Produced by prototypes/dfg-anim/make_data.py — baked in here so
// the slide renders with no runtime data read. Do not hand-edit weights; regenerate upstream.
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
    { id: 'cancelled', label: 'Cancelled', kind: 'activity', x: 80, y: 66 },
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
      truth: {
        start__create: 903, create__created: 903, created__sent: 823,
        sent__cancelled: 315, sent__returned: 497, returned__accepted: 343,
        accepted__end: 232, cancelled__end: 455,
      },
    },
  ],
}
