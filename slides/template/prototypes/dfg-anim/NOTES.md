# DFG-evolution animation — pipeline comparison (THROWAWAY prototype)

**Question this prototype answers:** which pipeline should render the slide-3
DFG-evolution animation, judged by the *actual display asset* each produces (not
ASCII mockups)?

All four pipelines render the **same** real drift: bpi2017 "offer" subgraph, three
observed weeks + one forecast week, every edge stable except the hero edge
`Sent → Cancelled`, whose weekly frequency collapses **281 → 262 → 88** (forecast
88 vs held-out truth 95). Shared data is baked once by `make_data.py` →
`data.json` / `data.js`, so only the rendering differs.

## Run

```bash
# regenerate shared data (real numbers from data/time_series/bpi2017.parquet)
uv run python slides/template/prototypes/dfg-anim/make_data.py
# (re)render the pre-rendered assets
uv run python slides/template/prototypes/dfg-anim/make_dfg_gif.py            # -> dfg.gif
uv run --python .venv manim -qm --media_dir /tmp/manim_media -o dfg \
    slides/template/prototypes/dfg-anim/manim_scene.py DfgEvolution
cp /tmp/manim_media/videos/manim_scene/720p30/dfg.mp4 \
   slides/template/prototypes/dfg-anim/dfg.mp4

# compare all four (toggle via the bottom bar, or ?pipeline=svg|gif|manim|cytoscape):
open slides/template/prototypes/dfg-anim/index.html
```

Live pipelines (svg, cytoscape) are presenter-paced — click **Next frame ▶** /
press Space/→ to advance t₁→t₄; ← resets. Pre-rendered ones (gif, manim)
auto-loop. The harness shows each asset inside a faithful 16:9 slide-3 mock.

## The four assets

| Pipeline | Paced? | Vector? | New deps | Verdict notes |
|---|---|---|---|---|
| **SVG component** ★ | presenter-paced (`v-click`) | yes | none | Cleanest fit. Crisp, on-brand, you control timing in sync with speech. Becomes `components/DfgEvolution.vue`. |
| matplotlib gif | auto-loop (~7s) | no (raster) | none (reuses make_figures.py) | Easy to produce; fixed clock fights the speaker; thick short edges read a little blobby. |
| Manim mp4 | auto-loop | no (raster) | **heavy** (manim + cairo/pango/ffmpeg) | Most cinematic / cleanest arrows. Biggest install + slowest iterate; fixed timing; overkill for 6 nodes. |
| cytoscape | presenter-paced | yes (canvas) | `cytoscape` (JS) | Data-driven and fine, but needs `layout:'preset'` (else grid), extra dep + code for a tiny fixed graph. |

## Recommendation (mine — user decides)

**SVG component.** For a 6-node graph on a presenter-driven talk slide, it wins on
the axes that matter: presenter-paced via Slidev `v-click` (timing follows your
speech, not a 7s loop), vector-crisp at any projector resolution, exactly on-brand
(monochrome + KU Leuven blue, no shadows), and zero new dependencies. Manim is the
only one that looks *better* in raw polish, but it's the heaviest pipeline and
auto-paced — not worth it here. Keep Manim in mind only if a future slide needs a
genuinely cinematic pre-render.

**If chosen → absorb:** promote `render_svg.js` into
`slides/template/components/DfgEvolution.vue` (v-click drives `setFrame`), bake the
`data.json` frames in, replace the slide-3 placeholder (`slides.md:99-104`), delete
this prototype dir + the cytoscape dep + `dfg.gif`/`dfg.mp4`. Also reword the slide-3
left column to drop the reserved term "window" (CONTEXT.md) — the harness mock
already previews the fix ("weekly sublogs" / "Each week → a time-indexed DFG").

**PIPELINE VERDICT:** SVG component (user choice, 2026-05-31).

---

# Round 2 — content & layout (`layout.html`)

**Question:** what should slide 3 *show*? One unified diagram that narrates the
workflow (event log → weekly DFGs → per-edge time series → forecast) AND lands
PMF's value — without stealing the drift story that slides 4 + 7b own.

**Locked:**
- **Value = capability, not drift-catch.** Calm Oct stretch (hero edge
  346→314→316, forecast 316 vs truth 315). No dramatic collapse — that's slide 4.
- **One DFG that morphs in place** (clicks t1→t4; t4 edges → accent = forecast).
- **Accumulating hero-edge sparkline** = step 3 ("DF edge → time series") made
  literal + the comparison-memory the morph loses.
- **Event-log-slice motif** + **lockstep step-labels** ①–④ define the vague terms
  ("weekly sublog", "time-indexed DFG") by showing them.

**Two layouts prototyped (`?layout=woven|twocol`):**

| Layout | Pros | Cons |
|---|---|---|
| **Woven full-width** ★ | One unified figure; words label each part of the picture (directly fixes "the text is vague"); reads as a workflow AND a money-shot. | Breaks the deck's `two-cols-header` consistency; some whitespace under the tall DFG. |
| Two-column | Keeps a readable numbered list (active step highlights in lockstep); consistent with other slides. | Doesn't fully become "one diagram" (text + picture separate); smaller diagram; looser right-side spacing. |

**Recommendation:** **Woven full-width** — it's the one that realises the user's
stated goal (a single diagram that both narrates the workflow and defines its
terms in place). Two-column is the safer, more consistent fallback.

**LAYOUT VERDICT:** Woven full-width (user choice, 2026-05-31). Refinement: the
event-log slice band **slides right one week per click** (tiles the log into the 4
consecutive weeks) so each click visibly grabs a later time-slice.
