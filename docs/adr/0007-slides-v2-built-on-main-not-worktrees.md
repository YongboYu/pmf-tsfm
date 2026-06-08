# The 19-slide v2 deck is rebuilt on `main`, not on the parallel v2 worktrees

**Status:** accepted (2026-06-06)

For the CAiSE 2026 slides content revision we build the 19-slide v2 deck **on top of `main`**
(post-#110) — its locked Assertion-Evidence template (#108), locked KU palette (#107,
`--brand #00407a` / `--accent #dd8a2e`), and ADR-0006 paper-faithful figure provenance — and treat
the two near-complete v2 worktrees as **reference only**. We do **not** build on either worktree.

## Context

Three divergent lines of work existed simultaneously:

- **`main`** — the "visual-verify" track. Carries all the locked rigor (#107/#108, ADR-0006) but
  only ~14 content + 8 backup slides on the older 11-beat outline, ~3 migrated to the template.
- **`../pmf-tsfm-slides-revision`** (Claude v2) and **`../pmf-tsfm-slides-codex`** (Codex v2) — two
  uncommitted worktrees branched from the *old* base `5f62ada`, each a feature-complete 19-slide
  deck matching the revision brief, but **predating** (and therefore lacking) the locked palette,
  the AE template, and ADR-0006 provenance, and using their own non-locked themes + (Codex) PNG
  figures. Plus GitHub PRD #97 + arc issues #98–#106 written for the worktree-AFK flow.

## Considered options

1. **Build on `main`** *(chosen)* — rebuild the 19-slide structure on main's verified foundation,
   mining the worktrees for wording, figure data, the sparsity pipeline, and the `DfgSnapshot`
   idea.
2. **Build on `slides-revision`** (port `main`'s locked work forward) — rejected: porting means
   regenerating every figure onto the locked palette and re-homing onto the AE template anyway —
   i.e. redoing most of the locked value — while inheriting a stale off-`5f62ada` branch with heavy
   `slides.md` / `figure_manifest.py` / `make_figures.py` rebase conflicts and a non-locked theme.
3. **Build on `slides-codex`** — same staleness, plus PNG (non-regenerable) figures and a `#1d4ed8`
   theme further from the brand.

## Consequences

- The two v2 worktrees become **reference material**, not a foundation. They are kept around to mine
  during the revision and removed afterwards (`git worktree remove`).
- PRD #97 is refreshed to the main-based, JIT-reviewed approach; arc issues **#98–#106 are folded
  and closed** (they encoded the abandoned worktree-AFK flow).
- The content revision proceeds in a fresh worktree `../pmf-tsfm-slides-content`
  (branch `slides/content-revision`) off updated `origin/main`.
- A future reader who finds two abandoned 19-slide worktree decks + a closed PRD should read this
  ADR before assuming work was lost: the *content* was carried forward; only the *base* changed.
