# /03-content-pass

Revise ONE beat of `slides/template/slides.md` in place. Run a compliance check.
Report a tight summary.

## Inputs

- Required: a **beat identifier** as args — beat number (e.g. `3`), beat title
  (e.g. `Why TSFMs`), or `backup-<name>` (e.g. `backup-rmse`).
  Match against headers in `slides/talk_design/outline.md` and the backup slides
  in `slides/template/slides.md`.
- Optional: feedback after a colon. Example:
  `/03-content-pass 7a: bullet 3 overstates the BPI2017 finding`.
- Context files (read these first):
  - `slides/SLIDES.md` — project brief and non-negotiable critique-derived constraints
  - `slides/talk_design/outline.md` — beat-by-beat content and time budgets
  - `slides/talk_design/workflow.md` — figure naming convention
  - `slides/template/slides.md` — the live deck

## Mode

REVISE IN PLACE. Use `Edit` on `slides/template/slides.md`. Do not append.
Do not output the revised markdown into chat — it goes into the file. Chat output
is the compliance report only.

## Process

1. **Resolve beat ID** against `outline.md` beat headers (or backup-slide headers in
   `slides.md`). If ambiguous, ask the user.
2. **Read context**:
   - `SLIDES.md` (all sections — small file)
   - `outline.md` (just the matched beat block)
   - `slides.md` (just the slide(s) corresponding to this beat — usually 1, sometimes 2)
3. **Scan for non-negotiable constraints**: in `SLIDES.md`, read the
   "Critique-derived constraints" section. List every rule that mentions this beat
   or its slide. These MUST be honored.
4. **Smart-detect the beat's figure** (if applicable per the table in `workflow.md`):
   Bash-check whether the expected file exists at
   `slides/template/public/figures/<expected-name>.{png,svg,webp,gif,mp4}`.
   - If present → replace the placeholder `<div>` with `<img src="/figures/<name>" ... />`.
   - If absent → keep the placeholder; refine its descriptive copy if it's unclear.
5. **Edit `slides.md`** for this beat: title, bullets, layout, speaker notes.
   If user gave feedback after a colon, address it specifically while keeping all
   other constraints.
6. **Output the compliance report** in chat (template below). Nothing else.

## Constraints

**Allowed:**
- Slidev built-in layouts: `default`, `two-cols`, `center`, `end`, `section`
- UnoCSS utility classes (`grid`, `border-2`, `opacity-70`, `pl-4`, `gap-6`, etc.)
- Minimal inline `style` for accent colors only (e.g., `style="color: #1d4ed8"`)
- KaTeX (rarely needed)

**Forbidden:**
- `.vue` components — out of scope; defer to a future components pass
- Broken `<img>` references — only wire `<img>` if smart-detect confirmed the file exists
- More than 5 bullets / 12 words per bullet on a main slide
  - **Relaxed for backup slides** — note "backup" in the compliance report and skip the check
- Modifying any beat other than the one being revised
- Rewriting `outline.md` or `SLIDES.md` — structural changes belong to `/01-outline`

**Non-negotiable per-beat constraints (from SLIDES.md):**
- **Beat 4 (Why TSFMs)**: slide must say *"no event logs in pretraining, to our knowledge"* —
  NOT "no process data"
- **Beat 7a (Results bars)**: speaker notes must include the baseline framing —
  *"two strongest baselines from our prior benchmark"*
- **Beat 9 (ER)**: memorized rebuttal embedded in speaker notes verbatim
  (see SLIDES.md "Spoken anchor lines")
- **Beat 10 (Signals)**: signal 1 must contain the hedge clause —
  *"four logs is not a paradigm"*
- One-line inline definition required when first used: **zero-shot**, **LoRA**,
  **Entropic Relevance**

## Compliance report template

Emit exactly this in chat after the Edit:

```
Beat <N: title>  ·  slides <X>(-Y)  ·  layout: <chosen layout>

  audience Q in speaker notes ........ PASS / FAIL
  ≤5 bullets / ≤12 words per bullet .. PASS / FAIL / N/A (backup)
  spoken anchor line in notes ........ PASS / FAIL / N/A (no anchor for this beat)
  critique-derived constraint ........ PASS / FAIL / N/A — <name the constraint>
  figure smart-detect ................ WIRED <filename> / PLACEHOLDER kept (<filename> not found) / N/A

layout choice: <one-phrase justification>

open question (if any): <single question where intent was unclear>
```

If a check FAILs, fix it before reporting. Only emit FAIL for constraints you
deliberately did not honor and want the user to override.

## Feedback mode

If user passes feedback after a colon, prioritize that feedback. Report what changed
in addition to the standard compliance summary. Example output prefix:

```
Addressed: "bullet 3 overstates the BPI2017 finding"
Change: "TSFMs always recover the drift" → "TSFMs adapt to the drift over time"

Beat 7a · slides 9 · layout: default
  ...
```
