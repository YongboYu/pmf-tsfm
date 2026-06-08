# Verbatim manuscript — CAiSE 2026 talk

Spoken script for the 20-minute talk. **Prose is written in the content-revision phase** —
this file establishes the structure and the sync convention only.

## How this stays in sync with the deck

Every slide carries a stable anchor `[S-xx]` (two digits, in deck order). The anchor appears
in **both** places, so the manuscript and the slides never drift:

- **Here**, as the section heading for that slide's script: `## [S-03] From event log to DF time series`.
- **In `slides.md`**, as the first line of that slide's Slidev presenter-note comment:

  ```md
  <!--
  [S-03]
  Spoken cue / delivery note for the presenter view.
  -->
  ```

Renumber anchors only when the deck order changes; keep the `[S-xx]` ↔ slide mapping 1:1.
The short per-slide `<!-- -->` note is the **delivery cue** (transitions, timing, the one line
to land); the manuscript below is the **full verbatim**.

## Per-slide budget

20 min ≈ the v2 outline's beat budget. Keep each slide to its one key message; if a slide's
script runs long, the slide is doing too much (split it or cut).

---

## [S-01] Title

**Opening anchor — rehearse COLD, deliver before advancing (verbatim, locked):**

> "I'm going to show you something that took me a while to believe: the best forecaster for your
> **process model** isn't one you **trained** — and probably isn't one you should train."

Pause. Let it land. Then advance.

## [S-02] Process discovery gives one static model, but processes drift

We start in familiar territory. **Process discovery** takes an event log — cases, activities,
timestamps, like the rows on the left — and extracts a **process model**. One common kind of
process model is the **Directly-Follows Graph**, or **DFG**: each node is an activity, each arrow
is a directly-follows relation, and the number on the arrow is how often that transition fired.

Now look at the same BPI2017 loan-offer process over two different periods. Same activities, same
shape of graph — but the **counts move**. Here, *Accepted → End* falls from **465 to 306**, a 34%
drop. The process **drifted**. Discovery gives you a single, **static snapshot** of whatever window
you ran it on; it says nothing about where those numbers are heading.

That gap is exactly where **Process Model Forecasting — PMF** — begins: learn how the model changes
over time, and **predict the next one**.

_Transition out:_ "If the model drifts, the natural question is *how* — and that splits into two
very different prediction problems."

## [S-03] PPM forecasts one case · PMF the whole process model

Same loan log, two very different prediction problems. Most of us in this room know the first one.

**Predictive Process Monitoring — PPM — is case-level.** You take one ongoing case, this single
application part-way through, and you predict its future: what activity comes next, how long until
it's decided, or how it ends — *"will this loan be canceled?"* The horizon is **the rest of that
one case**.

What we do is the other one. **Process Model Forecasting is system-level.** You don't follow a
single case; you take a window of the whole log and forecast the **next process model** — how often
each transition will fire across *all* cases, for instance *"how often does offer sent → canceled
occur?"* The horizon is **the near-term system future**, the aggregate behaviour of the process, not
one trace.

So: same event log, completely different question. PPM zooms into one case; PMF steps back to the
whole system. _(The conceptual horizon here is weeks-to-months; the concrete 7-day experimental
horizon is setup detail for S-12 — do not say it here.)_

_Transition out:_ "Both need data, but PMF needs something different — let me show you the pipeline."

## [S-04] Each directly-follows edge becomes a time series we forecast

_Transition in:_ "Both need data, but PMF needs something different — let me show you the pipeline."

Here's how PMF becomes a forecasting problem. A **directly-follows relation** — a **DF** relation —
is just one transition, activity A directly followed by activity B. We take each such edge and count
how often it fires **per day**. _(Let the clicks carry it — don't talk over them.)_

So every arc in the DFG becomes its own **daily count series** _(click 1: the navy history window —
note there are several arcs, the point being we forecast all of them, not just one)_. For each series
we **forecast the next 7 days** _(click 2: the amber tail)_. And then we sum each arc's seven daily
forecasts back up and reassemble them into a single **forecasted DFG** _(click 3: ≈316 against a
held-out 315 — strikingly close)_.

That's the whole reduction: a process model forecast is just a bundle of short DF-series forecasts,
aggregated **daily** and projected **7 days** ahead. _(Keep it to that line. Do NOT mention stride /
expanding window / the 60-20-20 split here — that is S-12. "Past/future" is a conceptual
history→horizon motif, not the experimental window.)_

_Transition out:_ "So it's a forecasting problem. Why isn't it already solved?"

## [S-05] DF series are hard: drift, intermittency, and heterogeneity

_Transition in:_ "So it's a forecasting problem. Why isn't it already solved?"

Because these are nasty little series. Everything on this slide is **real ground truth** — no model
lines yet. Three things make them hard.

First, **drift**. On the left, BPI2017 *Sent → Canceled*: it fires around 46 times a day early on,
then genuinely **collapses** to about 3. That's a real level shift, not noise.

Second, **intermittency**. On the right, BPI2019-1 *Cancel → Record Invoice Receipt*: roughly 80%
zeros, long flat stretches, then a **sudden spike** to nearly 30. Neither series is white noise, and
neither is a smooth trend.

And third — this is the one S5 owns — they're **heterogeneous**. The pattern differs **within** a
single log from one DF relation to the next, and **across** logs: these two panels are two different
logs with completely different shapes. So there's no single regime a model can lock onto.

_(These exact two plots come back on S-14 with the model line revealed — so do NOT show any model
line here.)_

_Transition out:_ "These are hard. So — do the methods we already have handle them?"

## [S-06] Machine Learning (ML) / Deep Learning (DL) underperform

_Transition in:_ "These are hard. So — do the methods we already have handle them?"

The honest answer is no — and let me show you with the strongest method available. The model on
these panels is **XGBoost**: in our prior PMF benchmark, **Yu et al. 2025**, it was the
**recommended top forecaster** on exactly this task, so it's the fairest possible opponent. It
becomes our trained ML baseline from here on.

Here it is on the **same two series** from the last slide. On the left, **drift**: XGBoost stays
stuck high — the boxed back-half — while the truth collapses to near zero. It completely misses the
level shift. On the right, **intermittency**: XGBoost **overshoots** badly, hallucinating spikes of
40 to 70 — the amber arrow — where the truth is mostly zero.

Why does the strongest trained model fail like this? Because the data is **small, complex, and
heterogeneous** — only a few hundred daily points per log — and a model trained from scratch on that
**overfits**. That diagnosis is the whole reason we'll reach for a model we *don't* train. _(Keep
"overfit" for here in speech and for S-9 on the slide — it's deliberately off this slide.)_

_(Honest nuance, speech only: the TSFM we reveal on S-14 wins by staying **controlled** — not
overshooting — not by catching the rare spikes. Do NOT show any TSFM line here; that's S-14.)_

_Transition out:_ "It's not bad luck — these series are statistically off the charts, and the logs
are tiny."

## [S-07] DF series are shorter and statistically harder

_Transition in:_ "It's not bad luck — these series are statistically off the charts, and the logs
are tiny."

This is the quantitative *why* behind that overfitting. Two facts, one message.

First, these series are **complex**. The table flags three of the paper's complexity metrics —
**Transition** (abrupt regime changes), **Shifting** (the level and timing move), and
**Non-Gaussianity** (spiky, not bell-curved). _(Those one-line definitions are spoken — they're off
the slide.)_ Read across the four logs and every one of them runs high. And, as the note under those
columns says, they run **higher than the 21 public forecasting benchmarks** in Li et al. 2025.
_(Say that qualitatively; we don't plot the benchmark numbers.)_

Second, the data is **small**. Each log is one short multivariate series: only **307 to 726 daily
steps**, with up to 149 DF relations to forecast. Sepsis is the extreme — 999 cases, 16,000 events
over 459 days — and that sparsity is what drives its later ER failure in the backup.

Put those together — _and click the punchline_ — a **complex signal with very little data** is
exactly the regime where a **trained-from-scratch** model **overfits**, which is what we just saw
XGBoost do.

_Transition out:_ "Small, complex data is exactly where a model you *don't* train might win."

---

## [S-08] Foundation models are the new direction in forecasting

_Transition in:_ "Small, complex, heterogeneous data is exactly where a model you *don't* train
might win."

This is the load-bearing slide, so let me be unambiguous about *what kind* of model we mean —
because the moment people hear "foundation model," half the room assumes we fine-tuned GPT. We did
not.

The current direction in forecasting is **foundation models**. You already know one: a **Large
Language Model** is a foundation model for language — pretrained on web-scale text, it answers "I am
a math ___" with "teacher." A **Time Series Foundation Model (TSFM)** is the exact same recipe for a
different modality: pretrained on **millions of diverse time series**, it reads "…12, 9, 14, 11, ?"
and forecasts "13." **Same recipe** — pretrain at scale, then generalize across tasks — **different
data**. So to be clear: we are forecasting with a TSFM, **not an LLM**.

Two terms I'll define here so the later slides can lean on them. **Zero-shot**: take the pretrained
model and run it on a new log with frozen parameters, no training at all. **Fine-tuning**: keep
training it on your own task data. _(The LoRA and full variants come later, S11 and S15.)_

_Transition out (to S9):_ "Same recipe as LLMs. So why would that help *our* problem?"

---

## [S-09] Foundation models are built not to overfit

_Transition in (from S8):_ "Same recipe as LLMs. So why would that help *our* problem?"

Here's the bet, and it's a one-line argument. Recall the diagnosis: our DF series carry a **complex
signal**, and they're **short** — each log is only a few hundred daily steps, so the training sample
is **tiny**. That combination, complex signal with few samples, is exactly where a model trained
from scratch **overfits**. So instead of fitting a fresh model on each short log, we reach for one
that was **pretrained on millions of diverse series** — it arrives carrying a **broad prior**, a
general sense of what time series look like, rather than starting from a blank slate and over-fitting
the noise. That's the whole bet: *foundation models are built not to overfit.*

One honest caveat, and I want to state it plainly: **no event logs in pretraining, to our knowledge.**
None of these models were trained on process data. So this is a genuine **zero-shot transfer** test —
we're asking whether generic forecasting knowledge carries over to directly-follows series it has
never seen.

_Transition out (to S10):_ "We have a candidate. Here's what we ask of it."

---

## [S-10] Three questions this talk answers

_Transition in (from S9):_ "We have a candidate. Here's what we ask of it."

This talk is built around three questions, and everything that follows is an answer to one of them.

**Question one: can zero-shot TSFMs give better DF time-series forecasts?** And by "better" I mean
**against the strongest PMF baselines** — seasonal-naive and a tuned XGBoost, the top forecaster from
the prior benchmark. So this is purely about **forecast accuracy**, not process-model quality yet.

**Question two: does fine-tuning improve the results further?** Once we have the zero-shot number, is
there headroom — does adapting the model on the task data add anything on top?

**Question three, the open one: does a better forecast give us a better forecasted *process model*?**
This is the one that matters for our field — a lower error on the series is only useful if it
**translates into a better DFG**. We answer it with Entropic Relevance later.

_Transition out (to S11):_ "Three questions. Here's what we point at them."

---

## [S-11] 3 model families, 12 variants, 3 settings

_Transition in (from S10):_ "Three questions. Here's what we point at them."

I won't read the roster — let the timeline carry it. The shape is what matters: we cover **three
model families** — Chronos, MOIRAI, and TimesFM — and each has shipped **several generations**. The
**latest of each**, highlighted in amber, is what the results use. And notice the trend: **newer is
smaller, yet better** — MOIRAI-2.0 is only about 11 million parameters, far smaller than the older
models, and still the strongest of its line.

We test them in **three settings**: **zero-shot**, with twelve variants; **LoRA — Low-Rank
Adaptation**, small trainable adapters; and **full fine-tuning**. Inference is **univariate**
throughout.

_Transition out (to S12):_ "That's the coverage — now exactly how we evaluated it."

---

## [S-12] Each step adds one day, then re-forecasts the next seven

_Transition in (from S11):_ "That's the coverage — now exactly how we evaluated it."

One slide on the protocol, because the design follows directly from how a process actually runs.
**You re-plan as new data arrives each day** — so we use a **stride of one day**. **You care about
the coming week** — so the **horizon is seven days**. And **you never throw history away** — so it's
an **expanding window**, not a sliding one: every step the history grows by a day, and we re-forecast
the next seven.

The part that makes the comparison fair: the **same expanding window and 7-day horizon apply to
every model** — baselines and TSFMs alike. And the two baselines aren't strawmen — **seasonal-naive
and a tuned XGBoost**, the strongest forecasters from the prior PMF benchmark. Inference is
**univariate** throughout.

_Transition out (to S13):_ "With that setup, here are the zero-shot results."

---

## [S-13] The latest zero-shot TSFMs beat both baselines on every log

_Transition in (the locked entry line):_ **"Three families. Twelve model variants. No event logs in
pretraining. Here's what happens when you point them at DF time series."**

First, the framing, so no one thinks we cherry-picked: **we compare against the two strongest
baselines from our prior benchmark — seasonal-naive and a tuned XGBoost — and the full ranking is in
the backup.** These bars show **Mean Absolute Error**, lower is better; gray is the baselines, color
is the latest TSFM of each family.

The result is clean. On all four logs, the **three latest models — Chronos-2, MOIRAI-2.0,
TimesFM-2.5 — beat both baselines**. _(Click.)_ Averaged, that's **−21% MAE versus the best
baseline**. And it's not just the top three: _(click)_ across **all twelve zero-shot variants on four
logs, 92% of results beat the best baseline**, a −15% mean — only four older, smaller models miss,
all on BPI2017.

And note Sepsis: even there MOIRAI-2.0 is down about 28%. Sepsis is **not** a MAE exception — it only
becomes the hard case later, on Entropic Relevance.

_Transition out (to S14):_ "Same data, same window — here's what that win looks like."

---

## [S-14] Zero-shot tracks the drift, stays controlled on sparsity

_Transition in (from S13):_ "Same data, same window — here's what that win looks like."

This is the callback. You've seen these two plots before — the truth, then XGBoost failing on each.
Now watch one click. _(Click.)_

On the **left, drift**. Same plot, same window. MOIRAI-2.0 misses the initial drop, but then it
**adapts** — it comes down toward the truth right where XGBoost stayed stuck high, the boxed
back-half. And it does that with **online context, no retraining**: the model just keeps an
expanding history window at inference.

On the **right, sparsity**, and this is the honest beat. The truth is mostly zero with rare spikes.
XGBoost **hallucinated bursts** of 40 to 71; MOIRAI-2.0 **holds near zero** — about an 89% lower
error, no false bursts. But I'll be straight: it does **not** catch the rare spikes either. On a
near-empty signal, the honest answer is to stay near zero — not to invent a miracle spike detector.

_Transition out (to S15):_ "TSFMs win zero-shot. Natural next question: can we make them better?"

---

## [S-15] Fine-tuning barely helps

_Transition in (from S14):_ "TSFMs win zero-shot. Natural next question — can we make them better?"

So we tried — LoRA and full fine-tuning, every model, every log. Read the slope: each line is a
model's MAE relative to its zero-shot result. Most lines **hug the 1.0 line** — that gray bundle is
fine-tuning **barely moving accuracy**. And a few full-fine-tuning lines shoot **up** in amber:
they **overfit** the small logs — MOIRAI-1.1 up 87% on BPI2019-1, others up 16 to 31% on Sepsis.

_(Click.)_ The tally: across all 36 LoRA and full-fine-tuning runs, **53% landed worse than
zero-shot**. The few real gains are small and dataset-specific.

And it **never comes for free** — this is the cost, and I say it aloud because it's not on the slide.
Zero-shot is **one forward pass** per window. LoRA and full fine-tuning each add a **whole training
stage**, repeated for every model and every log — and in the worst case full fine-tuning nearly
**doubled** the error. The payoff is a coin-flip and sometimes catastrophic. So at PMF data scale,
**skip it.**

_Transition out (to S16):_ "Fine-tuning isn't the win. So does the forecasting win even translate
into a better process model?"

---

## [S-16] Forecasting beats reuse, but better forecasts don't make better models

_Transition in (open with the QUESTION, not the result — this is the talk's lowest-energy moment):_
"But forecasting accuracy isn't the only thing we care about in process mining. Does the forecasting
win translate into a better process *model*?"

To answer that we need a model-level metric. **Entropic Relevance** — ER — is the expected number of
**bits to encode a log trace under the forecasted process model**. Lower is better, and the Truth bar
is the floor.

Two findings. _(Click.)_ **Forecasting beats reuse**: the Training bar — that's reusing the
historical, discovered model, the static snapshot from the start of the talk — towers at 5.8 bits,
while every forecast lands around 2.1 to 2.5. So **forecasting clearly has value**. _(Click.)_ But
**the TSFMs are no better than the baselines** — 2.4-ish against 2.1-ish, differences that small.
A better forecast did **not** give us a better process model. That answers question three.

_(The memorized rebuttal — deliver verbatim, do not skip:)_ **"ER parity means we're not producing
better DFG structures — we're producing DFGs with better edge weights. For tasks where the forecast
itself is the deliverable — capacity planning, drift detection, anomaly baselines — edge-weight
accuracy IS the contribution. For tasks where you need a different process model, ER says the
bottleneck is now the DFG representation, not the forecaster."**

_Transition out (to S17):_ "So the forecasting is solved — which means the open problem has moved
somewhere else."

---

## [S-17] The bottleneck has moved from forecasting accuracy to process-aware representation

_Transition in (from S16):_ "So the forecasting is solved — which means the open problem has moved
somewhere else."

That's the discussion in one line: the bottleneck is no longer the forecaster, it's the
**representation**. So here's where the field goes next — three forward directions.

**Beyond control flow.** A DFG captures only the control-flow skeleton. The richer prize is a
process-aware representation that also carries **resources, decisions, and richer relations** between
activities.

**Smarter fine-tuning.** There's clearly room for **lightweight, incremental tuning for PMF** —
cheaper ways to specialize a foundation model to a process without the cost we saw.

**Larger log corpora.** And ultimately, **bigger, higher-quality logs** — the kind of corpus that
could pretrain a genuinely process-native foundation model.

_Transition out (to S18):_ "But you don't have to wait for any of that to use this today."

---

## [S-18] Off-the-shelf forecasting is strong enough to build on

_Transition in (from S17):_ "So what do we actually walk away with?"

Here's the answer in one line: **off-the-shelf forecasting is now strong enough to build on.** Let me
unpack that, narrow to broad.

**For process mining**, two signals. **Zero-shot is the new default** — the latest TSFMs beat the
strongest baselines with no training at all. And **fine-tuning is marginal** at this data scale, so
you rarely need it.

**For practitioners**, the practical part: these TSFMs are **tiny beside LLMs**. Deploy them locally,
forecast fast on a laptop, even a cheap GPU, with very little compute.

And broadening beyond process mining, the takeaway is simple: **try a time-series foundation model
for your forecasts.**

_(The closing — rehearse cold, deliver verbatim:)_ **"Zero-shot TSFMs are the new PMF default. Four
logs is not a paradigm — but it is a strong enough signal that the right new default is to try a
zero-shot TSFM first. The deeper signal is that time-series forecasting just became cheap enough that
process mining can borrow from it without paying the training tax. What else can we borrow?"**

_Transition out (to S19):_ "And all of this is yours to run."

---

## [S-20] Thank you

_(S19 — the live demo / artifacts slide — is delivered by its own track; this manuscript skips it.)_

"Thank you — and I'm happy to take questions."

Then hold for the **10-minute Q&A**. The backup slides (B1–B10) follow, hidden from the table of
contents; pull them up on demand for the rehearsed hostile-question answers (strongest-baselines
justification, full MAE/RMSE rankings, the LoRA/full-fine-tuning recipe, the Sepsis ER exception, and
the DF-complexity numbers). Deliver those cold — see `SLIDES.md` "Required backup slides" and the Q&A
defense notes.

---

_S-19 is owned by the concurrent demo track. Backups B1–B10 are unchanged this round._
