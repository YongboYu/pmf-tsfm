"""PROTOTYPE — wipe me when slide 3 is finalised.

Pipeline B: pre-rendered DFG-evolution gif via matplotlib (same house style as
slides/scripts/make_figures.py). Reads data.json, tweens between the four weekly
frames for a smooth morph, snaps to accent-blue dashed on the forecast frame
with a ghost of the held-out ground truth. Outputs dfg.gif (~10s loop).

Run:  uv run python slides/template/prototypes/dfg-anim/make_dfg_gif.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
ACCENT = "#1d4ed8"
NEUTRAL = "#475569"
INK = "#0f172a"
GHOST = "#94a3b8"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)


def lw(freq, max_f):
    return 1.6 + 14.0 * (freq / max_f)


def build_tweens(frames, steps=14, hold=8):
    """List of (weights, kind, truth, label) — interpolated + held keyframes."""
    seq = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        # hold on keyframe a
        seq += [(a["weights"], a["kind"], a.get("truth"), a["label"])] * hold
        for s in range(1, steps + 1):
            f = s / steps
            w = {k: a["weights"][k] * (1 - f) + b["weights"][k] * f for k in a["weights"]}
            # adopt the forecast styling as we cross into the last segment
            kind = b["kind"] if (b["kind"] == "forecast") else a["kind"]
            seq.append((w, kind, b.get("truth"), b["label"] if f > 0.5 else a["label"]))
    last = frames[-1]
    seq += [(last["weights"], last["kind"], last.get("truth"), last["label"])] * (hold * 2)
    return seq


def main():
    data = json.loads((HERE / "data.json").read_text())
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = data["edges"]
    hero = next(e for e in edges if e["hero"])
    max_f = data["max_freq"]
    tweens = build_tweens(data["frames"])

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.02)

    def draw(idx):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(104, 0)  # invert y (canvas coords grow down)
        ax.set_xticks([])
        ax.set_yticks([])
        weights, kind, truth, label = tweens[idx]
        is_f = kind == "forecast"

        # ghost of held-out truth behind the hero edge (forecast frame only)
        if is_f and truth:
            s, t = nodes[hero["source"]], nodes[hero["target"]]
            ax.annotate(
                "",
                xy=(t["x"], t["y"]),
                xytext=(s["x"], s["y"]),
                arrowprops=dict(
                    arrowstyle="-|>", color=GHOST, lw=lw(truth[hero["id"]], max_f), alpha=0.6
                ),
            )

        for e in edges:
            s, t = nodes[e["source"]], nodes[e["target"]]
            col = ACCENT if is_f else (INK if e["hero"] else NEUTRAL)
            style = "dashed" if is_f else "solid"
            ax.annotate(
                "",
                xy=(t["x"], t["y"]),
                xytext=(s["x"], s["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=lw(weights[e["id"]], max_f),
                    linestyle=style,
                    shrinkA=16,
                    shrinkB=18,
                ),
            )

        # hero weight label
        s, t = nodes[hero["source"]], nodes[hero["target"]]
        hv = weights[hero["id"]]
        txt = f"≈{hv:.0f} (truth {truth[hero['id']]})" if (is_f and truth) else f"{hv:.0f}"
        ax.text(
            (s["x"] + t["x"]) / 2 + 7,
            (s["y"] + t["y"]) / 2 - 1,
            txt,
            color=ACCENT if is_f else NEUTRAL,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )

        # nodes
        for n in data["nodes"]:
            if n["kind"] == "activity":
                w, h = 21, 8.6
                ax.add_patch(
                    FancyBboxPatch(
                        (n["x"] - w / 2, n["y"] - h / 2),
                        w,
                        h,
                        boxstyle="round,pad=0.4,rounding_size=2",
                        fc="white",
                        ec="#cbd5e1",
                        lw=1.0,
                        zorder=5,
                    )
                )
                ax.text(
                    n["x"],
                    n["y"],
                    n["label"],
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=INK,
                    zorder=6,
                )
            else:
                marker = "o" if n["kind"] == "start" else "s"
                ax.plot(n["x"], n["y"], marker, color=INK, markersize=12, zorder=6)

        ax.set_title(label, fontsize=13, color=ACCENT if is_f else INK, fontweight="bold", pad=10)
        return []

    anim = FuncAnimation(fig, draw, frames=len(tweens), interval=80, blit=False)
    out = HERE / "dfg.gif"
    anim.save(out, writer=PillowWriter(fps=12))
    print(f"wrote {out} ({len(tweens)} frames)")


if __name__ == "__main__":
    main()
