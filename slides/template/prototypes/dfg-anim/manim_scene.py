"""PROTOTYPE — wipe me when slide 3 is finalised.

Pipeline C: cinematic DFG-evolution via Manim. Reads data.json, morphs edge
weights smoothly t1->t4, recolours to accent blue on the forecast frame and
draws a dashed ghost of the held-out ground truth behind the hero edge.

Render:
  uv run --python .venv manim -qm --media_dir /tmp/manim_media -o dfg \\
      slides/template/prototypes/dfg-anim/manim_scene.py DfgEvolution
  cp /tmp/manim_media/videos/manim_scene/720p30/dfg.mp4 \\
      slides/template/prototypes/dfg-anim/dfg.mp4
"""

import json
from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    UP,
    WHITE,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    Line,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
)

HERE = Path(__file__).parent
DATA = json.loads((HERE / "data.json").read_text())
ACCENT, INK, NEUTRAL, GHOST = "#1d4ed8", "#0f172a", "#475569", "#94a3b8"
MAXF = DATA["max_freq"]
K = 0.08


def pos(n):
    return np.array([(n["x"] - 57) * K, (51 - n["y"]) * K - 0.25, 0.0])


def lw(f):
    return 2.0 + 9.0 * (f / MAXF)


class DfgEvolution(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        nodes = {n["id"]: n for n in DATA["nodes"]}
        frames = DATA["frames"]
        hero = next(e for e in DATA["edges"] if e["hero"])
        f0 = frames[0]

        nmobs = VGroup()
        for n in DATA["nodes"]:
            p = pos(n)
            if n["kind"] == "activity":
                box = RoundedRectangle(
                    corner_radius=0.08,
                    width=1.75,
                    height=0.5,
                    stroke_color="#cbd5e1",
                    stroke_width=1.5,
                    fill_color=WHITE,
                    fill_opacity=1,
                ).move_to(p)
                nmobs.add(VGroup(box, Text(n["label"], font_size=15, color=INK).move_to(p)))
            elif n["kind"] == "start":
                nmobs.add(Dot(p, radius=0.12, color=INK))
            else:
                nmobs.add(
                    Square(
                        side_length=0.22, fill_opacity=1, fill_color=INK, stroke_width=0
                    ).move_to(p)
                )

        emobs = {}
        for e in DATA["edges"]:
            s, t = pos(nodes[e["source"]]), pos(nodes[e["target"]])
            u = (t - s) / np.linalg.norm(t - s)
            ln = Line(
                s + u * 0.30,
                t - u * 0.36,
                stroke_color=(INK if e["hero"] else NEUTRAL),
                stroke_width=lw(f0["weights"][e["id"]]),
            )
            ln.add_tip(tip_length=0.13, tip_width=0.13)
            emobs[e["id"]] = ln
        edge_group = VGroup(*emobs.values())

        hs, ht = pos(nodes[hero["source"]]), pos(nodes[hero["target"]])
        hmid = (hs + ht) / 2 + np.array([0.5, 0.2, 0])

        def hero_text(fr):
            v = fr["weights"][hero["id"]]
            txt = f"≈{v} (truth {fr['truth'][hero['id']]})" if fr["kind"] == "forecast" else f"{v}"
            col = ACCENT if fr["kind"] == "forecast" else NEUTRAL
            return Text(txt, font_size=16, weight=BOLD, color=col).move_to(hmid)

        title = Text(f0["label"], font_size=22, weight=BOLD, color=INK).to_edge(UP, buff=0.3)
        hlabel = hero_text(f0)

        self.play(FadeIn(nmobs), FadeIn(edge_group), FadeIn(title), FadeIn(hlabel), run_time=0.8)
        self.wait(0.6)

        for i in range(1, len(frames)):
            fr = frames[i]
            is_f = fr["kind"] == "forecast"
            anims = []
            for e in DATA["edges"]:
                col = ACCENT if is_f else (INK if e["hero"] else NEUTRAL)
                anims.append(
                    emobs[e["id"]].animate.set_stroke(width=lw(fr["weights"][e["id"]]), color=col)
                )
            anims.append(
                Transform(
                    title,
                    Text(
                        fr["label"], font_size=22, weight=BOLD, color=(ACCENT if is_f else INK)
                    ).to_edge(UP, buff=0.3),
                )
            )
            anims.append(Transform(hlabel, hero_text(fr)))
            if is_f and fr.get("truth"):
                u = (ht - hs) / np.linalg.norm(ht - hs)
                ghost = DashedLine(
                    hs + u * 0.30,
                    ht - u * 0.36,
                    stroke_color=GHOST,
                    stroke_width=lw(fr["truth"][hero["id"]]),
                    dash_length=0.08,
                ).set_opacity(0.6)
                anims.append(Create(ghost))
            self.play(*anims, run_time=1.3)
            self.wait(0.7)
        self.wait(1.2)
