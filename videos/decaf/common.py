"""Shared style, voice-over timing and diagram helpers for the DeCAF video."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from manimlib import *

import toy
from narration import SCENES

HERE = Path(__file__).resolve().parent
SOUND_DIR = HERE / "sounds"
TIMING_DIR = HERE / "build" / "timings"

# --------------------------------------------------------------------------
# Palette: one colour per mathematical role, used consistently in every scene
# --------------------------------------------------------------------------
RHO_C = "#58C4DD"       # source noise level rho
SIG_C = "#F4D345"       # target noise level sigma
TAU_C = "#C55F73"       # a third level tau
TEACHER_C = "#FF862F"   # teacher denoiser D and teacher velocity v
STUDENT_C = "#5CD0B3"   # student D_theta, u_theta, D-hat
NOISE_C = "#A0A0A0"     # epsilon
REWARD_C = "#E35DC5"    # rewards / potentials
ALIGN_C = "#83C167"     # rigid alignment
WARN_C = "#FC6255"      # the problematic chain-rule factor, failures
TRAJ_C = "#3F7DA8"      # background ODE trajectories
FONT = "CMU Serif"

# Single-letter roles are matched with regexes so they don't hit letters inside
# other commands (the v in \varepsilon, the D in D_\theta).
TEACHER_D = re.compile(r"(?<![a-zA-Z\\])D(?![_a-zA-Z])")
V_KEY = re.compile(r"(?<![a-zA-Z\\])v(?![a-zA-Z])")

T2C = {
    R"\rho": RHO_C,
    R"\sigma": SIG_C,
    R"\tau": TAU_C,
    R"D_\theta": STUDENT_C,
    R"u_\theta": STUDENT_C,
    R"D_{\mathrm{T}}": TEACHER_C,
    R"\varepsilon": NOISE_C,
}


def tex(s, font_size=40, t2c=None, isolate=(), **kw):
    """Tex with the shared colour map.  Regex keys in `t2c` are isolated and
    coloured after construction (manimgl only accepts string keys in t2c)."""
    cmap = dict(T2C)
    patterns = {}
    for k, v in (t2c or {}).items():
        (cmap if isinstance(k, str) else patterns)[k] = v
    cmap = {k: v for k, v in cmap.items() if k in s}
    mob = Tex(s, font_size=font_size, t2c=cmap, isolate=[*isolate, *patterns], **kw)
    for pat, col in patterns.items():
        mob.build_parts_from_indices_lists(mob.get_submob_indices_lists_by_selector(pat)).set_color(col)
    return mob


def text(s, font_size=30, color=WHITE, **kw):
    return Text(s, font=FONT, font_size=font_size, color=color, **kw)


def with_bg(mob, opacity=0.8, buff=0.06):
    """A label on a black backing so it stays legible over diagrams."""
    rect = BackgroundRectangle(mob, fill_opacity=opacity, buff=buff)
    return VGroup(rect, mob)


def section_label(s, number=None):
    label = text(s, font_size=30, color=GREY_A)
    if number is not None:
        num = text(f"{number}.", font_size=30, color=YELLOW)
        label = VGroup(num, label).arrange(RIGHT, buff=0.15)
    label.to_corner(UL, buff=0.35)
    underline = Line(label.get_left(), label.get_right(), stroke_width=1.5, color=GREY_C)
    underline.next_to(label, DOWN, buff=0.08)
    return VGroup(label, underline)


# --------------------------------------------------------------------------
# Voice-over: one audio clip per narration beat, with estimated phrase times
# --------------------------------------------------------------------------
_DUR = json.loads((SOUND_DIR / "durations.json").read_text())
_TEXT = {f"{scene}__{key}": t for scene, beats in SCENES.items() for key, t in beats}

# Per-character speaking-time model fitted on the synthesized clips
# (letters, spaces, and pauses at punctuation); used only to place animations
# near the phrase they illustrate.
_W_LETTER, _W_SPACE = 0.0588, 0.049
_W_PUNCT = {".": 0.199, ",": 0.229, ":": 0.361, ";": 0.361, "?": 0.199}


def _char_weights(s):
    w = np.zeros(len(s))
    for i, ch in enumerate(s):
        if ch.isalnum():
            w[i] = _W_LETTER
        elif ch == " ":
            w[i] = _W_SPACE
        elif ch in _W_PUNCT:
            w[i] = _W_PUNCT[ch]
    return w


class Beat:
    def __init__(self, scene, key, pad=0.5):
        self.scene = scene
        self.name = f"{type(scene).__name__}__{key}"
        self.text = _TEXT[self.name]
        self.duration = _DUR[self.name]["duration"]
        self.path = SOUND_DIR / f"{self.name}.wav"
        self.pad = pad
        w = _char_weights(self.text)
        self._cum = np.concatenate([[0], np.cumsum(w)]) * (self.duration / w.sum())

    def __enter__(self):
        self.t0 = self.scene.time
        self.scene.add_sound(str(self.path))
        self.scene.beat_log.append(
            dict(name=self.name, start=self.t0, duration=self.duration, text=self.text)
        )
        return self

    def at(self, fragment):
        idx = self.text.find(fragment)
        if idx < 0:
            raise ValueError(f"{fragment!r} not in beat {self.name}")
        return float(self._cum[idx])

    def elapsed(self):
        return self.scene.time - self.t0

    def remaining(self):
        return self.duration - self.elapsed()

    def until(self, fragment_or_time, extra=0.0):
        """Wait until the narrator reaches `fragment` (or a time offset)."""
        target = self.at(fragment_or_time) if isinstance(fragment_or_time, str) else fragment_or_time
        dt = target + extra - self.elapsed()
        if dt > 1 / 30:
            self.scene.wait(dt)

    def span(self, start, end):
        """Seconds between two fragments (for sizing an animation's run_time)."""
        t0 = self.at(start) if isinstance(start, str) else start
        t1 = self.at(end) if isinstance(end, str) else end
        return max(t1 - t0, 0.3)

    def __exit__(self, *exc):
        rem = self.remaining()
        if rem > 1 / 30:
            self.scene.wait(rem)
        if self.pad > 0:
            self.scene.wait(self.pad)
        return False


class DecafScene(Scene):
    default_pad = 0.5

    def setup(self):
        super().setup()
        self.beat_log = []

    def beat(self, key, pad=None):
        return Beat(self, key, pad=self.default_pad if pad is None else pad)

    def tear_down(self):
        TIMING_DIR.mkdir(parents=True, exist_ok=True)
        out = dict(scene=type(self).__name__, total=self.time, beats=self.beat_log)
        (TIMING_DIR / f"{type(self).__name__}.json").write_text(json.dumps(out, indent=1))
        super().tear_down()

    def clear_all(self, run_time=1.0, keep=(), **kw):
        mobs = [m for m in self.mobjects if m is not self.camera.frame and m not in keep]
        if mobs:
            self.play(*map(FadeOut, mobs), run_time=run_time, **kw)


# --------------------------------------------------------------------------
# The (sigma, x) plane used for every 1D picture
# --------------------------------------------------------------------------
class SigmaPlane(VGroup):
    """Noise level on the horizontal axis (sigma_max on the left, 0 on the
    right), a 1D stand-in for the structure on the vertical axis."""

    def __init__(
        self,
        width=8.5,
        height=5.6,
        x_range=(-3.0, 3.0),
        sigma_max=toy.SIGMA_MAX,
        n_traj=23,
        density_width=0.9,
        show_densities=True,
        **kw,
    ):
        super().__init__(**kw)
        self.w, self.h = width, height
        self.x_range = x_range
        self.sigma_max = sigma_max
        self.origin_shift = np.zeros(3)

        left, right = -width / 2, width / 2
        bottom, top = -height / 2, height / 2
        self.clean_axis = Line([right, bottom, 0], [right, top, 0], stroke_width=2.5, color=GREY_B)
        self.noise_axis = Line([left, bottom, 0], [left, top, 0], stroke_width=1.5, color=GREY_D)
        self.sigma_axis = Arrow(
            [right, bottom - 0.25, 0], [left - 0.3, bottom - 0.25, 0],
            stroke_width=2, buff=0, color=GREY_B,
        )
        self.sigma_axis_label = tex(R"\text{noise level } \sigma", font_size=28)
        self.sigma_axis_label.next_to(self.sigma_axis, DOWN, buff=0.12)
        self.zero_label = tex("0", font_size=26, t2c={}).next_to(self.sigma_axis.get_start(), DOWN, buff=0.12)
        self.smax_label = tex(R"\sigma_{\max}", font_size=26).next_to(
            [left + 0.15, bottom - 0.25, 0], DOWN, buff=0.12
        )
        self.axes = VGroup(self.noise_axis, self.clean_axis, self.sigma_axis,
                           self.sigma_axis_label, self.zero_label, self.smax_label)
        self.add(self.axes)

        # Densities at the two ends
        xs = np.linspace(*x_range, 300)
        self.left_density = self._density_shape(xs, sigma_max, left, -1, density_width, GREY_C, 0.12)
        self.right_density = self._density_shape(xs, 0.0, right, +1, density_width, GREY_B, 0.22)
        self.densities = VGroup(self.left_density, self.right_density)
        if show_densities:
            self.add(self.densities)

        # Background trajectories from quantiles of p_{sigma_max}
        starts = self._start_quantiles(n_traj)
        self.trajectories = VGroup(*[self.trajectory_curve(x0) for x0 in starts])
        self.trajectories.set_stroke(TRAJ_C, width=1.6, opacity=0.65)
        self.add(self.trajectories)

    # ---- coordinates -----------------------------------------------------
    def c2p(self, sigma, x):
        u = self.w / 2 - (sigma / self.sigma_max) * self.w
        lo, hi = self.x_range
        v = -self.h / 2 + (x - lo) / (hi - lo) * self.h
        return np.array([u, v, 0.0]) + self.get_center_shift()

    def get_center_shift(self):
        # The group may have been moved; track via the clean axis position.
        ref = np.array([self.w / 2, -self.h / 2, 0.0])
        return self.clean_axis.get_start() - ref

    def x_to_y(self, x):
        return self.c2p(0, x)[1]

    # ---- building blocks -------------------------------------------------
    def _density_shape(self, xs, sigma, base_u, direction, width, color, fill_opacity=0.2):
        p = toy.density(xs, sigma)
        p = p / p.max() * width
        lo, hi = self.x_range
        pts = [[base_u + direction * pi, -self.h / 2 + (x - lo) / (hi - lo) * self.h, 0]
               for x, pi in zip(xs, p)]
        pts = [[base_u, pts[0][1], 0]] + pts + [[base_u, pts[-1][1], 0]]
        shape = VMobject().set_points_as_corners(pts)
        shape.set_fill(color, fill_opacity).set_stroke(color, 1.5, 0.8)
        return shape

    def _start_quantiles(self, n):
        xs = np.linspace(-8, 8, 4001)
        cdf = np.cumsum(toy.density(xs, self.sigma_max))
        cdf /= cdf[-1]
        qs = np.linspace(0.04, 0.96, n)
        starts = np.interp(qs, cdf, xs)
        lo, hi = self.x_range
        return [s for s in starts if lo + 0.1 < s < hi - 0.1]

    def trajectory_curve(self, x_start, rho=None, sigma_end=0.0, n=160):
        rho = self.sigma_max if rho is None else rho
        sig, xs = toy.trajectory(x_start, rho=rho, n=n, sigma_end=sigma_end)
        curve = VMobject()
        curve.set_points_as_corners([self.c2p(s, x) for s, x in zip(sig, xs)])
        return curve

    def dot(self, sigma, x, color=WHITE, radius=0.07):
        return Dot(self.c2p(sigma, x), radius=radius, fill_color=color).set_stroke(BLACK, 1)

    def vline(self, sigma, color=GREY_B, label=None, label_tex=None, dashed=True):
        top = self.c2p(sigma, self.x_range[1])
        bot = self.c2p(sigma, self.x_range[0])
        line = DashedLine(bot, top, dash_length=0.08) if dashed else Line(bot, top)
        line.set_stroke(color, 1.5, 0.7)
        group = VGroup(line)
        if label_tex is not None:
            lab = tex(label_tex, font_size=30).next_to(bot, DOWN, buff=0.35)
            lab.set_color(color)
            group.add(lab)
        return group

    def line_through(self, p_sigma, p_x, q_sigma, q_x, extend_to_sigma=0.0):
        """Straight line in the (sigma, x) plane from p through q, extended to
        the level `extend_to_sigma` (the clean axis by default)."""
        slope = (q_x - p_x) / (q_sigma - p_sigma)
        x_end = p_x + (extend_to_sigma - p_sigma) * slope
        return Line(self.c2p(p_sigma, p_x), self.c2p(extend_to_sigma, x_end)), x_end


# --------------------------------------------------------------------------
# A small 2D "molecule" for the alignment illustrations
# --------------------------------------------------------------------------
def molecule_coords():
    """A short chain (protein-like) plus a six-ring ligand, in 2D."""
    t = np.linspace(0, 1, 14)
    chain = np.stack([2.6 * t - 1.3, 0.55 * np.sin(2 * np.pi * 1.2 * t) - 0.6], axis=1)
    ang = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    ring = np.stack([0.38 * np.cos(ang) + 0.25, 0.38 * np.sin(ang) + 0.55], axis=1)
    tail = np.array([[0.25 + 0.38 * np.cos(np.pi / 6) + 0.35, 0.55 + 0.38 * np.sin(np.pi / 6) + 0.2]])
    coords = np.concatenate([chain, ring, tail])
    kinds = ["p"] * len(chain) + ["l"] * (len(ring) + 1)
    bonds = [(i, i + 1) for i in range(len(chain) - 1)]
    r0 = len(chain)
    bonds += [(r0 + i, r0 + (i + 1) % 6) for i in range(6)]
    bonds += [(r0 + 1, r0 + 6)]
    return coords, kinds, bonds


def kabsch_2d(P, Q, w=None):
    """Rotation R and translation t minimizing sum_i w_i |R Q_i + t - P_i|^2."""
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    cp, cq = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((Q - cq) * w[:, None]).T @ (P - cp)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, d])
    R = Vt.T @ D @ U.T
    t = cp - R @ cq
    return R, t


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def molecule_mobject(coords, kinds, bonds, scale=1.0, center=ORIGIN, protein_color=GREY_B,
                     ligand_color=TEACHER_C, atom_radius=0.075, bond_width=2.5, opacity=1.0):
    pts = [np.array([x, y, 0]) * scale + center for x, y in coords]
    bond_lines = VGroup(*[Line(pts[i], pts[j]) for i, j in bonds])
    bond_lines.set_stroke(GREY_C, bond_width, opacity * 0.8)
    atoms = VGroup(*[
        Dot(p, radius=atom_radius * (1.15 if k == "l" else 1.0),
            fill_color=(ligand_color if k == "l" else protein_color), fill_opacity=opacity)
        for p, k in zip(pts, kinds)
    ])
    return VGroup(bond_lines, atoms)
