"""Scenes 1-5: intro, the EDM teacher, flow maps, sigma-space, denoiser map.

Render (from this folder):  manimgl scenes_a.py S02_Teacher -w --hd
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import *  # noqa: E402,F401

PLANE_KW = dict(width=8.4, height=4.9, x_range=(-2.7, 2.7))
PLANE_SHIFT = DOWN * 0.62


def make_plane(**kw):
    plane = SigmaPlane(**{**PLANE_KW, **kw})
    plane.shift(PLANE_SHIFT)
    return plane


def jump_arrow(p, q, angle=-0.55, color=STUDENT_C, width=3.5, tip=None):
    """A flow-map jump: an arc from p to q with a small tip sized to the arc."""
    arrow = ArcBetweenPoints(p, q, angle=angle)
    arrow.set_stroke(color, width)
    size = tip if tip is not None else float(np.clip(0.16 * np.linalg.norm(np.array(q) - np.array(p)), 0.09, 0.2))
    arrow.add_tip(width=size, length=size)
    arrow.set_fill(opacity=0)
    for t in arrow.get_tips():
        t.set_fill(color, 1).set_stroke(color, 0)
    return arrow


def complex_coords(seed=3):
    """Stylized protein chain wrapped around a pocket, plus a ligand."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(np.radians(40), np.radians(330), 36)
    r = 1.75 + 0.18 * np.sin(5 * ang) + 0.05 * rng.normal(size=ang.size)
    chain = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    ring_ang = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    ring = np.stack([0.34 * np.cos(ring_ang) + 0.25, 0.34 * np.sin(ring_ang) + 0.05], axis=1)
    tail = np.array([[0.25 + 0.34 + 0.3, 0.25], [0.25 + 0.34 + 0.62, 0.05]])
    coords = np.concatenate([chain, ring, tail])
    kinds = ["p"] * len(chain) + ["l"] * (len(ring) + len(tail))
    bonds = [(i, i + 1) for i in range(len(chain) - 1)]
    r0 = len(chain)
    bonds += [(r0 + i, r0 + (i + 1) % 6) for i in range(6)]
    bonds += [(r0 + 0, r0 + 6), (r0 + 6, r0 + 7)]
    return coords, kinds, bonds


class NoisyComplex(VGroup):
    """The stylized complex at noise level s: x0 + s * eps (the exact PF-ODE
    trajectory when the data distribution is a single structure)."""

    def __init__(self, center=ORIGIN, scale=1.15, seed=11, **kw):
        super().__init__(**kw)
        self.coords, self.kinds, self.bonds = complex_coords()
        self.eps = np.random.default_rng(seed).normal(size=self.coords.shape)
        self.center, self.scale_ = np.array(center), scale
        self.level = ValueTracker(0.0)
        self.add(self._build(0.0))
        self.add_updater(lambda m: m._refresh())

    def _positions(self, s):
        xy = (self.coords + s * self.eps) * self.scale_
        return [np.array([x, y, 0]) + self.center for x, y in xy]

    def _build(self, s):
        pts = self._positions(s)
        bonds = VGroup(*[Line(pts[i], pts[j]) for i, j in self.bonds])
        bonds.set_stroke(GREY_C, 2.2, 0.75)
        atoms = VGroup(*[
            Dot(p, radius=0.085 if k == "l" else 0.075,
                fill_color=TEACHER_C if k == "l" else BLUE_B)
            for p, k in zip(pts, self.kinds)
        ])
        return VGroup(bonds, atoms)

    def _refresh(self):
        pts = self._positions(self.level.get_value())
        bonds, atoms = self[0]
        for line, (i, j) in zip(bonds, self.bonds):
            line.put_start_and_end_on(pts[i], pts[j])
        for dot, p in zip(atoms, pts):
            dot.move_to(p)


# ==========================================================================
class S01_Intro(DecafScene):
    def construct(self):
        # ---- title ------------------------------------------------------
        with self.beat("title") as b:
            title = text("Few-step Cofolding with All-Atom Flow Maps", font_size=50)
            authors = text(
                "G. Scarpellini, R. Shprints, P. Holderrieth, J. Nam, P. Murugan,\n"
                "R. Gómez-Bombarelli, T. Jaakkola, M. Al-Shedivat, N. M. Boffi, A. J. Bose",
                font_size=24, color=GREY_B,
            )
            affil = text("Genesis Molecular AI · MIT · CMU · Imperial College London · Mila",
                         font_size=24, color=GREY_B)
            arxiv = text("arXiv:2606.08375 (v2)", font_size=24, color=GREY_C)
            header = VGroup(title, authors, affil, arxiv).arrange(DOWN, buff=0.35)
            header.move_to(UP * 0.6)
            self.play(Write(title, run_time=2.5))
            self.play(LaggedStartMap(FadeIn, VGroup(authors, affil, arxiv), shift=0.2 * UP,
                                     lag_ratio=0.35, run_time=2.5))
            b.until("The method is called")
            decaf = Text("DeCAF", font=FONT, font_size=110)
            decaf["De"].set_color(STUDENT_C)
            decaf["C"].set_color(TEACHER_C)
            decaf["A"].set_color(TEACHER_C)
            decaf["F"].set_color(SIG_C)
            expand = Text("Denoiser  Cofolding  All-atom  Flow map", font=FONT, font_size=34)
            for w, c in [("Denoiser", STUDENT_C), ("Cofolding", TEACHER_C), ("All-atom", TEACHER_C),
                         ("Flow map", SIG_C)]:
                expand[w].set_color(c)
            VGroup(decaf, expand).arrange(DOWN, buff=0.4).move_to(DOWN * 2.3)
            self.play(header.animate.shift(UP * 0.9).scale(0.85),
                      FadeIn(decaf, scale=1.3), run_time=1.2)
            self.play(Write(expand), run_time=1.5)

        # ---- cost of diffusion sampling -----------------------------------
        with self.beat("cost") as b:
            self.play(FadeOut(VGroup(header, decaf, expand)), run_time=0.8)
            cx = NoisyComplex(center=LEFT * 2.9 + DOWN * 0.1)
            cx.level.set_value(1.25)
            prot_label = text("protein", font_size=28, color=BLUE_B).next_to(cx, UP, buff=0.2)
            lig_label = text("ligand", font_size=28, color=TEACHER_C)
            self.play(FadeIn(cx), run_time=0.8)
            b.until("Their last stage")
            counter_label = text("diffusion-module calls:", font_size=30)
            counter = Integer(0, font_size=40, color=YELLOW)
            counter_group = VGroup(counter_label, counter).arrange(RIGHT, buff=0.25)
            counter_group.move_to(RIGHT * 3.2 + UP * 1.2)
            note = VGroup(
                text("AlphaFold 3, Boltz, Chai-1:", font_size=26, color=GREY_B),
                text("≈ 200 denoising steps per sample", font_size=26, color=GREY_B),
            ).arrange(DOWN, buff=0.15).next_to(counter_group, DOWN, buff=0.5)
            self.play(FadeIn(counter_group), FadeIn(note, shift=0.2 * UP))
            steps = ValueTracker(0)
            counter.add_updater(lambda m: m.set_value(int(round(steps.get_value()))))
            sig200 = toy.karras_sigmas(200, sigma_max=1.25, sigma_min=0.002)[:-1]
            self.add(cx.level)
            cx.level.add_updater(
                lambda m: m.set_value(
                    float(np.interp(steps.get_value(), np.arange(200), sig200))
                    if steps.get_value() < 199.5 else 0.0))
            run = max(b.remaining() - 1.5, 4.0)
            self.play(steps.animate.set_value(200), run_time=run, rate_func=linear)
            cx.level.clear_updaters()
            counter.clear_updaters()
            lig_label.next_to(cx, DOWN, buff=0.25)
            self.play(FadeIn(prot_label), FadeIn(lig_label), run_time=0.6)

        # ---- flow map: a few jumps --------------------------------------------
        with self.beat("jump") as b:
            self.play(FadeOut(VGroup(prot_label, lig_label)),
                      cx.level.animate.set_value(1.25), run_time=0.8)
            counter.set_value(0)
            new_label = text("flow-map calls:", font_size=30)
            new_label.move_to(counter_label, aligned_edge=RIGHT)
            self.play(FadeTransform(counter_label, new_label), run_time=0.6)
            b.until("jumps along it")
            levels = toy.karras_sigmas(5, sigma_max=1.25, sigma_min=0.002)[1:]
            for i, lev in enumerate(levels):
                self.play(cx.level.animate.set_value(lev), counter.animate.set_value(i + 1),
                          run_time=0.35, rate_func=rush_into)
                self.wait(0.35)
            b.until("The same network")
            options = VGroup(
                text("1 jump  ·  5 jumps  ·  10 jumps", font_size=30, color=GREY_A),
                text("same network", font_size=30, color=STUDENT_C),
            ).arrange(DOWN, buff=0.15).next_to(note, DOWN, buff=0.6)
            self.play(Write(options), run_time=1.5)

        # ---- the plan -------------------------------------------------------
        with self.beat("plan") as b:
            self.play(FadeOut(VGroup(cx, new_label, counter, note, options)), run_time=0.8)
            top = VGroup(
                text("Flow maps: not new.", font_size=34, color=GREY_B),
                text("New here: making them work for an all-atom, EDM-style teacher.",
                     font_size=34),
            ).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.6)
            self.play(FadeIn(top[0]), run_time=0.8)
            b.until("The contribution")
            self.play(FadeIn(top[1]), run_time=0.8)
            cards = VGroup(*[self.card(n, s) for n, s in [
                (1, "Index the flow map by noise level σ, not by time t"),
                (2, "Parameterize it as a denoiser: the loss compares structures,\n"
                    "so it can be rigidly aligned like the teacher's"),
                (3, "Use the map's cheap look-ahead to guide inference-time search"),
            ]]).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(top, DOWN, buff=0.7)
            for card, cue in zip(cards, ["First, index", "Second, param", "Third, use"]):
                b.until(cue)
                self.play(FadeIn(card, shift=0.3 * RIGHT), run_time=0.9)

    @staticmethod
    def card(n, s):
        num = Text(str(n), font=FONT, font_size=40, color=YELLOW)
        circ = Circle(radius=0.32).set_stroke(YELLOW, 2).move_to(num)
        body = Text(s, font=FONT, font_size=32, alignment="LEFT")
        body.next_to(circ, RIGHT, buff=0.35)
        return VGroup(VGroup(circ, num), body)


# ==========================================================================
class S02_Teacher(DecafScene):
    def construct(self):
        label = section_label("The teacher: an EDM-style diffusion model")
        with self.beat("noising") as b:
            self.play(FadeIn(label), run_time=0.8)
            line1 = VGroup(
                tex(R"x \in \mathbb{R}^{3N}", font_size=44),
                text("all atom coordinates, stacked", font_size=30, color=GREY_B),
            ).arrange(RIGHT, buff=0.5)
            line2 = tex(R"x_{\sigma} = x_0 + \sigma\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)",
                        font_size=48)
            VGroup(line1, line2).arrange(DOWN, buff=0.6).move_to(UP * 1.6)
            b.until("Stack all")
            self.play(Write(line1[0]), FadeIn(line1[1]), run_time=1.2)
            b.until("The noising process")
            self.play(Write(line2), run_time=2)
            b.until("In Boltz")
            scale = self.log_scale_bar().next_to(line2, DOWN, buff=1.1)
            self.play(ShowCreation(scale[0]), FadeIn(scale[1]), run_time=1.2)
            self.play(LaggedStartMap(FadeIn, scale[2:], lag_ratio=0.3), run_time=2)

        with self.beat("denoiser") as b:
            den_l = tex(R"D(x;\sigma)", font_size=48, t2c={TEACHER_D: TEACHER_C})
            den_r = tex(R"\approx\  \mathbb{E}\big[\,x_0 \mid x_{\sigma} = x\,\big]", font_size=48)
            den = VGroup(den_l, den_r).arrange(RIGHT, buff=0.25)
            den.next_to(line2, DOWN, buff=0.7)
            self.play(FadeOut(scale), run_time=0.6)
            self.play(Write(den_l), run_time=1)
            b.until("Ideally")
            self.play(Write(den_r), run_time=1.5)
            post = text("posterior mean", font_size=28, color=TEACHER_C)
            post.next_to(den_r, DOWN, buff=0.3)
            self.play(FadeIn(post, shift=0.1 * UP))

        with self.beat("ode") as b:
            ode_l = tex(R"\frac{dx}{d\sigma} \ =\  \frac{x - D(x;\sigma)}{\sigma}",
                        font_size=52, t2c={TEACHER_D: TEACHER_C})
            ode_r = tex(R"=:\  v(x,\sigma)", font_size=52, t2c={V_KEY: TEACHER_C})
            ode = VGroup(ode_l, ode_r).arrange(RIGHT, buff=0.25)
            ode.next_to(den, DOWN, buff=0.9)
            b.until("Written directly")
            self.play(FadeOut(post), Write(ode_l), run_time=2)
            b.until("We'll call")
            self.play(Write(ode_r), run_time=1)
            vel = text("σ-velocity", font_size=30, color=TEACHER_C).next_to(ode_r, DOWN, buff=0.3)
            self.play(FadeIn(vel, shift=0.1 * UP))

        with self.beat("picture") as b:
            self.play(FadeOut(VGroup(line1, line2, den, vel)), FadeOut(label),
                      ode.animate.scale(0.62).to_corner(UR, buff=0.4), run_time=1.2)
            plane = make_plane()
            self.play(ShowCreation(plane.axes), run_time=1.5)
            b.until("pure noise")
            noise_lab = text("pure noise", font_size=28, color=GREY_B).next_to(
                plane.left_density, UP, buff=0.15)
            data_lab = text("data", font_size=28, color=GREY_B).next_to(
                plane.right_density, UP, buff=0.15)
            self.play(FadeIn(plane.left_density), FadeIn(noise_lab), run_time=1)
            self.play(FadeIn(plane.right_density), FadeIn(data_lab), run_time=1)
            b.until("The vertical axis")
            ylab = text("structure (1D stand-in)", font_size=24, color=GREY_B)
            ylab.rotate(PI / 2).next_to(plane.left_density, LEFT, buff=0.15)
            self.play(FadeIn(ylab), run_time=0.8)
            b.until("think of two")
            poses = VGroup(*[
                text(s, font_size=24, color=GREY_A).next_to(plane.c2p(0, y), RIGHT, buff=1.0)
                for s, y in [("pose A", 1.2), ("pose B", -1.2)]
            ])
            self.play(FadeIn(poses), run_time=0.8)
            b.until("Every curve")
            self.play(ShowCreation(plane.trajectories, lag_ratio=0.04), run_time=4.5)
        self.plane, self.ode = plane, ode

        with self.beat("euler") as b:
            x0 = 0.1
            exact = plane.trajectory_curve(x0).set_stroke(WHITE, 3)
            start = plane.dot(plane.sigma_max, x0)
            self.play(plane.trajectories.animate.set_stroke(opacity=0.3),
                      ShowCreation(exact), FadeIn(start), run_time=1.5)
            b.until("With many steps")
            many = toy.karras_sigmas(40)
            xs_many = toy.euler_path(x0, many)
            path_many = VMobject().set_points_as_corners(
                [plane.c2p(s, x) for s, x in zip(many, xs_many)]).set_stroke(ALIGN_C, 2.5)
            dots_many = VGroup(*[plane.dot(s, x, ALIGN_C, radius=0.035)
                                 for s, x in zip(many, xs_many)])
            lab_many = text("40 steps", font_size=26, color=ALIGN_C)
            lab_many.next_to(plane.c2p(0.25, xs_many[-5]), UP, buff=0.35)
            self.play(ShowCreation(path_many), LaggedStartMap(FadeIn, dots_many, lag_ratio=0.05),
                      FadeIn(lab_many), run_time=2)
            b.until("With only five")
            few = toy.karras_sigmas(5)
            xs_few = toy.euler_path(x0, few)
            segs = VGroup(*[Line(plane.c2p(s0, a), plane.c2p(s1, c))
                            for s0, s1, a, c in zip(few[:-1], few[1:], xs_few[:-1], xs_few[1:])])
            segs.set_stroke(WARN_C, 3.5)
            dots_few = VGroup(*[plane.dot(s, x, WARN_C, radius=0.06) for s, x in zip(few, xs_few)])
            lab_few = text("5 steps = 5 network calls", font_size=26, color=WARN_C)
            lab_few.next_to(plane.c2p(0.9, xs_few[1]), DOWN, buff=0.45)
            self.play(FadeIn(lab_few), FadeIn(dots_few[0]))
            for seg, d in zip(segs, dots_few[1:]):
                self.play(ShowCreation(seg), FadeIn(d, scale=0.5), run_time=0.7)
            b.until("between the modes")
            end = plane.c2p(0, xs_few[-1])
            miss = with_bg(text("lands between the modes:\nno data here", font_size=26, color=WARN_C))
            miss.next_to(end, DOWN + LEFT, buff=0.35)
            self.play(Flash(end, color=WARN_C), FadeIn(miss), run_time=1)

    @staticmethod
    def log_scale_bar():
        width = 9.0
        lo, hi = -3, 4
        axis = Line(LEFT * width / 2, RIGHT * width / 2).set_stroke(GREY_B, 2)

        def pos(v):
            return axis.get_left() + (np.log10(v) - lo) / (hi - lo) * width * RIGHT

        ticks = VGroup()
        for k in range(lo, hi + 1):
            t = Line(UP * 0.08, DOWN * 0.08).move_to(pos(10.0**k)).set_stroke(GREY_B, 2)
            lab = tex(rf"10^{{{k}}}", font_size=24).next_to(t, DOWN, buff=0.12).set_color(GREY_B)
            ticks.add(VGroup(t, lab))
        unit = text("Å", font_size=26, color=GREY_B).next_to(axis, RIGHT, buff=0.3)
        smin = VGroup(
            Dot(pos(0.0064), radius=0.08, fill_color=SIG_C),
            tex(R"\sigma_{\min} = 0.0064\,\text{\AA}", font_size=30),
        )
        smin[1].next_to(smin[0], UP, buff=0.2)
        smax = VGroup(
            Dot(pos(2560), radius=0.08, fill_color=SIG_C),
            tex(R"\sigma_{\max} = 2560\,\text{\AA}", font_size=30),
        )
        smax[1].next_to(smax[0], UP, buff=0.2)
        span = text("almost 6 orders of magnitude", font_size=26, color=GREY_B)
        span.next_to(axis, DOWN, buff=0.7)
        return VGroup(axis, VGroup(ticks, unit), smin, smax, span)


# ==========================================================================
class S03_FlowMaps(DecafScene):
    RHO, XR, SIG, TAU = 1.0, 0.9, 0.38, 0.1
    ROWS = [3.55, 2.95, 2.38]  # y of the three text rows above the plane

    def row(self, mob, i, x=1.2):
        return mob.move_to([x, self.ROWS[i], 0])

    def construct(self):
        plane = make_plane()
        plane.trajectories.set_stroke(opacity=0.35)
        label = section_label("Flow maps")
        rho, xr, sig = self.RHO, self.XR, self.SIG
        xs = float(toy.flow(xr, rho, sig))

        with self.beat("define") as b:
            self.play(FadeIn(label), FadeIn(plane), run_time=1.2)
            traj = self.full_trajectory(plane, xr, rho)
            p_start = plane.dot(rho, xr, RHO_C, radius=0.09)
            lab_start = tex(R"x_{\rho}", font_size=34).next_to(p_start, DOWN + LEFT, buff=0.05)
            v_rho = plane.vline(rho, RHO_C, label_tex=R"\rho")
            v_sig = plane.vline(sig, SIG_C, label_tex=R"\sigma")
            self.play(ShowCreation(traj), FadeIn(p_start), FadeIn(lab_start),
                      FadeIn(v_rho), run_time=1.5)
            b.until("and returns")
            p_end = plane.dot(sig, xs, SIG_C, radius=0.09)
            arc = jump_arrow(p_start.get_center(), p_end.get_center(), angle=-0.9)
            self.play(FadeIn(v_sig), ShowCreation(arc), run_time=1.2)
            lab_end = tex(R"X_{\rho,\sigma}(x_{\rho})", font_size=34).next_to(p_end, DOWN + RIGHT, buff=0.08)
            self.play(FadeIn(p_end, scale=0.5), Write(lab_end), run_time=1)
            b.until("It is the solution")
            eq = tex(R"X_{\rho,\sigma}(x_{\rho}) = x_{\sigma} \quad \text{(solution operator of the ODE)}",
                     font_size=36)
            self.row(eq, 0)
            self.play(Write(eq), run_time=1.5)

        with self.beat("props") as b:
            b.until("Mapping a level")
            ident = tex(R"X_{\rho,\rho}(x) = x", font_size=36)
            comp = tex(R"X_{\sigma,\tau}\circ X_{\rho,\sigma} = X_{\rho,\tau}", font_size=36)
            props = VGroup(ident, comp).arrange(RIGHT, buff=1.2)
            self.row(props, 1)
            self.play(Write(ident), run_time=1.2)
            b.until("And maps compose")
            tau = self.TAU
            xt = float(toy.flow(xr, rho, tau))
            v_tau = plane.vline(tau, TAU_C, label_tex=R"\tau")
            p_tau = plane.dot(tau, xt, TAU_C, radius=0.09)
            arc2 = jump_arrow(p_end.get_center(), p_tau.get_center(), angle=-1.2, color=STUDENT_C)
            arc3 = jump_arrow(p_start.get_center(), p_tau.get_center(), angle=-0.75, color=TAU_C)
            self.play(Write(comp), FadeIn(v_tau), run_time=1.2)
            self.play(ShowCreation(arc2), FadeIn(p_tau), run_time=1)
            self.play(ShowCreation(arc3), run_time=1.2)
            b.until("A consistency model")
            cm_arcs = VGroup()
            for s0, x0 in [(1.4, -1.2), (1.1, -0.2), (0.7, 1.9), (0.5, -1.7)]:
                x_end = float(toy.flow(x0, s0, 0))
                cm_arcs.add(jump_arrow(plane.c2p(s0, x0), plane.c2p(0, x_end), angle=-0.4,
                                       color=GREY_A, width=2.5))
            cm_lab = text("consistency model: only jumps to σ = 0", font_size=28, color=GREY_A)
            self.row(cm_lab, 2)
            self.play(FadeOut(VGroup(arc2, arc3)), LaggedStartMap(ShowCreation, cm_arcs, lag_ratio=0.2),
                      FadeIn(cm_lab), run_time=2)
            b.until("A flow map can")
            fm_lab = text("flow map: any pair (ρ, σ)  →  any number of steps", font_size=28,
                          color=STUDENT_C)
            self.row(fm_lab, 2)
            self.play(FadeOut(cm_arcs), FadeTransform(cm_lab, fm_lab), run_time=1)

        with self.beat("lagrange") as b:
            self.play(FadeOut(VGroup(p_tau, v_tau, fm_lab, props)), run_time=0.8)
            lag = tex(R"\text{Lagrangian:}\quad \partial_{\sigma} X_{\rho,\sigma}(x) = "
                      R"v\big(X_{\rho,\sigma}(x),\,\sigma\big)", font_size=36, t2c={V_KEY: TEACHER_C})
            self.row(lag, 1)
            self.play(Write(lag), run_time=1.5)
            b.until("the derivative of X")
            s_tr = ValueTracker(sig)
            self.remove(arc, p_end, lab_end)
            arc_d = always_redraw(lambda: jump_arrow(
                p_start.get_center(), plane.c2p(s_tr.get_value(), toy.flow(xr, rho, s_tr.get_value())),
                angle=-0.9))
            end_d = always_redraw(lambda: plane.dot(
                s_tr.get_value(), toy.flow(xr, rho, s_tr.get_value()), SIG_C, radius=0.09))
            tan_d = always_redraw(lambda: self.tangent_arrow(plane, s_tr.get_value(),
                                                             float(toy.flow(xr, rho, s_tr.get_value()))))
            v_sig_d = always_redraw(lambda: plane.vline(s_tr.get_value(), SIG_C, label_tex=R"\sigma"))
            self.remove(v_sig)
            self.add(v_sig_d, arc_d, end_d)
            self.play(FadeIn(tan_d), run_time=0.5)
            self.play(s_tr.animate.set_value(0.12), run_time=3, rate_func=there_and_back_with_pause)
            self.play(FadeOut(tan_d), run_time=0.4)

        with self.beat("euler_eq") as b:
            eul = tex(R"\text{Eulerian:}\quad \partial_{\rho} X_{\rho,\sigma}(x) + "
                      R"\nabla_x X_{\rho,\sigma}(x)\, v(x,\rho) = 0",
                      font_size=36, t2c={V_KEY: TEACHER_C})
            self.row(eul, 2)
            self.play(Write(eul), run_time=1.8)
            r_tr = ValueTracker(rho)
            self.remove(arc_d, end_d, p_start, lab_start, v_sig_d)
            fixed_end = plane.dot(sig, xs, SIG_C, radius=0.09)

            def start_x():
                return float(toy.flow(xr, rho, r_tr.get_value()))

            start_d = always_redraw(lambda: plane.dot(r_tr.get_value(), start_x(), RHO_C, radius=0.09))
            arc_e = always_redraw(lambda: jump_arrow(
                plane.c2p(r_tr.get_value(), start_x()), fixed_end.get_center(),
                angle=-0.9 * min(1, (r_tr.get_value() - sig) / (rho - sig) + 0.25)))
            v_rho_d = always_redraw(lambda: plane.vline(r_tr.get_value(), RHO_C, label_tex=R"\rho"))
            self.remove(v_rho)
            self.add(v_sig, v_rho_d, arc_e, start_d, fixed_end)
            b.until("Slide the start")
            self.play(r_tr.animate.set_value(1.5), run_time=2.2)
            self.play(r_tr.animate.set_value(0.6), run_time=3.0)
            self.play(r_tr.animate.set_value(rho), run_time=1.5)
            b.until("This is the equation")
            tag = with_bg(text("DeCAF enforces this, with the teacher's velocity v", font_size=26,
                               color=STUDENT_C))
            tag.next_to(eul, DOWN, buff=0.12)
            self.play(FadeIn(tag, shift=0.1 * UP), FlashAround(eul, color=STUDENT_C), run_time=1.2)

        with self.beat("segue", pad=0.3) as b:
            self.clear_all(run_time=1.2)

    @staticmethod
    def full_trajectory(plane, x, rho):
        x_top = float(toy.flow(x, rho, plane.sigma_max))
        curve = plane.trajectory_curve(x_top)
        return curve.set_stroke(WHITE, 3)

    @staticmethod
    def tangent_arrow(plane, s, x, length=1.1):
        # direction of motion as sigma decreases: (d sigma, dx) = (-1, -v)
        v = float(toy.velocity(x, max(s, 1e-3)))
        p = plane.c2p(s, x)
        q = plane.c2p(s - 0.1, x - 0.1 * v)
        d = (q - p) / np.linalg.norm(q - p)
        arr = Arrow(p, p + length * d, buff=0, thickness=3, fill_color=TEACHER_C)
        return arr

# ==========================================================================
class S04_SigmaSpace(DecafScene):
    def construct(self):
        label = section_label("Noise level, not time", number=1)
        with self.beat("schedule") as b:
            self.play(FadeIn(label), run_time=0.8)
            left = text("flow-map recipes: time t ∈ [0, 1]", font_size=30)
            right = text("teacher: noise level σ", font_size=30, color=SIG_C)
            VGroup(left, right).arrange(RIGHT, buff=1.4).to_edge(UP, buff=1.1)
            self.play(FadeIn(left), run_time=0.8)
            b.until("The teacher lives")
            self.play(FadeIn(right), run_time=0.8)
            b.until("a Karras schedule")
            sched = tex(
                R"\sigma(t) = \Big(\sigma_{\min}^{1/7} + t\,\big(\sigma_{\max}^{1/7}-\sigma_{\min}^{1/7}\big)\Big)^{7}",
                font_size=40).next_to(VGroup(left, right), DOWN, buff=0.45)
            self.play(Write(sched), run_time=2)
            axes, ylabels = self.log_axes()
            g_sigma = axes.get_graph(lambda t: np.log10(self.sigma(t)), x_range=(0, 1, 0.005))
            g_sigma.set_stroke(SIG_C, 3.5)
            g_lab = tex(R"\sigma(t)", font_size=32).next_to(axes.c2p(0.9, np.log10(self.sigma(0.9))),
                                                         DOWN + RIGHT, buff=0.08)
            self.play(ShowCreation(axes), FadeIn(ylabels), run_time=1.2)
            self.play(ShowCreation(g_sigma), FadeIn(g_lab), run_time=1.5)

        with self.beat("chain") as b:
            chain = tex(R"\frac{dx}{dt} = \dot\sigma(t)\  v\big(x,\sigma(t)\big), \qquad "
                        R"\dot\sigma(t) = \frac{d\sigma}{dt}",
                        font_size=40, t2c={R"\dot\sigma(t)": WARN_C, V_KEY: TEACHER_C})
            chain.move_to(sched)
            self.play(FadeOut(sched, shift=0.2 * UP), FadeIn(chain, shift=0.2 * UP), run_time=1.2)
            b.until("For this schedule")
            g_dot = axes.get_graph(lambda t: np.log10(self.sigma_dot(t)), x_range=(0, 1, 0.005))
            g_dot.set_stroke(WARN_C, 3.5)
            d_lab = tex(R"\dot\sigma(t)", font_size=32, t2c={R"\dot\sigma(t)": WARN_C})
            d_lab.next_to(axes.c2p(0.42, np.log10(self.sigma_dot(0.42))), UP + LEFT, buff=0.08)
            self.play(ShowCreation(g_dot), FadeIn(d_lab), run_time=1.5)
            b.until("about a quarter")
            lo_pt = axes.c2p(0, np.log10(self.sigma_dot(0)))
            hi_pt = axes.c2p(1, np.log10(self.sigma_dot(1)))
            lo_lab = tex(R"0.24\ \text{\AA}", font_size=30).set_color(WARN_C).next_to(lo_pt, RIGHT, buff=0.2)
            hi_lab = tex(R"15\,082\ \text{\AA}", font_size=30).set_color(WARN_C).next_to(hi_pt, LEFT, buff=0.2)
            self.play(FadeIn(Dot(lo_pt, fill_color=WARN_C)), FadeIn(lo_lab), run_time=0.8)
            b.until("and about fifteen")
            self.play(FadeIn(Dot(hi_pt, fill_color=WARN_C)), FadeIn(hi_lab), run_time=0.8)
            b.until("That is a factor")
            ratio = VGroup(text("× 63 000", font_size=34, color=WARN_C),
                           text("≈ 5 orders of magnitude", font_size=28, color=WARN_C))
            ratio.arrange(DOWN, buff=0.12).next_to(axes, RIGHT, buff=0.55).shift(UP * 0.4)
            arr = Arrow(axes.c2p(1.02, np.log10(self.sigma_dot(0))), axes.c2p(1.02, np.log10(self.sigma_dot(1))),
                        buff=0, thickness=3, fill_color=WARN_C)
            self.play(GrowArrow(arr), FadeIn(ratio), run_time=1.2)
        plot = VGroup(axes, ylabels, g_sigma, g_lab, g_dot, d_lab, lo_lab, hi_lab, arr, ratio)
        plot.add(*[m for m in self.mobjects if isinstance(m, Dot)])

        with self.beat("problem") as b:
            self.play(FadeOut(plot), FadeOut(VGroup(left, right)),
                      chain.animate.scale(0.85).to_edge(UP, buff=1.0), run_time=1.2)
            avg = tex(R"X_{t,r}(x) = x - (t-r)\,U_\theta(x,t,r)", font_size=38,
                      t2c={R"U_\theta": STUDENT_C})
            lt = tex(R"\mathcal{L}_t = \Big\Vert\, U_\theta + (t-r)\,\mathrm{sg}\Big[\tfrac{d}{dt}U_\theta\Big]"
                     R" - \dot\sigma(t)\, v \,\Big\Vert^2",
                     font_size=44, t2c={R"\dot\sigma(t)": WARN_C, R"U_\theta": STUDENT_C, V_KEY: TEACHER_C})
            jvp = tex(R"\tfrac{d}{dt}U_\theta = \partial_t U_\theta + \nabla_x U_\theta\,"
                      R"\big(\dot\sigma(t)\, v\big)",
                      font_size=40, t2c={R"\dot\sigma(t)": WARN_C, R"U_\theta": STUDENT_C, V_KEY: TEACHER_C})
            VGroup(avg, lt, jvp).arrange(DOWN, buff=0.5).next_to(chain, DOWN, buff=0.6)
            b.until("It regresses")
            self.play(FadeIn(avg), run_time=0.8)
            self.play(Write(lt), run_time=2)
            b.until("and it differentiates")
            self.play(Write(jvp), run_time=2)
            jvp_note = text("Jacobian-vector product, tangent  (σ̇ v, 1)", font_size=26, color=GREY_B)
            jvp_note.next_to(jvp, DOWN, buff=0.25)
            self.play(FadeIn(jvp_note), run_time=0.8)
            b.until("both carry")
            reds = VGroup(lt[R"\dot\sigma(t)"], jvp[R"\dot\sigma(t)"])
            self.play(*[FlashAround(r, color=WARN_C, time_width=1.5) for r in reds], run_time=1.5)
            b.until("Depending on where")
            scale_note = text("target scale ∝ σ̇(t):  0.24  …  15 082", font_size=30, color=WARN_C)
            scale_note.next_to(jvp_note, DOWN, buff=0.45)
            self.play(FadeIn(scale_note, shift=0.1 * UP), run_time=1)
            b.until("The paper reports")
            unstable = text("paper: time-parameterized objectives are numerically unstable",
                            font_size=28, color=GREY_A)
            unstable.next_to(scale_note, DOWN, buff=0.35)
            self.play(FadeIn(unstable), run_time=1)

        with self.beat("fix") as b:
            b.until("Index the flow map")
            lsig = tex(R"\mathcal{L}_{\sigma} = \Big\Vert\, u_\theta + (\rho-\sigma)\,"
                       R"\mathrm{sg}\Big[\tfrac{d}{d\rho}u_\theta\Big] - v \,\Big\Vert^2",
                       font_size=44, t2c={V_KEY: TEACHER_C})
            jvps = tex(R"\tfrac{d}{d\rho}u_\theta = \partial_{\rho} u_\theta + \nabla_x u_\theta\, v",
                       font_size=40, t2c={V_KEY: TEACHER_C})
            avg_s = tex(R"X_{\rho,\sigma}(x) = x - (\rho-\sigma)\,u_\theta(x,\rho,\sigma)", font_size=38)
            VGroup(avg_s, lsig, jvps).arrange(DOWN, buff=0.5).move_to(VGroup(avg, lt, jvp))
            self.play(FadeOut(VGroup(scale_note, unstable, jvp_note)), run_time=0.6)
            self.play(TransformMatchingTex(avg, avg_s), run_time=1.5)
            self.play(TransformMatchingTex(lt, lsig), TransformMatchingTex(jvp, jvps), run_time=2)
            eqs = VGroup(avg_s, lsig, jvps)
            self.play(FadeOut(chain, shift=0.3 * UP), eqs.animate.to_edge(UP, buff=0.9), run_time=1)
            b.until("The sigma velocity is")
            vdef = tex(R"v(x,\rho) = \frac{x - D(x;\rho)}{\rho}\ \approx\ \hat\varepsilon"
                       R"\quad\text{(order one at every }\rho)",
                       font_size=36, t2c={V_KEY: TEACHER_C, TEACHER_D: TEACHER_C})
            vdef.next_to(jvps, DOWN, buff=0.5)
            self.play(Write(vdef), run_time=2)
            b.until("The tangent of")
            tang = text("JVP tangent: (v, 1)", font_size=30, color=GREY_A).next_to(vdef, DOWN, buff=0.35)
            self.play(FadeIn(tang), run_time=0.8)
            b.until("The factor d sigma d t")
            gone = text("σ̇(t): never appears", font_size=30, color=WARN_C)
            gone.next_to(tang, RIGHT, buff=0.8)
            self.play(FadeIn(gone), run_time=0.6)
            mag = self.magnitude_plot().to_edge(DOWN, buff=0.9)
            self.play(FadeIn(mag[0]), ShowCreation(mag[1]), FadeIn(mag[3]), run_time=1.2)
            self.play(ShowCreation(mag[2]), FadeIn(mag[4]), run_time=1.2)
            b.until("The schedule survives")
            surv = text("the schedule only chooses which (ρ, σ) to train on and to step through",
                        font_size=28, color=GREY_A)
            surv.to_edge(DOWN, buff=0.3)
            self.play(FadeIn(surv, shift=0.1 * UP), run_time=1)

    def magnitude_plot(self):
        axes = Axes((0, 1, 0.25), (-1, 5, 1), width=6.0, height=2.2,
                    axis_config=dict(stroke_color=GREY_B, stroke_width=2))
        ylabs = VGroup(*[tex(rf"10^{{{k}}}", font_size=22).set_color(GREY_B)
                         .next_to(axes.c2p(0, k), LEFT, buff=0.12) for k in (0, 2, 4)])
        title = text("size of the velocity target (per coordinate)", font_size=24, color=GREY_B)
        title.next_to(axes, UP, buff=0.12)
        tl = tex("t", font_size=26).next_to(axes.c2p(1, -1), DOWN, buff=0.1)
        frame = VGroup(axes, ylabs, title, tl)
        red = axes.get_graph(lambda t: np.log10(self.sigma_dot(t)), x_range=(0, 1, 0.01))
        red.set_stroke(WARN_C, 3.5)
        teal = axes.get_graph(lambda t: 0.0, x_range=(0, 1, 0.05)).set_stroke(STUDENT_C, 3.5)
        red_lab = tex(R"t\text{-space: } \dot\sigma(t)\,\hat\varepsilon", font_size=28,
                      t2c={R"\dot\sigma(t)": WARN_C}).set_color(WARN_C)
        red_lab.next_to(axes.c2p(0.6, np.log10(self.sigma_dot(0.6))), UP + LEFT, buff=0.05)
        teal_lab = tex(R"\sigma\text{-space: } \hat\varepsilon", font_size=28).set_color(STUDENT_C)
        teal_lab.next_to(axes.c2p(1, 0), RIGHT, buff=0.15)
        return VGroup(frame, red, teal, red_lab, teal_lab)

    @staticmethod
    def sigma(t):
        a, c = 0.0064 ** (1 / 7), 2560 ** (1 / 7)
        return (a + t * (c - a)) ** 7

    @staticmethod
    def sigma_dot(t):
        a, c = 0.0064 ** (1 / 7), 2560 ** (1 / 7)
        return 7 * (a + t * (c - a)) ** 6 * (c - a)

    @staticmethod
    def log_axes():
        axes = Axes((0, 1, 0.25), (-3, 5, 1), width=7.0, height=3.6,
                    axis_config=dict(stroke_color=GREY_B, stroke_width=2))
        axes.to_edge(DOWN, buff=0.55).shift(LEFT * 1.9)
        labels = VGroup()
        for k in range(-2, 5, 2):
            lab = tex(rf"10^{{{k}}}", font_size=24).set_color(GREY_B)
            lab.next_to(axes.c2p(0, k), LEFT, buff=0.15)
            labels.add(lab)
        for t in [0, 0.5, 1]:
            lab = tex(f"{t:g}", font_size=24).set_color(GREY_B)
            lab.next_to(axes.c2p(t, -3), DOWN, buff=0.15)
            labels.add(lab)
        tl = tex("t", font_size=30).next_to(axes.c2p(1, -3), DOWN + RIGHT, buff=0.15)
        clean = text("clean", font_size=22, color=GREY_B).next_to(axes.c2p(0, -3), DOWN, buff=0.45)
        noisy = text("noisy", font_size=22, color=GREY_B).next_to(axes.c2p(1, -3), DOWN, buff=0.45)
        unit = text("Å (log scale)", font_size=22, color=GREY_B).next_to(axes.c2p(0, 5), UP, buff=0.1)
        labels.add(tl, clean, noisy, unit)
        return axes, labels


# ==========================================================================
class S05_DenoiserMap(DecafScene):
    RHO, XR, SIG = 1.0, 0.2, 0.5
    ZOOM = dict(x_range=(-2.2, 2.2))

    def construct(self):
        label = section_label("Parameterize the map as a denoiser", number=2)
        rho, xr, sig = self.RHO, self.XR, self.SIG
        with self.beat("ddim") as b:
            self.play(FadeIn(label), run_time=0.8)
            e1 = tex(R"x_{\sigma} = x_{\rho} - (\rho-\sigma)\, v(x_{\rho},\rho)", font_size=46,
                     t2c={V_KEY: TEACHER_C})
            e1.move_to(UP * 1.5)
            b.until("one Euler step")
            self.play(Write(e1), run_time=1.8)
            tag = text("one Euler step  =  DDIM", font_size=28, color=GREY_B).next_to(e1, DOWN, buff=0.35)
            b.until("that is exactly")
            self.play(FadeIn(tag), run_time=0.8)
            b.until("Rearranged")
            e2 = tex(R"x_{\sigma} = \frac{\sigma}{\rho}\, x_{\rho} + \Big(1-\frac{\sigma}{\rho}\Big)\, D(x_{\rho};\rho)",
                     font_size=46, t2c={TEACHER_D: TEACHER_C}, isolate=[R"D(x_{\rho};\rho)"])
            e2.next_to(tag, DOWN, buff=0.6)
            self.play(TransformMatchingTex(e1.copy(), e2), run_time=2)
            b.until("the denoised estimate")
            br = Brace(e2[R"D(x_{\rho};\rho)"], DOWN, buff=0.15)
            br_lab = text("denoised estimate", font_size=26, color=TEACHER_C).next_to(br, DOWN, buff=0.1)
            self.play(GrowFromCenter(br), FadeIn(br_lab), run_time=1)

        plane = make_plane(**self.ZOOM)
        plane.trajectories.set_stroke(opacity=0.3)
        d_teacher = float(toy.denoiser(xr, rho))
        with self.beat("tangent") as b:
            self.play(FadeOut(VGroup(e1, tag, br, br_lab)),
                      e2.animate.scale(0.75).to_edge(UP, buff=0.35).shift(RIGHT * 1.6), run_time=1)
            self.play(FadeIn(plane), run_time=1)
            traj = S03_FlowMaps.full_trajectory(plane, xr, rho)
            p0 = plane.dot(rho, xr, RHO_C, radius=0.09)
            lab0 = tex(R"x_{\rho}", font_size=32).next_to(p0, DOWN + LEFT, buff=0.05)
            v_rho = plane.vline(rho, RHO_C, label_tex=R"\rho")
            self.play(ShowCreation(traj), FadeIn(p0), FadeIn(lab0), FadeIn(v_rho), run_time=1.2)
            b.until("An Euler step moves")
            tline = Line(plane.c2p(rho, xr), plane.c2p(0, d_teacher)).set_stroke(TEACHER_C, 3)
            self.play(ShowCreation(tline), run_time=1.5)
            b.until("exactly at the denoiser")
            dT = plane.dot(0, d_teacher, TEACHER_C, radius=0.1)
            dT_lab = with_bg(tex(R"D(x_{\rho};\rho)", font_size=32, t2c={TEACHER_D: TEACHER_C}))
            dT_lab.next_to(dT, RIGHT, buff=0.15)
            self.play(FadeIn(dT, scale=0.5), FadeIn(dT_lab), run_time=1)
            self.play(Flash(dT, color=TEACHER_C), run_time=0.8)
            x_euler = xr + (sig - rho) * float(toy.velocity(xr, rho))
            pe = plane.dot(sig, x_euler, TEACHER_C, radius=0.08).set_fill(opacity=0)
            pe.set_stroke(TEACHER_C, 2.5)
            pe_lab = with_bg(text("Euler / DDIM", font_size=24, color=TEACHER_C)).next_to(pe, DOWN, buff=0.15)
            vs = plane.vline(sig, SIG_C, label_tex=R"\sigma")
            self.play(FadeIn(vs), FadeIn(pe), FadeIn(pe_lab), run_time=1)

        with self.beat("decaf_form") as b:
            e3 = tex(R"X_{\rho,\sigma}(x) = \frac{\sigma}{\rho}\, x + \Big(1-\frac{\sigma}{\rho}\Big)\,"
                     R"D_\theta(x,\rho,\sigma)", font_size=46, isolate=[R"D_\theta(x,\rho,\sigma)"])
            e3.scale(0.75).move_to(e2)
            self.play(TransformMatchingTex(e2, e3), run_time=2)
            b.until("a denoiser that takes")
            box = SurroundingRectangle(e3[R"D_\theta(x,\rho,\sigma)"], buff=0.08).set_stroke(STUDENT_C, 2)
            two = text("two noise levels", font_size=26, color=STUDENT_C).next_to(box, RIGHT, buff=0.2)
            self.play(ShowCreation(box), FadeIn(two), run_time=1)

        s_tr = ValueTracker(sig)

        def x_end_of_chord():
            s = s_tr.get_value()
            return d_teacher if s > rho - 1e-3 else float(toy.flowmap_denoiser(xr, rho, s))

        def chord():
            return Line(plane.c2p(rho, xr), plane.c2p(0, x_end_of_chord())).set_stroke(STUDENT_C, 3.5)

        def chord_dot():
            return plane.dot(0, x_end_of_chord(), STUDENT_C, radius=0.1)

        with self.beat("chord") as b:
            xs = float(toy.flow(xr, rho, sig))
            px = plane.dot(sig, xs, SIG_C, radius=0.09)
            px_lab = with_bg(tex(R"X_{\rho,\sigma}(x_{\rho})", font_size=30)).next_to(px, UP + LEFT, buff=0.05)
            self.play(FadeOut(VGroup(pe, pe_lab)), FadeIn(px, scale=0.5), FadeIn(px_lab), run_time=1)
            b.until("Draw the chord")
            ch = chord()
            self.play(ShowCreation(ch), run_time=1.5)
            b.until("Where it crosses")
            cd = chord_dot()
            cd_lab = with_bg(tex(R"D_\theta(x_{\rho},\rho,\sigma)", font_size=32)).next_to(cd, RIGHT, buff=0.15)
            self.play(FadeIn(cd, scale=0.5), FadeIn(cd_lab), run_time=1)
            b.until("The tangent gives")
            t1 = text("tangent  →  teacher's denoiser", font_size=28, color=TEACHER_C)
            t2 = text("chord  →  flow map's denoiser", font_size=28, color=STUDENT_C)
            legend = VGroup(t1, t2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            legend.move_to([-3.2, 2.45, 0])
            self.play(FadeIn(t1), run_time=0.8)
            b.until("The chord gives")
            self.play(FadeIn(t2), run_time=0.8)

        with self.beat("free") as b:
            ch_d = always_redraw(chord)
            cd_d = always_redraw(chord_dot)
            px_d = always_redraw(lambda: plane.dot(
                s_tr.get_value(), toy.flow(xr, rho, s_tr.get_value()), SIG_C, radius=0.09))
            vs_d = always_redraw(lambda: plane.vline(
                s_tr.get_value(), SIG_C,
                label_tex=R"\sigma" if 0.1 < s_tr.get_value() < rho - 0.1 else None))
            self.remove(ch, cd, px, vs)
            self.add(vs_d, ch_d, cd_d, px_d)
            self.play(FadeOut(VGroup(px_lab, cd_lab, legend)), run_time=0.5)
            caps = [
                tex(R"\sigma=\rho:\quad X_{\rho,\rho}(x) = x \qquad\text{built in}", font_size=34),
                tex(R"\sigma\to\rho:\quad D_\theta(x,\rho,\rho) = D(x;\rho)\qquad"
                    R"\text{tangent condition = denoiser distillation}",
                    font_size=34, t2c={TEACHER_D: TEACHER_C}),
                tex(R"\sigma=0:\quad X_{\rho,0}(x) = D_\theta(x,\rho,0)\qquad"
                    R"\text{one call, a sharp clean structure}", font_size=34),
            ]
            for c in caps:
                c.move_to([0.4, 2.45, 0])
                if c.get_width() > FRAME_WIDTH - 1:
                    c.set_width(FRAME_WIDTH - 1)
            b.until("When sigma equals rho")
            self.play(FadeIn(caps[0], shift=0.1 * UP), run_time=1)
            b.until("As sigma approaches")
            self.play(FadeOut(caps[0], shift=0.1 * UP), FadeIn(caps[1], shift=0.1 * UP),
                      s_tr.animate.set_value(rho - 0.002), run_time=3)
            self.play(Flash(plane.c2p(0, d_teacher), color=STUDENT_C), run_time=0.8)
            b.until("And at sigma equals zero")
            self.play(FadeOut(caps[1], shift=0.1 * UP), FadeIn(caps[2], shift=0.1 * UP),
                      s_tr.animate.set_value(0.0), run_time=3.5)
            b.until("So a single network")
            x_end = float(toy.flow(xr, rho, 0))
            sharp = with_bg(text("trajectory end point", font_size=24, color=STUDENT_C))
            sharp.next_to(plane.c2p(0, x_end), RIGHT, buff=0.2)
            mean = with_bg(text("posterior mean", font_size=24, color=TEACHER_C))
            mean.next_to(dT, RIGHT, buff=0.2)
            self.play(FadeOut(dT_lab), FadeIn(sharp), FadeIn(mean), run_time=1)

        with self.beat("arch") as b:
            self.clear_all(run_time=1)
            trunk = self.block("Pairformer trunk", "frozen", GREY_B, BLUE_E)
            teacher = self.block("teacher diffusion module", "frozen", TEACHER_C, BLUE_E)
            student = self.block("DeCAF head: same architecture\n+ second noise-level embedding",
                                 "trained", STUDENT_C, GREEN_E)
            trunk.move_to(LEFT * 4.3)
            VGroup(teacher, student).arrange(DOWN, buff=1.0).move_to(RIGHT * 0.9)
            a1 = Arrow(trunk.get_right(), teacher.get_left(), buff=0.15, thickness=2.5)
            a2 = Arrow(trunk.get_right(), student.get_left(), buff=0.15, thickness=2.5)
            ins = tex(R"\text{inputs: } x_{\rho},\ \log\rho,\ \log\sigma", font_size=32)
            ins.next_to(student, DOWN, buff=0.4)
            out = tex(R"\longrightarrow\ D_\theta(x_{\rho},\rho,\sigma)", font_size=34)
            out.next_to(student, RIGHT, buff=0.3)
            src = text("(released code)", font_size=24, color=GREY_C).to_corner(DR, buff=0.4)
            self.play(FadeIn(trunk), run_time=0.8)
            self.play(GrowArrow(a1), FadeIn(teacher), run_time=0.8)
            b.until("plus a second")
            self.play(GrowArrow(a2), FadeIn(student), FadeIn(ins), FadeIn(out), run_time=1.2)
            b.until("The Pairformer trunk")
            self.play(FadeIn(src), *[Indicate(m[1][1], color=WHITE) for m in (trunk, teacher)],
                      run_time=1.2)
            b.until("only this head")
            self.play(Indicate(student[1][1], color=GREEN), run_time=1)

    @staticmethod
    def block(title, status, color, status_color):
        body = Text(title, font=FONT, font_size=28)
        rect = RoundedRectangle(width=max(body.get_width() + 0.6, 3.2), height=body.get_height() + 0.7,
                                corner_radius=0.15)
        rect.set_stroke(color, 2.5).set_fill(color, 0.08)
        body.move_to(rect)
        tag = text(status, font_size=22, color=WHITE)
        tag_bg = RoundedRectangle(width=tag.get_width() + 0.3, height=tag.get_height() + 0.18,
                                  corner_radius=0.08).set_fill(status_color, 1).set_stroke(width=0)
        tag.move_to(tag_bg)
        badge = VGroup(tag_bg, tag).next_to(rect, UP, buff=0.08).align_to(rect, RIGHT)
        return VGroup(rect, VGroup(body, badge))
