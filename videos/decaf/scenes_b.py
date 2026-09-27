"""Scenes 6-10: aligned structure-space loss, sampling, DeCAF-Search, results, takeaways.

Render (from this folder):  manimgl scenes_b.py S06_AlignedLoss -w --hd
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import *  # noqa: E402,F401
from scenes_a import jump_arrow, make_plane, S03_FlowMaps  # noqa: E402


def bullet_list(items, font_size=30, buff=0.28, color=WHITE):
    rows = VGroup()
    for item in items:
        dot = Dot(radius=0.05, fill_color=GREY_B)
        body = Text(item, font=FONT, font_size=font_size, color=color, alignment="LEFT")
        body.next_to(dot, RIGHT, buff=0.25)
        dot.align_to(body[0], UP).shift(DOWN * 0.1)
        rows.add(VGroup(dot, body))
    rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    return rows


# ==========================================================================
class S06_AlignedLoss(DecafScene):
    def construct(self):
        label = section_label("The loss lives in structure space", number=2)
        with self.beat("identity") as b:
            self.play(FadeIn(label), run_time=0.8)
            e1 = tex(R"X_{\rho,\sigma}(x) = x - (\rho-\sigma)\, u_\theta(x,\rho,\sigma)", font_size=40)
            e1b = tex(R"u_\theta = \frac{x - D_\theta(x,\rho,\sigma)}{\rho}", font_size=40)
            row1 = VGroup(e1, e1b).arrange(RIGHT, buff=1.0)
            e2 = tex(R"\text{Eulerian} \ \Rightarrow\ \ u = v - (\rho-\sigma)\,\frac{du}{d\rho}",
                     font_size=40, t2c={V_KEY: TEACHER_C})
            e3 = tex(R"\frac{du}{d\rho} = \partial_{\rho} u + (\nabla_x u)\, v", font_size=40,
                     t2c={V_KEY: TEACHER_C})
            jvp = text("one Jacobian-vector product, tangent (v, 1)", font_size=26, color=GREY_B)
            col = VGroup(row1, e2, VGroup(e3, jvp).arrange(RIGHT, buff=0.6)).arrange(DOWN, buff=0.45)
            col.next_to(label, DOWN, buff=0.35).set_x(0)
            b.until("Write the jump")
            self.play(Write(e1), run_time=1.5)
            b.until("where u is")
            self.play(Write(e1b), run_time=1.5)
            b.until("Substituting")
            self.play(Write(e2), run_time=2)
            b.until("That derivative costs")
            self.play(Write(e3), run_time=1.5)
            self.play(FadeIn(jvp), run_time=0.8)

        with self.beat("loss") as b:
            loss = tex(R"\mathcal{L} = \Big\Vert\, u_\theta + (\rho-\sigma)\,\mathrm{sg}\Big[\frac{du_\theta}{d\rho}\Big]"
                       R" - v \,\Big\Vert^2", font_size=46, t2c={V_KEY: TEACHER_C})
            loss.next_to(col, DOWN, buff=0.5)
            b.until("So the student")
            self.play(Write(loss), run_time=2)
            b.until("On the diagonal")
            diag = tex(R"\sigma=\rho:\quad \mathcal{L} = \frac{1}{\rho^2}\,"
                       R"\big\Vert D_\theta(x,\rho,\rho) - D(x;\rho) \big\Vert^2",
                       font_size=36, t2c={TEACHER_D: TEACHER_C})
            diag.next_to(loss, DOWN, buff=0.35)
            note = text("plain denoiser distillation", font_size=26, color=GREY_B)
            note.next_to(diag, DOWN, buff=0.18)
            self.play(FadeIn(diag, shift=0.1 * UP), run_time=1.2)
            self.play(FadeIn(note), run_time=0.6)

        with self.beat("structures") as b:
            self.play(FadeOut(VGroup(col, diag, note)), loss.animate.to_edge(UP, buff=0.9), run_time=1)
            b.until("Multiply the residual")
            e5 = tex(R"\rho\,\Big(u_\theta + (\rho-\sigma)\,\mathrm{sg}[\cdots] - v\Big)"
                     R" = D(x;\rho) - \hat D_\theta", font_size=42,
                     t2c={V_KEY: TEACHER_C, TEACHER_D: TEACHER_C})
            e5.next_to(loss, DOWN, buff=0.35)
            self.play(Write(e5), run_time=2)
            b.until("call it D hat")
            e6 = tex(R"\hat D_\theta := D_\theta - \rho(\rho-\sigma)\,\mathrm{sg}\Big[\frac{du_\theta}{d\rho}\Big]",
                     font_size=40)
            e6.next_to(e5, DOWN, buff=0.3)
            self.play(FadeIn(e6, shift=0.1 * UP), run_time=1.2)
            b.until("The loss becomes")
            e7 = tex(R"\mathcal{L} = \frac{1}{\rho^2}\,\big\Vert \hat D_\theta - D(x;\rho)\big\Vert^2",
                     font_size=46, t2c={TEACHER_D: TEACHER_C})
            e7.next_to(e6, DOWN, buff=0.55)
            box = SurroundingRectangle(e7, buff=0.15).set_stroke(STUDENT_C, 2)
            self.play(Write(e7), ShowCreation(box), run_time=1.5)
            clouds = self.clouds_pair(0.0, (0, 0)).scale(0.6).to_edge(DOWN, buff=0.2)
            self.play(FadeIn(clouds), run_time=1)

        with self.beat("why") as b:
            keep = VGroup(e7, box)
            self.play(FadeOut(VGroup(loss, e5, e6, clouds)), keep.animate.to_edge(UP, buff=0.9), run_time=1)
            head = text("How the teacher was trained", font_size=30, color=TEACHER_C)
            teacher_loss = tex(
                R"\mathcal{L}_{\text{teacher}} = \big\Vert D(x_{\sigma};\sigma) - "
                R"\mathrm{Align}\big(x_0 \rightarrow D(x_{\sigma};\sigma)\big)\big\Vert^2"
                R" + \mathcal{L}_{\text{smooth-LDDT}}", font_size=38, t2c={TEACHER_D: TEACHER_C})
            pts = bullet_list([
                "non-equivariant network, random rotations as augmentation",
                "the loss is blind to the global pose of D's output",
                "Boltz-1's own sampler re-aligns the noisy structure to D before every step",
            ], font_size=28)
            grp = VGroup(head, teacher_loss, pts).arrange(DOWN, buff=0.45)
            grp.next_to(keep, DOWN, buff=0.6)
            b.until("AlphaFold-style")
            self.play(FadeIn(head), run_time=0.6)
            b.until("They are trained")
            self.play(Write(teacher_loss), FadeIn(pts[0]), run_time=2)
            b.until("So nothing")
            self.play(FadeIn(pts[1]), run_time=0.8)
            b.until("Boltz's own sampler")
            self.play(FadeIn(pts[2]), run_time=0.8)

        with self.beat("kabsch") as b:
            self.play(FadeOut(grp), run_time=0.8)
            P, Qraw, kinds, bonds, R0, t0 = self.molecule_pair()
            scale, center = 1.55, DOWN * 1.0
            stu = molecule_mobject(P, kinds, bonds, scale=scale, center=center, ligand_color=STUDENT_C,
                                   protein_color=GREY_B)
            tea_pts = (Qraw @ R0.T) + t0
            tea = molecule_mobject(tea_pts, kinds, bonds, scale=scale, center=center, ligand_color=TEACHER_C,
                                   protein_color=TEACHER_C, opacity=0.9)
            lab_s = tex(R"\text{student } \hat D_\theta", font_size=32).set_color(STUDENT_C)
            lab_t = tex(R"\text{teacher } D(x;\rho)", font_size=32).set_color(TEACHER_C)
            legend = VGroup(lab_s, lab_t).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(DL, buff=0.6)
            self.play(FadeIn(stu), FadeIn(tea), FadeIn(legend), run_time=1.2)

            def residuals(Q):
                arrows = VGroup()
                for p, q in zip(P, Q):
                    a = np.array([*p, 0]) * scale + center
                    c = np.array([*q, 0]) * scale + center
                    if np.linalg.norm(c - a) > 0.02:
                        arrows.add(Line(a, c).set_stroke(WARN_C, 2.5))
                return arrows

            def rmsd(Q):
                return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))

            res0 = residuals(tea_pts)
            m0 = tex(rf"\text{{residual RMS}} = {rmsd(tea_pts):.2f}", font_size=32).set_color(WARN_C)
            m0.to_corner(DR, buff=0.6)
            b.until("Those residuals")
            self.play(ShowCreation(res0, lag_ratio=0.05), FadeIn(m0), run_time=1.5)
            b.until("In structure space")
            w = np.array([10.0 if k == "l" else 1.0 for k in kinds])
            R, t = kabsch_2d(P, tea_pts, w)
            aligned = tea_pts @ R.T + t
            kab = text("weighted Kabsch (ligand ×10)", font_size=28, color=ALIGN_C).next_to(m0, UP, buff=0.3)
            self.play(FadeIn(kab), run_time=0.8)
            b.until("align the teacher's")
            tea_target = molecule_mobject(aligned, kinds, bonds, scale=scale, center=center,
                                          ligand_color=TEACHER_C, protein_color=TEACHER_C, opacity=0.9)
            self.play(FadeOut(res0), Transform(tea, tea_target), run_time=2.5)
            res1 = residuals(aligned)
            m1 = tex(rf"\text{{residual RMS}} = {rmsd(aligned):.2f}", font_size=32).set_color(ALIGN_C)
            m1.move_to(m0)
            self.play(ShowCreation(res1), Transform(m0, m1), run_time=1.2)
            b.until("What remains")
            self.play(Indicate(res1, color=WARN_C, scale_factor=1.0), run_time=1.2)

        with self.beat("summary") as b:
            self.play(FadeOut(VGroup(stu, tea, res1, m0, kab, legend)), run_time=0.8)
            full = tex(R"\mathcal{L}(\theta) = \frac{w}{\rho^2}\Big\Vert \hat D_\theta - "
                       R"\mathrm{Align}\big(D(x_{\rho};\rho) \rightarrow \hat D_\theta\big)\Big\Vert^2"
                       R" + \mathcal{L}_{\text{smooth-LDDT}}", font_size=40, t2c={TEACHER_D: TEACHER_C})
            full.move_to(keep)
            box2 = SurroundingRectangle(full, buff=0.15).set_stroke(STUDENT_C, 2)
            self.play(FadeOut(keep), FadeIn(full), ShowCreation(box2), run_time=1.2)
            b.until("The paper reports")
            rep = text("paper: alignment substantially reduces gradient variance; critical for accuracy",
                       font_size=26, color=GREY_A).next_to(box2, DOWN, buff=0.3)
            self.play(FadeIn(rep), run_time=1)
            steps = [
                R"1.\ \ \rho \ge \sigma \text{ drawn through the teacher's schedule}",
                R"2.\ \ x_{\rho} = x_0 + \rho\,\varepsilon \quad(\text{randomly rotated } x_0)",
                R"3.\ \ \text{one teacher call: } D(x_{\rho};\rho),\ \ v = (x_{\rho} - D)/\rho",
                R"4.\ \ \text{one student call } D_\theta(x_{\rho},\rho,\sigma) \text{ + one JVP along } (v, 1)",
                R"5.\ \ \text{aligned loss in structure space}",
            ]
            rows = VGroup(*[tex(s_, font_size=32, t2c={TEACHER_D: TEACHER_C, V_KEY: TEACHER_C})
                            for s_ in steps])
            rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT).next_to(rep, DOWN, buff=0.5)
            cues = ["draw two noise", "noise a training", "make one teacher", "and one student", "and take an aligned"]
            for row, cue in zip(rows, cues):
                b.until(cue)
                self.play(FadeIn(row, shift=0.2 * RIGHT), run_time=0.7)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def molecule_pair(seed=4):
        P, kinds, bonds = molecule_coords()
        rng = np.random.default_rng(seed)
        Q = P + 0.035 * rng.normal(size=P.shape)
        lig = np.array([k == "l" for k in kinds])
        Q[lig] += np.array([0.06, -0.04])
        R0 = rot2(np.radians(38))
        t0 = np.array([0.55, 0.25])
        return P, Q, kinds, bonds, R0, t0

    def clouds_pair(self, angle, shift):
        P, Q, kinds, bonds, _, _ = self.molecule_pair()
        a = molecule_mobject(P, kinds, bonds, scale=1.0, center=LEFT * 2.2, ligand_color=STUDENT_C)
        b_ = molecule_mobject(Q, kinds, bonds, scale=1.0, center=RIGHT * 2.2, ligand_color=TEACHER_C,
                              protein_color=TEACHER_C)
        la = tex(R"\hat D_\theta", font_size=36).next_to(a, DOWN, buff=0.2)
        lb = tex(R"D(x;\rho)", font_size=36, t2c={TEACHER_D: TEACHER_C}).next_to(b_, DOWN, buff=0.2)
        vs = tex(R"\longleftrightarrow", font_size=48).move_to(ORIGIN)
        return VGroup(a, b_, la, lb, vs)


# ==========================================================================
class S07_Sampling(DecafScene):
    LEVELS = [toy.SIGMA_MAX, 1.0, 0.55, 0.25, 0.08, 0.0]
    STARTS = [-2.2, -1.3, -0.45, 0.25, 1.15, 2.1]

    def construct(self):
        label = section_label("Sampling")
        plane = make_plane()
        plane.trajectories.set_stroke(opacity=0.25)
        with self.beat("chain") as b:
            self.play(FadeIn(label), FadeIn(plane), run_time=1.2)
            vlines = VGroup(*[plane.vline(s, GREY_B) for s in self.LEVELS[1:-1]])
            b.until("Deterministically")
            self.play(FadeIn(vlines), run_time=0.8)
            dots = VGroup(*[plane.dot(self.LEVELS[0], x0, RHO_C, radius=0.07) for x0 in self.STARTS])
            self.play(FadeIn(dots), run_time=0.6)
            counter = VGroup(text("network calls:", font_size=30), Integer(0, font_size=36, color=YELLOW))
            counter.arrange(RIGHT, buff=0.2).to_edge(UP, buff=0.35).shift(RIGHT * 3)
            self.play(FadeIn(counter), run_time=0.5)
            xs = list(self.STARTS)
            arcs = VGroup()
            for k, (s0, s1) in enumerate(zip(self.LEVELS[:-1], self.LEVELS[1:])):
                new_xs = [float(toy.flow(x, s0, s1)) for x in xs]
                step_arcs = VGroup(*[jump_arrow(plane.c2p(s0, a), plane.c2p(s1, c), angle=-0.5, width=2.5)
                                     for a, c in zip(xs, new_xs)])
                new_dots = VGroup(*[plane.dot(s1, c, SIG_C if s1 > 0 else STUDENT_C, radius=0.07)
                                    for c in new_xs])
                self.play(LaggedStartMap(ShowCreation, step_arcs, lag_ratio=0.1),
                          FadeIn(new_dots), counter[1].animate.set_value(k + 1), run_time=0.9)
                arcs.add(step_arcs, new_dots)
                xs = new_xs
            b.until("and in this toy")
            karras5 = toy.karras_sigmas(5)
            euler_ends = [toy.euler_path(x0, karras5)[-1] for x0 in self.STARTS]
            e_dots = VGroup(*[
                Dot(plane.c2p(0, e) + LEFT * 0.35, radius=0.075).set_fill(opacity=0).set_stroke(WARN_C, 3)
                for e in euler_ends])
            f_lab = with_bg(text("5 flow-map jumps: on the modes", font_size=26, color=STUDENT_C))
            e_lab = with_bg(text("5 Euler steps (scene 2): pulled inward", font_size=26, color=WARN_C))
            labs = VGroup(f_lab, e_lab).arrange(DOWN, buff=0.12, aligned_edge=RIGHT)
            labs.next_to(plane.c2p(0.3, plane.x_range[0]), UP, buff=0.25).align_to(plane.c2p(0.05, 0), RIGHT)
            self.play(FadeIn(e_dots), FadeIn(labs), run_time=1)

        with self.beat("gamma") as b:
            self.play(FadeOut(VGroup(arcs, dots, e_dots, labs, vlines, counter)), run_time=0.8)
            g1 = tex(R"\tilde\sigma = \sqrt{1-\gamma^2}\;\sigma", font_size=38, t2c={R"\tilde\sigma": SIG_C})
            g2 = tex(R"\tilde x = X_{\rho,\tilde\sigma}(x_{\rho})", font_size=38, t2c={R"\tilde\sigma": SIG_C})
            g3 = tex(R"x_{\sigma} = \tilde x + \gamma\,\sigma\,\varepsilon", font_size=38)
            eqs = VGroup(g1, g2, g3).arrange(RIGHT, buff=0.8).to_edge(UP, buff=0.35).shift(RIGHT * 0.9)
            rho, sig, gam, eps = 1.45, 0.8, 0.8, -1.25
            x0 = 0.9
            st = sqrt_ = np.sqrt(1 - gam**2) * sig
            xt = float(toy.flow(x0, rho, st))
            xn = xt + gam * sig * eps
            p0 = plane.dot(rho, x0, RHO_C, radius=0.09)
            p0_lab = with_bg(tex(R"x_{\rho}", font_size=30)).next_to(p0, DOWN, buff=0.1)
            v_r = plane.vline(rho, RHO_C, label_tex=R"\rho")
            v_s = plane.vline(sig, SIG_C, label_tex=R"\sigma")
            v_t = plane.vline(st, SIG_C, label_tex=R"\tilde\sigma")
            self.play(FadeIn(p0), FadeIn(p0_lab), FadeIn(v_r), FadeIn(v_s), run_time=1)
            b.until("Jump to a slightly")
            pt = plane.dot(st, xt, SIG_C, radius=0.08)
            arc = jump_arrow(p0.get_center(), pt.get_center(), angle=-0.6)
            self.play(Write(g1), FadeIn(v_t), run_time=1.2)
            self.play(Write(g2), ShowCreation(arc), FadeIn(pt), run_time=1.5)
            b.until("Then add gamma")
            pn = plane.dot(sig, xn, SIG_C, radius=0.09)
            inj = DashedLine(pt.get_center(), pn.get_center(), dash_length=0.08).set_stroke(NOISE_C, 3)
            inj_lab = with_bg(text("noise injection", font_size=24, color=NOISE_C)).next_to(inj, UP, buff=0.1)
            self.play(Write(g3), ShowCreation(inj), FadeIn(pn), FadeIn(inj_lab), run_time=1.5)
            b.until("The total noise")
            tot = tex(R"\tilde\sigma^2 + \gamma^2\sigma^2 = \sigma^2", font_size=36, t2c={R"\tilde\sigma": SIG_C})
            tot.next_to(eqs, DOWN, buff=0.3).to_edge(LEFT, buff=0.8)
            self.play(FadeIn(tot, shift=0.1 * UP), run_time=1)
            b.until("Gamma equals zero")
            c0 = tex(R"\gamma = 0:\ \text{pure flow map (deterministic)}", font_size=30)
            c1 = tex(R"\gamma = 1:\ \text{jump to clean, re-noise (multistep consistency)}", font_size=30)
            cases = VGroup(c0, c1).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            cases.next_to(eqs, DOWN, buff=0.25).to_edge(RIGHT, buff=0.5)
            cases_bg = BackgroundRectangle(cases, fill_opacity=0.85, buff=0.1)
            self.play(FadeIn(cases_bg), FadeIn(c0), run_time=0.8)
            b.until("Gamma equals one")
            self.play(FadeIn(c1), run_time=0.8)


# ==========================================================================
def reward(x):
    return np.exp(-((x - 1.25) ** 2) / (2 * 0.28**2))


class S08_Search(DecafScene):
    RHO, X0 = 1.1, 0.3

    def construct(self):
        label = section_label("Search with the flow-map look-ahead", number=3)
        plane = make_plane()
        plane.trajectories.set_stroke(opacity=0.25)
        base_u = plane.c2p(0, 0)[0] + 1.05
        xs = np.linspace(*plane.x_range, 300)
        rcurve = VMobject().set_points_as_corners(
            [[base_u + 0.75 * reward(x), plane.x_to_y(x), 0] for x in xs]).set_stroke(REWARD_C, 3)
        rbase = Line([base_u, plane.x_to_y(plane.x_range[0]), 0], [base_u, plane.x_to_y(plane.x_range[1]), 0])
        rbase.set_stroke(REWARD_C, 1.5, 0.5)
        rlab = tex("R(x)", font_size=30).set_color(REWARD_C).next_to(rcurve, UP, buff=0.1)
        rho, x0 = self.RHO, self.X0
        with self.beat("steer") as b:
            self.play(FadeIn(label), FadeIn(plane), run_time=1.2)
            b.until("Boltz one x steers")
            top = VGroup(
                text("Boltz-1x: steer with physics potentials R", font_size=30),
                text("(clashes, bond geometry, chirality)", font_size=26, color=GREY_B),
            ).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.3).shift(RIGHT * 1.5)
            self.play(FadeIn(top), ShowCreation(rbase), ShowCreation(rcurve), FadeIn(rlab), run_time=1.5)
            b.until("To score a noisy")
            p0 = plane.dot(rho, x0, RHO_C, radius=0.09)
            p0_lab = with_bg(tex(R"x_{\rho}", font_size=30)).next_to(p0, DOWN, buff=0.1)
            score = tex(R"\text{score}(x_{\rho}) = R\big(D(x_{\rho};\rho)\big)", font_size=34,
                        t2c={TEACHER_D: TEACHER_C})
            score.next_to(top, DOWN, buff=0.25)
            self.play(FadeIn(p0), FadeIn(p0_lab), Write(score), run_time=1.5)

        d_t = float(toy.denoiser(x0, rho))
        with self.beat("mean") as b:
            tline = Line(plane.c2p(rho, x0), plane.c2p(0, d_t)).set_stroke(TEACHER_C, 3)
            dT = plane.dot(0, d_t, TEACHER_C, radius=0.1)
            self.play(ShowCreation(tline), FadeIn(dT), run_time=1.2)
            b.until("At high noise")
            mean_lab = with_bg(text("posterior mean", font_size=24, color=TEACHER_C))
            mean_lab.next_to(dT, LEFT, buff=0.2).shift(DOWN * 0.15)
            r_mark = VGroup(
                DashedLine(dT.get_center(), [base_u + 0.75 * reward(d_t), dT.get_y(), 0], dash_length=0.06)
                .set_stroke(TEACHER_C, 2),
                Dot([base_u + 0.75 * reward(d_t), dT.get_y(), 0], radius=0.08, fill_color=TEACHER_C),
            )
            r_val = with_bg(tex(rf"R \approx {reward(d_t):.2f}", font_size=28).set_color(TEACHER_C))
            r_val.next_to(r_mark[1], RIGHT, buff=0.12)
            self.play(FadeIn(mean_lab), ShowCreation(r_mark[0]), FadeIn(r_mark[1]), FadeIn(r_val), run_time=1)
            b.until("The reward of the mean")
            jensen = tex(R"R\big(\mathbb{E}[x_0\mid x_{\rho}]\big)\ \neq\ \mathbb{E}\big[R(x_0)\mid x_{\rho}\big]",
                         font_size=36)
            jensen.next_to(score, DOWN, buff=0.25)
            self.play(Write(jensen), run_time=1.5)

        x_end = float(toy.flow(x0, rho, 0))
        with self.beat("lookahead") as b:
            end = plane.dot(0, x_end, STUDENT_C, radius=0.1)
            arc = jump_arrow(p0.get_center(), end.get_center(), angle=-0.5)
            self.play(ShowCreation(arc), FadeIn(end), run_time=1.5)
            r_mark2 = VGroup(
                DashedLine(end.get_center(), [base_u + 0.75 * reward(x_end), end.get_y(), 0], dash_length=0.06)
                .set_stroke(STUDENT_C, 2),
                Dot([base_u + 0.75 * reward(x_end), end.get_y(), 0], radius=0.08, fill_color=STUDENT_C),
            )
            r_val2 = with_bg(tex(rf"R \approx {reward(x_end):.2f}", font_size=28).set_color(STUDENT_C))
            r_val2.next_to(r_mark2[1], RIGHT, buff=0.12)
            look = with_bg(tex(R"X_{\rho,0}(x_{\rho})", font_size=30)).next_to(end, UP + LEFT, buff=0.25)
            self.play(ShowCreation(r_mark2[0]), FadeIn(r_mark2[1]), FadeIn(r_val2), FadeIn(look), run_time=1)
            b.until("The paper calls")
            tag = with_bg(text("exact look-ahead: one network call", font_size=28, color=STUDENT_C))
            tag.move_to(plane.c2p(0.75, -1.45))
            self.play(FadeIn(tag), run_time=0.8)
            b.until("Exact with respect")
            caveat = with_bg(text("(exact for the ODE; as good as the learned map)", font_size=24, color=GREY_B))
            caveat.next_to(tag, DOWN, buff=0.1)
            self.play(FadeIn(caveat), run_time=0.8)

        with self.beat("loop") as b:
            self.clear_all(run_time=1, keep=[label])
            names = [
                ("population of particles", R"\{x^{i}_{\rho}\}_{i=1}^{K}"),
                ("look ahead with the flow map", R"\hat x^{i}_0 = X_{\rho,0}(x^{i}_{\rho})"),
                ("refine the clean prediction", R"\hat x^{i}_0 \leftarrow \hat x^{i}_0 + \eta\,\nabla R(\hat x^{i}_0)"),
                ("re-noise to the next level", R"x^{i}_{\sigma} = \hat x^{i}_0 + \sigma\,\varepsilon^{i}\ \ (\text{or } \gamma\text{-step})"),
                ("select: reallocate compute", R"\text{keep / copy / expand particles by } R"),
            ]
            nodes = VGroup()
            for i, (t_, f_) in enumerate(names):
                t_m = text(t_, font_size=28)
                f_m = tex(f_, font_size=30)
                box = VGroup(t_m, f_m).arrange(DOWN, buff=0.12)
                rect = RoundedRectangle(width=box.get_width() + 0.5, height=box.get_height() + 0.35,
                                        corner_radius=0.12).set_stroke(GREY_B, 2)
                rect.move_to(box)
                nodes.add(VGroup(rect, box))
            angles = [90, 18, -54, -126, 162]
            for node, a in zip(nodes, angles):
                node.move_to(np.array([4.3 * np.cos(np.radians(a)), 2.3 * np.sin(np.radians(a)) - 0.35, 0]))
            arrows = VGroup(*[
                Arrow(nodes[i].get_center(), nodes[(i + 1) % 5].get_center(), buff=0.0, thickness=2.5,
                      fill_color=GREY_B)
                for i in range(5)
            ])
            for arr, i in zip(arrows, range(5)):
                start = nodes[i].get_center()
                end = nodes[(i + 1) % 5].get_center()
                d = (end - start) / np.linalg.norm(end - start)
                arr.put_start_and_end_on(start + d * 0.45 * np.linalg.norm(end - start),
                                         start + d * 0.62 * np.linalg.norm(end - start))
            nodes[1][0].set_stroke(STUDENT_C, 2.5)
            nodes[2][0].set_stroke(REWARD_C, 2.5)
            b.until("Keep a population")
            self.play(FadeIn(nodes[0]), run_time=0.8)
            for i, cue in zip(range(1, 5), ["look ahead with", "refine the clean", "re-noise to", "and reallocate"]):
                b.until(cue)
                self.play(GrowArrow(arrows[i - 1]), FadeIn(nodes[i]), run_time=0.8)
            self.play(GrowArrow(arrows[4]), run_time=0.6)
        self.loop = VGroup(nodes, arrows)

        with self.beat("rules") as b:
            self.play(self.loop.animate.scale(0.66).move_to(UP * 1.35), run_time=1)
            cols = VGroup()
            for rule, name, color in [
                ("resample ∝ exp(λ · reward gain)", "Feynman–Kac steering", TEACHER_C),
                ("UCB / UCT over a tree of particles", "Monte Carlo tree search", STUDENT_C),
                ("no reallocation, keep the best", "best-of-N", SIG_C),
            ]:
                r = text(rule, font_size=24)
                arrow = tex(R"\Downarrow", font_size=34).set_color(GREY_B)
                n = text(name, font_size=30, color=color)
                col = VGroup(r, arrow, n).arrange(DOWN, buff=0.2)
                rect = RoundedRectangle(width=max(4.1, col.get_width() + 0.35), height=col.get_height() + 0.5,
                                        corner_radius=0.12)
                rect.set_stroke(color, 2)
                col.move_to(rect)
                cols.add(VGroup(rect, col))
            cols.arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.6)
            for col, cue in zip(cols, ["Resampling in", "An upper-confidence", "Never reallocating"]):
                b.until(cue)
                self.play(FadeIn(col, shift=0.2 * UP), run_time=0.9)
            b.until("One algorithm")
            one = text("one algorithm, any compute budget", font_size=30, color=GREY_A)
            one.next_to(cols, UP, buff=0.3)
            self.play(FadeIn(one), run_time=0.8)


# ==========================================================================
RESULTS = [
    # name, success (RMSD<2 & PB-valid, %), PB-valid (%), kind
    ("Pearl (full budget)", 78.3, 98.8, "teacher"),
    ("DeCAF-Pearl", 77.0, 99.5, "decaf"),
    ("AlphaFold 3", 69.6, 84.8, "other"),
    ("Boltz-1x", 66.9, 96.6, "other"),
    ("DeCAF-Boltz", 66.8, 98.0, "decaf"),
    ("Chai-1", 62.3, 88.2, "other"),
    ("Boltz-2", 61.6, 81.6, "other"),
]


class S09_Results(DecafScene):
    def construct(self):
        label = section_label("What the paper reports")
        with self.beat("claims") as b:
            self.play(FadeIn(label), run_time=0.8)
            head = text("DeCAF-Boltz  (teacher: Boltz-1, 200 steps)", font_size=34, color=STUDENT_C)
            items = bullet_list([
                "near-parity with the teacher at ≈ 5× fewer network calls",
                "10–50 calls: beats Boltz-1x run at the same budget, on RMSD and\nphysical validity (Runs N' Poses)",
                "PoseBusters: matches full-budget Boltz-1x (600 calls) at ≈ 20× fewer calls",
            ], font_size=30, buff=0.4)
            grp = VGroup(head, items).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(UP * 0.3)
            src = text("as reported in the paper (arXiv:2606.08375)", font_size=24, color=GREY_C)
            src.to_edge(DOWN, buff=0.5)
            b.until("For Decaf Boltz")
            self.play(FadeIn(head), FadeIn(src), run_time=0.8)
            for item, cue in zip(items, ["near-parity", "At tight budgets", "And on PoseBusters"]):
                b.until(cue)
                self.play(FadeIn(item, shift=0.2 * RIGHT), run_time=0.8)

        with self.beat("benchmark") as b:
            self.play(FadeOut(VGroup(head, items, src)), run_time=0.8)
            title = text("Runs N' Poses (released after 2023), best of 5 samples", font_size=32)
            title.to_edge(UP, buff=0.9)
            sub = text("success = ligand RMSD < 2 Å  and  PoseBusters-valid", font_size=26, color=GREY_B)
            sub.next_to(title, DOWN, buff=0.15)
            self.play(FadeIn(title), run_time=0.8)
            b.until("Success means")
            self.play(FadeIn(sub), run_time=0.8)
            chart = self.bar_chart().next_to(sub, DOWN, buff=0.45)
            self.play(FadeIn(chart["frame"]), run_time=1)

        with self.beat("bars") as b:
            cues = ["The Pearl teacher", "The distilled", "Decaf Boltz scores", "level with Boltz",
                    "AlphaFold three:", "Chai one:", "Boltz two:"]
            order = [0, 1, 4, 3, 2, 5, 6]
            for idx, cue in zip(order, cues):
                b.until(cue)
                bar, val = chart["bars"][idx], chart["values"][idx]
                self.play(GrowFromEdge(bar, LEFT), FadeIn(val), run_time=0.7)

        with self.beat("honest") as b:
            b.until("Relative to its own")
            br1 = self.gap_brace(chart, 0, 1, "−1.3")
            br2 = self.gap_brace(chart, 3, 4, "−0.1")
            self.play(FadeIn(br1), run_time=0.8)
            b.until("That is the actual")
            self.play(FadeIn(br2), run_time=0.8)
            b.until("The margin over")
            dim = [m.animate.set_opacity(0.4) for i in (2, 5, 6) for m in chart["rows"][i][:3]]
            note = text("dimmed rows: the gap to them comes from the Pearl teacher, not from distillation",
                        font_size=26, color=GREY_A)
            note.next_to(chart["bars"][6], DOWN, buff=0.45).set_x(0)
            self.play(*dim, FadeIn(note), run_time=1)
            b.until("What the distilled models")
            self.play(FadeIn(chart["pb"]), run_time=1)
            self.play(*[FlashAround(chart["pb"][i], color=STUDENT_C) for i in (1, 4)], run_time=1.2)
            src = text("numbers: the authors' benchmark figure (DeCAF GitHub README)", font_size=22, color=GREY_C)
            src.to_corner(DR, buff=0.3)
            self.play(FadeIn(src), run_time=0.6)

    def bar_chart(self, width=6.4, row_h=0.56):
        rows, bars, values, pbs = VGroup(), VGroup(), VGroup(), VGroup()
        colors = {"teacher": "#4A78F0", "decaf": "#7FB0E8", "other": GREY_D}
        for i, (name, succ, pb, kind) in enumerate(RESULTS):
            y = -i * row_h
            name_m = text(name, font_size=26, color=WHITE if kind != "other" else GREY_A)
            name_m.move_to([-0.3, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=width * succ / 100, height=row_h * 0.72)
            bar.set_fill(colors[kind], 1).set_stroke(WHITE if kind == "other" else colors[kind], 1)
            bar.move_to([0, y, 0], aligned_edge=LEFT)
            val = text(f"{succ:.1f}", font_size=26).next_to(bar, RIGHT, buff=0.12)
            pb_m = text(f"{pb:.1f}", font_size=24, color=STUDENT_C if kind == "decaf" else GREY_B)
            pb_m.move_to([width + 1.3, y, 0])
            rows.add(VGroup(name_m, bar, val, pb_m))
            bars.add(bar)
            values.add(val)
            pbs.add(pb_m)
        axis = Line([0, 0.4, 0], [0, -(len(RESULTS) - 1) * row_h - 0.4, 0]).set_stroke(GREY_B, 2)
        names = VGroup(*[r[0] for r in rows])
        pb_head = text("PB-valid %", font_size=22, color=GREY_B).move_to([width + 1.3, 0.55, 0])
        succ_head = text("success %", font_size=22, color=GREY_B).move_to([width * 0.4, 0.55, 0])
        frame = VGroup(axis, names, succ_head)
        chart = VGroup(frame, bars, values, VGroup(pb_head, pbs))
        chart.move_to(ORIGIN)
        self.chart_parts = dict(frame=frame, bars=bars, values=values, pb=VGroup(pb_head, *pbs), rows=rows)
        return _ChartView(chart, self.chart_parts)

    @staticmethod
    def gap_brace(chart, i, j, label):
        top = chart["bars"][i].get_right()
        bot = chart["bars"][j].get_right()
        x = max(top[0], bot[0]) + 0.75
        line = VGroup(
            Line([top[0] + 0.62, top[1], 0], [x, top[1], 0]),
            Line([x, top[1], 0], [x, bot[1], 0]),
            Line([bot[0] + 0.62, bot[1], 0], [x, bot[1], 0]),
        ).set_stroke(YELLOW, 2)
        lab = text(label, font_size=26, color=YELLOW).next_to(line, RIGHT, buff=0.12)
        return VGroup(line, lab)


class _ChartView(VGroup):
    """A VGroup that also lets scenes look up its parts by name."""

    def __init__(self, group, parts):
        super().__init__(*group)
        self.parts = parts

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.parts[key]
        return super().__getitem__(key)


# ==========================================================================
class S10_Takeaways(DecafScene):
    def construct(self):
        label = section_label("Summary")
        with self.beat("three") as b:
            self.play(FadeIn(label), run_time=0.8)
            cards = VGroup(*[self.card(n, t, d) for n, t, d in [
                (1, "Index the flow map by noise level",
                 "removes a schedule factor spanning ~5 orders of magnitude from the objective"),
                (2, "Parameterize it as a two-level denoiser",
                 "boundary + tangent conditions built in; distillation becomes a distance between\n"
                 "structures, rigidly aligned exactly as the teacher was trained"),
                (3, "Search with the flow-map look-ahead",
                 "a sharp one-call estimate of the end point; one algorithm covers\n"
                 "FK steering, tree search and best-of-N"),
            ]]).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(DOWN * 0.1)
            for card, cue in zip(cards, ["One:", "Two:", "Three:"]):
                b.until(cue)
                self.play(FadeIn(card, shift=0.3 * RIGHT), run_time=0.9)

        with self.beat("caveats") as b:
            self.play(FadeOut(cards), run_time=0.8)
            head = text("Caveats", font_size=36, color=WARN_C)
            items = bullet_list([
                "distillation: the ceiling is the teacher",
                "savings are in the structure module; the trunk still runs once per complex,\n"
                "so end-to-end speed-ups depend on system size and number of samples",
                "training needs Jacobian-vector products through the structure module",
                "the look-ahead is only as exact as the learned map",
            ], font_size=30, buff=0.38)
            grp = VGroup(head, items).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(DOWN * 0.1)
            self.play(FadeIn(head), run_time=0.6)
            for item, cue in zip(items, ["This is distillation", "The savings", "Training needs", "And the look-ahead"]):
                b.until(cue)
                self.play(FadeIn(item, shift=0.2 * RIGHT), run_time=0.8)

        with self.beat("close", pad=1.5) as b:
            self.play(FadeOut(VGroup(head, items)), run_time=0.8)
            refs = VGroup(
                text("Few-step Cofolding with All-Atom Flow Maps", font_size=38),
                text("Scarpellini, Shprints, Holderrieth, Nam, Murugan, Gómez-Bombarelli,\n"
                     "Jaakkola, Al-Shedivat, Boffi, Bose — arXiv:2606.08375", font_size=26, color=GREY_B),
                text("code: github.com/genesistherapeutics/decaf", font_size=28, color=STUDENT_C),
                text("DeCAF-Boltz weights: huggingface.co/genesisml/decaf", font_size=28, color=STUDENT_C),
            ).arrange(DOWN, buff=0.35)
            note = text("The 1D pictures are exact computations for a two-mode Gaussian toy,\n"
                        "not outputs of the cofolding model.", font_size=24, color=GREY_C)
            note.to_edge(DOWN, buff=0.5)
            b.until("The paper is")
            self.play(FadeIn(refs[0]), FadeIn(refs[1]), run_time=1.2)
            b.until("and the code")
            self.play(FadeIn(refs[2]), FadeIn(refs[3]), FadeIn(note), run_time=1.2)

    @staticmethod
    def card(n, title, detail):
        num = Text(str(n), font=FONT, font_size=40, color=YELLOW)
        circ = Circle(radius=0.32).set_stroke(YELLOW, 2).move_to(num)
        t = Text(title, font=FONT, font_size=34)
        d = Text(detail, font=FONT, font_size=26, color=GREY_B, alignment="LEFT")
        body = VGroup(t, d).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        body.next_to(circ, RIGHT, buff=0.35)
        circ.align_to(body, UP)
        num.move_to(circ)
        return VGroup(VGroup(circ, num), body)
