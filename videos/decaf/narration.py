"""Voice-over script for the DeCAF explainer, split into beats.

Each scene class in ``scenes.py`` plays its beats in order.  The spoken text is
written for the TTS engine (e.g. "Oilerian" so that it is pronounced like
"Eulerian"); ``caption()`` maps it back to normal spelling for the subtitles.

Sources for every factual statement are listed in README.md.  Numbers quoted
from the paper are marked "the paper reports" in the narration.
"""

SCENES = {
    "S01_Intro": [
        ("title",
         "This video is about one paper: Few-step co-folding with all-atom flow maps, "
         "from Genesis Molecular AI, with collaborators at MIT, Carnegie Mellon, "
         "Imperial College and Mila. The method is called Decaf."),
        ("cost",
         "Co-folding models, like AlphaFold three or Boltz, predict a protein together "
         "with the small molecule bound to it. Their last stage is a diffusion model over "
         "the coordinates of every atom, and producing a single structure means running "
         "that diffusion module about two hundred times."),
        ("jump",
         "Decaf distills such a model into a flow map: a network that, instead of walking "
         "along the denoising trajectory in small steps, jumps along it. The same network "
         "can take one jump, or five, or ten."),
        ("plan",
         "Flow maps themselves are not new. The contribution is making them work for an "
         "all-atom teacher of the E D M kind, and it comes down to three decisions. "
         "First, index the flow map by noise level instead of time. Second, parameterize "
         "it as a denoiser, so that the training loss compares structures and can be "
         "rigidly aligned, exactly like the teacher's own loss. Third, use the map's cheap "
         "look-ahead to guide search at inference time. Let's take them in order."),
    ],
    "S02_Teacher": [
        ("noising",
         "First, the teacher. Stack all the atom coordinates into one long vector, x. "
         "The noising process is variance exploding: x sigma equals the clean structure, "
         "x zero, plus sigma times standard Gaussian noise. In Boltz, sigma ranges from "
         "under a hundredth of an angstrom to about two and a half thousand angstroms."),
        ("denoiser",
         "The network is a denoiser, D of x and sigma, which predicts the clean structure. "
         "Ideally, it is the posterior mean: the expected clean structure, given the noisy one."),
        ("ode",
         "Sampling integrates the probability flow O D E, or a stochastic version of it. "
         "Written directly in terms of the noise level, the O D E is remarkably simple: "
         "d x, d sigma, equals x minus D, over sigma. "
         "We'll call the right-hand side the sigma velocity, v."),
        ("picture",
         "Here is a picture we'll use throughout. The horizontal axis is the noise level: "
         "pure noise on the left, clean data on the right. The vertical axis is a "
         "one-dimensional stand-in for the structure. The data distribution has two modes; "
         "think of two different binding poses. Every curve is an exact trajectory of the "
         "O D E, carrying a point from the broad Gaussian on the left to one of the modes "
         "on the right."),
        ("euler",
         "A standard sampler walks along a trajectory with small Euler steps, one network "
         "call per step. With many steps, it tracks the curve. With only five, each step "
         "follows the local tangent, the curvature makes it undershoot, and it ends up "
         "between the modes, where there is no data at all."),
    ],
    "S03_FlowMaps": [
        ("define",
         "A flow map skips the walking. Capital X, sub rho sigma, takes a point at noise "
         "level rho, and returns where its trajectory is at the lower noise level sigma. "
         "It is the solution operator of the O D E."),
        ("props",
         "Two properties pin it down. Mapping a level to itself does nothing. And maps "
         "compose: going from rho to sigma and then on to tau is the same as going from rho "
         "to tau directly. A consistency model is the special case that only ever jumps to "
         "zero. A flow map can jump between any two levels, so one network supports any "
         "number of steps."),
        ("lagrange",
         "To learn it, you need a differential equation that it satisfies. The Lagrangian "
         "view moves the end point: the derivative of X with respect to the target level is "
         "the velocity at the end point. The Oilerian view moves the starting point instead."),
        ("euler_eq",
         "Slide the start along its own trajectory, and the destination does not move. "
         "So the derivative with respect to rho, taken along the velocity field, is zero. "
         "This is the equation the released Decaf code enforces, with the teacher's velocity "
         "as the field."),
        ("segue",
         "Before that can work, there is a problem that looks like bookkeeping, but turns "
         "out to matter a lot."),
    ],
    "S04_SigmaSpace": [
        ("schedule",
         "Generic flow map recipes are written in a time variable, t, between zero and one. "
         "The teacher lives in noise level, and the two are connected by the schedule it was "
         "trained with: a Karras schedule, which interpolates linearly in sigma to the power "
         "one seventh."),
        ("chain",
         "Changing variables brings in the chain rule. The velocity in t is the sigma "
         "velocity times d sigma, d t. For this schedule, d sigma d t is about a quarter of "
         "an angstrom per unit time at the clean end, and about fifteen thousand at the noisy "
         "end. That is a factor of sixty thousand, almost five orders of magnitude."),
        ("problem",
         "Now look at what the Oilerian objective needs. It regresses onto the teacher's "
         "velocity, and it differentiates the network along the trajectory, which is a "
         "Jacobian-vector product whose tangent is that same velocity. In time coordinates, "
         "both carry the factor d sigma d t. Depending on where the pair of times lands, the "
         "targets and the derivatives change scale by up to that factor of sixty thousand. "
         "The paper reports that time-parameterized objectives become numerically unstable "
         "because of it."),
        ("fix",
         "The fix is a change of variables. Index the flow map by noise levels directly, and "
         "write every objective and every sampling step in sigma. The sigma velocity is just "
         "the teacher's noise prediction, so it is of order one at every noise level. The "
         "tangent of the Jacobian-vector product is simply v, and one. The factor d sigma d t "
         "never appears. The schedule survives only as a choice of which noise levels to "
         "train on and to step through."),
    ],
    "S05_DenoiserMap": [
        ("ddim",
         "Second decision: how to parameterize the map. Start from the simplest possible "
         "jump, one Euler step of the O D E from rho down to sigma. For this process, that "
         "is exactly a D D I M step. Rearranged, the new point is a weighted average: sigma "
         "over rho times the current point, plus one minus sigma over rho, times the "
         "denoised estimate."),
        ("tangent",
         "In our picture, this has a clean geometric meaning. An Euler step moves along the "
         "tangent line of the trajectory. And the tangent line at level rho crosses the clean "
         "axis, sigma equals zero, exactly at the denoiser's output. The teacher's denoiser "
         "is where the tangent line lands."),
        ("decaf_form",
         "Decaf keeps this form, but lets the denoiser know where the jump is going. "
         "X rho sigma of x equals sigma over rho times x, plus one minus sigma over rho, "
         "times D theta of x, rho, and sigma: a denoiser that takes two noise levels."),
        ("chord",
         "Geometrically, the exact flow map lands on the trajectory, not on the tangent. "
         "Draw the chord from the starting point through that landing point, and extend it "
         "to the clean axis. Where it crosses is exactly what D theta must output. The "
         "tangent gives the teacher's denoiser. The chord gives the flow map's denoiser."),
        ("free",
         "Three properties come for free. When sigma equals rho, the map is the identity, "
         "whatever the network outputs. As sigma approaches rho, the chord turns into the "
         "tangent, so the diagonal, D theta of x, rho, rho, must equal the teacher's "
         "denoiser. That is the tangent condition, and training it is ordinary denoiser "
         "distillation. And at sigma equals zero, the chord ends at the trajectory's end "
         "point. So a single network call, from any noise level, returns a sharp clean "
         "structure rather than a posterior mean."),
        ("arch",
         "In the released code, the student has the same architecture as the teacher's "
         "diffusion module, plus a second noise-level embedding. The Pairformer trunk and "
         "the teacher are frozen, and only this head is trained."),
    ],
    "S06_AlignedLoss": [
        ("identity",
         "Now the loss. Write the jump with an average velocity, u. X equals x, minus rho "
         "minus sigma, times u, where u is x minus D theta, over rho. Substituting into the "
         "Oilerian equation gives an identity that the exact map satisfies: u equals v, "
         "minus rho minus sigma, times the derivative of u along the trajectory. That "
         "derivative costs one Jacobian-vector product, in the direction v in space, and "
         "one in rho."),
        ("loss",
         "So the student regresses u, plus rho minus sigma times the stop-gradient "
         "derivative, onto the teacher's velocity. On the diagonal, where sigma equals rho, "
         "the correction vanishes, and this is plain denoiser distillation."),
        ("structures",
         "Here is the key step. Multiply the residual by rho. Both sides turn into clean "
         "structures. On one side, the student's clean prediction, corrected by the "
         "derivative term; call it D hat. On the other side, the teacher's denoised "
         "structure. The loss becomes a distance between two point clouds, weighted by one "
         "over rho squared."),
        ("why",
         "Why does that matter? AlphaFold-style networks are not built to be equivariant. "
         "They are trained with random rotations, and with a loss that first rigidly aligns "
         "the ground truth onto the prediction. So nothing in the teacher's training loss "
         "depends on the global orientation of its output. Boltz's own sampler even "
         "re-aligns the noisy coordinates to the denoised prediction before every step."),
        ("kabsch",
         "A velocity-matching loss would ask the student to reproduce that arbitrary pose. "
         "Those residuals can be large and systematic, and they say nothing about the "
         "structure. In structure space, you can "
         "do what the teacher's loss did: find the best rigid motion with the Kabsch "
         "algorithm, weighted toward the ligand atoms, align the teacher's structure onto "
         "the student's, and only then take the difference. What remains is the part that "
         "matters."),
        ("summary",
         "The paper reports that this alignment substantially reduces gradient variance, "
         "and that it is critical for accuracy. The teacher's smooth L D D T loss carries "
         "over unchanged, too. So one training step is: draw two noise levels from the "
         "teacher's schedule, noise a training structure, make one teacher call and one "
         "student call with its Jacobian-vector product, and take an aligned loss in "
         "structure space."),
    ],
    "S07_Sampling": [
        ("chain",
         "Sampling with the trained map is simple. Deterministically, you chain jumps along "
         "a short schedule of noise levels. Five jumps, five network calls, and in this toy, "
         "with an exact map, the end points land on the data, not between the modes."),
        ("gamma",
         "Decaf also uses gamma sampling, a knob between determinism and fresh noise. Jump "
         "to a slightly lower level, sigma tilde, equal to root one minus gamma squared, "
         "times sigma. Then add gamma sigma worth of Gaussian noise. The total noise level "
         "is back to sigma. Gamma equals zero is the pure flow map. Gamma equals one jumps "
         "all the way to a clean structure and re-noises, like multistep consistency "
         "sampling."),
    ],
    "S08_Search": [
        ("steer",
         "Third decision: search at inference time. Boltz one x steers sampling with "
         "physics-based potentials, penalizing clashes, bad bond geometry and wrong "
         "chirality. To score a noisy intermediate state, the potential is evaluated on the "
         "denoiser's prediction."),
        ("mean",
         "But a denoiser returns a posterior mean. At high noise, that is an average over "
         "very different outcomes, and it can sit between the modes, where no real "
         "structure lives. The reward of the mean is not the mean reward."),
        ("lookahead",
         "The flow map's full jump lands where the trajectory actually ends: a concrete "
         "structure, for the same single network call. The paper calls this an exact "
         "look-ahead. Exact with respect to the probability flow, and only as good as the "
         "learned map."),
        ("loop",
         "Decaf Search is built around that look-ahead. Keep a population of particles. At "
         "each noise level, look ahead with the flow map, refine the clean prediction with "
         "reward gradients, re-noise to the next level, and reallocate compute with a "
         "selection rule."),
        ("rules",
         "The choice of rule recovers familiar methods. Resampling in proportion to "
         "exponentiated reward gives Feynman-Kats steering. An upper-confidence rule over a "
         "tree gives Monte Carlo tree search. Never reallocating, and keeping the best at "
         "the end, gives best-of-N. One algorithm, spanning every compute budget."),
    ],
    "S09_Results": [
        ("claims",
         "What does this buy? For Decaf Boltz, distilled from Boltz one, the paper reports "
         "near-parity with the teacher at about five times fewer network evaluations. At "
         "tight budgets of ten to fifty evaluations, it beats Boltz one x run at the same "
         "budget, on both accuracy and physical validity. And on PoseBusters, it matches the "
         "full-budget, six-hundred-evaluation Boltz one x, with roughly twenty times less "
         "compute."),
        ("benchmark",
         "Here is the headline benchmark: Runs and Poses, complexes released after 2023, "
         "best of five samples. Success means a ligand R M S D under two angstroms, and "
         "passing the PoseBusters validity checks."),
        ("bars",
         "The Pearl teacher, at full budget, scores seventy-eight point three. The distilled "
         "Decaf Pearl scores seventy-seven. Decaf Boltz scores sixty-six point eight, level "
         "with Boltz one x at sixty-six point nine. AlphaFold three: sixty-nine point six. "
         "Chai one: sixty-two point three. Boltz two: sixty-one point six."),
        ("honest",
         "Read this carefully. Relative to its own teacher, distillation cost about one "
         "point, at a fraction of the compute. That is the actual result. The margin over "
         "AlphaFold three, Chai one and Boltz two comes from the teacher, Pearl, which is "
         "Genesis's own model; it is not created by distillation. Decaf Boltz sits level "
         "with Boltz one x. What the distilled models do show is high physical validity: "
         "ninety-eight to ninety-nine and a half percent."),
    ],
    "S10_Takeaways": [
        ("three",
         "To summarize. One: index the flow map by noise level. This removes a "
         "schedule-induced factor spanning five orders of magnitude from the training "
         "objective. Two: parameterize the map as a two-level denoiser. This builds in the "
         "boundary and tangent conditions, and turns distillation into a distance between "
         "structures, which can be rigidly aligned exactly the way the teacher was trained. "
         "Three: use the flow map's look-ahead for reward-guided search, which unifies "
         "steering, tree search and best-of-N."),
        ("caveats",
         "And the caveats. This is distillation, so the ceiling is the teacher. The savings "
         "are in the structure module: the trunk still runs once per complex, so end-to-end "
         "speedups depend on system size and on how many samples you draw. Training needs "
         "Jacobian-vector products through the structure module. And the look-ahead is only "
         "as exact as the learned map."),
        ("close",
         "Within those limits, it is a clean example of a general idea, flow maps, adapted "
         "to a domain with its own geometry. The paper is arXiv twenty-six oh six, point, "
         "oh eight three seven five, and the code and the Decaf Boltz weights are public."),
    ],
}

# Spoken spelling -> written spelling, for subtitles.
_CAPTION_FIXES = [
    ("Decaf Search", "DeCAF-Search"),
    ("Decaf Boltz", "DeCAF-Boltz"),
    ("Decaf Pearl", "DeCAF-Pearl"),
    ("Decaf", "DeCAF"),
    ("Oilerian", "Eulerian"),
    ("co-folding", "cofolding"),
    ("Co-folding", "Cofolding"),
    ("Feynman-Kats", "Feynman-Kac"),
    ("Runs and Poses", "Runs N' Poses"),
    ("E D M", "EDM"),
    ("O D E", "ODE"),
    ("D D I M", "DDIM"),
    ("L D D T", "LDDT"),
    ("R M S D", "RMSD"),
    ("Boltz one x", "Boltz-1x"),
    ("Boltz one", "Boltz-1"),
    ("Boltz two", "Boltz-2"),
    ("Chai one", "Chai-1"),
    ("AlphaFold three", "AlphaFold 3"),
    ("arXiv twenty-six oh six, point, oh eight three seven five", "arXiv 2606.08375"),
]


def caption(text: str) -> str:
    for spoken, written in _CAPTION_FIXES:
        text = text.replace(spoken, written)
    return text


def word_count() -> int:
    return sum(len(t.split()) for beats in SCENES.values() for _, t in beats)


if __name__ == "__main__":
    print(word_count(), "words")
