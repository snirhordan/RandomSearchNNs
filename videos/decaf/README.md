# DeCAF, explained: a 3Blue1Brown-style video

An explainer video (≈ 16 min, 1080p, voice-over + soft English subtitles) of

> **Few-step Cofolding with All-Atom Flow Maps** — Scarpellini, Shprints, Holderrieth, Nam,
> Murugan, Gómez-Bombarelli, Jaakkola, Al-Shedivat, Boffi, Bose. arXiv:2606.08375 (v2).

It is aimed at a PhD student who knows diffusion / flow matching and roughly what
AlphaFold-3-style cofolding is, but not this paper. It is built with
[3b1b/manim](https://github.com/3b1b/manim) (`manimgl`), with the same look as
[3b1b/videos](https://github.com/3b1b/videos) (black background, CMU Serif, colour-coded
math). It is not affiliated with 3Blue1Brown.

## What the video covers

| # | Scene | Content |
|---|-------|---------|
| 1 | `S01_Intro` | the cost of ~200 diffusion steps; the three decisions the paper makes |
| 2 | `S02_Teacher` | EDM/Boltz teacher: VE noising, denoiser, σ-space PF-ODE; few Euler steps undershoot |
| 3 | `S03_FlowMaps` | flow map X_{ρ,σ}; identity/composition; Lagrangian vs Eulerian characterization |
| 4 | `S04_SigmaSpace` | **decision 1**: why t-indexed flow maps break with a Karras schedule (dσ/dt spans 0.24 → 15 082 Å, ×63 000) and the σ-space change of variables |
| 5 | `S05_DenoiserMap` | **decision 2**: X_{ρ,σ}(x) = (σ/ρ)x + (1−σ/ρ)D_θ(x,ρ,σ). Geometry: the teacher's denoiser is where the *tangent* hits the clean axis, the flow map's denoiser is where the *chord* does; boundary/tangent conditions for free |
| 6 | `S06_AlignedLoss` | the Eulerian (MeanFlow-type) distillation loss, rewritten as a distance between two structures, hence rigidly alignable (weighted Kabsch) exactly like the teacher's loss |
| 7 | `S07_Sampling` | chained jumps; γ-sampling |
| 8 | `S08_Search` | **decision 3**: posterior-mean look-ahead vs flow-map look-ahead; DeCAF-Search and its special cases (FK steering, MCTS, best-of-N) |
| 9 | `S09_Results` | reported results, read critically (distillation ≈ −1.3 points vs its Pearl teacher; DeCAF-Boltz ≈ Boltz-1x) |
| 10 | `S10_Takeaways` | contributions and caveats |

All 1D pictures (trajectories, tangents, chords, flow-map end points, posterior means) are
**exact computations** for a two-component Gaussian mixture (`toy.py`: closed-form denoiser,
RK4 in log σ). They illustrate the geometry; they are not outputs of the cofolding model.

## Sources, and how much to trust each statement

arxiv.org was not reachable from the environment this was produced in, so the paper's full
text could not be read directly. Statements in the video come from:

1. **The authors' released code**, [genesistherapeutics/decaf](https://github.com/genesistherapeutics/decaf)
   (`src/boltz/model/modules/decaf.py`, `decaf_sampler.py`, `models/boltz1.py`). The σ-space
   flow map, the two-noise-level denoiser parameterization, the Eulerian JVP loss with
   `(v, 1, 0)` tangent and stop-gradient, the 1/σ² weighting, the ligand-weighted rigid
   alignment + smooth-LDDT, the frozen trunk/teacher, γ-sampling, FK/MCTS samplers, and the
   Boltz-1 sampler's per-step re-alignment (`alignment_reverse_diff=True`) are all read off
   this code. The video says "in the released code" where that matters.
2. **The authors' README and figures** (same repository): the three design decisions, the
   DeCAF-Search description, the Runs N' Poses bar chart numbers (success and PB-valid for
   Pearl, DeCAF-Pearl, AF3, Boltz-1x, DeCAF-Boltz, Chai-1, Boltz-2), and "5× speed-up at
   near-parity" for DeCAF-Boltz.
3. **Search-engine excerpts of the paper's text** (abstract and result sentences): the
   numerical instability of t-parameterized objectives under the EDM schedule, "alignment is
   critical / reduces gradient variance", DeCAF-Pearl matching its teacher at 5× fewer NFEs,
   10–50 NFE comparisons against Boltz-1x, and the PoseBusters 600-NFE / ≈20× comparison.
   These are marked "the paper reports" in the narration; they are the least verified items.
4. **Derivations done for the video** (independently checkable): dσ/dt range of the Karras
   schedule for Boltz-1's σ_min = 0.0064 Å, σ_max = 2560 Å, exponent 7; the Eulerian identity
   u = v − (ρ−σ) du/dρ in σ-space; the equivalence of the velocity loss with
   ‖D̂_θ − D‖²/ρ²; the tangent/chord geometry of the denoiser parameterization.

One discrepancy worth knowing: the README text says DeCAF-Boltz beats AF3, Chai-1, Boltz-1x
and Boltz-2 by 3–15 points on Runs N' Poses, but the README's own figure shows DeCAF-Boltz at
66.8 vs AF3 69.6 and Boltz-1x 66.9; the gains over those baselines belong to DeCAF-Pearl
(77.0), whose teacher (Pearl, 78.3) is Genesis's proprietary model. The video follows the
figure.

## Rebuilding

Requirements: Python ≥ 3.10, `manimgl==1.7.2`, LaTeX (`texlive-latex-extra`,
`texlive-fonts-extra`, `texlive-science`, `tipa`, `dvisvgm`), `ffmpeg`, `fonts-cmu`, an X
server (or `xvfb-run`), and for the voice-over `kokoro-onnx` + `soundfile` with the Kokoro
v1.0 model files from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases).

```bash
cd videos/decaf
python tts.py --model kokoro-v1.0.onnx --voices voices-v1.0.bin   # sounds/*.wav + durations.json
python check_cues.py scenes_a.py scenes_b.py                     # every animation cue exists in the narration
for s in S01_Intro S02_Teacher S03_FlowMaps S04_SigmaSpace S05_DenoiserMap; do ./render.sh scenes_a.py $s --hd; done
for s in S06_AlignedLoss S07_Sampling S08_Search S09_Results S10_Takeaways; do ./render.sh scenes_b.py $s --hd; done
python assemble.py                                               # build/DeCAF_explained.mp4 + .srt
```

Files: `narration.py` (the script, one entry per beat), `common.py` (palette, voice-over
timing, the σ–x plane), `scenes_a.py` / `scenes_b.py` (scenes), `toy.py` (exact 1D model),
`tts.py`, `assemble.py`, `render.sh`, `contact.sh` (frame contact sheets for review).
Each animation is placed at the moment the narrator reaches a given phrase (`Beat.until`),
using a per-character speaking-time model fitted to the synthesized clips.
