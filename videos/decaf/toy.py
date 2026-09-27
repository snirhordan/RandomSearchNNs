"""Exact 1D toy model used for every (sigma, x) picture in the video.

Data: a two-component Gaussian mixture in one dimension.  The forward process
is variance exploding, x_sigma = x_0 + sigma * eps, exactly as in EDM / Boltz,
so the ideal denoiser D(x; sigma) = E[x_0 | x_sigma = x] has a closed form and
the probability-flow ODE  dx/dsigma = (x - D(x; sigma)) / sigma  can be
integrated to high accuracy.  Nothing in the pictures is hand-drawn: tangent
lines, chords, flow-map endpoints and denoiser outputs are all computed here.
"""
from __future__ import annotations

import numpy as np

MUS = np.array([-1.2, 1.2])
STDS = np.array([0.2, 0.2])
WEIGHTS = np.array([0.5, 0.5])
SIGMA_MAX = 1.6


def _component_posteriors(x, sigma):
    x = np.asarray(x, dtype=float)
    var = STDS**2 + sigma**2
    logp = (
        np.log(WEIGHTS)
        - 0.5 * np.log(2 * np.pi * var)
        - 0.5 * (x[..., None] - MUS) ** 2 / var
    )
    logp -= logp.max(axis=-1, keepdims=True)
    w = np.exp(logp)
    return w / w.sum(axis=-1, keepdims=True), var


def density(x, sigma):
    x = np.asarray(x, dtype=float)
    var = STDS**2 + sigma**2
    comps = WEIGHTS * np.exp(-0.5 * (x[..., None] - MUS) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return comps.sum(axis=-1)


def denoiser(x, sigma):
    """Posterior mean E[x_0 | x_sigma = x] (the ideal teacher D(x; sigma))."""
    w, var = _component_posteriors(x, sigma)
    shrink = STDS**2 / var
    comp_means = MUS + shrink * (np.asarray(x, dtype=float)[..., None] - MUS)
    return (w * comp_means).sum(axis=-1)


def velocity(x, sigma):
    """sigma-velocity of the PF-ODE: dx/dsigma = (x - D(x; sigma)) / sigma."""
    return (x - denoiser(x, sigma)) / sigma


def flow(x, rho, sigma, n=400):
    """Exact flow map X_{rho -> sigma}(x), integrated with RK4 in log(sigma)."""
    x = np.asarray(x, dtype=float)
    if sigma == rho:
        return x.copy()
    s_end = max(sigma, 1e-5)
    lam = np.linspace(np.log(rho), np.log(s_end), n + 1)
    f = lambda xx, l: xx - denoiser(xx, np.exp(l))  # dx/dlog(sigma)
    for l0, l1 in zip(lam[:-1], lam[1:]):
        h = l1 - l0
        k1 = f(x, l0)
        k2 = f(x + 0.5 * h * k1, l0 + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, l0 + 0.5 * h)
        k4 = f(x + h * k3, l1)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if sigma == 0:
        x = denoiser(x, s_end)
    return x


def trajectory(x_start, rho=SIGMA_MAX, n=240, sigma_end=0.0):
    """Samples (sigma_i, x_i) along the ODE trajectory from (rho, x_start)."""
    sigmas = np.concatenate([np.geomspace(rho, 2e-3, n), [0.0]])
    sigmas = sigmas[sigmas >= sigma_end]
    xs = [float(x_start)]
    for s0, s1 in zip(sigmas[:-1], sigmas[1:]):
        xs.append(float(flow(xs[-1], s0, s1, n=8)))
    return sigmas, np.array(xs)


def flowmap_denoiser(x, rho, sigma):
    """The D_theta(x, rho, sigma) an exact DeCAF flow map must output.

    Solves X = (sigma/rho) x + (1 - sigma/rho) D for D; at sigma = rho it is
    the tangent-line limit, i.e. the teacher's denoiser.
    """
    if np.isclose(sigma, rho):
        return denoiser(x, rho)
    X = flow(x, rho, sigma)
    return (X - (sigma / rho) * x) / (1 - sigma / rho)


def karras_sigmas(n, sigma_max=SIGMA_MAX, sigma_min=2e-3, p=7.0):
    i = np.arange(n) / max(n - 1, 1)
    s = (sigma_max ** (1 / p) + i * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
    return np.concatenate([s, [0.0]])


def euler_path(x_start, sigmas):
    """Plain Euler / DDIM in sigma: one teacher call per step."""
    xs = [float(x_start)]
    for s0, s1 in zip(sigmas[:-1], sigmas[1:]):
        x = xs[-1]
        xs.append(float(x + (s1 - s0) * velocity(x, s0)))
    return np.array(xs)
