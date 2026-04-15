#!/usr/bin/env python3
"""
QPO_TripleA.py  —  Analytic-gradient Adaptive Amplitude Ascent
===============================================================
Custom Whittle-likelihood optimiser for the QPO fitting pipeline.

Replaces the Stingray / Powell / Nelder-Mead optimisation path with
L-BFGS-B driven by an analytically computed gradient.  All upstream
pipeline logic (IC model selection, guardrails, multi-track fitting,
component seeding, struct I/O) is untouched.

Why the standard methods struggle
----------------------------------
Powell and Nelder-Mead are gradient-free.  The Whittle objective

    L(θ) = Σᵢ  [ log M(fᵢ; θ)  +  P(fᵢ) / M(fᵢ; θ) ]

has three features that defeat them:

  1. Hard const-amplitude ridges: a broad Lorentzian and the white-noise
     constant can exchange power along a nearly flat valley.  Powell slides
     along the ridge indefinitely.

  2. Discontinuous prior cliffs: Stingray encodes bounds as log-likelihood
     penalties.  At a cliff, prior(x) → 0, so log-posterior → -∞.  Gradient-
     free methods probe outside physical bounds, receive -∞, and stall.

  3. Simplex degeneracy (Nelder-Mead): above ~6 dimensions the simplex
     collapses onto a subspace and declares a false convergence.

TripleA fixes all three simultaneously:

  a. Log-reparameterisation of amp and fwhm.
     · amp_k  = exp(θ[3k]),   fwhm_k = exp(θ[3k+2]),  ν₀_k = θ[3k+1]  (linear)
     · Positivity is automatic — no penalty needed.
     · Log-uniform priors on amp/fwhm become flat priors in θ-space.
     · Parameter scales become comparable, which is essential for the
       L-BFGS-B Hessian approximation to work.

  b. Analytical gradient — no finite-difference cost.

  c. True box constraints passed to L-BFGS-B — no penalty cliffs.

Gradient derivation
--------------------
M_i  = C  +  Σ_k  L_{k,i}
L_{k,i} = A_k · g_k² / d_{k,i}    where  d_{k,i} = (fᵢ − ν₀_k)² + g_k²,  g_k = fwhm_k/2
cf_i = ∂L/∂M_i = (M_i − P_i) / M_i²

∂L/∂ log_A_k    = Σᵢ  cf_i · L_{k,i}
∂L/∂ ν₀_k       = Σᵢ  cf_i · 2(fᵢ − ν₀_k) · L_{k,i} / d_{k,i}
∂L/∂ log_fwhm_k  = Σᵢ  cf_i · 2(fᵢ − ν₀_k)² · L_{k,i} / d_{k,i}
∂L/∂ log_C      = C · Σᵢ  cf_i

All four are O(n_freq) vectorised dot-products; the entire gradient evaluation
is faster than a single finite-difference step with Powell.

Public API
----------
tripleA_fit_once(ps, *, nlor, t0, include_const,
                 x0_lims, fwhm_lims, amp_caps,
                 amp_lo_list=None, const_max, eps_amp=1e-30)
    → (TripleAResult | None,  error_str | None)

    Drop-in for _try_fit_once when fitmethod == "TripleA".
    ps.freq / ps.power are used directly; ps.m is ignored (m-averaging
    does not change the argmin of the per-bin Whittle objective).

Integration with the pipeline (minimal diff)
--------------------------------------------
QPO_Parameter.py:
    FIT_METHOD = "TripleA"
    + AAA_* parameter block (see below)

QPO_fit.py:
    + from QPO_TripleA import tripleA_fit_once as _tripleA_fit_once
    + 6-line branch in _try_fit_once
    + 5-line _aaa_kw dict in _run_stage  (passed as **_aaa_kw)

QPO_interactive.py:
    + from QPO_TripleA import tripleA_fit_once as _tripleA_fit_once
    + 12-line short-circuit block in _run_direct_fit (before the Stingray loop)

Runtime-tunable knobs (QPO_Parameter.py)
-----------------------------------------
AAA_N_STARTS         = 5      Multi-start count.  TripleA manages its own
                               jitter; each start is cheap (~50-200 gradient
                               evaluations vs ~1000-5000 for Powell).
AAA_JITTER_STD_LOG   = 0.30   Std of log-space perturbation for amp/fwhm.
                               0.30 ≈ ±35 % amplitude, stays well within bounds.
AAA_JITTER_STD_NU0   = 0.10   Std for ν₀ as fraction of the x0 range.
AAA_FTOL             = 1e-11  L-BFGS-B function-value convergence tolerance.
                               Tighter than scipy default (2.22e-9).
AAA_GTOL             = 1e-7   Gradient-norm convergence tolerance.
AAA_MAXITER          = 1000   Max L-BFGS-B iterations per start.

Validation / debugging
-----------------------
To verify the gradient numerically against finite differences:

    from scipy.optimize import check_grad
    from QPO_TripleA import _whittle_loss_and_grad, _pack_theta, _build_bounds

    err = check_grad(
        lambda t: _whittle_loss_and_grad(t, freq, power, nlor, True)[0],
        lambda t: _whittle_loss_and_grad(t, freq, power, nlor, True)[1],
        theta0,
    )
    print(f"gradient error: {err:.3e}")   # should be < 1e-5

Typical values are 1e-7 to 1e-9, confirming the derivation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

import QPO_Parameter as P

# Floor applied to M(f) before log and division to prevent log(0) / 1/0.
# Set lower than any physically meaningful power level.
_EPS_M: float = 1e-300


# ============================================================================
# Result container
# ============================================================================

@dataclass
class TripleAResult:
    """
    Mimics the subset of the Stingray OptimizationResults that _wrap reads.

    Attribute names match what QPOFitter._wrap accesses via getattr:
        p_opt    — parameter vector in Stingray order [amp_0, x0_0, fwhm_0, ..., const]
        mfit     — model evaluated at ps.freq
        aic      — Akaike information criterion  (deviance + 2·npar)
        bic      — Bayesian information criterion (deviance + npar·ln N)
        deviance — 2 × Whittle NLL  (consistent with Stingray convention)
    """
    p_opt:    np.ndarray
    mfit:     np.ndarray
    aic:      float
    bic:      float
    deviance: float
    p_cov:    Optional[np.ndarray] = None  # linear-space covariance [amp,nu0,fwhm,...,const]


# ============================================================================
# Parameter packing / unpacking
# ============================================================================

def _pack_theta(
    t0:           np.ndarray,
    nlor:         int,
    include_const: bool,
    eps_amp:      float = 1e-30,
) -> np.ndarray:
    """
    Convert linear Stingray t0 to log-space theta.

    Stingray order: [amp_0, x0_0, fwhm_0,  amp_1, x0_1, fwhm_1, ...,  const]
    theta order:  [log_amp_0, x0_0, log_fwhm_0,  ...,  log_const]

    x0 / ν₀ stays in linear Hz; amp and fwhm are log-transformed.
    """
    t     = np.asarray(t0, float)
    theta = t.copy()
    for k in range(nlor):
        theta[3 * k]     = np.log(max(float(t[3 * k]),     eps_amp))
        # theta[3*k+1]   = t[3*k+1]   — x0 unchanged
        theta[3 * k + 2] = np.log(max(float(t[3 * k + 2]), eps_amp))
    if include_const:
        theta[-1] = np.log(max(float(t[-1]), eps_amp))
    return theta


def _unpack_theta(
    theta:         np.ndarray,
    nlor:          int,
    include_const: bool,
) -> np.ndarray:
    """
    Convert log-space theta back to linear Stingray order.
    Inverse of _pack_theta; safe for any finite theta.
    """
    t = theta.copy()
    for k in range(nlor):
        t[3 * k]     = np.exp(theta[3 * k])       # amp
        # t[3*k+1]   = theta[3*k+1]   — x0 unchanged
        t[3 * k + 2] = np.exp(theta[3 * k + 2])   # fwhm
    if include_const:
        t[-1] = np.exp(theta[-1])
    return t


# ============================================================================
# Bounds construction
# ============================================================================

def _build_bounds(
    nlor:          int,
    x0_lims:       List[Tuple[float, float]],
    fwhm_lims:     List[Tuple[float, float]],
    amp_caps:      List[float],
    amp_lo_list:   Optional[List[float]],
    include_const: bool,
    const_max:     float,
    eps_amp:       float = 1e-30,
) -> List[Tuple[float, float]]:
    """
    Build L-BFGS-B bounds in log-space theta ordering.

    amp and fwhm bounds are log-transformed; x0 bounds are kept linear.
    These replace the prior-penalty approach entirely — L-BFGS-B enforces
    them as hard constraints, so no boundary probing occurs.

    Parameters
    ----------
    amp_lo_list : lower amplitude bound per component.
        None → eps_amp for all (standard case).
        Set per-component to a tight window around the current value
        for frozen amplitude components (interactive fitter).
    """
    if amp_lo_list is None:
        amp_lo_list = [eps_amp] * nlor

    bounds: List[Tuple[float, float]] = []

    for k in range(nlor):
        a_lo = max(float(amp_lo_list[k]), eps_amp)
        a_hi = max(float(amp_caps[k]),    a_lo * 2.0)
        x_lo = float(x0_lims[k][0])
        x_hi = float(x0_lims[k][1])
        f_lo = max(float(fwhm_lims[k][0]), eps_amp)
        f_hi = max(float(fwhm_lims[k][1]), f_lo * 2.0)

        bounds.append((np.log(a_lo), np.log(a_hi)))   # log_amp
        bounds.append((x_lo, x_hi))                    # x0  (linear)
        bounds.append((np.log(f_lo), np.log(f_hi)))   # log_fwhm

    if include_const:
        c_lo = eps_amp
        c_hi = max(float(const_max), c_lo * 2.0)
        bounds.append((np.log(c_lo), np.log(c_hi)))

    return bounds


# ============================================================================
# Whittle loss + analytical gradient
# ============================================================================

def _whittle_loss_and_grad(
    theta:         np.ndarray,
    freq:          np.ndarray,
    power:         np.ndarray,
    nlor:          int,
    include_const: bool,
) -> Tuple[float, np.ndarray]:
    """
    Whittle negative log-likelihood and its exact gradient w.r.t. theta.

    L(θ) = Σᵢ [ log M_i + P_i / M_i ]

    where M_i = C + Σ_k L_{k,i},  L_{k,i} = A_k g_k² / d_{k,i},
    d_{k,i} = (fᵢ − ν₀_k)² + g_k²,  g_k = fwhm_k / 2.

    Returns (loss, grad) in the format expected by scipy.optimize.minimize
    with jac=True — i.e. a single callable that returns both values,
    avoiding redundant model evaluations.

    Numerics
    --------
    M is floored at _EPS_M before log and division.  This is safe because
    the optimiser never needs to visit the M→0 region (it is guarded by
    log-amp bounds from below), but avoids NaN propagation from any
    floating-point underflow.
    """
    # ---- Unpack theta -------------------------------------------------------
    # Index mapping (same positions as Stingray, amp/fwhm are logged):
    #   theta[3k]   = log_amp_k
    #   theta[3k+1] = nu0_k   (linear Hz)
    #   theta[3k+2] = log_fwhm_k
    #   theta[-1]   = log_C   (if include_const)

    log_amps = theta[0::3][:nlor]            # (nlor,)
    nu0s     = theta[1::3][:nlor]            # (nlor,)
    log_fwhm = theta[2::3][:nlor]            # (nlor,)

    amps = np.exp(log_amps)                  # (nlor,)
    fwhm = np.exp(log_fwhm)                  # (nlor,)
    g    = 0.5 * fwhm                        # half-widths (nlor,)

    C = np.exp(float(theta[-1])) if include_const else 0.0

    # ---- Build per-component arrays (nlor, n_freq) -------------------------
    # Broadcasting: freq is (n,), nu0s is (nlor,)
    delta = freq[np.newaxis, :] - nu0s[:, np.newaxis]   # (nlor, n)  = fᵢ − ν₀_k
    g2    = (g ** 2)[:, np.newaxis]                       # (nlor, 1)
    d     = delta ** 2 + g2                               # (nlor, n)  = d_{k,i}

    # L_{k,i} = A_k · g_k² / d_{k,i}
    L = amps[:, np.newaxis] * g2 / d                     # (nlor, n)

    # ---- Total model and Whittle loss --------------------------------------
    M = C + L.sum(axis=0)                                # (n,)
    M = np.maximum(M, _EPS_M)

    ratio = power / M                                     # P_i / M_i
    loss  = float(np.sum(np.log(M) + ratio))

    if not np.isfinite(loss):
        # Return a large but finite value so L-BFGS-B does not crash.
        # Zero gradient points away from this degenerate region.
        return 1e300, np.zeros_like(theta)

    # ---- Common factor: cf_i = (M_i − P_i) / M_i² ------------------------
    cf = (M - power) / (M * M)                           # (n,)

    # ---- Gradient -----------------------------------------------------------
    grad = np.zeros_like(theta)

    for k in range(nlor):
        Lk  = L[k]           # (n,)
        dk  = d[k]           # (n,)  = d_{k,i}
        dlt = delta[k]       # (n,)  = fᵢ − ν₀_k

        # ∂L/∂ log_A_k = Σᵢ cf_i · L_{k,i}
        # (chain rule: ∂A_k/∂ log_A_k = A_k;  ∂L_{k,i}/∂A_k = L_{k,i}/A_k)
        grad[3 * k] = np.dot(cf, Lk)

        # ∂L/∂ ν₀_k = Σᵢ cf_i · 2(fᵢ−ν₀_k) · L_{k,i} / d_{k,i}
        grad[3 * k + 1] = np.dot(cf, (2.0 * dlt / dk) * Lk)

        # ∂L/∂ log_fwhm_k = Σᵢ cf_i · 2(fᵢ−ν₀_k)² · L_{k,i} / d_{k,i}
        # Derivation:
        #   ∂L_{k,i}/∂g_k = 2g_k(fᵢ−ν₀_k)² · A_k / d_{k,i}²
        #                  = 2(fᵢ−ν₀_k)² · L_{k,i} / (g_k · d_{k,i})
        #   ∂L_{k,i}/∂fwhm_k = (1/2) · ∂L_{k,i}/∂g_k
        #   ∂fwhm_k/∂ log_fwhm_k = fwhm_k = 2g_k
        #   → combine: factor 2g_k · (1/2) / g_k = 1  (g_k cancels cleanly)
        grad[3 * k + 2] = np.dot(cf, (2.0 * dlt * dlt / dk) * Lk)

    if include_const:
        # ∂L/∂ log_C = C · Σᵢ cf_i
        grad[-1] = C * float(np.sum(cf))

    return loss, grad


# ============================================================================
# Single L-BFGS-B run
# ============================================================================

def _run_lbfgsb(
    theta0:        np.ndarray,
    bounds:        List[Tuple[float, float]],
    freq:          np.ndarray,
    power:         np.ndarray,
    nlor:          int,
    include_const: bool,
    ftol:          float,
    gtol:          float,
    maxiter:       int,
) -> Optional[np.ndarray]:
    """
    Run one L-BFGS-B minimisation from theta0.

    Returns the optimised theta array on success, None on any failure.

    Solver options
    --------------
    maxcor=20  : Hessian memory (default 10).  20 past vectors give a
                 better curvature estimate for correlated parameters (the
                 amp-const ridge, overlapping FWHM components).
    maxls=40   : Line-search steps (default 20).  Extra budget for narrow
                 ridges where the Wolfe conditions are hard to satisfy.
    """
    try:
        result = minimize(
            _whittle_loss_and_grad,
            theta0,
            args=(freq, power, nlor, include_const),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "ftol":    float(ftol),
                "gtol":    float(gtol),
                "maxcor":  20,
                "maxls":   40,
            },
        )
    except Exception:
        return None

    if not np.all(np.isfinite(result.x)):
        return None

    return result.x


# ============================================================================
# Multi-start jitter
# ============================================================================

def _jitter_theta(
    theta:               np.ndarray,
    bounds:              List[Tuple[float, float]],
    nlor:                int,
    include_const:       bool,
    rng:                 np.random.Generator,
    jitter_std_log:      float,
    jitter_std_nu0_frac: float,
) -> np.ndarray:
    """
    Perturb theta, then clip to bounds.

    amp / fwhm : additive Gaussian in log-space (std = jitter_std_log).
        A value of 0.30 corresponds to a typical multiplicative factor
        of exp(±0.30) ≈ 0.74 – 1.35, which explores the neighbourhood
        without stepping across component boundaries.
    ν₀         : additive Gaussian with std = jitter_std_nu0_frac × (hi − lo).
    const      : small perturbation (10 % of jitter_std_log) — the constant
        is usually well-determined and should not move far.
    """
    t = theta.copy()

    for k in range(nlor):
        lo_a, hi_a = bounds[3 * k]
        lo_x, hi_x = bounds[3 * k + 1]
        lo_f, hi_f = bounds[3 * k + 2]

        t[3 * k]     += jitter_std_log * rng.standard_normal()
        t[3 * k + 1] += jitter_std_nu0_frac * (hi_x - lo_x) * rng.standard_normal()
        t[3 * k + 2] += jitter_std_log * rng.standard_normal()

        t[3 * k]     = float(np.clip(t[3 * k],     lo_a, hi_a))
        t[3 * k + 1] = float(np.clip(t[3 * k + 1], lo_x, hi_x))
        t[3 * k + 2] = float(np.clip(t[3 * k + 2], lo_f, hi_f))

    if include_const:
        lo_c, hi_c = bounds[-1]
        t[-1] += 0.10 * jitter_std_log * rng.standard_normal()
        t[-1]  = float(np.clip(t[-1], lo_c, hi_c))

    return t


# ============================================================================
# Parameter covariance  (Hessian inversion at the Whittle MLE)
# ============================================================================

def _compute_covariance(
    theta_opt:     np.ndarray,
    freq:          np.ndarray,
    power:         np.ndarray,
    nlor:          int,
    include_const: bool,
) -> Optional[np.ndarray]:
    """
    Estimate the covariance matrix of theta_opt via numerical Hessian.

    The Whittle log-likelihood is asymptotically quadratic near the MLE,
    so the Fisher information matrix equals the negative expected Hessian.
    At the MLE the observed Hessian H satisfies:

        Cov(theta_opt) ≈ H^{-1}

    The Hessian is computed by central finite differences of the analytical
    gradient, which is far more accurate than FD of the loss itself:

        H_jk ≈ [∇_j(θ + ε_k e_k) − ∇_j(θ − ε_k e_k)] / (2 ε_k)

    This costs 2*npar gradient evaluations.

    Returns None when the Hessian is numerically singular or indefinite
    (active box constraints at convergence, degenerate components).
    """
    npar = len(theta_opt)
    # Step size: relative to |θ_j|, floored so near-zero components get a
    # reasonable step.  1e-4 is a reliable choice for smooth functions with
    # analytical gradients.
    eps_vec = np.maximum(np.abs(theta_opt), 0.1) * 1e-4

    H = np.zeros((npar, npar), dtype=float)
    for k in range(npar):
        tp = theta_opt.copy(); tp[k] += eps_vec[k]
        tm = theta_opt.copy(); tm[k] -= eps_vec[k]
        _, gp = _whittle_loss_and_grad(tp, freq, power, nlor, include_const)
        _, gm = _whittle_loss_and_grad(tm, freq, power, nlor, include_const)
        H[:, k] = (gp - gm) / (2.0 * eps_vec[k])

    # Symmetrise (FD rounding can break exact symmetry)
    H = 0.5 * (H + H.T)

    # The Hessian of the *loss* (minimised) should be positive-definite at a
    # proper minimum.  Check and invert.
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    # Sanity: diagonal of the covariance must be non-negative.
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
        return None

    return cov


def _cov_logspace_to_linear(
    cov_theta:     np.ndarray,
    theta_opt:     np.ndarray,
    nlor:          int,
    include_const: bool,
) -> np.ndarray:
    """
    Convert the log-space covariance to linear-space via the delta method.

    theta layout: [log_amp_0, nu0_0, log_fwhm_0, ..., log_C]
    output layout: [amp_0, nu0_0, fwhm_0, ..., C]  (Stingray p_opt order)

    Jacobian J = diag(j) where:
        j[3k]   = exp(theta[3k])   = amp_k    (∂amp/∂log_amp = amp)
        j[3k+1] = 1                           (nu0 is already linear)
        j[3k+2] = exp(theta[3k+2]) = fwhm_k  (∂fwhm/∂log_fwhm = fwhm)
        j[-1]   = exp(theta[-1])   = C
    """
    npar = len(theta_opt)
    j = np.ones(npar, dtype=float)
    for k in range(nlor):
        j[3 * k]     = float(np.exp(theta_opt[3 * k]))      # amp
        # j[3*k+1] = 1  (nu0 linear, no transform)
        j[3 * k + 2] = float(np.exp(theta_opt[3 * k + 2]))  # fwhm
    if include_const:
        j[-1] = float(np.exp(theta_opt[-1]))                 # const

    # Cov_linear = J @ Cov_theta @ J^T  where J = diag(j)
    # = j[:, None] * cov_theta * j[None, :]
    return j[:, None] * cov_theta * j[None, :]


# ============================================================================
# Result builder
# ============================================================================

def _make_triplea_result(
    theta_opt:     np.ndarray,
    freq:          np.ndarray,
    power:         np.ndarray,
    nlor:          int,
    include_const: bool,
) -> TripleAResult:
    """
    Convert optimal log-space theta to a TripleAResult.

    Reconstructs M(f) from scratch for numerical consistency (avoids
    reusing intermediate arrays from the final gradient call, which may
    have been discarded).

    Deviance and IC definitions match Stingray's convention:
        deviance = 2 · Σᵢ [log M_i + P_i / M_i]
        AIC      = deviance + 2 · npar
        BIC      = deviance + npar · ln(N)
    """
    p_opt = _unpack_theta(theta_opt, nlor, include_const)

    amps = p_opt[0::3][:nlor]
    nu0s = p_opt[1::3][:nlor]
    fwhm = p_opt[2::3][:nlor]
    g    = 0.5 * fwhm

    C = float(p_opt[-1]) if include_const else 0.0

    delta = freq[np.newaxis, :] - nu0s[:, np.newaxis]
    g2    = (g ** 2)[:, np.newaxis]
    d     = delta ** 2 + g2
    L     = amps[:, np.newaxis] * g2 / d
    M     = np.maximum(C + L.sum(axis=0), _EPS_M)

    nll      = float(np.sum(np.log(M) + power / M))
    deviance = 2.0 * nll
    npar     = 3 * nlor + (1 if include_const else 0)
    N        = int(freq.size)
    aic      = deviance + 2.0 * float(npar)
    bic      = deviance + float(npar) * np.log(float(N))

    # Compute parameter covariance via Hessian inversion.
    # Costs 2*npar gradient evaluations (~ms for typical npar<=10).
    # Returns None if the Hessian is singular (active bounds, degenerate fit).
    p_cov_theta  = _compute_covariance(theta_opt, freq, power, nlor, include_const)
    p_cov_linear = (
        _cov_logspace_to_linear(p_cov_theta, theta_opt, nlor, include_const)
        if p_cov_theta is not None else None
    )

    return TripleAResult(
        p_opt    = p_opt,
        mfit     = M,
        aic      = float(aic),
        bic      = float(bic),
        deviance = float(deviance),
        p_cov    = p_cov_linear,
    )


# ============================================================================
# Public entry point
# ============================================================================

def tripleA_fit_once(
    ps: Any,
    *,
    nlor:          int,
    t0:            list,
    include_const: bool,
    x0_lims:       List[Tuple[float, float]],
    fwhm_lims:     List[Tuple[float, float]],
    amp_caps:      List[float],
    amp_lo_list:   Optional[List[float]] = None,
    const_max:     float,
    eps_amp:       float = 1e-30,
) -> Tuple[Optional[TripleAResult], Optional[str]]:
    """
    TripleA single-stage multi-start L-BFGS-B fit.

    Drop-in replacement for the Stingray _try_fit_once call when
    fitmethod == "TripleA".  Returns a (TripleAResult, None) pair on
    success or (None, error_str) on total failure, matching the
    _try_fit_once contract.

    Parameters
    ----------
    ps            : Stingray Powerspectrum.  Only ps.freq and ps.power
                    are used.  ps.m is deliberately ignored: the scalar
                    m passed to Stingray is a summary statistic and does
                    not change the argmin of the Whittle objective.
    nlor          : number of Lorentzian components.
    t0            : initial parameters in Stingray order:
                    [amp_0, x0_0, fwhm_0,  amp_1, x0_1, fwhm_1, ...,  const]
    include_const : whether a free white-noise constant is included.
    x0_lims       : per-component centroid bounds in Hz.
    fwhm_lims     : per-component FWHM bounds in Hz.
    amp_caps      : per-component amplitude upper bounds.
    amp_lo_list   : per-component amplitude lower bounds.
                    None → eps_amp for all (standard case).
                    Pass tight windows around the current value to
                    implement frozen-amplitude behaviour (interactive fitter).
    const_max     : upper bound for the white-noise constant.
    eps_amp       : hard floor for amplitudes and FWHMs (default 1e-30).

    Runtime configuration (read from QPO_Parameter at call time)
    -------------------------------------------------------------
    AAA_N_STARTS       (int,   default 5)
    AAA_JITTER_STD_LOG (float, default 0.30)
    AAA_JITTER_STD_NU0 (float, default 0.10)
    AAA_FTOL           (float, default 1e-11)
    AAA_GTOL           (float, default 1e-7)
    AAA_MAXITER        (int,   default 1000)
    FIT_RANDOM_SEED    (int,   default 42)     shared with the rest of the pipeline

    Notes
    -----
    * TripleA manages its own multi-start loop (AAA_N_STARTS restarts).
      The outer _jittered_starts loop in _run_stage is suppressed to a
      single pass when fitmethod == "TripleA" (see _run_stage fix in
      QPO_fit.py), so exactly AAA_N_STARTS starts run per _run_stage
      call.  In the interactive fitter (_run_direct_fit) the outer loop
      is already short-circuited.

    * No internal good-bin mask is applied.  _run_stage feeds ps from
      self.f / self.p which are already filtered for finite positive
      values by QPOFitter.__init__.  The _EPS_M floor in
      _whittle_loss_and_grad handles numerical edge cases, and the
      `if not np.isfinite(loss)` guard prevents NaN propagation.
      Masking inside tripleA_fit_once would give mfit a different length
      from self.p, breaking _compute_rchi2 in _wrap.
    """
    # ---- Runtime config from QPO_Parameter ---------------------------------
    n_starts       = int(getattr(P, "AAA_N_STARTS",       5))
    jitter_std_log = float(getattr(P, "AAA_JITTER_STD_LOG", 0.30))
    jitter_std_nu0 = float(getattr(P, "AAA_JITTER_STD_NU0", 0.10))
    ftol           = float(getattr(P, "AAA_FTOL",           1e-11))
    gtol           = float(getattr(P, "AAA_GTOL",           1e-7))
    maxiter        = int(getattr(P, "AAA_MAXITER",          1000))
    random_seed    = int(getattr(P, "FIT_RANDOM_SEED",      42))

    # ---- Data ---------------------------------------------------------------
    # Do NOT apply a good-bin mask here.  _run_stage pre-filters self.f and
    # self.p so ps.freq / ps.power are already finite and positive.  Masking
    # would change the length of mfit relative to self.p, causing a shape
    # mismatch in _wrap → _compute_rchi2.
    freq  = np.asarray(ps.freq,  float)
    power = np.asarray(ps.power, float)

    n_bins = int(freq.size)
    n_pars = 3 * nlor + (1 if include_const else 0)
    if n_bins < max(30, n_pars + 5):
        return None, (
            f"TripleA: too few usable bins ({n_bins}) "
            f"for {n_pars} free parameters"
        )

    # ---- Bounds in log-space -----------------------------------------------
    bounds = _build_bounds(
        nlor,
        x0_lims=x0_lims,
        fwhm_lims=fwhm_lims,
        amp_caps=amp_caps,
        amp_lo_list=amp_lo_list,
        include_const=include_const,
        const_max=float(const_max),
        eps_amp=float(eps_amp),
    )

    # ---- Base start point ---------------------------------------------------
    t0_arr = np.asarray(t0, float)
    theta0 = _pack_theta(t0_arr, nlor, include_const, eps_amp)
    # Clip to bounds in case t0 was on or outside a limit.
    for j, (lo, hi) in enumerate(bounds):
        theta0[j] = float(np.clip(theta0[j], lo + 1e-12 * (hi - lo),
                                             hi - 1e-12 * (hi - lo)))

    rng = np.random.default_rng(int(random_seed))

    best_loss:  float                = np.inf
    best_theta: Optional[np.ndarray] = None

    # ---- Multi-start L-BFGS-B ----------------------------------------------
    for start_idx in range(max(1, n_starts)):
        if start_idx == 0:
            t_start = theta0.copy()
        else:
            t_start = _jitter_theta(
                theta0, bounds, nlor, include_const,
                rng, jitter_std_log, jitter_std_nu0,
            )

        t_opt = _run_lbfgsb(
            t_start, bounds, freq, power,
            nlor, include_const, ftol, gtol, maxiter,
        )
        if t_opt is None:
            continue

        loss, _ = _whittle_loss_and_grad(t_opt, freq, power, nlor, include_const)
        if np.isfinite(loss) and loss < best_loss:
            best_loss  = loss
            best_theta = t_opt.copy()

    # ---- Outcome -----------------------------------------------------------
    if best_theta is None:
        return None, (
            "TripleA: all L-BFGS-B starts failed or returned "
            "non-finite Whittle loss"
        )

    result = _make_triplea_result(best_theta, freq, power, nlor, include_const)
    return result, None
