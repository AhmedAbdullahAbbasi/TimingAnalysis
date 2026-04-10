#!/usr/bin/env python3
"""
QPO_fit.py  (v2.4)
==================
Lorentzian fitting engine for X-ray power density spectra.

Changes from v2.3
-----------------
- BUG FIX (_run_smart): grouping key changed from nq_post to
  (nq_post, n_cont_post) where n_cont_post = nlor - nq_post.

  Root cause of the cont=4/5 problem: a cont3+QPO config where _wrap
  relabels the QPO to "cont" produces nq_post=0, n_cont_post=4.  With
  the old nq_post-only key this fell into bucket 0 and competed against
  genuine cont2/cont3 fits (nc=2,3) without any continuum IC gate.

  Fix: each (nq_post, n_cont_post) tuple is a separate bucket.  Upgrades
  from nc=2 to nc=3 still require cont_ic_delta_min (Step 1).  QPO
  upgrades still require qpo_ic_delta_min (Step 4).

- BUG FIX (_run_smart): hard max_cont cap.  Any fit where n_cont_post
  exceeds max_cont (3 when FIT_CONT4_ENABLE=False, 4 when True) is
  discarded before IC comparison.  This enforces the cont4 switch for
  relabelled configs that would otherwise sneak extra components through.

Changes from v2.2
-----------------
- BUG FIX: _wrap relabels QPO-typed components with Q < qpo_detect_qmin
  to "cont" after fitting (parameter values unchanged).
- _grow_qpos uses comp_types count (not _detect_qpos) for the max guard
  and collects used-frequencies from all QPO-typed components.
- Stage 1b applies multi_qpo_ic_delta_min vs the 1-QPO best.

Changes from v2
---------------
- BUG FIX: _grow_qpos guard for target_nqpo using comp_types count.
- BUG FIX: _find_qpos_in_residual floor lowered from 2 to 1.

Changes from v1
---------------
- Multi-scale candidate finder in z-score space.
- Per-bin m_eff in rchi2 for log-rebinned PDS.
- Per-component continuum centroid limits (narrow / wide / free).
- cont4 fallback for complex spectral states.
- Log-uniform priors for amp and fwhm.
- Reduced Whittle deviance alongside rchi2.
- Guardrail overshoot threshold raised for log-rebinned data.

Public API
----------
fit_lorentzian_family(freq, power, power_err=None, ...)  -> FitResult
QPOFitter — construct and .run()

Parameter representation
------------------------
pars[i] = (nu0_i, fwhm_i, amp_i)       # centre Hz, width Hz, peak amplitude
stingray_p_opt = [amp_0, x0_0, fwhm_0,  amp_1, x0_1, fwhm_1, ...,  const]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter

from stingray import Powerspectrum
from stingray.modeling.scripts import fit_lorentzians


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def lorentz(freq: np.ndarray, nu0: float, fwhm: float, amp: float) -> np.ndarray:
    """Lorentzian profile.  Peak value at f=nu0 equals amp."""
    f = np.asarray(freq, float)
    g = 0.5 * float(fwhm)
    return float(amp) * g * g / ((f - float(nu0)) ** 2 + g * g)


def lorentz_integral_exact(amp: float, fwhm: float) -> float:
    """Exact integral of lorentz() over (-inf, +inf) = pi/4 * amp * fwhm."""
    return float(np.pi / 4.0 * float(amp) * float(fwhm))


def component_power_integral(
    freq: np.ndarray, comp: np.ndarray, fmin: float, fmax: float
) -> float:
    f = np.asarray(freq, float)
    y = np.asarray(comp, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(f) & np.isfinite(y)
    if np.sum(m) < 2:
        return 0.0
    return float(np.trapz(y[m], f[m]))


def extract_qpo_params_list(
    fitres,
    *,
    qpo_fmin: float,
    qpo_fmax: float,
    qmin: float = 3.0,
    sort_by: str = "area",
) -> List[Dict[str, Any]]:
    """Return a list of dicts, one per QPO-like Lorentzian in fitres."""
    if fitres is None or not getattr(fitres, "ok", False):
        return []
    pars = np.asarray(getattr(fitres, "pars", []), float)
    if pars.ndim != 2 or pars.shape[1] != 3 or pars.size == 0:
        return []
    freq = np.asarray(getattr(fitres, "freq", []), float)
    if freq.size < 2:
        return []

    out = []
    for i, (nu0, fwhm, amp) in enumerate(pars):
        if not (np.isfinite(nu0) and np.isfinite(fwhm) and np.isfinite(amp)):
            continue
        if fwhm <= 0 or not (qpo_fmin <= nu0 <= qpo_fmax):
            continue
        Q = nu0 / fwhm
        if Q < float(qmin):
            continue
        area = float(np.trapz(lorentz(freq, nu0, fwhm, amp), freq))
        out.append(dict(
            qpo_index=int(i),
            qpo_nu0_hz=float(nu0),
            qpo_fwhm_hz=float(fwhm),
            qpo_Q=float(Q),
            qpo_area=float(area),
        ))

    key = str(sort_by).strip().lower()
    if key == "freq":
        out.sort(key=lambda d: (d["qpo_nu0_hz"], -d["qpo_area"]))
    elif key == "q":
        out.sort(key=lambda d: (-d["qpo_Q"], -d["qpo_area"], d["qpo_nu0_hz"]))
    else:
        out.sort(key=lambda d: (-d["qpo_area"], d["qpo_nu0_hz"]))
    return out


def extract_qpo_params(
    fitres, *, qpo_fmin: float, qpo_fmax: float, qmin: float = 3.0
) -> Optional[Dict[str, Any]]:
    lst = extract_qpo_params_list(
        fitres, qpo_fmin=qpo_fmin, qpo_fmax=qpo_fmax, qmin=qmin, sort_by="area"
    )
    return lst[0] if lst else None


# ---------------------------------------------------------------------------
# FitResult container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Result of a single Lorentzian-family fit.

    rchi2 : float
        Reduced chi-squared using per-bin sigma = P/sqrt(m_bin).
        Uses Gaussian approximation; kept as final diagnostic 
    red_deviance : float
        Reduced Whittle deviance = deviance / (N - npar).
        Uses the same likelihood framework as AIC/BIC. 
    """
    ok: bool
    message: str
    nlor: int
    pars: np.ndarray               # (nlor, 3): nu0, fwhm, amp
    comp_types: List[str]          # 'cont' or 'qpo', length nlor
    const: float
    freq: np.ndarray
    model: np.ndarray
    aic: float
    bic: float
    deviance: float
    rchi2: Optional[float] = None
    red_deviance: Optional[float] = None
    stingray_p_opt: Optional[np.ndarray] = None
    p_err: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Private helpers (module-level; no state)
# ---------------------------------------------------------------------------

def _rolling_median(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    w = int(w)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    return median_filter(y, size=w, mode="nearest").astype(float)


def _safe_scalar_m(m) -> int:
    try:
        arr = np.asarray(m, float)
        v = float(arr) if arr.ndim == 0 else float(np.nanmedian(arr))
        if (not np.isfinite(v)) or v < 1:
            return 1
        return int(v)
    except Exception:
        return 1


def _m_as_array(m, n: int) -> np.ndarray:
    """Convert m (scalar, array, or Powerspectrum.m) to a float array of length n."""
    arr = np.asarray(m, float)
    if arr.ndim == 0:
        return np.full(n, max(1.0, float(arr)))
    if arr.size == n:
        out = np.where(np.isfinite(arr) & (arr >= 1), arr, 1.0)
        return out.astype(float)
    # Length mismatch: fall back to median
    med = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 1.0
    return np.full(n, max(1.0, med))


def _seed_const(freq: np.ndarray, power: np.ndarray, fmin: float) -> float:
    m = (freq >= fmin) & np.isfinite(power) & (power > 0)
    if np.sum(m) < 10:
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else 0.0
    return float(np.nanmedian(power[m]))


def _seed_lowf_level(freq: np.ndarray, power: np.ndarray, fmax: float = 2.0) -> float:
    m = (freq > 0) & (freq <= fmax) & np.isfinite(power) & (power > 0)
    if np.sum(m) < 10:
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else 0.0
    return float(np.nanmedian(power[m]))


def _seed_amp_at(freq: np.ndarray, power: np.ndarray, nu0: float,
                 half_width: int = 2) -> float:
    """Robust amplitude seed: median of 2*half_width+1 bins centred on nu0."""
    i = int(np.argmin(np.abs(freq - float(nu0))))
    lo = max(0, i - half_width)
    hi = min(len(power), i + half_width + 1)
    chunk = power[lo:hi]
    good = np.isfinite(chunk) & (chunk > 0)
    if np.any(good):
        return float(np.median(chunk[good]))
    return float(power[i])


def _local_median_around(
    freq: np.ndarray, power: np.ndarray, nu0: float, width_hz: float = 0.5
) -> float:
    if not (np.isfinite(nu0) and np.isfinite(width_hz) and width_hz > 0):
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else np.nan
    m = (freq >= nu0 - width_hz) & (freq <= nu0 + width_hz) & np.isfinite(power) & (power > 0)
    if np.sum(m) < 5:
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else np.nan
    return float(np.nanmedian(power[m]))


def _compute_rchi2(
    power: np.ndarray, model: np.ndarray, power_err, m_avg, npar: int
) -> float:
    """
    Reduced chi-squared using per-bin sigma.

    m_avg can be a scalar (applied uniformly) or a per-bin array.
    When per-bin m is available (log-rebinned PDS), this correctly accounts
    for the frequency-dependent averaging.
    """
    p   = np.asarray(power, float)
    mod = np.asarray(model, float)
    n   = p.size

    if power_err is not None:
        err = np.asarray(power_err, float)
    else:
        m_arr = _m_as_array(m_avg, n)
        err   = p / np.sqrt(m_arr)

    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(err) & (err > 0)
    dof  = int(np.sum(good)) - int(npar)
    if np.sum(good) < 20 or dof <= 0:
        return np.nan
    chi2 = np.sum(((p[good] - mod[good]) / err[good]) ** 2)
    return float(chi2 / dof)


def _compute_red_deviance(deviance: float, n_bins: int, npar: int) -> float:
    """Reduced Whittle deviance = deviance / (N - npar)."""
    dof = int(n_bins) - int(npar)
    if dof <= 0 or not np.isfinite(deviance):
        return np.nan
    return float(deviance / dof)


def _estimate_sigma_local(
    *, cont: np.ndarray, p: np.ndarray, m_eff: int, mode: str = "cont"
) -> np.ndarray:
    m_eff = max(1, int(m_eff))
    base  = np.asarray(cont if mode.lower().strip() == "cont" else p, float)
    base  = np.where(np.isfinite(base) & (base > 0), base, np.nan)
    med   = float(np.nanmedian(base)) if np.any(np.isfinite(base)) else 1.0
    base  = np.where(np.isfinite(base), base, med)
    return base / np.sqrt(float(m_eff))


def _ic_value(fr: FitResult, criterion: str) -> float:
    c = str(criterion).strip().lower()
    if c == "bic":
        return float(fr.bic) if np.isfinite(fr.bic) else np.inf
    return float(fr.aic) if np.isfinite(fr.aic) else np.inf


def _accept_upgrade_ic(
    simple: FitResult, complex_: FitResult, *, criterion: str, delta_min: float
) -> bool:
    d = _ic_value(simple, criterion) - _ic_value(complex_, criterion)
    return np.isfinite(d) and (d >= float(delta_min))


# ---------------------------------------------------------------------------
# Multi-scale candidate peak finder (z-score based)
# ---------------------------------------------------------------------------

def _find_candidates_single_scale(
    f: np.ndarray, p: np.ndarray,
    *,
    smooth_hz: float,
    min_width_bins: int,
    prominence: float,
    min_sep_hz: float,
    m_eff: int,
    require_ksigma: Optional[float],
    cand_sigma_mode: str,
) -> List[Dict[str, float]]:
    """
    Find peaks at a single smoothing scale using z-score whitening.

    Returns list of dicts: [{"nu_hz", "z_prominence", "excess_sigma", "z_peak"}, ...]
    """
    if f.size < 25:
        return []

    df = float(np.median(np.diff(f)))
    if not np.isfinite(df) or df <= 0:
        return []

    w = max(int(min_width_bins), int(np.round(smooth_hz / df)))
    if w % 2 == 0:
        w += 1
    if w >= len(p):
        w = len(p) - 2 if (len(p) > 2) else 3
        if w % 2 == 0:
            w -= 1
    if w < 3 or w >= len(p):
        return []

    cont = _rolling_median(p, w)
    good = np.isfinite(cont) & (cont > 0)
    if not np.any(good):
        return []
    cont = np.where(good, cont, np.nanmedian(cont[good]))

    # z-score whitening: (P - cont) / sigma
    sigma = _estimate_sigma_local(cont=cont, p=p, m_eff=int(m_eff), mode=cand_sigma_mode)
    z = (p - cont) / sigma

    distance = int(max(1, np.round(min_sep_hz / df)))
    peaks, props = scipy.signal.find_peaks(
        z, prominence=float(prominence), distance=distance
    )
    if peaks.size == 0:
        return []

    prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), float)

    # Optional hard sigma gate
    if require_ksigma is not None and np.isfinite(require_ksigma) and require_ksigma > 0:
        keep = z[peaks] >= float(require_ksigma)
        peaks, prom = peaks[keep], prom[keep]
        if peaks.size == 0:
            return []

    return [
        dict(
            nu_hz=float(f[idx]),
            z_prominence=float(pr),
            excess_sigma=float(z[idx]) if np.isfinite(z[idx]) else float("nan"),
            z_peak=float(z[idx]),
            ratio_peak=float(p[idx] / cont[idx]) if cont[idx] > 0 else float("nan"),
        )
        for idx, pr in zip(peaks.tolist(), prom.tolist())
    ]


def find_qpo_candidates(
    freq,
    power,
    *,
    cand_fmin: float,
    cand_fmax: float,
    smooth_scales: Optional[List[float]] = None,
    smooth_hz: float = 0.5,
    prominence: float = 0.5,
    min_sep_hz: float = 0.15,
    max_candidates: int = 5,
    min_width_bins: int = 7,
    m_eff: int = 1,
    require_ksigma: Optional[float] = None,
    cand_sigma_mode: str = "cont",
) -> List[Dict[str, float]]:
    """
    Multi-scale candidate QPO finder using z-score whitening.

    Runs the peak finder at each scale in smooth_scales (or just smooth_hz
    if smooth_scales is None), unions the results, deduplicates, and returns
    sorted by z-score prominence descending.
    """
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    m = (f >= cand_fmin) & (f <= cand_fmax) & np.isfinite(p) & (p > 0)
    f, p = f[m], p[m]
    if f.size < 25:
        return []

    scales = list(smooth_scales) if smooth_scales else [float(smooth_hz)]

    all_cands: List[Dict[str, float]] = []
    for scale in scales:
        cands_this = _find_candidates_single_scale(
            f, p,
            smooth_hz=float(scale),
            min_width_bins=min_width_bins,
            prominence=float(prominence),
            min_sep_hz=min_sep_hz,
            m_eff=int(m_eff),
            require_ksigma=require_ksigma,
            cand_sigma_mode=cand_sigma_mode,
        )
        all_cands.extend(cands_this)

    if not all_cands:
        return []

    # Deduplicate: keep the entry with the highest z_prominence for each cluster
    all_cands.sort(key=lambda c: -c.get("z_prominence", 0.0))
    deduped: List[Dict[str, float]] = []
    for c in all_cands:
        nu = float(c.get("nu_hz", np.nan))
        if not np.isfinite(nu):
            continue
        if any(abs(nu - float(d.get("nu_hz", np.nan))) < min_sep_hz for d in deduped):
            continue
        deduped.append(c)

    # Keep legacy key "prominence" pointing to z_prominence for backward compat
    for c in deduped:
        c["prominence"] = c.get("z_prominence", c.get("prominence", 0.0))

    return deduped[:int(max_candidates)]


def _nearest_candidate_metrics(
    cands: List[Dict[str, float]], nu: float, *, tol_hz: float
) -> Tuple[float, float]:
    if not cands:
        return np.nan, np.nan
    nu   = float(nu)
    best = min(cands, key=lambda c: abs(float(c.get("nu_hz", np.inf)) - nu))
    if abs(float(best.get("nu_hz", np.inf)) - nu) > float(tol_hz):
        return np.nan, np.nan
    return float(best.get("prominence", np.nan)), float(best.get("excess_sigma", np.nan))


# ---------------------------------------------------------------------------
# Prior builders
# ---------------------------------------------------------------------------

def _hard_trunc_uniform(lo: float, hi: float):
    w = max(float(hi - lo), 1e-300)
    def prior(x):
        x = float(x)
        if not np.isfinite(x) or x < lo or x > hi:
            return 0.0
        return 1.0 / w
    return prior


def _half_uniform(hi: float):
    hi = max(float(hi), 1e-300)
    return _hard_trunc_uniform(0.0, hi)


def _log_uniform(lo: float, hi: float):
    """
    Log-uniform prior p(x) ∝ 1/x for lo ≤ x ≤ hi.
    Provides a natural gradient away from zero and from the upper bound.
    Used for amplitude and FWHM (positive-definite, scale-free quantities).
    Falls back to half-uniform if lo is too small.
    """
    lo = max(float(lo), 1e-30)
    hi = max(float(hi), lo * 2.0)
    log_ratio = np.log(hi / lo)
    if not (np.isfinite(log_ratio) and log_ratio > 0):
        return _half_uniform(hi)
    def prior(x):
        x = float(x)
        if not np.isfinite(x) or x < lo or x > hi:
            return 0.0
        return 1.0 / (x * log_ratio)
    return prior


def _build_priors(
    nlor: int,
    *,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    amp_max_list: List[float],
    include_const: bool,
    const_max: float,
    eps_amp: float = 1e-30,
) -> Dict[str, Any]:
    priors: Dict[str, Any] = {}
    for i in range(nlor):
        # Half-uniform for amplitude (flat prior — log-uniform with eps_amp=1e-30
        # creates a ~20 nat prior gradient that overwhelms the likelihood and
        # collapses all amplitudes to zero).
        priors[f"amplitude_{i}"] = _half_uniform(amp_max_list[i])
        priors[f"x_0_{i}"]       = _hard_trunc_uniform(*x0_lims[i])
        # Log-uniform for FWHM: range is moderate (~3 nats for typical limits),
        # provides gentle gradient away from the extremes.
        priors[f"fwhm_{i}"]      = _log_uniform(fwhm_lims[i][0], fwhm_lims[i][1])
    if include_const:
        priors[f"amplitude_{nlor}"] = _half_uniform(const_max)
    return priors


def _pack_t0(
    comps: List[Tuple[float, float, float]], const: Optional[float]
) -> List[float]:
    t0 = []
    for amp, x0, fwhm in comps:
        t0 += [float(amp), float(x0), float(fwhm)]
    if const is not None:
        t0.append(float(const))
    return t0


# ---------------------------------------------------------------------------
# Parameter repair / clamping
# ---------------------------------------------------------------------------

def _repair_params(
    t: np.ndarray,
    *,
    nlor: int,
    include_const: bool,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    const_max: float,
    eps_amp: float,
) -> np.ndarray:
    out = np.asarray(t, float).copy()
    for i in range(nlor):
        ai, xi, wi = 3*i, 3*i+1, 3*i+2

        if not np.isfinite(out[ai]) or out[ai] < eps_amp:
            out[ai] = float(eps_amp)

        lo, hi = x0_lims[i]
        if not np.isfinite(out[xi]):
            out[xi] = float(np.clip((lo + hi) * 0.5, lo, hi))
        else:
            out[xi] = float(np.clip(out[xi], lo, hi))

        flo, fhi = fwhm_lims[i]
        if not np.isfinite(out[wi]) or out[wi] <= 0:
            out[wi] = float(max(flo, 1e-6))
        else:
            out[wi] = float(np.clip(out[wi], flo, fhi))

    if include_const and out.size >= (3*nlor + 1):
        ci = -1
        if not np.isfinite(out[ci]) or out[ci] < eps_amp:
            out[ci] = float(eps_amp)
        out[ci] = float(np.clip(out[ci], 0.0, float(const_max)))

    return out


def _param_sanity_check(
    p_opt: np.ndarray,
    *,
    nlor: int,
    include_const: bool,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    amp_max_list: List[float],
    const_max: float,
    df: float,
) -> Tuple[bool, str]:
    p_opt  = np.asarray(p_opt, float)
    x0_tol = float(max(2.0 * float(df), 1e-6))
    for i in range(nlor):
        amp_i  = float(p_opt[3*i])
        x0_i   = float(p_opt[3*i+1])
        fwhm_i = float(p_opt[3*i+2])
        if not (np.isfinite(amp_i) and np.isfinite(x0_i) and np.isfinite(fwhm_i)):
            return False, f"non-finite param in component {i}"
        if fwhm_i <= 0:
            return False, f"non-positive fwhm in component {i}"
        if amp_i < 0:
            return False, f"negative amplitude in component {i}"
        lo, hi = x0_lims[i]
        if x0_i < (lo - x0_tol) or x0_i > (hi + x0_tol):
            return False, f"x0 out of bounds in component {i}: {x0_i:.4g}"
        flo, fhi = fwhm_lims[i]
        if fwhm_i < flo or fwhm_i > fhi:
            return False, f"fwhm out of bounds in component {i}: {fwhm_i:.4g}"
        if amp_i > (float(amp_max_list[i]) * 1.05):
            return False, f"amp above cap in component {i}: {amp_i:.4g}"
    if include_const:
        c = float(p_opt[-1])
        if not np.isfinite(c):
            return False, "non-finite const"
        if c < 0:
            return False, "negative const"
        if c > (float(const_max) * 1.05):
            return False, f"const above cap: {c:.4g}"
    return True, ""


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _guardrail_overshoot(
    freq, data_p, model_p, power_err, m_eff, *, ksigma=4.0, max_run_bins=6, max_frac=0.10
) -> Tuple[bool, str]:
    """
    Detect systematic model overshoot.

    m_eff can be scalar or per-bin array.  Default ksigma raised to 4.0
    because log-rebinned bins have non-Gaussian tails at low m.
    """
    p   = np.asarray(data_p, float)
    mod = np.asarray(model_p, float)
    n   = p.size

    if power_err is not None:
        sig = np.asarray(power_err, float)
    else:
        m_arr = _m_as_array(m_eff, n)
        sig   = p / np.sqrt(m_arr)

    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(sig) & (sig > 0)
    ngood = int(np.sum(good))

    if ngood < 10:
        return True, ""
    if ngood < 30:
        warnings.warn(
            f"Overshoot guardrail: only {ngood} good bins — applying relaxed threshold.",
            RuntimeWarning, stacklevel=4,
        )
        ksigma        = max(float(ksigma), 5.0)
        max_run_bins  = max(int(max_run_bins), 4)
        max_frac      = max(float(max_frac), 0.15)

    resid = mod[good] - p[good]
    bad   = resid > float(ksigma) * sig[good]
    if not np.any(bad):
        return True, ""

    frac   = float(np.mean(bad))
    run = maxrun = 0
    for b in bad.astype(int):
        run = run + 1 if b else 0
        maxrun = max(maxrun, run)

    if maxrun > int(max_run_bins) or frac > float(max_frac):
        return False, f"overshoot FAIL run={maxrun} frac={frac:.4g}"
    return True, ""


def _guardrail_component_local_amp(
    freq, power, fit_pars, *, local_amp_factor=6.0, local_width_hz=0.5
) -> Tuple[bool, str]:
    for (nu0, fwhm, amp) in np.asarray(fit_pars, float):
        if not (np.isfinite(nu0) and np.isfinite(amp)) or amp <= 0:
            continue
        local = _local_median_around(freq, power, nu0, local_width_hz)
        if not np.isfinite(local) or local <= 0:
            continue
        if (amp / local) > float(local_amp_factor):
            return False, f"CompAmp FAIL at {nu0:.3g} Hz ratio={amp/local:.3g}"
    return True, ""


# ---------------------------------------------------------------------------
# Stingray call wrapper
# ---------------------------------------------------------------------------

def _try_fit_once(
    ps: Powerspectrum,
    *,
    nlor: int,
    t0: List[float],
    priors: Dict[str, Any],
    include_const: bool,
    fitmethod: str,
) -> Tuple[Any, Optional[str]]:
    try:
        parest, res = fit_lorentzians(
            ps, nlor, list(np.asarray(t0, float)),
            fit_whitenoise=include_const,
            max_post=True,
            priors=priors,
            fitmethod=str(fitmethod),
        )
        return res, None
    except Exception as exc:
        return None, repr(exc)


# ---------------------------------------------------------------------------
# QPOFitter class
# ---------------------------------------------------------------------------

class QPOFitter:
    """
    Fits a power density spectrum (fractional-rms normalisation) with a sum of
    Lorentzians (continuum components + optional QPO components).

    v2 changes:
      - Per-component continuum centroid limits (narrow / wide / free).
      - Multi-scale candidate finding in z-score space.
      - cont4 fallback for complex spectral states.
      - Per-bin m in rchi2 computation.
      - Log-uniform priors for amp and fwhm.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        freq,
        power,
        power_err=None,
        *,
        cand_freq=None,
        cand_power=None,
        cand_m_eff=None,
        m=1,
        fit_fmin=0.05,
        fit_fmax=64.0,
        cand_fmin=0.05,
        cand_fmax=10.0,
        const_seed_fmin=30.0,
        include_const=True,
        include_harmonic=False,
        harm_fwhm_lim=(0.03, 8.0),
        harm_amp_factor=5.0,
        # ---- multi-scale smoothing ----
        smooth_scales=None,
        smooth_hz=0.5,
        prominence=0.5,
        min_sep_hz=0.15,
        max_candidates=6,
        cand_require_ksigma=None,
        cand_sigma_mode="cont",
        stage1_n_seeds=3,
        max_qpos=1,
        multi_qpo_ic_delta_min=None,
        multi_qpo_require_improvement=True,
        reseed_enable=True,
        reseed_rchi_bad=1.8,
        reseed_edge_frac=0.08,
        reseed_area_min=0.0,
        reseed_exclude_hz_min=0.5,
        reseed_exclude_df_mult=10.0,
        reseed_prom_factor=1.25,
        reseed_sigma_factor=1.10,
        random_seed=42,
        fitmethod="Powell",
        rchi_target=1.3,
        cont_ic_criterion="bic",
        cont_ic_delta_min=10.0,
        qpo_ic_criterion="aic",
        qpo_ic_delta_min=2.0,
        # ---- per-component centroid limits ----
        cont_x0_narrow_hz=0.3,
        cont_x0_wide_hz=3.0,
        cont_x0_free_hz=8.0,
        cont_x0_max_hz=None,       # legacy: if set, overrides narrow
        cont_fwhm_lim=(0.3, 64.0),
        qpo_fwhm_lim=(0.03, 5.0),
        cont_amp_factor=10.0,
        qpo_amp_factor=5.0,
        qpo_fwhm_frac=0.06,
        qpo_fwhm_min=0.05,
        n_starts=6,
        jitter_frac=0.18,
        seed_peak_hz=None,
        qpo_detect_qmin=3.0,
        guard_enable=True,
        guard_overshoot_ksigma=4.0,
        guard_overshoot_max_run_bins=6,
        guard_overshoot_max_frac=0.10,
        guard_comp_local_amp_factor=6.0,
        max_retries=5,
        eps_amp=1e-30,
        const_cap_factor=5.0,
        postqpo_cont3_enable=True,
        postqpo_cont3_trigger_rchi=1.6,
        postqpo_cont3_rchi_improve_min=0.06,
        postqpo_cont3_ic_delta_min=0.0,
        postqpo_cont3_rchi_not_worse_tol=0.02,
        rchi_override_enable=True,
        rchi_override_threshold=None,
        rchi_override_min_improve=0.02,
        force_cont3_rchi: float = np.inf,
        forced_qpo_seeds: Optional[List[float]] = None,
        # ---- cont4 fallback ----
        cont4_enable: bool = True,
        cont4_trigger_rchi: float = 1.5,
        cont4_ic_criterion: str = "bic",
        cont4_ic_delta_min: float = 3.0,
        # ---- warm start (from QPO_cache / interactive fitter) ----
        warm_start_comps: Optional[Dict[str, Any]] = None,
        **_ignored,
    ):
        # ---- data selection ----
        f_all = np.asarray(freq, float)
        p_all = np.asarray(power, float)
        e_all = None if power_err is None else np.asarray(power_err, float)

        sel = (f_all >= fit_fmin) & (f_all <= fit_fmax) & np.isfinite(p_all) & (p_all > 0)
        self.f = f_all[sel]
        self.p = p_all[sel]
        self.e = None if e_all is None else e_all[sel]

        if self.f.size < 30:
            self._too_few = True
            return
        self._too_few = False

        # ---- per-bin m (for rchi2) ----
        self.m_raw = m                     # keep original
        m_full = _m_as_array(m, f_all.size)
        self.m_arr = m_full[sel]           # per-bin m in the fit band

        # ---- candidate grid ----
        if cand_freq is None or cand_power is None:
            cf, cp = self.f, self.p
        else:
            cf = np.asarray(cand_freq, float)
            cp = np.asarray(cand_power, float)
        self.cf = cf
        self.cp = cp

        # ---- Powerspectrum object for stingray ----
        # Pass scalar m to stingray (it expects m to be consistent with freq length).
        # Per-bin m_arr is used only for our rchi2 computation.
        self.m_fit       = _safe_scalar_m(m)
        self.ps          = self._make_ps(self.f, self.p, self.e, self.m_fit)
        self.df          = float(np.median(np.diff(self.f)))
        self.cand_df     = float(np.median(np.diff(cf))) if cf.size > 1 else self.df
        self.m_cand      = _safe_scalar_m(cand_m_eff if cand_m_eff is not None else m)

        # ---- amplitude seeds ----
        self.c0   = _seed_const(self.f, self.p, const_seed_fmin)
        self.lowf = max(_seed_lowf_level(self.f, self.p, fmax=min(2.0, fit_fmax)), float(eps_amp))

        # ---- per-component continuum centroid ranges ----
        # Legacy compat: cont_x0_max_hz overrides narrow if explicitly set
        if cont_x0_max_hz and np.isfinite(cont_x0_max_hz) and cont_x0_max_hz > 0:
            narrow = float(cont_x0_max_hz)
        else:
            narrow = float(cont_x0_narrow_hz)
        wide = float(cont_x0_wide_hz)
        free = float(cont_x0_free_hz)

        self.cont_x0_lims_narrow = (-narrow, narrow)
        self.cont_x0_lims_wide   = (-wide, wide)
        self.cont_x0_lims_free   = (-free, free)
        # Legacy alias (used by QPO x0 bounds, guardrails, etc.)
        self.cont_x0_lims        = self.cont_x0_lims_narrow

        # ---- amplitude caps ----
        self.cont_amp_cap = float(cont_amp_factor) * self.lowf
        self.qpo_amp_cap  = float(qpo_amp_factor)  * self.lowf
        self.const_cap    = float(max(eps_amp, float(const_cap_factor) * max(self.c0, eps_amp)))

        # ---- base continuum seeds (amp, x0, fwhm) ----
        cf_lim = cont_fwhm_lim
        self.cont1          = (min(self.cont_amp_cap, self.lowf), 0.0, float(np.clip(2.0,  cf_lim[0], cf_lim[1])))
        self.cont2          = (min(self.cont_amp_cap, self.lowf), 0.0, float(np.clip(10.0, cf_lim[0], cf_lim[1])))
        self.cont3_default  = (min(self.cont_amp_cap, self.lowf), 0.0, float(np.clip(30.0, cf_lim[0], cf_lim[1])))
        self.cont4_default  = (min(self.cont_amp_cap, self.lowf * 0.5), 0.0, float(np.clip(0.5, cf_lim[0], cf_lim[1])))

        # ---- store all config knobs ----
        self.include_const   = bool(include_const)
        self.fit_fmin        = float(fit_fmin)
        self.fit_fmax        = float(fit_fmax)
        self.cand_fmin       = float(cand_fmin)
        self.cand_fmax       = float(cand_fmax)
        self.smooth_scales   = list(smooth_scales) if smooth_scales else [float(smooth_hz)]
        self.smooth_hz       = float(smooth_hz)
        self.prominence      = float(prominence)
        self.min_sep_hz      = float(min_sep_hz)
        self.max_candidates  = int(max_candidates)
        self.cand_require_ksigma = cand_require_ksigma
        self.cand_sigma_mode = str(cand_sigma_mode)
        self.stage1_n_seeds  = int(stage1_n_seeds)
        self.max_qpos        = int(max_qpos)
        self.multi_qpo_ic_delta_min       = multi_qpo_ic_delta_min
        self.multi_qpo_require_improvement = bool(multi_qpo_require_improvement)
        self.reseed_enable         = bool(reseed_enable)
        self.reseed_rchi_bad       = float(reseed_rchi_bad)
        self.reseed_edge_frac      = float(reseed_edge_frac)
        self.reseed_area_min       = float(reseed_area_min) if reseed_area_min else 0.0
        self.reseed_exclude_hz_min = float(reseed_exclude_hz_min)
        self.reseed_exclude_df_mult = float(reseed_exclude_df_mult)
        self.reseed_prom_factor    = float(reseed_prom_factor)
        self.reseed_sigma_factor   = float(reseed_sigma_factor)
        self.fitmethod    = str(fitmethod)
        self.rchi_target  = float(rchi_target)
        self.cont_ic_criterion = str(cont_ic_criterion)
        self.cont_ic_delta_min = float(cont_ic_delta_min)
        self.qpo_ic_criterion  = str(qpo_ic_criterion)
        self.qpo_ic_delta_min  = float(qpo_ic_delta_min)
        self.cont_fwhm_lim  = tuple(cont_fwhm_lim)
        self.qpo_fwhm_lim   = tuple(qpo_fwhm_lim)
        self.qpo_fwhm_frac  = float(qpo_fwhm_frac)
        self.qpo_fwhm_min   = float(qpo_fwhm_min)
        self.n_starts       = int(n_starts)
        self.jitter_frac    = float(jitter_frac)
        self.seed_peak_hz   = float(seed_peak_hz) if (seed_peak_hz is not None and np.isfinite(seed_peak_hz)) else None
        self.qpo_detect_qmin = float(qpo_detect_qmin)
        self.guard_enable   = bool(guard_enable)
        self.guard_overshoot_ksigma       = float(guard_overshoot_ksigma)
        self.guard_overshoot_max_run_bins = int(guard_overshoot_max_run_bins)
        self.guard_overshoot_max_frac     = float(guard_overshoot_max_frac)
        self.guard_comp_local_amp_factor  = float(guard_comp_local_amp_factor)
        self.max_retries    = int(max_retries)
        self.eps_amp        = float(eps_amp)
        self.postqpo_cont3_enable           = bool(postqpo_cont3_enable)
        self.postqpo_cont3_trigger_rchi     = float(postqpo_cont3_trigger_rchi)
        self.postqpo_cont3_rchi_improve_min = float(postqpo_cont3_rchi_improve_min)
        self.postqpo_cont3_ic_delta_min     = float(postqpo_cont3_ic_delta_min)
        self.postqpo_cont3_rchi_not_worse_tol = float(postqpo_cont3_rchi_not_worse_tol)
        self.rchi_override_enable    = bool(rchi_override_enable)
        self.rchi_override_threshold = (
            float(rchi_override_threshold)
            if (rchi_override_threshold is not None and np.isfinite(rchi_override_threshold))
            else None
        )
        self.rchi_override_min_improve = float(rchi_override_min_improve)
        self.force_cont3_rchi = float(force_cont3_rchi) if np.isfinite(force_cont3_rchi) else np.inf
        self.forced_qpo_seeds: List[float] = [
            float(s) for s in (forced_qpo_seeds or [])
            if np.isfinite(s) and self.cand_fmin <= float(s) <= self.cand_fmax
        ]
        # cont4
        self.cont4_enable       = bool(cont4_enable)
        self.cont4_trigger_rchi = float(cont4_trigger_rchi)
        self.cont4_ic_criterion = str(cont4_ic_criterion)
        self.cont4_ic_delta_min = float(cont4_ic_delta_min)

        seed = int(random_seed) if random_seed is not None else 12345
        self.rng = np.random.default_rng(seed)

        # ---- warm start override ----
        # Must be last: overrides cont1/cont2/cont3_default and forced_qpo_seeds
        # that were set above. All new params default to None so existing callers
        # are completely unaffected.
        if warm_start_comps is not None:
            self._apply_warm_start(warm_start_comps, cont_fwhm_lim, float(eps_amp))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> FitResult:
        """
        Two-track fitting pipeline.

        Track 1 (Smart): Analytical continuum + peak-shape QPO estimation
        → enumerate ~6 model configs → polish each with 2-3 optimizer calls.
        Total: ~15-20 optimizer calls.

        Track 2 (Fallback): If smart path gives rchi2 > target, run up to
        2 brute-force attempts with the full seed-screening machinery.
        """
        if self._too_few:
            return self._make_failure("Too few bins in fit band")

        # ---- Track 1: smart path ----
        best = self._run_smart()

        # ---- Track 2: limited brute-force fallback ----
        if best is None or (np.isfinite(best.rchi2) and best.rchi2 > self.rchi_target):
            ladder = self._build_ladder()
            first  = ladder[0]
            fit_cont = self._fit_continuum(
                first["fitmethod"], first["jitter"], first["n_starts"]
            )
            if fit_cont is not None:
                for att in ladder[:2]:
                    result = self._attempt(fit_cont=fit_cont, **att)
                    if result is None:
                        continue
                    if best is None or self._rchi_is_better(result, best):
                        best = result
                    if np.isfinite(best.rchi2) and best.rchi2 <= self.rchi_target:
                        break

        if best is None:
            return self._make_failure("All fitting approaches failed")
        return best

    # ------------------------------------------------------------------
    # Track 1: Analytical estimation → polish
    # ------------------------------------------------------------------

    def _run_smart(self) -> Optional[FitResult]:
        """
        Smart fitting: analytical seed → one-shot polish.

        Steps:
          1. Estimate continuum via iterative peak-masked smoothing.
          2. Decompose smooth continuum into Lorentzians via curve_fit (~5ms).
          3. Find QPOs in the continuum-subtracted residual.
          4. Estimate each QPO's (nu0, fwhm, amp) from peak shape.
          5. Build ~6 model configurations.
          6. Polish each with 2-3 optimizer calls via _run_stage.
          7. Select best by IC.
        """
        cont_smooth = self._estimate_continuum_shape()
        cont2, noise2 = self._decompose_continuum(cont_smooth, n_lor=2)
        cont3, noise3 = self._decompose_continuum(cont_smooth, n_lor=3)
        qpos = self._find_qpos_in_residual(cont2, noise2)
        configs = self._build_smart_configs(cont2, noise2, cont3, noise3, qpos)

        # ── Collect fits grouped by (nq_post, n_cont_post) ──────────────
        #
        # nq_post    : quality QPOs after fitting (from fit.comp_types, which
        #              _wrap has already corrected — low-Q QPO-typed → "cont").
        # n_cont_post: actual continuum count after fitting = nlor - nq_post.
        #
        # Why this key (root-cause of the cont=4/5 bugs):
        #   Using only nq_post caused relabelled fits with more continuum
        #   components to compete against simpler models without any IC gate.
        #   E.g. cont3+QPO where the QPO→cont gives nq_post=0, n_cont_post=4.
        #   Using (nq_post, n_cont_post) keeps each distinct component-count
        #   model in its own bucket so upgrades are always IC-gated.
        #
        # max_cont cap: enforce the cont4 switch.  Any fit where the post-fit
        #   continuum count exceeds max_cont is discarded before IC comparison.
        #   This prevents relabelled QPO-typed components from inflating the
        #   effective continuum order beyond what the user has enabled.
        max_cont = 4 if self.cont4_enable else 3

        best_by_key: Dict[tuple, FitResult] = {}
        for comps, x0l, fwl, ampc, ctypes, tag in configs:
            fit = self._run_stage(
                comps=comps, x0_lims=x0l, fwhm_lims=fwl,
                amp_caps=ampc, comp_types=ctypes,
                fitmethod=self.fitmethod, jitter=self.jitter_frac,
                n_starts=max(2, self.n_starts // 2), tag=tag,
            )
            if fit is None:
                continue
            nq_post    = sum(1 for t in fit.comp_types if t == "qpo")
            n_cont_post = int(fit.nlor) - nq_post
            # Hard cap: respect FIT_CONT4_ENABLE
            if n_cont_post > max_cont:
                continue
            key = (nq_post, n_cont_post)
            if key not in best_by_key or (
                _ic_value(fit, self.qpo_ic_criterion)
                < _ic_value(best_by_key[key], self.qpo_ic_criterion)
            ):
                best_by_key[key] = fit

        # ── Step 1: best continuum-only model (0 QPOs) ───────────────────
        # cont2 (nc=2) vs cont3 (nc=3) gated by cont_ic_delta_min.
        # A relabelled cont2+QPO has nc=3 and legitimately competes with
        # cont3 — both have the same number of free parameters.
        fit_c2 = best_by_key.get((0, 2))
        fit_c3 = best_by_key.get((0, 3))
        if fit_c2 is None:
            fit_0q = fit_c3
        elif fit_c3 is None:
            fit_0q = fit_c2
        else:
            fit_0q = (fit_c3 if _accept_upgrade_ic(
                fit_c2, fit_c3,
                criterion=self.cont_ic_criterion,
                delta_min=self.cont_ic_delta_min,
            ) else fit_c2)

        # ── Step 2: best 1-QPO fit (select across nc values by IC) ───────
        fit_1q: Optional[FitResult] = None
        for nc in range(2, max_cont + 1):
            cand = best_by_key.get((1, nc))
            if cand is None:
                continue
            if fit_1q is None or (
                _ic_value(cand, self.qpo_ic_criterion)
                < _ic_value(fit_1q, self.qpo_ic_criterion)
            ):
                fit_1q = cand

        # ── Step 3: best 2-QPO fit (select across nc values by IC) ───────
        fit_2q: Optional[FitResult] = None
        for nc in range(2, max_cont + 1):
            cand = best_by_key.get((2, nc))
            if cand is None:
                continue
            if fit_2q is None or (
                _ic_value(cand, self.qpo_ic_criterion)
                < _ic_value(fit_2q, self.qpo_ic_criterion)
            ):
                fit_2q = cand

        # ── Step 4: 0-QPO → 1-QPO upgrade (qpo_ic_delta_min) ────────────
        if fit_0q is None:
            best = fit_1q
        elif fit_1q is None:
            best = fit_0q
        else:
            best = (fit_1q if _accept_upgrade_ic(
                fit_0q, fit_1q,
                criterion=self.qpo_ic_criterion,
                delta_min=self.qpo_ic_delta_min,
            ) else fit_0q)

        # ── Step 5: 1-QPO → 2-QPO upgrade (multi_qpo_ic_delta_min) ──────
        multi_delta = (
            float(self.multi_qpo_ic_delta_min)
            if self.multi_qpo_ic_delta_min is not None
            else float(self.qpo_ic_delta_min)
        )
        if fit_2q is not None:
            ref = best if best is not None else fit_0q
            if ref is None or _accept_upgrade_ic(
                ref, fit_2q,
                criterion=self.qpo_ic_criterion,
                delta_min=multi_delta,
            ):
                best = fit_2q

        if best is not None and np.isfinite(best.rchi2) and best.rchi2 > self.rchi_target:
            refit = self._repolish(best, "Nelder-Mead")
            if refit is not None and self._rchi_is_better(refit, best):
                best = refit

        if best is not None:
            best.meta = best.meta or {}
            best.meta["track"] = "smart"
        return best

    # --- analytical continuum estimation ---

    def _estimate_continuum_shape(self) -> np.ndarray:
        """
        Iterative peak-masked median smoothing.

        Returns a smooth array tracing the continuum with QPO peaks removed.
        """
        f, p = self.f, self.p
        n = len(p)
        w = max(7, int(np.round(1.0 / max(self.df, 1e-6))))
        if w % 2 == 0:
            w += 1
        w = min(w, n - 2)
        if w < 3:
            return p.copy()

        cont = p.copy()
        m_eff = max(1, self.m_fit)

        for _ in range(3):
            smooth = _rolling_median(cont, w)
            good = np.isfinite(smooth) & (smooth > 0)
            if not np.any(good):
                break
            med = float(np.nanmedian(smooth[good]))
            smooth = np.where(good, smooth, med)
            sigma = smooth / np.sqrt(float(m_eff))
            excess = (p - smooth) / np.where(sigma > 0, sigma, 1.0)
            is_peak = excess > 2.5
            if not np.any(is_peak):
                cont = smooth
                break
            cont = np.where(is_peak, smooth, p)
        return cont

    def _decompose_continuum(
        self, cont_smooth: np.ndarray, n_lor: int = 2,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Fit n_lor zero-centred Lorentzians + constant to the smooth
        continuum via scipy.optimize.curve_fit (~5ms, not Whittle).
        """
        from scipy.optimize import curve_fit

        f, target = self.f, np.asarray(cont_smooth, float)
        good = np.isfinite(target) & (target > 0)
        if np.sum(good) < 20:
            return self._default_cont_params(n_lor)

        fg, tg = f[good], target[good]
        sigma_w = np.maximum(tg, 1e-30)
        noise_est = float(self.c0)
        lowf_band = fg < min(2.0, fg[-1] * 0.1)
        lowf_est = float(np.nanmedian(tg[lowf_band])) if np.any(lowf_band) else float(tg[0])

        if n_lor == 2:
            def model(freq, a1, w1, a2, w2, c):
                return lorentz(freq, 0, w1, a1) + lorentz(freq, 0, w2, a2) + c
            p0 = [lowf_est, 2.0, lowf_est * 0.3, 12.0, noise_est]
            blo = [1e-30, 0.1, 1e-30, 0.5, 0.0]
            bhi = [lowf_est * 20, 100, lowf_est * 20, 200, noise_est * 10]
        elif n_lor == 3:
            def model(freq, a1, w1, a2, w2, a3, w3, c):
                return (lorentz(freq, 0, w1, a1) + lorentz(freq, 0, w2, a2)
                        + lorentz(freq, 0, w3, a3) + c)
            p0 = [lowf_est, 1.5, lowf_est * 0.3, 8.0, lowf_est * 0.1, 30.0, noise_est]
            blo = [1e-30, 0.1, 1e-30, 0.3, 1e-30, 0.5, 0.0]
            bhi = [lowf_est * 20, 100, lowf_est * 20, 200, lowf_est * 20, 200, noise_est * 10]
        else:
            return self._default_cont_params(n_lor)

        try:
            popt, _ = curve_fit(model, fg, tg, p0=p0, bounds=(blo, bhi),
                                sigma=sigma_w, maxfev=300)
            if n_lor == 2:
                comps = [(float(popt[0]), 0.0, float(popt[1])),
                         (float(popt[2]), 0.0, float(popt[3]))]
                return comps, float(popt[4])
            else:
                comps = [(float(popt[0]), 0.0, float(popt[1])),
                         (float(popt[2]), 0.0, float(popt[3])),
                         (float(popt[4]), 0.0, float(popt[5]))]
                return comps, float(popt[6])
        except Exception:
            return self._default_cont_params(n_lor)

    def _default_cont_params(self, n_lor: int):
        if n_lor >= 3:
            return [self.cont1, self.cont2, self.cont3_default], self.c0
        return [self.cont1, self.cont2], self.c0

    # --- QPO finding in residual ---

    def _find_qpos_in_residual(
        self,
        cont_params: List[Tuple[float, float, float]],
        noise: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Find QPO peaks in the continuum-subtracted residual.
        Estimate each peak's (amp, nu0, fwhm) from its shape — no optimizer.

        v3.1: Merges self.forced_qpo_seeds (cross-band hints) with residual
        peaks.  For each forced seed, estimates (amp, fwhm) from the residual
        at that frequency, even if the residual peak is too weak to be found
        by find_peaks alone.
        """
        f, p = self.f, self.p
        cont_model = np.full_like(f, float(noise))
        for amp, nu0, fwhm in cont_params:
            cont_model += lorentz(f, nu0, fwhm, amp)

        resid = p - cont_model
        band = (f >= self.cand_fmin) & (f <= self.cand_fmax)
        if np.sum(band) < 10:
            # Even with no band, try to use forced seeds
            return self._params_from_forced_seeds(f, resid)

        fb, rb = f[band], resid[band]
        sigma_r = np.abs(cont_model[band]) / np.sqrt(max(1, self.m_fit))
        z = rb / np.where(sigma_r > 0, sigma_r, 1.0)

        df_b = float(np.median(np.diff(fb))) if len(fb) > 1 else self.df
        distance = int(max(1, np.round(self.min_sep_hz / df_b)))
        peaks, props = scipy.signal.find_peaks(z, prominence=1.5, distance=distance)

        out: List[Tuple[float, float, float]] = []

        if peaks.size > 0:
            prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), float)
            order = np.argsort(prom)[::-1]
            # FIX: use max_qpos as the natural ceiling; floor at 1 (not 2)
            # so that the returned seed list respects the user-configured limit.
            peaks = peaks[order][:max(self.max_qpos, 1)]

            for pidx in peaks:
                nu0, fwhm, amp = self._estimate_peak_shape(fb, rb, int(pidx))
                if amp > 0 and fwhm > 0 and self.cand_fmin <= nu0 <= self.cand_fmax:
                    out.append((float(amp), float(nu0), float(fwhm)))

        # Merge cross-band forced seeds.  For each forced seed not already
        # covered by a residual peak, estimate (amp, fwhm) from the residual
        # at that frequency.  This is the key bridge between cross-band
        # reseeding and the smart path.
        sep_tol = max(self.min_sep_hz, 5.0 * self.df)
        for fs in self.forced_qpo_seeds:
            if any(abs(fs - q[1]) < sep_tol for q in out):
                continue   # already have a peak near this frequency
            # Find nearest bin in fit band
            i_full = int(np.argmin(np.abs(f - fs)))
            if not (self.cand_fmin <= f[i_full] <= self.cand_fmax):
                continue
            # Estimate amp from a small median window around the seed
            # (use raw power, not residual, in case continuum subtraction
            # over-corrected at this frequency)
            half_w = max(2, int(np.round(0.05 / max(self.df, 1e-6))))
            lo, hi = max(0, i_full - half_w), min(len(p), i_full + half_w + 1)
            local_power = float(np.nanmedian(p[lo:hi]))
            local_cont  = float(np.nanmedian(cont_model[lo:hi]))
            seed_amp = max(local_power - local_cont, local_power * 0.1)
            if seed_amp <= 0:
                continue
            # Estimate fwhm from peak shape if there's a residual bump,
            # else fall back to a narrow default consistent with a QPO.
            i_band = int(np.argmin(np.abs(fb - fs)))
            if 0 <= i_band < len(rb) and rb[i_band] > 0:
                _, fwhm_est, _ = self._estimate_peak_shape(fb, rb, i_band)
            else:
                fwhm_est = max(self.qpo_fwhm_min, self.qpo_fwhm_frac * fs)
            fwhm_est = float(np.clip(fwhm_est, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))
            out.append((float(seed_amp), float(fs), float(fwhm_est)))

        # Sort by frequency for stable downstream behaviour
        out.sort(key=lambda q: q[1])
        # FIX: cap at max(max_qpos, forced_seeds).  The old code had a
        # hard floor of 2 which allowed extra seeds to leak through even
        # when max_qpos=1.  Forced seeds are still honoured above max_qpos
        # because they carry cross-band physical information.
        return out[:max(self.max_qpos, len(self.forced_qpo_seeds))]

    def _params_from_forced_seeds(
        self, f: np.ndarray, resid: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Fallback: build QPO seeds purely from forced_qpo_seeds (no peak finding)."""
        out: List[Tuple[float, float, float]] = []
        for fs in self.forced_qpo_seeds:
            i = int(np.argmin(np.abs(f - fs)))
            if not (self.cand_fmin <= f[i] <= self.cand_fmax):
                continue
            half_w = max(2, int(np.round(0.05 / max(self.df, 1e-6))))
            lo, hi = max(0, i - half_w), min(len(self.p), i + half_w + 1)
            seed_amp = float(np.nanmedian(self.p[lo:hi])) * 0.5
            fwhm_est = float(np.clip(
                max(self.qpo_fwhm_min, self.qpo_fwhm_frac * fs),
                self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1],
            ))
            if seed_amp > 0:
                out.append((float(seed_amp), float(fs), float(fwhm_est)))
        return out

    def _estimate_peak_shape(
        self, f: np.ndarray, resid: np.ndarray, peak_idx: int
    ) -> Tuple[float, float, float]:
        """Estimate (nu0, fwhm, amp) from half-maximum interpolation."""
        nu0 = float(f[peak_idx])
        amp = float(resid[peak_idx])
        if amp <= 0:
            return nu0, self.qpo_fwhm_min, 0.0
        half = amp / 2.0
        left = peak_idx
        while left > 0 and resid[left] > half:
            left -= 1
        right = peak_idx
        while right < len(resid) - 1 and resid[right] > half:
            right += 1
        fwhm = float(f[min(right, len(f) - 1)] - f[max(left, 0)])
        fwhm = float(np.clip(fwhm, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))
        return nu0, fwhm, amp

    # --- smart model configuration builder ---

    def _build_smart_configs(self, c2, n2, c3, n3, qpos):
        """
        Build ~3-6 model configurations from analytical estimates.

        Each config is (comps, x0_lims, fwhm_lims, amp_caps, comp_types, tag).
        """
        configs = []

        def _xlims(params):
            lims = []
            for i, _ in enumerate(params):
                if i == 0:   lims.append(self.cont_x0_lims_narrow)
                elif i == 1: lims.append(self.cont_x0_lims_wide)
                else:        lims.append(self.cont_x0_lims_free)
            return lims

        def _cc(params):
            return [(float(np.clip(a, self.eps_amp, self.cont_amp_cap)),
                     float(np.clip(x, self.cont_x0_lims_free[0], self.cont_x0_lims_free[1])),
                     float(np.clip(w, self.cont_fwhm_lim[0], self.cont_fwhm_lim[1])))
                    for a, x, w in params]

        def _cq(params):
            return [(float(np.clip(a, self.eps_amp, self.qpo_amp_cap)),
                     float(np.clip(x, self.cand_fmin, self.cand_fmax)),
                     float(np.clip(w, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1])))
                    for a, x, w in params]

        cc2, cc3 = _cc(c2), _cc(c3)

        # A: cont2 only
        configs.append((cc2, _xlims(cc2), [self.cont_fwhm_lim]*2,
                        [self.cont_amp_cap]*2, ["cont"]*2, "smart_cont2"))
        # B: cont3 only
        configs.append((cc3, _xlims(cc3), [self.cont_fwhm_lim]*3,
                        [self.cont_amp_cap]*3, ["cont"]*3, "smart_cont3"))

        if qpos:
            q = _cq(qpos)

            # Single-QPO configs: try each QPO individually with cont2 and cont3.
            # When there are multiple seeds (e.g. forced from cross-band), each
            # gets its own single-QPO trial — otherwise the IC selector only
            # ever sees the lowest-frequency seed paired with the continuum.
            for qi, qq in enumerate(q):
                configs.append((
                    cc2 + [qq],
                    _xlims(cc2) + [(self.cand_fmin, self.cand_fmax)],
                    [self.cont_fwhm_lim]*2 + [self.qpo_fwhm_lim],
                    [self.cont_amp_cap]*2 + [self.qpo_amp_cap],
                    ["cont", "cont", "qpo"],
                    f"smart_c2+q@{qq[1]:.2g}Hz",
                ))
                configs.append((
                    cc3 + [qq],
                    _xlims(cc3) + [(self.cand_fmin, self.cand_fmax)],
                    [self.cont_fwhm_lim]*3 + [self.qpo_fwhm_lim],
                    [self.cont_amp_cap]*3 + [self.qpo_amp_cap],
                    ["cont", "cont", "cont", "qpo"],
                    f"smart_c3+q@{qq[1]:.2g}Hz",
                ))

            if len(q) >= 2 and self.max_qpos >= 2:
                # E: cont2 + 2QPO (top two by area heuristic — first two in
                # the freq-sorted list, which will exercise both extremes).
                configs.append((
                    cc2 + [q[0], q[1]],
                    _xlims(cc2) + [(self.cand_fmin, self.cand_fmax)]*2,
                    [self.cont_fwhm_lim]*2 + [self.qpo_fwhm_lim]*2,
                    [self.cont_amp_cap]*2 + [self.qpo_amp_cap]*2,
                    ["cont", "cont", "qpo", "qpo"],
                    f"smart_c2+2q@{q[0][1]:.2g}+{q[1][1]:.2g}Hz",
                ))
                # F: cont3 + 2QPO
                configs.append((
                    cc3 + [q[0], q[1]],
                    _xlims(cc3) + [(self.cand_fmin, self.cand_fmax)]*2,
                    [self.cont_fwhm_lim]*3 + [self.qpo_fwhm_lim]*2,
                    [self.cont_amp_cap]*3 + [self.qpo_amp_cap]*2,
                    ["cont", "cont", "cont", "qpo", "qpo"],
                    f"smart_c3+2q@{q[0][1]:.2g}+{q[1][1]:.2g}Hz",
                ))
        return configs

    def _repolish(self, fit: FitResult, fitmethod: str = "Nelder-Mead") -> Optional[FitResult]:
        """Re-polish an existing fit with a different optimizer."""
        if fit.pars.ndim != 2 or fit.pars.shape[0] < 1:
            return None
        comps, x0l, fwl, ampc, ctypes = [], [], [], [], []
        stored = list(getattr(fit, "comp_types", []) or [])
        for idx, (nu0, fwhm, amp) in enumerate(fit.pars):
            nu0, fwhm, amp = float(nu0), float(fwhm), float(amp)
            ct = stored[idx] if idx < len(stored) else "cont"
            if ct == "qpo":
                comps.append((float(np.clip(amp, self.eps_amp, self.qpo_amp_cap)), nu0,
                              float(np.clip(fwhm, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))))
                x0l.append((self.cand_fmin, self.cand_fmax))
                fwl.append(self.qpo_fwhm_lim)
                ampc.append(self.qpo_amp_cap)
            else:
                comps.append((float(np.clip(amp, self.eps_amp, self.cont_amp_cap)), nu0,
                              float(np.clip(fwhm, self.cont_fwhm_lim[0], self.cont_fwhm_lim[1]))))
                x0l.append(self._infer_cont_x0_lims(nu0, fwhm))
                fwl.append(self.cont_fwhm_lim)
                ampc.append(self.cont_amp_cap)
            ctypes.append(ct)
        return self._run_stage(comps=comps, x0_lims=x0l, fwhm_lims=fwl,
                               amp_caps=ampc, comp_types=ctypes,
                               fitmethod=fitmethod, jitter=self.jitter_frac,
                               n_starts=3, tag="repolish")

    # ------------------------------------------------------------------
    # Attempt runner
    # ------------------------------------------------------------------

    def _attempt(
        self, *, fit_cont: FitResult, fitmethod: str, jitter: float, n_starts: int, label: str
    ) -> Optional[FitResult]:
        """
        One fitting attempt using a pre-fitted continuum.

        v2 changes:
          - Continuum is pre-fitted (passed in), not refitted per attempt.
          - Stage 1 tries simultaneous 2-QPO pairs when max_qpos >= 2.
          - _grow_qpos uses residual-based candidate finding.
          - cont3+QPO rescue fires for nlor >= 3 (not just == 3).
        """
        fit_cont.meta = fit_cont.meta or {}
        fit_cont.meta.update(dict(retry_label=label, optimizer=fitmethod, jitter=jitter, n_starts=n_starts))

        thr = self.rchi_override_threshold if self.rchi_override_threshold is not None \
              else self.postqpo_cont3_trigger_rchi
        cont_bad     = np.isfinite(fit_cont.rchi2) and float(fit_cont.rchi2) > thr
        override_active = self.rchi_override_enable and cont_bad
        fit_cont.meta["override_active_at_continuum"] = bool(override_active)
        fit_cont.meta["override_thr"]                 = float(thr)

        # ---------- Build initial candidates (multi-scale) ----------
        cands, seeds = self._build_candidates(
            prom=self.prominence,
            require_ks=self.cand_require_ksigma,
            exclude_center=None,
            exclude_hw=0.0,
            max_cands=self.max_candidates,
            seed_peak=self.seed_peak_hz,
            k=self.stage1_n_seeds,
            forced_seeds=self.forced_qpo_seeds,
        )

        # Relaxed fallback if no seeds found
        if not seeds:
            if override_active or (self.reseed_enable and self._cont_rchi_bad(fit_cont)):
                cands2, seeds2 = self._build_candidates(
                    prom=self.prominence / max(1e-6, self.reseed_prom_factor),
                    require_ks=self._relax_ks(self.cand_require_ksigma, self.reseed_sigma_factor),
                    exclude_center=None, exclude_hw=0.0,
                    max_cands=max(self.max_candidates, 2*self.stage1_n_seeds),
                    seed_peak=self.seed_peak_hz,
                    k=max(self.stage1_n_seeds, 2),
                    forced_seeds=self.forced_qpo_seeds,
                )
                if seeds2:
                    cands, seeds = cands2, seeds2

        if not seeds:
            fit_cont.message = "OK (continuum-only; no candidate seeds)"
            fit_cont.meta["cands"] = cands
            return fit_cont

        cand_tol_hz = max(2.0 * self.cand_df, 0.05)

        # Use reduced n_starts for seed screening, full for polishing
        screen_ns = max(2, n_starts // 3)

        # ---------- Stage 1: single-QPO seed screening + polish ----------
        fit_best, best_qpo_real, best_seed = self._stage1_best_qpo(
            seeds, cands, fit_cont, fitmethod, jitter,
            screen_n_starts=screen_ns, polish_n_starts=n_starts,
            override_relax_ic=override_active,
            cand_tol_hz=cand_tol_hz,
        )

        # ---------- Stage 1b: simultaneous 2-QPO pairs ----------
        if self.max_qpos >= 2 and len(seeds) >= 2:
            fit_2q = self._stage1_2qpo_pairs(
                seeds, cands, fit_cont, fitmethod, jitter,
                screen_n_starts=screen_ns, polish_n_starts=n_starts,
                override_relax_ic=override_active,
                cand_tol_hz=cand_tol_hz,
            )
            if fit_2q is not None:
                # FIX: compare 2-QPO result against the best 1-QPO fit using
                # multi_qpo_ic_delta_min, not against the continuum.  The old
                # code accepted any AIC improvement (delta=0 effectively), which
                # let marginal sidelobe-like secondary QPOs through.
                multi_delta = (
                    float(self.multi_qpo_ic_delta_min)
                    if self.multi_qpo_ic_delta_min is not None
                    else float(self.qpo_ic_delta_min)
                )
                ref_for_2q = fit_best if fit_best is not None else fit_cont
                accept_2q  = (
                    override_active
                    or fit_best is None
                    or _accept_upgrade_ic(
                        ref_for_2q, fit_2q,
                        criterion=self.qpo_ic_criterion,
                        delta_min=multi_delta,
                    )
                )
                if accept_2q:
                    fit_best = fit_2q
                    best_qpo_real = self._detect_qpo(fit_2q)
                    best_seed = None  # came from a pair

        # No QPO accepted: conditional reseed
        if fit_best is None:
            cands_r, seeds_r = self._build_candidates(
                prom=self.prominence / max(1e-6, self.reseed_prom_factor),
                require_ks=self._relax_ks(self.cand_require_ksigma, self.reseed_sigma_factor),
                exclude_center=None, exclude_hw=0.0,
                max_cands=max(self.max_candidates, 2*self.stage1_n_seeds),
                seed_peak=self.seed_peak_hz,
                k=max(self.stage1_n_seeds, 2),
            )
            if seeds_r:
                fit_best, best_qpo_real, best_seed = self._stage1_best_qpo(
                    seeds_r, cands_r, fit_cont, fitmethod, jitter,
                    screen_n_starts=screen_ns, polish_n_starts=n_starts,
                    override_relax_ic=bool(override_active),
                    cand_tol_hz=cand_tol_hz,
                )
                if fit_best is not None:
                    cands, seeds = cands_r, seeds_r

        if fit_best is None or best_qpo_real is None:
            fit_cont.message = "OK (continuum-only; Stage1+reseed rejected QPO)"
            fit_cont.meta.update({"cands": cands})
            return fit_cont

        fit_best.meta = fit_best.meta or {}
        fit_best.meta.update({"cands": cands, "override_active": bool(override_active)})
        if best_seed is not None:
            fit_best.meta["seed_hz"] = float(best_seed)

        # ---------- Reseed ladder ----------
        if self.reseed_enable and np.isfinite(fit_best.rchi2):
            rbad  = float(fit_best.rchi2) > self.reseed_rchi_bad
            qnu   = float(best_qpo_real.get("qpo_nu0_hz", np.nan))
            edgey = self._is_edge(qnu)
            tiny  = self._is_tiny_area(fit_best, best_qpo_real)

            if rbad or edgey or tiny:
                excl_hw = max(self.reseed_exclude_hz_min, self.reseed_exclude_df_mult * self.df)
                cands_r, seeds_r = self._build_candidates(
                    prom=self.prominence * self.reseed_prom_factor,
                    require_ks=self._tighten_ks(self.cand_require_ksigma, self.reseed_sigma_factor),
                    exclude_center=qnu if np.isfinite(qnu) else None,
                    exclude_hw=excl_hw,
                    max_cands=max(self.max_candidates, 2*self.stage1_n_seeds),
                    seed_peak=self.seed_peak_hz,
                    k=max(self.stage1_n_seeds, 2),
                )
                fb2, bqr2, bs2 = self._stage1_best_qpo(
                    seeds_r, cands_r, fit_cont, fitmethod, jitter,
                    screen_n_starts=screen_ns, polish_n_starts=n_starts,
                    override_relax_ic=bool(override_active),
                    cand_tol_hz=cand_tol_hz,
                )
                if fb2 is not None and self._rchi_is_better(fb2, fit_best):
                    fit_best     = fb2
                    best_qpo_real = bqr2
                    best_seed    = bs2
                    fit_best.meta = fit_best.meta or {}
                    fit_best.meta.update({
                        "reseed_triggered": True, "reseed_reason": (rbad, edgey, tiny),
                        "reseed_improved": True,
                    })
                else:
                    fit_best.meta["reseed_triggered"] = True
                    fit_best.meta["reseed_improved"]  = False

        # ---------- Multi-QPO growth (residual-based) ----------
        if self.max_qpos > 1:
            fit_best = self._grow_qpos(fit_best, cands, fitmethod, jitter, n_starts, override_active)

        # ---------- cont3+QPO rescue ----------
        # FIXED: fires for nlor >= 3 (was nlor == 3, blocking rescue after 2-QPO fits)
        do_override  = self.rchi_override_enable and np.isfinite(fit_best.rchi2) and float(fit_best.rchi2) > thr
        do_postqpo   = (self.postqpo_cont3_enable and np.isfinite(fit_best.rchi2)
                        and float(fit_best.rchi2) > self.postqpo_cont3_trigger_rchi)
        n_cont = sum(1 for t in getattr(fit_best, "comp_types", []) if t == "cont")
        n_qpo  = sum(1 for t in getattr(fit_best, "comp_types", []) if t == "qpo")
        try_rescue = (n_cont <= 2 and n_qpo >= 1) and (do_override or do_postqpo) and isinstance(best_qpo_real, dict)

        if try_rescue:
            fit_best = self._cont3_qpo_rescue(
                fit_best, best_qpo_real, thr, fitmethod, jitter, n_starts,
                do_override=do_override,
            )

        # ---------- cont4 fallback ----------
        if (self.cont4_enable
                and np.isfinite(fit_best.rchi2)
                and float(fit_best.rchi2) > self.cont4_trigger_rchi):
            fit_best = self._try_cont4(fit_best, best_qpo_real, fitmethod, jitter, n_starts)

        return fit_best

    # ------------------------------------------------------------------
    # Continuum fitting (per-component x0_lims)
    # ------------------------------------------------------------------

    def _fit_continuum(
        self, fitmethod: str, jitter: float, n_starts: int
    ) -> Optional[FitResult]:
        """
        Fit cont2 and cont3, select by IC.  Returns None on hard crash.

        cont1 uses narrow centroid limits (broadest Lorentzian, stays near 0).
        cont2 uses wide limits (Lb can sit at 0.5–3 Hz).
        cont3 uses free limits (anywhere in the low-frequency range).
        """
        fit2 = self._run_stage(
            comps=[self.cont1, self.cont2],
            x0_lims=[self.cont_x0_lims_narrow, self.cont_x0_lims_wide],
            fwhm_lims=[self.cont_fwhm_lim, self.cont_fwhm_lim],
            amp_caps=[self.cont_amp_cap, self.cont_amp_cap],
            comp_types=["cont", "cont"],
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag="cont2",
        )
        if fit2 is None:
            return None

        cont3_seed = self._seed_cont3_residual(fit2)
        fit3 = self._run_stage(
            comps=[self.cont1, self.cont2, cont3_seed],
            x0_lims=[self.cont_x0_lims_narrow, self.cont_x0_lims_wide, self.cont_x0_lims_free],
            fwhm_lims=[self.cont_fwhm_lim]*3,
            amp_caps=[self.cont_amp_cap]*3,
            comp_types=["cont", "cont", "cont"],
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag="cont3",
        )

        if fit3 is None:
            return fit2

        if _accept_upgrade_ic(fit2, fit3, criterion=self.cont_ic_criterion, delta_min=self.cont_ic_delta_min):
            fit3.message = f"OK (cont3 accepted by {self.cont_ic_criterion.upper()})"
            return fit3

        cont2_rchi = float(getattr(fit2, "rchi2", np.nan))
        cont3_rchi = float(getattr(fit3, "rchi2", np.nan))
        if (np.isfinite(cont2_rchi)
                and cont2_rchi > self.force_cont3_rchi
                and np.isfinite(cont3_rchi)
                and cont3_rchi <= cont2_rchi + 0.05):
            fit3.message = (
                f"OK (cont3 forced — cont2 rchi2={cont2_rchi:.3f} "
                f"> threshold={self.force_cont3_rchi:.3f})"
            )
            fit3.meta = fit3.meta or {}
            fit3.meta["cont3_forced"] = True
            fit3.meta["cont2_rchi2"]  = cont2_rchi
            return fit3

        return fit2

    # ------------------------------------------------------------------
    # cont4 fallback
    # ------------------------------------------------------------------

    def _try_cont4(
        self, fit_best: FitResult, best_qpo_real: Optional[Dict],
        fitmethod: str, jitter: float, n_starts: int
    ) -> FitResult:
        """
        Try adding a fourth continuum Lorentzian (free centroid).
        Only accepted if BIC improves by cont4_ic_delta_min.
        """
        if not hasattr(fit_best, "pars") or fit_best.pars.ndim != 2:
            return fit_best

        comps, x0_lims, fwhm_lims, amp_caps, comp_types = [], [], [], [], []
        stored_types = list(getattr(fit_best, "comp_types", []) or [])

        for idx, (nu0_i, fwhm_i, amp_i) in enumerate(fit_best.pars):
            nu0_i, fwhm_i, amp_i = float(nu0_i), float(fwhm_i), float(amp_i)
            ctype = stored_types[idx] if idx < len(stored_types) else "cont"

            if ctype == "qpo":
                comps.append((float(np.clip(amp_i, self.eps_amp, self.qpo_amp_cap)),
                              nu0_i,
                              float(np.clip(fwhm_i, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))))
                x0_lims.append((self.cand_fmin, self.cand_fmax))
                fwhm_lims.append(self.qpo_fwhm_lim)
                amp_caps.append(self.qpo_amp_cap)
            else:
                comps.append((float(np.clip(amp_i, self.eps_amp, self.cont_amp_cap)),
                              nu0_i,
                              float(np.clip(fwhm_i, self.cont_fwhm_lim[0], self.cont_fwhm_lim[1]))))
                # Determine x0_lims based on component position
                x0_lims.append(self._infer_cont_x0_lims(nu0_i, fwhm_i))
                fwhm_lims.append(self.cont_fwhm_lim)
                amp_caps.append(self.cont_amp_cap)
            comp_types.append(ctype)

        # Add cont4 with free centroid
        comps.append(self.cont4_default)
        x0_lims.append(self.cont_x0_lims_free)
        fwhm_lims.append(self.cont_fwhm_lim)
        amp_caps.append(self.cont_amp_cap)
        comp_types.append("cont")

        fit4 = self._run_stage(
            comps=comps, x0_lims=x0_lims, fwhm_lims=fwhm_lims, amp_caps=amp_caps,
            comp_types=comp_types,
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag="cont4_fallback",
        )

        if fit4 is None:
            return fit_best

        if _accept_upgrade_ic(fit_best, fit4,
                              criterion=self.cont4_ic_criterion,
                              delta_min=self.cont4_ic_delta_min):
            fit4.message = "OK (cont4 accepted)"
            fit4.meta = fit4.meta or {}
            fit4.meta["cont4_accepted"] = True
            return fit4

        # Also accept if rchi2 substantially improves
        r0 = float(getattr(fit_best, "rchi2", np.nan))
        r4 = float(getattr(fit4, "rchi2", np.nan))
        if (np.isfinite(r0) and np.isfinite(r4)
                and r0 > self.cont4_trigger_rchi
                and r4 < r0 - 0.05):
            fit4.message = f"OK (cont4 forced — rchi {r0:.3f} -> {r4:.3f})"
            fit4.meta = fit4.meta or {}
            fit4.meta["cont4_forced"] = True
            return fit4

        return fit_best

    def _infer_cont_x0_lims(self, nu0: float, fwhm: float) -> Tuple[float, float]:
        """Infer appropriate x0 limits for a continuum component based on where it sits."""
        if abs(nu0) <= self.cont_x0_lims_narrow[1]:
            return self.cont_x0_lims_narrow if fwhm > 5.0 else self.cont_x0_lims_wide
        elif abs(nu0) <= self.cont_x0_lims_wide[1]:
            return self.cont_x0_lims_wide
        else:
            return self.cont_x0_lims_free

    # ------------------------------------------------------------------
    # Generic stage runner with multi-start
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        comps: List[Tuple[float, float, float]],
        x0_lims: List[Tuple[float, float]],
        fwhm_lims: List[Tuple[float, float]],
        amp_caps: List[float],
        comp_types: List[str],
        fitmethod: str,
        jitter: float,
        n_starts: int,
        tag: str,
    ) -> Optional[FitResult]:
        nlor    = len(comps)
        priors  = _build_priors(
            nlor, x0_lims=x0_lims, fwhm_lims=fwhm_lims,
            amp_max_list=amp_caps, include_const=self.include_const,
            const_max=self.const_cap, eps_amp=self.eps_amp,
        )
        base_t0 = np.array(_pack_t0(comps, self.c0 if self.include_const else None), float)

        best_res = None
        last_exc = None

        for t0 in self._jittered_starts(base_t0, n_starts, jitter, x0_lims, fwhm_lims, nlor):
            res, exc = _try_fit_once(
                self.ps, nlor=nlor, t0=t0.tolist(), priors=priors,
                include_const=self.include_const, fitmethod=fitmethod,
            )
            if res is None:
                last_exc = exc
                continue

            p0 = np.asarray(res.p_opt, float)
            t1 = _repair_params(
                p0, nlor=nlor, include_const=self.include_const,
                x0_lims=x0_lims, fwhm_lims=fwhm_lims,
                const_max=self.const_cap, eps_amp=self.eps_amp,
            )
            if np.any(t1 != p0):
                res2, _ = _try_fit_once(
                    self.ps, nlor=nlor, t0=t1.tolist(), priors=priors,
                    include_const=self.include_const, fitmethod=fitmethod,
                )
                if res2 is not None:
                    res = res2

            if best_res is None or getattr(res, "aic", np.inf) < getattr(best_res, "aic", np.inf):
                best_res = res

        if best_res is None:
            return None

        fit = self._wrap(best_res, nlor, self.include_const, tag, comp_types)
        ok_g, msg_g = self._check_guardrails(fit, x0_lims, fwhm_lims, amp_caps, tag)
        if not ok_g:
            return None
        return fit

    # ------------------------------------------------------------------
    # Jitter generator
    # ------------------------------------------------------------------

    def _jittered_starts(
        self,
        base_t0: np.ndarray,
        n_starts: int,
        jitter_frac: float,
        x0_lims: List[Tuple[float, float]],
        fwhm_lims: List[Tuple[float, float]],
        nlor: int,
    ) -> Generator[np.ndarray, None, None]:
        for start_idx in range(max(1, n_starts)):
            t = base_t0.copy()
            if jitter_frac > 0 and start_idx > 0:
                for i in range(nlor):
                    lo, hi = x0_lims[i]
                    t[3*i]   *= max(0.5, 1.0 + jitter_frac * self.rng.standard_normal())
                    t[3*i+1] += jitter_frac * (hi - lo) * 0.5 * self.rng.standard_normal()
                    t[3*i+2] *= max(0.5, 1.0 + jitter_frac * self.rng.standard_normal())
            t = _repair_params(
                t, nlor=nlor, include_const=self.include_const,
                x0_lims=x0_lims, fwhm_lims=fwhm_lims,
                const_max=self.const_cap, eps_amp=self.eps_amp,
            )
            yield t

    # ------------------------------------------------------------------
    # Result wrapping
    # ------------------------------------------------------------------

    def _wrap(
        self,
        stingray_res,
        nlor: int,
        include_const: bool,
        stage: str,
        comp_types: List[str],
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        p_opt = np.asarray(stingray_res.p_opt, float)

        pars_out = np.array(
            [(p_opt[3*i+1], p_opt[3*i+2], p_opt[3*i]) for i in range(nlor)],
            float,
        )

        # ── Post-fit QPO quality correction ──────────────────────────────
        # The optimizer can move a QPO-typed seed to a position where
        # Q = nu0/fwhm < qpo_detect_qmin, making it physically equivalent
        # to a broad continuum feature.  Keeping the "qpo" label then:
        #   (a) pollutes the legend and struct with spurious QPO entries;
        #   (b) lets a low-Q "2-QPO" config win the IC comparison in
        #       _run_smart even when no real QPO was found;
        #   (c) inflates the comp_types QPO count used by _grow_qpos.
        # Relabelling such components to "cont" (parameter values unchanged)
        # fixes all three downstream effects.
        corrected_types: List[str] = list(comp_types[:nlor])
        n_relabelled = 0
        for i in range(min(nlor, len(pars_out))):
            if corrected_types[i] == "qpo":
                nu0_i  = float(pars_out[i, 0])
                fwhm_i = float(pars_out[i, 1])
                Q_i    = abs(nu0_i) / max(fwhm_i, 1e-12)
                if Q_i < float(self.qpo_detect_qmin):
                    corrected_types[i] = "cont"
                    n_relabelled += 1

        const_val = float(p_opt[-1]) if include_const else 0.0
        model     = np.asarray(stingray_res.mfit, float)
        # Per-bin m for rchi2
        rchi2     = _compute_rchi2(self.p, model, self.e, m_avg=self.m_arr, npar=p_opt.size)

        deviance = float(getattr(stingray_res, "deviance", np.nan))
        n_good   = int(np.sum(np.isfinite(self.p) & np.isfinite(model)))
        red_dev  = _compute_red_deviance(deviance, n_good, p_opt.size)

        meta = dict(
            stage=stage,
            df=self.df,
            cand_df=self.cand_df,
            m_fit=int(self.m_fit),
            m_cand=int(self.m_cand),
            cont_ic_criterion=self.cont_ic_criterion,
            cont_ic_delta_min=self.cont_ic_delta_min,
            qpo_ic_criterion=self.qpo_ic_criterion,
            qpo_ic_delta_min=self.qpo_ic_delta_min,
            qpo_detect_qmin=self.qpo_detect_qmin,
            qpo_relabelled=int(n_relabelled),
        )
        if isinstance(meta_extra, dict):
            meta.update(meta_extra)

        return FitResult(
            ok=True,
            message=f"OK ({stage})",
            nlor=int(nlor),
            pars=pars_out,
            comp_types=corrected_types,
            const=const_val,
            freq=self.f,
            model=model,
            aic=float(getattr(stingray_res, "aic",     np.nan)),
            bic=float(getattr(stingray_res, "bic",     np.nan)),
            deviance=deviance,
            rchi2=rchi2,
            red_deviance=red_dev,
            stingray_p_opt=p_opt,
            p_err=(np.asarray(stingray_res.err, float) if hasattr(stingray_res, "err") else None),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Guardrail evaluation
    # ------------------------------------------------------------------

    def _check_guardrails(
        self,
        fit: FitResult,
        x0_lims: List[Tuple[float, float]],
        fwhm_lims: List[Tuple[float, float]],
        amp_caps: List[float],
        tag: str,
    ) -> Tuple[bool, str]:
        if fit.stingray_p_opt is None:
            return True, ""

        ok_p, msg_p = _param_sanity_check(
            fit.stingray_p_opt, nlor=fit.nlor,
            include_const=self.include_const,
            x0_lims=x0_lims, fwhm_lims=fwhm_lims,
            amp_max_list=amp_caps, const_max=self.const_cap, df=self.df,
        )
        if not ok_p:
            return False, f"{tag}: {msg_p}"

        if not self.guard_enable:
            return True, ""

        ok_o, msg_o = _guardrail_overshoot(
            self.f, self.p, fit.model, self.e, self.m_arr,
            ksigma=self.guard_overshoot_ksigma,
            max_run_bins=self.guard_overshoot_max_run_bins,
            max_frac=self.guard_overshoot_max_frac,
        )
        if not ok_o:
            return False, f"{tag}: {msg_o}"

        ok_c, msg_c = _guardrail_component_local_amp(
            self.f, self.p, fit.pars,
            local_amp_factor=self.guard_comp_local_amp_factor,
            local_width_hz=max(0.5, 6.0 * self.df),
        )
        if not ok_c:
            return False, f"{tag}: {msg_c}"

        return True, ""

    # ------------------------------------------------------------------
    # Candidate + seed building (now passes smooth_scales)
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        *,
        prom: float,
        require_ks,
        exclude_center,
        exclude_hw: float,
        max_cands: int,
        seed_peak,
        k: int,
        forced_seeds: Optional[List[float]] = None,
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        cands0 = find_qpo_candidates(
            self.cf, self.cp,
            cand_fmin=self.cand_fmin, cand_fmax=self.cand_fmax,
            smooth_scales=self.smooth_scales,
            smooth_hz=self.smooth_hz,
            prominence=float(prom),
            min_sep_hz=self.min_sep_hz, max_candidates=int(max_cands),
            m_eff=self.m_cand, require_ksigma=require_ks,
            cand_sigma_mode=self.cand_sigma_mode,
        )

        if exclude_center is not None and np.isfinite(exclude_center):
            lo_ex = float(exclude_center) - float(exclude_hw)
            hi_ex = float(exclude_center) + float(exclude_hw)
            cands = [c for c in cands0 if not (lo_ex <= float(c.get("nu_hz", np.nan)) <= hi_ex)]
        else:
            cands = cands0

        seeds: List[float] = []

        def _add_seed(nu: float) -> None:
            if not np.isfinite(nu):
                return
            if not (self.cand_fmin <= nu <= self.cand_fmax):
                return
            if any(abs(nu - s) <= max(0.5 * self.cand_df, 1e-6) for s in seeds):
                return
            seeds.append(float(nu))

        for fs in (forced_seeds or []):
            _add_seed(float(fs))

        if seed_peak is not None:
            _add_seed(float(seed_peak))

        for c in cands[:max(1, int(k))]:
            _add_seed(float(c.get("nu_hz", np.nan)))

        return cands, seeds

    # ------------------------------------------------------------------
    # Stage 1: try all seeds, return best accepted QPO fit
    # ------------------------------------------------------------------

    def _stage1_best_qpo(
        self,
        seeds: List[float],
        cands: List[Dict[str, float]],
        fit_cont: FitResult,
        fitmethod: str,
        jitter: float,
        screen_n_starts: int,
        polish_n_starts: int,
        override_relax_ic: bool,
        cand_tol_hz: float,
    ) -> Tuple[Optional[FitResult], Optional[Dict], Optional[float]]:
        """
        Try all seeds with fast screening, then polish the top candidates.

        v2: Two-phase approach:
          1. Screen all seeds with screen_n_starts (fast).
          2. Polish top 2 accepted seeds with polish_n_starts (thorough).
        This cuts optimizer calls by ~60% for typical seed counts.
        """
        # Phase 1: fast screen
        accepted: List[Tuple[FitResult, Dict, float, float]] = []  # (fit, qpo_dict, seed, ic)

        for s in seeds:
            fit1 = self._fit_cont2_qpo(float(s), fitmethod, jitter, screen_n_starts)
            if fit1 is None:
                continue

            qpo_real = self._detect_qpo(fit1)
            ok, _why = self._accept_qpo(
                fit_cont, fit1, cands, qpo_real,
                override_ic=override_relax_ic,
                cand_tol_hz=cand_tol_hz,
            )
            if not ok:
                continue

            ic = _ic_value(fit1, self.qpo_ic_criterion)
            accepted.append((fit1, qpo_real, float(s), ic))

        if not accepted:
            return None, None, None

        # Phase 2: polish top candidates (up to 2) with full n_starts
        if override_relax_ic:
            accepted.sort(key=lambda t: float(t[0].rchi2) if np.isfinite(t[0].rchi2) else np.inf)
        else:
            accepted.sort(key=lambda t: t[3])  # sort by IC

        n_polish = min(2, len(accepted))
        best_fit  : Optional[FitResult] = None
        best_qpo  : Optional[Dict]      = None
        best_seed : Optional[float]     = None

        for fit1, qpo_real, seed, _ in accepted[:n_polish]:
            if polish_n_starts > screen_n_starts:
                # Re-fit with more starts for a thorough optimization
                fit1p = self._fit_cont2_qpo(seed, fitmethod, jitter, polish_n_starts)
                if fit1p is not None:
                    qpo_p = self._detect_qpo(fit1p)
                    if qpo_p is not None:
                        fit1, qpo_real = fit1p, qpo_p

            if best_fit is None:
                best_fit, best_qpo, best_seed = fit1, qpo_real, seed
            elif override_relax_ic:
                if self._rchi_is_better(fit1, best_fit):
                    best_fit, best_qpo, best_seed = fit1, qpo_real, seed
            else:
                if _ic_value(fit1, self.qpo_ic_criterion) < _ic_value(best_fit, self.qpo_ic_criterion):
                    best_fit, best_qpo, best_seed = fit1, qpo_real, seed

        return best_fit, best_qpo, best_seed

    # ------------------------------------------------------------------
    # Stage 1b: simultaneous 2-QPO pair fitting
    # ------------------------------------------------------------------

    def _stage1_2qpo_pairs(
        self,
        seeds: List[float],
        cands: List[Dict[str, float]],
        fit_cont: FitResult,
        fitmethod: str,
        jitter: float,
        screen_n_starts: int,
        polish_n_starts: int,
        override_relax_ic: bool,
        cand_tol_hz: float,
    ) -> Optional[FitResult]:
        """
        Try simultaneous 2-QPO fits from the top candidate pairs.

        When both QPOs are present, a single-QPO Stage 1 model mismodels the
        PDS (the continuum distorts to absorb the second QPO).  Fitting both
        QPOs simultaneously avoids this problem.

        Only tries the top 3 unique pairs to limit cost.
        """
        if len(seeds) < 2:
            return None

        sep_tol = max(self.min_sep_hz, max(0.5 * self.df, 0.05))
        # Build pairs: top seeds that are separated by at least sep_tol
        pairs: List[Tuple[float, float]] = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i+1:]:
                if abs(s1 - s2) > sep_tol:
                    pairs.append((s1, s2))
                    if len(pairs) >= 3:
                        break
            if len(pairs) >= 3:
                break

        if not pairs:
            return None

        best_fit: Optional[FitResult] = None

        for s1, s2 in pairs:
            fit2q = self._fit_cont2_2qpo(s1, s2, fitmethod, jitter, screen_n_starts)
            if fit2q is None:
                continue

            # Check that at least 1 QPO passes Q gate
            qpos = self._detect_qpos(fit2q)
            if not qpos:
                continue

            # IC check against continuum-only
            if not override_relax_ic:
                if not _accept_upgrade_ic(fit_cont, fit2q,
                                          criterion=self.qpo_ic_criterion,
                                          delta_min=self.qpo_ic_delta_min):
                    continue

            if best_fit is None or _ic_value(fit2q, self.qpo_ic_criterion) < _ic_value(best_fit, self.qpo_ic_criterion):
                best_fit = fit2q

        # Polish the winner
        if best_fit is not None and polish_n_starts > screen_n_starts:
            qpos = self._detect_qpos(best_fit)
            if len(qpos) >= 2:
                s1 = float(qpos[0]["qpo_nu0_hz"])
                s2 = float(qpos[1]["qpo_nu0_hz"])
                fit2p = self._fit_cont2_2qpo(s1, s2, fitmethod, jitter, polish_n_starts)
                if fit2p is not None and self._detect_qpos(fit2p):
                    best_fit = fit2p

        return best_fit

    def _fit_cont2_2qpo(
        self, qseed1: float, qseed2: float, fitmethod: str, jitter: float, n_starts: int
    ) -> Optional[FitResult]:
        """Fit cont2 + 2 QPO components simultaneously."""
        def _qpo_comp(seed: float):
            fwhm = float(np.clip(
                max(self.qpo_fwhm_min, self.qpo_fwhm_frac * seed),
                self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1],
            ))
            amp = float(min(self.qpo_amp_cap, _seed_amp_at(self.f, self.p, seed)))
            return (amp, seed, fwhm)

        return self._run_stage(
            comps=[self.cont1, self.cont2, _qpo_comp(qseed1), _qpo_comp(qseed2)],
            x0_lims=[self.cont_x0_lims_narrow, self.cont_x0_lims_wide,
                     (self.cand_fmin, self.cand_fmax), (self.cand_fmin, self.cand_fmax)],
            fwhm_lims=[self.cont_fwhm_lim, self.cont_fwhm_lim,
                       self.qpo_fwhm_lim, self.qpo_fwhm_lim],
            amp_caps=[self.cont_amp_cap, self.cont_amp_cap,
                      self.qpo_amp_cap, self.qpo_amp_cap],
            comp_types=["cont", "cont", "qpo", "qpo"],
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag=f"cont2+2qpo@{qseed1:.2g}+{qseed2:.2g}Hz",
        )

    # ------------------------------------------------------------------
    # QPO acceptance
    # ------------------------------------------------------------------

    def _accept_qpo(
        self,
        fit_cont: FitResult,
        fit_qpo: FitResult,
        cands: List[Dict[str, float]],
        qpo_real: Optional[Dict],
        *,
        override_ic: bool,
        cand_tol_hz: float,
    ) -> Tuple[bool, str]:
        if qpo_real is None:
            return False, "reject: Q gate failed"

        if not override_ic:
            if not _accept_upgrade_ic(
                fit_cont, fit_qpo,
                criterion=self.qpo_ic_criterion,
                delta_min=self.qpo_ic_delta_min,
            ):
                return False, f"reject: delta-{self.qpo_ic_criterion.upper()} below threshold"

        ks = self.cand_require_ksigma
        if ks is not None and np.isfinite(ks) and float(ks) > 0:
            _, exsig = _nearest_candidate_metrics(
                cands, float(qpo_real["qpo_nu0_hz"]), tol_hz=cand_tol_hz
            )
            if not np.isfinite(exsig) or exsig < float(ks):
                return False, "reject: sigma gate failed at fitted QPO freq"

        return True, ""

    # ------------------------------------------------------------------
    # QPO detection (from a FitResult)
    # ------------------------------------------------------------------

    def _detect_qpos(self, fit: FitResult) -> List[Dict[str, Any]]:
        return extract_qpo_params_list(
            fit, qpo_fmin=self.cand_fmin, qpo_fmax=self.cand_fmax,
            qmin=self.qpo_detect_qmin, sort_by="area",
        )

    def _detect_qpo(self, fit: FitResult) -> Optional[Dict[str, Any]]:
        lst = self._detect_qpos(fit)
        return lst[0] if lst else None

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _fit_cont2_qpo(
        self, qseed: float, fitmethod: str, jitter: float, n_starts: int
    ) -> Optional[FitResult]:
        qpo_fwhm = float(np.clip(
            max(self.qpo_fwhm_min, self.qpo_fwhm_frac * qseed),
            self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1],
        ))
        qpo_amp  = float(min(self.qpo_amp_cap, _seed_amp_at(self.f, self.p, qseed)))
        qpo_comp = (qpo_amp, qseed, qpo_fwhm)

        return self._run_stage(
            comps=[self.cont1, self.cont2, qpo_comp],
            x0_lims=[self.cont_x0_lims_narrow, self.cont_x0_lims_wide, (self.cand_fmin, self.cand_fmax)],
            fwhm_lims=[self.cont_fwhm_lim, self.cont_fwhm_lim, self.qpo_fwhm_lim],
            amp_caps=[self.cont_amp_cap, self.cont_amp_cap, self.qpo_amp_cap],
            comp_types=["cont", "cont", "qpo"],
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag=f"cont2+qpo@{qseed:.3g}Hz",
        )

    def _seed_cont3_residual(self, fit_base: FitResult) -> Tuple[float, float, float]:
        """
        Seed cont3 from the low-frequency residual of a previous fit.

        FIXED: Centroid is NOT clipped to narrow cont_x0_lims.
        The whole purpose of cont3 is to capture power missed by cont1+cont2,
        which is often at intermediate frequencies (0.5–5 Hz).
        The centroid is instead clipped to cont_x0_lims_free.
        """
        try:
            f0  = np.asarray(fit_base.freq, float)
            mod = np.asarray(fit_base.model, float)
            p0  = self.p

            band = (f0 >= self.fit_fmin) & (f0 <= min(5.0, self.fit_fmax)) & np.isfinite(mod) & np.isfinite(p0)
            if np.sum(band) < 10:
                return self.cont3_default

            resid     = p0[band] - mod[band]
            ff        = f0[band]
            resid_pos = np.where(resid > 0, resid, 0.0)
            if np.sum(resid_pos) <= 0:
                return self.cont3_default

            # Centroid in real frequency space — clipped to FREE range, not narrow
            cen = float(np.sum(ff * resid_pos) / np.sum(resid_pos))
            cen = float(np.clip(cen, self.cont_x0_lims_free[0], self.cont_x0_lims_free[1]))
            amp = float(min(self.cont_amp_cap, max(self.lowf, self.eps_amp)))
            fwhm = float(np.clip(20.0, self.cont_fwhm_lim[0], self.cont_fwhm_lim[1]))
            return (amp, cen, fwhm)
        except Exception:
            return self.cont3_default

    def _fit_add_qpo(
        self,
        fit_seed: FitResult,
        qseed: float,
        fitmethod: str,
        jitter: float,
        n_starts: int,
    ) -> Optional[FitResult]:
        if not (hasattr(fit_seed, "pars") and fit_seed.pars.ndim == 2 and fit_seed.pars.shape[0] >= 1):
            return None

        comps, x0_lims, fwhm_lims, amp_caps, comp_types = [], [], [], [], []

        stored_types = list(getattr(fit_seed, "comp_types", []) or [])
        for idx, (nu0_i, fwhm_i, amp_i) in enumerate(fit_seed.pars):
            nu0_i, fwhm_i, amp_i = float(nu0_i), float(fwhm_i), float(amp_i)
            if idx < len(stored_types):
                ctype = stored_types[idx]
            else:
                ctype = "qpo" if (self.cand_fmin <= nu0_i <= self.cand_fmax
                                  and nu0_i / max(fwhm_i, 1e-12) >= self.qpo_detect_qmin) else "cont"

            if ctype == "qpo":
                comps.append((float(np.clip(amp_i, self.eps_amp, self.qpo_amp_cap)),
                               nu0_i,
                               float(np.clip(fwhm_i, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))))
                x0_lims.append((self.cand_fmin, self.cand_fmax))
                fwhm_lims.append(self.qpo_fwhm_lim)
                amp_caps.append(self.qpo_amp_cap)
            else:
                comps.append((float(np.clip(amp_i, self.eps_amp, self.cont_amp_cap)),
                               nu0_i,
                               float(np.clip(fwhm_i, self.cont_fwhm_lim[0], self.cont_fwhm_lim[1]))))
                x0_lims.append(self._infer_cont_x0_lims(nu0_i, fwhm_i))
                fwhm_lims.append(self.cont_fwhm_lim)
                amp_caps.append(self.cont_amp_cap)
            comp_types.append(ctype)

        qfwhm = float(np.clip(
            max(self.qpo_fwhm_min, self.qpo_fwhm_frac * qseed),
            self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1],
        ))
        comps.append((float(min(self.qpo_amp_cap, _seed_amp_at(self.f, self.p, qseed))),
                      float(qseed), qfwhm))
        x0_lims.append((self.cand_fmin, self.cand_fmax))
        fwhm_lims.append(self.qpo_fwhm_lim)
        amp_caps.append(self.qpo_amp_cap)
        comp_types.append("qpo")

        return self._run_stage(
            comps=comps, x0_lims=x0_lims, fwhm_lims=fwhm_lims, amp_caps=amp_caps,
            comp_types=comp_types,
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag=f"add_qpo@{qseed:.3g}Hz",
        )

    def _fit_cont3_qpo_rescue(
        self,
        fit_seed: FitResult,
        qnu0: float, qfwhm: float, qamp: float,
        fitmethod: str, jitter: float, n_starts: int,
    ) -> Optional[FitResult]:
        cont3s = self._seed_cont3_residual(fit_seed)
        qfwhm  = float(np.clip(qfwhm, self.qpo_fwhm_lim[0], self.qpo_fwhm_lim[1]))
        qamp   = float(np.clip(qamp,  self.eps_amp, self.qpo_amp_cap))

        return self._run_stage(
            comps=[self.cont1, self.cont2, cont3s, (qamp, qnu0, qfwhm)],
            x0_lims=[self.cont_x0_lims_narrow, self.cont_x0_lims_wide,
                     self.cont_x0_lims_free, (self.cand_fmin, self.cand_fmax)],
            fwhm_lims=[self.cont_fwhm_lim]*3 + [self.qpo_fwhm_lim],
            amp_caps=[self.cont_amp_cap]*3 + [self.qpo_amp_cap],
            comp_types=["cont", "cont", "cont", "qpo"],
            fitmethod=fitmethod, jitter=jitter, n_starts=n_starts,
            tag="cont3+qpo_rescue",
        )

    # ------------------------------------------------------------------
    # Multi-QPO growth
    # ------------------------------------------------------------------

    def _grow_qpos(
        self,
        fit_seed: FitResult,
        cands: List[Dict[str, float]],
        fitmethod: str,
        jitter: float,
        n_starts: int,
        override_ic: bool,
    ) -> FitResult:
        """
        Grow additional QPOs on top of the current fit.

        v2 changes:
          - Computes residual candidates from model-subtracted PDS.
          - Merges residual candidates with original candidates.
          - Wider tolerance for _find_new_qpo matching.

        BUG FIX (v2.1):
          Added guard so the loop breaks when existing >= target_nqpo.

        BUG FIX (v2.2):
          The v2.1 guard used len(_detect_qpos()) which only counts quality
          QPOs (Q >= qpo_detect_qmin).  When the incoming fit already had two
          QPO-typed components but one had Q < qpo_detect_qmin, _detect_qpos
          returned 1 rather than 2, the guard evaluated 1 >= 2 → False, and
          a third QPO-typed component was appended.

          Fix: count QPO-typed components directly from comp_types (the label
          assigned at fit time), which is the authoritative source regardless
          of the Q value.  _detect_qpos is still used to build the 'existing'
          list for seed exclusion (quality gate is appropriate there — we only
          want to anchor new growth on well-constrained features).
          The 'used' frequency set is now collected from ALL QPO-typed
          components so the new QPO is not placed on top of a low-Q component.
        """
        fit_curr    = fit_seed
        target_nqpo = max(1, self.max_qpos)
        if target_nqpo <= 1:
            return fit_curr

        delta_add = (float(self.qpo_ic_delta_min)
                     if self.multi_qpo_ic_delta_min is None
                     else float(self.multi_qpo_ic_delta_min))
        sep_tol   = max(self.min_sep_hz, max(0.5 * self.df, 0.05))

        for grow_idx in range(2, target_nqpo + 1):
            # ── Guard: count by comp_types, not by quality gate ──────────
            # Using _detect_qpos here was the root cause of the 3rd-QPO bug:
            # a low-Q QPO-typed component is invisible to _detect_qpos,
            # making the guard think there is room to grow when there is not.
            n_qpo_typed = sum(
                1 for t in getattr(fit_curr, "comp_types", []) if t == "qpo"
            )
            if n_qpo_typed >= target_nqpo:
                break

            # Quality-gated existing list — used only to verify we have a
            # solid foundation and to seed the residual candidate search.
            existing = self._detect_qpos(fit_curr)
            if len(existing) < (grow_idx - 1):
                break

            # ── Collect used frequencies from ALL QPO-typed components ───
            # This prevents the new seed from landing on an existing low-Q
            # component that _detect_qpos would not have included.
            pars_arr   = getattr(fit_curr, "pars", np.empty((0, 3)))
            ctypes_lst = list(getattr(fit_curr, "comp_types", []))
            used = [
                float(pars_arr[i, 0])
                for i, t in enumerate(ctypes_lst)
                if t == "qpo"
                and i < pars_arr.shape[0]
                and np.isfinite(pars_arr[i, 0])
            ]

            # Compute residual-based candidates: subtract model, find peaks
            resid_cands = self._residual_candidates(fit_curr, exclude_freqs=used, sep_tol=sep_tol)

            # Merge original + residual candidates, deduplicated
            all_cands = list(cands)
            for rc in resid_cands:
                nu = float(rc.get("nu_hz", np.nan))
                if np.isfinite(nu) and not any(abs(nu - float(c.get("nu_hz", np.nan))) < sep_tol for c in all_cands):
                    all_cands.append(rc)

            new_seeds = [
                float(c.get("nu_hz", np.nan)) for c in all_cands
                if np.isfinite(c.get("nu_hz", np.nan))
                and all(abs(float(c.get("nu_hz", np.nan)) - u) > sep_tol for u in used)
            ]
            if not new_seeds:
                break

            best_add = best_add_qpo = best_add_seed = None
            best_add_ic = np.inf
            # Wider tolerance for matching — the optimizer can shift the QPO
            match_tol = max(sep_tol, 1.0, 5.0 * self.df)

            for s in new_seeds:
                fit_add = self._fit_add_qpo(fit_curr, float(s), fitmethod, jitter, n_starts)
                if fit_add is None:
                    continue

                qpo_list  = self._detect_qpos(fit_add)
                added_qpo = self._find_new_qpo(qpo_list, existing, seed_hz=float(s), tol_hz=match_tol)
                if added_qpo is None:
                    continue

                ks = self.cand_require_ksigma
                if ks is not None and np.isfinite(ks) and float(ks) > 0:
                    _, exsig = _nearest_candidate_metrics(all_cands, float(added_qpo["qpo_nu0_hz"]), tol_hz=match_tol)
                    if not np.isfinite(exsig) or exsig < float(ks):
                        continue

                if self.multi_qpo_require_improvement and not override_ic:
                    if not _accept_upgrade_ic(fit_curr, fit_add,
                                              criterion=self.qpo_ic_criterion,
                                              delta_min=delta_add):
                        continue

                ic_now = _ic_value(fit_add, self.qpo_ic_criterion)
                if best_add is None or ic_now < best_add_ic - 1e-12:
                    best_add, best_add_qpo, best_add_seed, best_add_ic = fit_add, added_qpo, float(s), ic_now

            if best_add is None:
                break

            fit_curr = best_add
            fit_curr.message = f"OK ({grow_idx} QPOs accepted)"

        return fit_curr

    def _residual_candidates(
        self,
        fit: FitResult,
        exclude_freqs: List[float],
        sep_tol: float,
    ) -> List[Dict[str, float]]:
        """
        Find QPO candidates in the model-subtracted residual PDS.

        After fitting QPO1, the residual P - model + continuum reveals QPO2
        peaks that were invisible in the raw whitened PDS.
        """
        try:
            f = np.asarray(fit.freq, float)
            mod = np.asarray(fit.model, float)
            p = self.p

            # Residual = data - model (positive residuals = unmodeled power)
            resid = p - mod
            # Add back a smooth baseline so the candidate finder can work
            # Use the white-noise constant as the baseline
            baseline = float(fit.const) if np.isfinite(fit.const) and fit.const > 0 else 0.0
            resid_pds = np.maximum(resid + baseline, 1e-30)

            cands = find_qpo_candidates(
                f, resid_pds,
                cand_fmin=self.cand_fmin, cand_fmax=self.cand_fmax,
                smooth_scales=self.smooth_scales,
                smooth_hz=self.smooth_hz,
                prominence=max(0.3, self.prominence * 0.5),  # lower threshold on residuals
                min_sep_hz=self.min_sep_hz,
                max_candidates=self.max_candidates,
                m_eff=self.m_cand,
                require_ksigma=None,  # don't sigma-gate residuals
                cand_sigma_mode=self.cand_sigma_mode,
            )

            # Remove candidates near already-fitted QPOs
            filtered = []
            for c in cands:
                nu = float(c.get("nu_hz", np.nan))
                if np.isfinite(nu) and all(abs(nu - ef) > sep_tol for ef in exclude_freqs):
                    filtered.append(c)

            return filtered
        except Exception:
            return []

    @staticmethod
    def _find_new_qpo(
        qpo_list: List[Dict], existing: List[Dict], *, seed_hz: float, tol_hz: float
    ) -> Optional[Dict]:
        def _is_new(nu: float) -> bool:
            return all(abs(nu - float(q.get("qpo_nu0_hz", np.nan))) > tol_hz for q in existing)

        best = best_d = None
        for q in qpo_list:
            nu = float(q.get("qpo_nu0_hz", np.nan))
            if not np.isfinite(nu) or not _is_new(nu):
                continue
            d = abs(nu - seed_hz)
            if best is None or d < best_d:
                best, best_d = q, d
        if best is None or best_d > tol_hz:
            return None
        return best

    # ------------------------------------------------------------------
    # cont3+QPO rescue
    # ------------------------------------------------------------------

    def _cont3_qpo_rescue(
        self,
        fit_best: FitResult,
        best_qpo_real: Dict,
        thr: float,
        fitmethod: str,
        jitter: float,
        n_starts: int,
        do_override: bool,
    ) -> FitResult:
        qidx   = int(np.clip(int(best_qpo_real.get("qpo_index", 2)), 0, fit_best.pars.shape[0]-1))
        q_nu0  = float(fit_best.pars[qidx, 0])
        q_fwhm = float(fit_best.pars[qidx, 1])
        q_amp  = float(fit_best.pars[qidx, 2])

        fit_up = self._fit_cont3_qpo_rescue(fit_best, q_nu0, q_fwhm, q_amp, fitmethod, jitter, n_starts)

        if fit_up is None or not (np.isfinite(fit_up.rchi2) and np.isfinite(fit_best.rchi2)):
            return fit_best

        r0, r1  = float(fit_best.rchi2), float(fit_up.rchi2)
        dr      = r0 - r1
        dic     = _ic_value(fit_best, self.qpo_ic_criterion) - _ic_value(fit_up, self.qpo_ic_criterion)
        not_worse = r1 <= r0 + self.postqpo_cont3_rchi_not_worse_tol

        accept_override  = do_override and np.isfinite(dr) and (dr >= self.rchi_override_min_improve) and not_worse
        accept_postqpo   = (np.isfinite(dr) and dr >= self.postqpo_cont3_rchi_improve_min) or \
                           (np.isfinite(dic) and dic >= self.postqpo_cont3_ic_delta_min and not_worse)

        fit_best.meta = fit_best.meta or {}
        fit_best.meta.update(dict(
            cont3qpo_attempted=True, cont3qpo_override=bool(do_override),
            cont3qpo_rchi_delta=dr, cont3qpo_ic_delta=float(dic),
        ))

        if accept_override or accept_postqpo:
            fit_up.message = "OK (cont3+qpo rescue accepted)"
            fit_up.meta    = fit_up.meta or {}
            fit_up.meta["cont3qpo_accepted"] = True
            return fit_up

        fit_best.meta["cont3qpo_accepted"] = False
        return fit_best

    # ------------------------------------------------------------------
    # Warm-start seed injection
    # ------------------------------------------------------------------

    def _apply_warm_start(
        self,
        ws: Dict[str, Any],
        cf_lim: tuple,
        eps_amp: float,
    ) -> None:
        """
        Override default component seeds with parameters from a prior fit.

        Called at the end of __init__ when warm_start_comps is provided.
        ws comes from WarmStartCache.get_warm_comps_for_band() and has the form:
            {
              "cont": [(nu0, fwhm, amp), ...],   # FitResult.pars layout
              "qpo":  [(nu0, fwhm, amp), ...],
              "const": float | None,
            }

        FitResult.pars stores (nu0, fwhm, amp).
        Internal comp tuples are (amp, x0, fwhm).
        This method converts and clamps before overriding the seeds.
        """
        cont_ws  = ws.get("cont",  [])
        qpo_ws   = ws.get("qpo",   [])
        const_ws = ws.get("const", None)

        def _make_cont(nu0: float, fwhm: float, amp: float) -> tuple:
            # Convert (nu0, fwhm, amp) → (amp, x0=nu0, fwhm), clamped
            safe_amp  = float(np.clip(amp,  eps_amp, self.cont_amp_cap))
            safe_x0   = float(np.clip(nu0,  self.cont_x0_lims[0], self.cont_x0_lims[1]))
            safe_fwhm = float(np.clip(fwhm, cf_lim[0], cf_lim[1]))
            return (safe_amp, safe_x0, safe_fwhm)

        if len(cont_ws) >= 1:
            self.cont1         = _make_cont(*cont_ws[0])
        if len(cont_ws) >= 2:
            self.cont2         = _make_cont(*cont_ws[1])
        if len(cont_ws) >= 3:
            self.cont3_default = _make_cont(*cont_ws[2])

        # QPO frequencies: prepend to forced_qpo_seeds (highest-priority slot)
        # so they are always tried first in the seed screening stage.
        for (nu0, _fwhm, _amp) in qpo_ws:
            nu0 = float(nu0)
            if not (np.isfinite(nu0) and self.cand_fmin <= nu0 <= self.cand_fmax):
                continue
            if not any(abs(nu0 - s) < max(0.5 * self.cand_df, 1e-6)
                       for s in self.forced_qpo_seeds):
                self.forced_qpo_seeds.insert(0, nu0)

        # White-noise floor seed
        if const_ws is not None:
            c = float(const_ws)
            if np.isfinite(c) and c > 0:
                self.c0 = float(np.clip(c, eps_amp, self.const_cap))

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _build_ladder(self) -> List[Dict[str, Any]]:
        """
        Build the retry ladder.

        v2: n_starts escalation reduced since continuum is now cached.
        The ladder varies optimizer and jitter but keeps n_starts moderate.
        """
        frac = self.jitter_frac
        ns   = self.n_starts
        entries = [
            dict(fitmethod=self.fitmethod, jitter=frac,                   n_starts=ns,          label="attempt0"),
            dict(fitmethod=self.fitmethod, jitter=max(0.10, 0.60*frac),   n_starts=ns,          label="attempt1_less_jitter"),
            dict(fitmethod="Nelder-Mead", jitter=max(0.10, 0.70*frac),   n_starts=ns,          label="attempt2_nm"),
            dict(fitmethod="Powell",      jitter=max(0.16, 1.20*frac),   n_starts=max(8, ns),  label="attempt3_powell_wide"),
            dict(fitmethod="Nelder-Mead", jitter=max(0.20, 1.40*frac),   n_starts=max(8, ns),  label="attempt4_nm_wide"),
        ]
        return entries[:max(1, self.max_retries)]

    @staticmethod
    def _rchi_is_better(a: FitResult, b: FitResult, margin: float = 0.01) -> bool:
        ra = float(a.rchi2) if np.isfinite(a.rchi2) else np.inf
        rb = float(b.rchi2) if np.isfinite(b.rchi2) else np.inf
        return ra < rb - margin

    def _cont_rchi_bad(self, fit: FitResult) -> bool:
        return np.isfinite(fit.rchi2) and float(fit.rchi2) > self.reseed_rchi_bad

    def _is_edge(self, freq_hz: float) -> bool:
        if not np.isfinite(freq_hz):
            return False
        w = max(self.reseed_edge_frac * (self.cand_fmax - self.cand_fmin), 2.0 * self.df)
        return freq_hz <= self.cand_fmin + w or freq_hz >= self.cand_fmax - w

    def _is_tiny_area(self, fit: FitResult, qpo_real: Optional[Dict]) -> bool:
        if not (self.reseed_area_min > 0 and isinstance(qpo_real, dict)):
            return False
        try:
            qidx  = int(np.clip(int(qpo_real.get("qpo_index", 0)), 0, fit.pars.shape[0]-1))
            comp  = lorentz(fit.freq, fit.pars[qidx, 0], fit.pars[qidx, 1], fit.pars[qidx, 2])
            area  = component_power_integral(fit.freq, comp, self.cand_fmin, self.cand_fmax)
            return np.isfinite(area) and float(area) < self.reseed_area_min
        except Exception:
            return False

    @staticmethod
    def _relax_ks(ks, factor: float):
        if ks is None:
            return None
        return max(0.0, float(ks) / max(1e-6, float(factor)))

    @staticmethod
    def _tighten_ks(ks, factor: float):
        if ks is None:
            return None
        return float(ks) * float(factor)

    @staticmethod
    def _make_ps(freq, power, power_err, m) -> Powerspectrum:
        ps = Powerspectrum()
        ps.freq      = np.asarray(freq,  float)
        ps.power     = np.asarray(power, float)
        ps.power_err = None if power_err is None else np.asarray(power_err, float)
        ps.df        = float(np.median(np.diff(ps.freq)))
        ps.m         = m
        ps.norm      = "frac"
        return ps

    def _make_failure(self, msg: str) -> FitResult:
        f = self.f if not self._too_few else np.array([])
        return FitResult(
            ok=False, message=msg, nlor=0,
            pars=np.empty((0, 3)), comp_types=[],
            const=0.0, freq=f, model=np.full_like(f, np.nan),
            aic=np.nan, bic=np.nan, deviance=np.nan,
            rchi2=np.nan, red_deviance=np.nan, meta={},
        )


# ---------------------------------------------------------------------------
# Public wrapper function
# ---------------------------------------------------------------------------

def fit_lorentzian_family(
    freq,
    power,
    power_err=None,
    *,
    cand_freq=None,
    cand_power=None,
    cand_m_eff=None,
    random_seed=42,
    obsid_seed_offset: int = 0,
    force_cont3_rchi: float = np.inf,
    forced_qpo_seeds: Optional[List[float]] = None,
    warm_start_comps: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> FitResult:
    seed = (int(random_seed) + int(obsid_seed_offset)) if random_seed is not None else 12345
    fitter = QPOFitter(
        freq, power, power_err,
        cand_freq=cand_freq,
        cand_power=cand_power,
        cand_m_eff=cand_m_eff,
        random_seed=seed,
        force_cont3_rchi=force_cont3_rchi,
        forced_qpo_seeds=forced_qpo_seeds,
        warm_start_comps=warm_start_comps,
        **kwargs,
    )
    return fitter.run()
