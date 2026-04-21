#!/usr/bin/env python3
"""
QPO_fit.py  (v3.2 — TripleA-only, multi-QPO + trapz fix)
==========================================================
Lorentzian fitting engine for X-ray power density spectra.

Changes from v3.1
-----------------
- BUG FIX: np.trapz replaced by _trapz alias (np.trapezoid in NumPy >=2.0,
  np.trapz in older versions) to silence DeprecationWarning.

- BUG FIX: multi-QPO detection.  When max_qpos >= 2 and two or more QPO seeds
  are available, simultaneous 2-QPO configurations are now added:
    cont2 + QPO1 + QPO2  and  cont3 + QPO1 + QPO2
  (one pair per unique combination of QPO seeds, up to max_qpos pairs).
  Previously only single-QPO configs were ever built, so fit_1q was never
  populated for the 2-QPO case and multi-QPO detection was impossible.

- Added multi_qpo_ic_delta_min parameter (wired from FIT_MULTI_QPO_IC_DELTA_MIN).
  The 1-QPO → 2-QPO upgrade is gated by this threshold (or by rchi2 improvement),
  separately from the 0-QPO → 1-QPO gate.

- Refactored QPO component building into _build_qpo_comp() to avoid duplication
  between single- and multi-QPO config construction.

Changes from v3
---------------
- QPO FWHM upper bound capped at nu0/qpo_detect_qmin per seed.
- QPO amplitude seeded as excess above local continuum.
- rchi2 improvement fallback for QPO acceptance.
- No stingray.modeling imports.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter

# NumPy 2.0 renamed trapz → trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))


# ---------------------------------------------------------------------------
# Lorentzian profile
# ---------------------------------------------------------------------------

def lorentz(freq: np.ndarray, nu0: float, fwhm: float, amp: float) -> np.ndarray:
    f = np.asarray(freq, float)
    g = 0.5 * float(fwhm)
    return float(amp) * g * g / ((f - float(nu0)) ** 2 + g * g)


def component_power_integral(
    freq: np.ndarray, comp: np.ndarray, fmin: float, fmax: float
) -> float:
    f = np.asarray(freq, float)
    y = np.asarray(comp, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(f) & np.isfinite(y)
    if np.sum(m) < 2:
        return 0.0
    return float(_trapz(y[m], f[m]))


# ---------------------------------------------------------------------------
# QPO extraction from FitResult
# ---------------------------------------------------------------------------

def extract_qpo_params_list(
    fitres,
    *,
    qpo_fmin: float,
    qpo_fmax: float,
    qmin: float = 3.0,
    sort_by: str = "area",
) -> List[Dict[str, Any]]:
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
        area = float(_trapz(lorentz(freq, nu0, fwhm, amp), freq))
        out.append(dict(qpo_index=int(i), qpo_nu0_hz=float(nu0),
                        qpo_fwhm_hz=float(fwhm), qpo_Q=float(Q), qpo_area=float(area)))
    key = str(sort_by).strip().lower()
    if key == "freq":
        out.sort(key=lambda d: (d["qpo_nu0_hz"], -d["qpo_area"]))
    elif key == "q":
        out.sort(key=lambda d: (-d["qpo_Q"], -d["qpo_area"], d["qpo_nu0_hz"]))
    else:
        out.sort(key=lambda d: (-d["qpo_area"], d["qpo_nu0_hz"]))
    return out



# ---------------------------------------------------------------------------
# FitResult container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Result of a Lorentzian-family fit.

    rchi2        : reduced chi-squared  (Gaussian approx, for diagnostics)
    red_deviance : reduced Whittle deviance = deviance / (N − npar)
    stingray_p_opt : parameter vector [amp_0, x0_0, fwhm_0, ..., const]
                     kept for struct / plot / interactive-fitter compatibility.
    """
    ok:           bool
    message:      str
    nlor:         int
    pars:         np.ndarray
    comp_types:   List[str]
    const:        float
    freq:         np.ndarray
    model:        np.ndarray
    aic:          float
    bic:          float
    deviance:     float
    rchi2:        Optional[float] = None
    red_deviance: Optional[float] = None
    stingray_p_opt: Optional[np.ndarray] = None
    p_err:        Optional[np.ndarray] = None
    meta:         Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Private numerical helpers
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
        return int(v) if (np.isfinite(v) and v >= 1) else 1
    except Exception:
        return 1


def _m_as_array(m, n: int) -> np.ndarray:
    arr = np.asarray(m, float)
    if arr.ndim == 0:
        return np.full(n, max(1.0, float(arr)))
    if arr.size == n:
        return np.where(np.isfinite(arr) & (arr >= 1), arr, 1.0).astype(float)
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
    i = int(np.argmin(np.abs(freq - float(nu0))))
    lo, hi = max(0, i - half_width), min(len(power), i + half_width + 1)
    chunk = power[lo:hi]
    good = np.isfinite(chunk) & (chunk > 0)
    return float(np.median(chunk[good])) if np.any(good) else float(power[i])


def _local_median_around(
    freq: np.ndarray, power: np.ndarray, nu0: float, width_hz: float = 0.5
) -> float:
    if not (np.isfinite(nu0) and np.isfinite(width_hz) and width_hz > 0):
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else np.nan
    m = ((freq >= nu0 - width_hz) & (freq <= nu0 + width_hz)
         & np.isfinite(power) & (power > 0))
    if np.sum(m) < 5:
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else np.nan
    return float(np.nanmedian(power[m]))


def _seed_qpo_amp(
    freq: np.ndarray, power: np.ndarray, nu0: float, lowf: float, eps_amp: float
) -> float:
    """
    Seed QPO amplitude as the excess above the local continuum.

    Uses a wide (±2 Hz) median window to estimate the continuum at nu0,
    then returns the excess.  Falls back to 10 % of local power when
    no positive excess is found (weak or absent QPO).
    """
    total_at_nu0 = _seed_amp_at(freq, power, float(nu0))
    local_cont   = _local_median_around(freq, power, float(nu0), width_hz=2.0)
    local_cont   = max(local_cont if np.isfinite(local_cont) else lowf, lowf)
    excess       = total_at_nu0 - local_cont
    if excess <= 0:
        excess = local_cont * 0.10
    return max(float(excess), float(eps_amp))


def _compute_rchi2(
    power: np.ndarray, model: np.ndarray, power_err, m_avg, npar: int
) -> float:
    p = np.asarray(power, float)
    mod = np.asarray(model, float)
    n = p.size
    if power_err is not None:
        err = np.asarray(power_err, float)
    else:
        err = p / np.sqrt(_m_as_array(m_avg, n))
    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(err) & (err > 0)
    dof = int(np.sum(good)) - int(npar)
    if np.sum(good) < 20 or dof <= 0:
        return np.nan
    return float(np.sum(((p[good] - mod[good]) / err[good]) ** 2) / dof)


def _compute_red_deviance(deviance: float, n_bins: int, npar: int) -> float:
    dof = int(n_bins) - int(npar)
    if dof <= 0 or not np.isfinite(deviance):
        return np.nan
    return float(deviance / dof)


def _estimate_sigma_local(
    *, cont: np.ndarray, p: np.ndarray, m_eff: int, mode: str = "cont"
) -> np.ndarray:
    m_eff = max(1, int(m_eff))
    base = np.asarray(cont if mode.lower().strip() == "cont" else p, float)
    base = np.where(np.isfinite(base) & (base > 0), base, np.nan)
    med = float(np.nanmedian(base)) if np.any(np.isfinite(base)) else 1.0
    base = np.where(np.isfinite(base), base, med)
    return base / np.sqrt(float(m_eff))


# ---------------------------------------------------------------------------
# IC helpers
# ---------------------------------------------------------------------------

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


def _rchi2_of(fr: Optional[FitResult]) -> float:
    if fr is None:
        return np.inf
    r = getattr(fr, "rchi2", np.nan)
    return float(r) if np.isfinite(r) else np.inf


# ---------------------------------------------------------------------------
# Multi-scale candidate peak finder  (z-score based)
# ---------------------------------------------------------------------------

def _find_candidates_single_scale(
    f: np.ndarray, p: np.ndarray,
    *,
    smooth_hz: float, min_width_bins: int, prominence: float,
    min_sep_hz: float, m_eff: int, require_ksigma: Optional[float],
    cand_sigma_mode: str,
) -> List[Dict[str, float]]:
    if f.size < 25:
        return []
    df = float(np.median(np.diff(f)))
    if not np.isfinite(df) or df <= 0:
        return []
    w = max(int(min_width_bins), int(np.round(smooth_hz / df)))
    if w % 2 == 0:
        w += 1
    if w >= len(p):
        w = len(p) - 2 if len(p) > 2 else 3
        if w % 2 == 0:
            w -= 1
    if w < 3 or w >= len(p):
        return []
    cont = _rolling_median(p, w)
    good = np.isfinite(cont) & (cont > 0)
    if not np.any(good):
        return []
    cont = np.where(good, cont, np.nanmedian(cont[good]))
    sigma = _estimate_sigma_local(cont=cont, p=p, m_eff=int(m_eff), mode=cand_sigma_mode)
    z = (p - cont) / sigma
    distance = int(max(1, np.round(min_sep_hz / df)))
    peaks, props = scipy.signal.find_peaks(z, prominence=float(prominence), distance=distance)
    if peaks.size == 0:
        return []
    prom = np.asarray(props.get("prominences", np.zeros(len(peaks))), float)
    if require_ksigma is not None and np.isfinite(require_ksigma) and require_ksigma > 0:
        keep = z[peaks] >= float(require_ksigma)
        peaks, prom = peaks[keep], prom[keep]
        if peaks.size == 0:
            return []
    return [
        dict(nu_hz=float(f[idx]), z_prominence=float(pr),
             excess_sigma=float(z[idx]) if np.isfinite(z[idx]) else float("nan"),
             z_peak=float(z[idx]),
             ratio_peak=float(p[idx] / cont[idx]) if cont[idx] > 0 else float("nan"),
             prominence=float(pr))
        for idx, pr in zip(peaks.tolist(), prom.tolist())
    ]


def find_qpo_candidates(
    freq, power, *,
    cand_fmin: float, cand_fmax: float,
    smooth_scales: Optional[List[float]] = None, smooth_hz: float = 0.5,
    prominence: float = 0.5, min_sep_hz: float = 0.15, max_candidates: int = 5,
    min_width_bins: int = 7, m_eff: int = 1,
    require_ksigma: Optional[float] = None, cand_sigma_mode: str = "cont",
) -> List[Dict[str, float]]:
    """Multi-scale z-score QPO candidate finder."""
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    m = (f >= cand_fmin) & (f <= cand_fmax) & np.isfinite(p) & (p > 0)
    f, p = f[m], p[m]
    if f.size < 25:
        return []
    scales = list(smooth_scales) if smooth_scales else [float(smooth_hz)]
    all_cands: List[Dict[str, float]] = []
    for scale in scales:
        all_cands.extend(_find_candidates_single_scale(
            f, p, smooth_hz=float(scale), min_width_bins=min_width_bins,
            prominence=float(prominence), min_sep_hz=min_sep_hz, m_eff=int(m_eff),
            require_ksigma=require_ksigma, cand_sigma_mode=cand_sigma_mode,
        ))
    if not all_cands:
        return []
    all_cands.sort(key=lambda c: -c.get("z_prominence", 0.0))
    deduped: List[Dict[str, float]] = []
    for c in all_cands:
        nu = float(c.get("nu_hz", np.nan))
        if not np.isfinite(nu):
            continue
        if any(abs(nu - float(d.get("nu_hz", np.nan))) < min_sep_hz for d in deduped):
            continue
        deduped.append(c)
    return deduped[:int(max_candidates)]


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _param_sanity_check(
    p_opt: np.ndarray, *, nlor: int, include_const: bool,
    x0_lims: List[Tuple[float, float]], fwhm_lims: List[Tuple[float, float]],
    amp_max_list: List[float], const_max: float, df: float,
) -> Tuple[bool, str]:
    p_opt = np.asarray(p_opt, float)
    x0_tol = float(max(2.0 * float(df), 1e-6))
    for i in range(nlor):
        amp_i, x0_i, fwhm_i = float(p_opt[3*i]), float(p_opt[3*i+1]), float(p_opt[3*i+2])
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
        if amp_i > float(amp_max_list[i]) * 1.05:
            return False, f"amp above cap in component {i}: {amp_i:.4g}"
    if include_const:
        c = float(p_opt[-1])
        if not np.isfinite(c):
            return False, "non-finite const"
        if c < 0:
            return False, "negative const"
        if c > float(const_max) * 1.05:
            return False, f"const above cap: {c:.4g}"
    return True, ""


def _guardrail_overshoot(
    freq, data_p, model_p, power_err, m_eff,
    *, ksigma=4.0, max_run_bins=6, max_frac=0.10
) -> Tuple[bool, str]:
    p = np.asarray(data_p, float)
    mod = np.asarray(model_p, float)
    n = p.size
    sig = (np.asarray(power_err, float) if power_err is not None
           else p / np.sqrt(_m_as_array(m_eff, n)))
    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(sig) & (sig > 0)
    ngood = int(np.sum(good))
    if ngood < 10:
        return True, ""
    if ngood < 30:
        warnings.warn(f"Overshoot guardrail: only {ngood} bins — relaxed.",
                      RuntimeWarning, stacklevel=4)
        ksigma = max(float(ksigma), 5.0)
        max_run_bins = max(int(max_run_bins), 4)
        max_frac = max(float(max_frac), 0.15)
    resid = mod[good] - p[good]
    bad = resid > float(ksigma) * sig[good]
    if not np.any(bad):
        return True, ""
    frac = float(np.mean(bad))
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
# TripleA stage runner
# ---------------------------------------------------------------------------

class _PDS:
    __slots__ = ("freq", "power")
    def __init__(self, freq, power):
        self.freq  = np.asarray(freq,  float)
        self.power = np.asarray(power, float)


def _run_triplea_stage(
    f: np.ndarray, p: np.ndarray, e: Optional[np.ndarray], m_arr: np.ndarray,
    *,
    nlor: int, t0: List[float],
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    amp_caps: List[float],
    comp_types: List[str],
    include_const: bool, const_cap: float, eps_amp: float,
    qpo_detect_qmin: float, df: float, tag: str,
    guard_enable: bool, guard_overshoot_ksigma: float,
    guard_overshoot_max_run_bins: int, guard_overshoot_max_frac: float,
    guard_comp_local_amp_factor: float,
) -> Optional[FitResult]:
    from QPO_TripleA import tripleA_fit_once as _aaa

    res, _exc = _aaa(
        _PDS(f, p),
        nlor=nlor, t0=t0, include_const=include_const,
        x0_lims=x0_lims, fwhm_lims=fwhm_lims, amp_caps=amp_caps,
        const_max=float(const_cap), eps_amp=float(eps_amp),
    )
    if res is None:
        return None

    p_opt = np.asarray(res.p_opt, float)
    pars_out = np.array(
        [(p_opt[3*i+1], p_opt[3*i+2], p_opt[3*i]) for i in range(nlor)], float
    )

    corr_types = list(comp_types[:nlor])
    n_relabelled = 0
    for i in range(nlor):
        if corr_types[i] == "qpo":
            nu0_i, fwhm_i = float(pars_out[i, 0]), float(pars_out[i, 1])
            if fwhm_i > 0 and abs(nu0_i) / fwhm_i < float(qpo_detect_qmin):
                corr_types[i] = "cont"
                n_relabelled += 1

    const_val = float(p_opt[-1]) if include_const else 0.0
    model     = np.asarray(res.mfit, float)
    npar      = int(p_opt.size)
    rchi2     = _compute_rchi2(p, model, e, m_avg=m_arr, npar=npar)
    deviance  = float(res.deviance)
    n_good    = int(np.sum(np.isfinite(p) & np.isfinite(model)))
    red_dev   = _compute_red_deviance(deviance, n_good, npar)

    ok_p, _ = _param_sanity_check(
        p_opt, nlor=nlor, include_const=include_const,
        x0_lims=x0_lims, fwhm_lims=fwhm_lims,
        amp_max_list=amp_caps, const_max=const_cap, df=df,
    )
    if not ok_p:
        return None

    if guard_enable:
        ok_o, _ = _guardrail_overshoot(
            f, p, model, e, m_arr,
            ksigma=guard_overshoot_ksigma,
            max_run_bins=guard_overshoot_max_run_bins,
            max_frac=guard_overshoot_max_frac,
        )
        if not ok_o:
            return None
        ok_c, _ = _guardrail_component_local_amp(
            f, p, pars_out,
            local_amp_factor=guard_comp_local_amp_factor,
            local_width_hz=max(0.5, 6.0 * df),
        )
        if not ok_c:
            return None

    return FitResult(
        ok=True, message=f"OK ({tag})",
        nlor=nlor, pars=pars_out, comp_types=corr_types,
        const=const_val, freq=f, model=model,
        aic=float(res.aic), bic=float(res.bic), deviance=deviance,
        rchi2=rchi2, red_deviance=red_dev, stingray_p_opt=p_opt,
        # p_err stores the full linear-space parameter covariance matrix.
        # Shape: (npar, npar), Stingray order [amp_0, nu0_0, fwhm_0, ..., const].
        # None when the Hessian was singular (active bounds, degenerate fit).
        p_err=getattr(res, "p_cov", None),
        meta={"stage": tag, "method": "TripleA", "qpo_relabelled": int(n_relabelled)},
    )


# ---------------------------------------------------------------------------
# QPO component builder (shared by 1-QPO and multi-QPO configs)
# ---------------------------------------------------------------------------

def _build_qpo_comp(
    nu0: float, f: np.ndarray, p: np.ndarray, lowf: float, eps_amp: float,
    qpo_amp_cap: float, qpo_fwhm_lim: tuple, qpo_detect_qmin: float,
    qpo_fwhm_min: float, qpo_fwhm_frac: float,
    cand_fmin: float, cand_fmax: float,
) -> Tuple[Tuple[float, float, float],
           Tuple[float, float],
           Tuple[float, float]]:
    """
    Build one QPO component tuple + per-component bounds.

    Returns (comp, x0_lims, fwhm_lims) where:
      comp      = (amp_seed, nu0, fwhm_seed)  in Stingray t0 order
      x0_lims   = (cand_fmin, cand_fmax)
      fwhm_lims = (qf_lo, qf_hi)  with FWHM capped so Q >= qpo_detect_qmin
    """
    qf_lo = float(qpo_fwhm_lim[0])
    qf_hi = float(qpo_fwhm_lim[1])

    # Cap FWHM so TripleA cannot broaden the QPO below the Q threshold,
    # which would trigger relabelling to "cont" in _run_triplea_stage.
    if qpo_detect_qmin > 0 and nu0 > 0:
        qf_hi_q = float(nu0) / float(qpo_detect_qmin)
        if qf_hi_q > qf_lo:
            qf_hi = min(qf_hi, qf_hi_q)

    qfwhm = float(np.clip(
        max(float(qpo_fwhm_min), float(qpo_fwhm_frac) * nu0), qf_lo, qf_hi
    ))
    qamp = float(min(qpo_amp_cap,
                     max(eps_amp, _seed_qpo_amp(f, p, nu0, lowf, eps_amp))))

    comp   = (qamp, float(nu0), qfwhm)
    x0_lim = (float(cand_fmin), float(cand_fmax))
    fw_lim = (qf_lo, qf_hi)
    return comp, x0_lim, fw_lim


# ---------------------------------------------------------------------------
# Parameter error propagation
# ---------------------------------------------------------------------------

def _lorentz_integral_exact_derivs(
    nu0: float, fwhm: float, amp: float, fmin: float, fmax: float
) -> Tuple[float, float, float, float]:
    """
    Exact integral of a Lorentzian over [fmin, fmax] and its partial
    derivatives w.r.t. (amp, nu0, fwhm).

    L(f) = A * g² / ((f - ν₀)² + g²),   g = fwhm/2

    Integral: I = A * g * [arctan((fmax - ν₀)/g) - arctan((fmin - ν₀)/g)]

    Derivatives (for delta-method error propagation):
        ∂I/∂A    = I / A
        ∂I/∂ν₀  = A * g² * [1/d_min - 1/d_max]
        ∂I/∂fwhm = A/2 * [φ + g * ((fmin-ν₀)/d_min - (fmax-ν₀)/d_max)]

    where d_min = (fmin-ν₀)² + g²,  d_max = (fmax-ν₀)² + g²

    Returns (I, ∂I/∂A, ∂I/∂ν₀, ∂I/∂fwhm).
    """
    A, nu0, fwhm = float(amp), float(nu0), float(fwhm)
    if A <= 0 or fwhm <= 0:
        return 0.0, 0.0, 0.0, 0.0
    g = 0.5 * fwhm
    hi = float(fmax) - nu0
    lo = float(fmin) - nu0
    d_max = hi * hi + g * g
    d_min = lo * lo + g * g
    phi   = np.arctan(hi / g) - np.arctan(lo / g)
    I     = A * g * phi

    dI_dA    = g * phi                              # = I / A
    dI_dnu0  = A * g * g * (1.0 / d_min - 1.0 / d_max)
    dI_dfwhm = 0.5 * A * (phi + g * (lo / d_min - hi / d_max))

    return float(I), float(dI_dA), float(dI_dnu0), float(dI_dfwhm)


def _rms2_err_from_cov(
    nu0: float, fwhm: float, amp: float,
    cov3: np.ndarray,
    fmin: float, fmax: float,
) -> Tuple[float, float, float, float]:
    """
    Propagate parameter covariance to RMS amplitude errors via delta method.

    cov3 : 3×3 sub-covariance for [amp, nu0, fwhm] of one QPO component.

    Returns (rms2, rms2_err, rms, rms_err).
    All values are NaN if rms2 ≤ 0 or covariance is non-finite.
    """
    I, dI_dA, dI_dnu0, dI_dfwhm = _lorentz_integral_exact_derivs(
        nu0, fwhm, amp, fmin, fmax
    )
    if I <= 0 or not np.all(np.isfinite(cov3)):
        return float(I), np.nan, np.nan, np.nan

    # Gradient vector [∂I/∂A, ∂I/∂ν₀, ∂I/∂Δ]
    g = np.array([dI_dA, dI_dnu0, dI_dfwhm], dtype=float)

    # σ²(I) = gᵀ Σ g
    var_I = float(g @ cov3 @ g)
    if var_I < 0:
        var_I = 0.0

    rms2     = float(I)
    rms2_err = float(np.sqrt(var_I))
    rms      = float(np.sqrt(max(I, 0.0)))
    # σ(rms) = σ(√I) = σ(I) / (2√I)  by delta method
    rms_err  = float(rms2_err / (2.0 * rms)) if rms > 0 else np.nan

    return rms2, rms2_err, rms, rms_err


def _q_err_from_cov(
    nu0: float, fwhm: float, cov_nu0_fwhm: np.ndarray
) -> float:
    """
    Propagate parameter covariance to error on Q = ν₀ / fwhm.

    cov_nu0_fwhm : 2×2 sub-covariance [[σ²(ν₀), Cov(ν₀,Δ)],
                                         [Cov(Δ,ν₀), σ²(Δ)]]
    """
    if fwhm <= 0 or not np.all(np.isfinite(cov_nu0_fwhm)):
        return np.nan
    # Gradient of Q w.r.t. [ν₀, fwhm]: [1/fwhm, -ν₀/fwhm²]
    g = np.array([1.0 / fwhm, -nu0 / (fwhm * fwhm)], dtype=float)
    var_Q = float(g @ cov_nu0_fwhm @ g)
    return float(np.sqrt(max(var_Q, 0.0)))



def _nu_max(nu0: float, fwhm: float) -> float:
    """
    Characteristic frequency (Belloni convention): peak of ν × L(ν).

        ν_max = sqrt(ν₀² + (FWHM/2)²)

    For a narrow QPO (Q >> 1), ν_max ≈ ν₀.
    For a broad zero-centred Lorentzian, ν_max = FWHM/2.
    """
    g = 0.5 * float(fwhm)
    return float(np.sqrt(float(nu0) ** 2 + g ** 2))


def _nu_max_err_from_cov(
    nu0: float, fwhm: float, cov3: np.ndarray
) -> float:
    """
    Propagate parameter covariance to the error on ν_max via the delta method.

    cov3 : 3×3 sub-covariance for [amp, nu0, fwhm] of one component.

    ∂ν_max/∂ν₀   = ν₀ / ν_max
    ∂ν_max/∂FWHM = FWHM / (4 · ν_max)   [= g / (2 · ν_max)]
    """
    nmax = _nu_max(nu0, fwhm)
    if nmax <= 0 or not np.all(np.isfinite(cov3)):
        return np.nan
    # Gradient w.r.t. [amp, nu0, fwhm] — amp has no effect on ν_max
    g = np.array([0.0,
                  float(nu0)  / nmax,
                  float(fwhm) / (4.0 * nmax)], dtype=float)
    var = float(g @ cov3 @ g)
    return float(np.sqrt(max(var, 0.0)))


def _extract_component_cov(p_cov: np.ndarray, comp_idx: int) -> Optional[np.ndarray]:
    """
    Extract the 3×3 sub-covariance for component comp_idx from the full
    p_cov matrix.  Returns None if p_cov is None or too small.

    Stingray parameter order: [amp_0, nu0_0, fwhm_0, amp_1, nu0_1, fwhm_1, ..., const]
    Component k occupies indices [3k, 3k+1, 3k+2].
    """
    if p_cov is None or not np.all(np.isfinite(p_cov)):
        return None
    base = 3 * int(comp_idx)
    if base + 3 > p_cov.shape[0]:
        return None
    idx  = [base, base + 1, base + 2]
    cov3 = p_cov[np.ix_(idx, idx)]
    if np.any(np.diag(cov3) < 0):
        return None
    return cov3


# ---------------------------------------------------------------------------
# Failure helper
# ---------------------------------------------------------------------------

def _make_failure(msg: str, f: Optional[np.ndarray] = None) -> FitResult:
    safe_f = f if (f is not None and f.size > 0) else np.array([])
    return FitResult(
        ok=False, message=msg, nlor=0,
        pars=np.empty((0, 3)), comp_types=[],
        const=0.0, freq=safe_f, model=np.full(safe_f.size, np.nan),
        aic=np.nan, bic=np.nan, deviance=np.nan,
        rchi2=np.nan, red_deviance=np.nan, meta={},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_lorentzian_family(
    freq, power, power_err=None,
    *,
    # Data
    m=1, cand_freq=None, cand_power=None, cand_m_eff=None,
    fit_fmin: float = 0.05, fit_fmax: float = 64.0,
    cand_fmin: float = 0.05, cand_fmax: float = 10.0,
    include_const: bool = True, const_seed_fmin: float = 40.0,
    const_cap_factor: float = 5.0,
    # Component bounds
    cont_x0_narrow_hz: float = 0.3, cont_x0_wide_hz: float = 3.0,
    cont_x0_free_hz: float = 8.0,
    cont_fwhm_lim=(0.30, 64.0), qpo_fwhm_lim=(0.08, 5.0),
    cont_amp_factor: float = 12.0, qpo_amp_factor: float = 12.0,
    qpo_fwhm_frac: float = 0.03, qpo_fwhm_min: float = 0.08,
    # Candidate finder
    smooth_scales=None, smooth_hz: float = 0.5, prominence: float = 0.5,
    min_sep_hz: float = 0.15, max_candidates: int = 6,
    cand_require_ksigma=None, cand_sigma_mode: str = "cont",
    # Seeds
    seed_peak_hz=None, forced_qpo_seeds=None, max_qpos: int = 1,
    # IC selection
    cont_ic_criterion: str = "bic", cont_ic_delta_min: float = 15.0,
    qpo_ic_criterion: str = "aic", qpo_ic_delta_min: float = 10.0,
    qpo_detect_qmin: float = 3.0,
    # Multi-QPO: IC delta for the 1-QPO → 2-QPO upgrade (separate gate).
    # Typically set stricter than qpo_ic_delta_min so secondary QPOs are
    # only accepted when there is clear evidence for them.
    multi_qpo_ic_delta_min: Optional[float] = None,
    # rchi2 fallback: accept QPO/multi-QPO when IC gate fails but rchi2 improves.
    qpo_rchi_improve_min: float = 0.05,
    # Guardrails
    guard_enable: bool = True, guard_overshoot_ksigma: float = 4.0,
    guard_overshoot_max_run_bins: int = 6, guard_overshoot_max_frac: float = 0.10,
    guard_comp_local_amp_factor: float = 6.0,
    # Misc
    eps_amp: float = 1e-30,
    # Ignored pass-throughs from QPO_main (Powell-era params, fitmethod, etc.)
    **_ignored,
) -> FitResult:
    """
    Fit a PDS with a sum of Lorentzians using TripleA (L-BFGS-B).

    Pipeline
    --------
    1.  Filter data to [fit_fmin, fit_fmax].
    2.  Compute amplitude seeds from low-frequency power level.
    3.  Collect QPO seeds: forced_qpo_seeds → seed_peak_hz → find_qpo_candidates.
    4.  Build model configurations:
          cont2, cont3                         (always)
          cont2+QPO@s, cont3+QPO@s            (per seed s, 1-QPO)
          cont2+QPO@s1+QPO@s2, cont3+...      (per pair, when max_qpos >= 2)
    5.  Run _run_triplea_stage for each configuration.
    6.  IC selection:
          BIC gate  cont2  → cont3      (cont_ic_delta_min)
          AIC/rchi2 cont   → 1-QPO     (qpo_ic_delta_min + rchi2 fallback)
          AIC/rchi2 1-QPO  → 2-QPO     (multi_qpo_ic_delta_min + rchi2 fallback)
    """
    # ── 1. Data ───────────────────────────────────────────────────────────
    f_all = np.asarray(freq,  float)
    p_all = np.asarray(power, float)
    e_all = None if power_err is None else np.asarray(power_err, float)

    sel = ((f_all >= fit_fmin) & (f_all <= fit_fmax)
           & np.isfinite(p_all) & (p_all > 0))
    f = f_all[sel]
    p = p_all[sel]
    e = None if e_all is None else e_all[sel]

    if f.size < 30:
        return _make_failure("Too few bins in fit band", f)

    df    = float(np.median(np.diff(f)))
    m_arr = _m_as_array(m, f_all.size)[sel]

    # ── 2. Amplitude seeds ────────────────────────────────────────────────
    lowf         = max(_seed_lowf_level(f, p, fmax=min(2.0, fit_fmax)), eps_amp)
    c0           = _seed_const(f, p, const_seed_fmin)
    const_cap    = float(max(eps_amp, const_cap_factor * max(c0, eps_amp)))
    cont_amp_cap = float(cont_amp_factor) * lowf
    qpo_amp_cap  = float(qpo_amp_factor)  * lowf

    # multi_qpo_ic_delta_min defaults to qpo_ic_delta_min if not supplied
    _multi_delta = (float(multi_qpo_ic_delta_min)
                    if multi_qpo_ic_delta_min is not None
                    else float(qpo_ic_delta_min))

    # ── 3. Centroid bounds ────────────────────────────────────────────────
    x0_narrow = (-float(cont_x0_narrow_hz), float(cont_x0_narrow_hz))
    x0_wide   = (-float(cont_x0_wide_hz),   float(cont_x0_wide_hz))
    x0_free   = (-float(cont_x0_free_hz),   float(cont_x0_free_hz))
    cf_lo, cf_hi = float(cont_fwhm_lim[0]), float(cont_fwhm_lim[1])

    # ── 4. Continuum component seeds ──────────────────────────────────────
    def _cc(amp, fwhm):
        return (float(np.clip(amp, eps_amp, cont_amp_cap)),
                0.0,
                float(np.clip(fwhm, cf_lo, cf_hi)))

    cont1 = _cc(lowf,       2.0)
    cont2 = _cc(lowf * 0.5, 10.0)
    cont3 = _cc(lowf * 0.3,  1.5)

    # ── 5. QPO seeds (forced → peak → candidates) ─────────────────────────
    qpo_seeds: List[float] = []

    def _add_seed(nu: float) -> None:
        nu = float(nu)
        if not (np.isfinite(nu) and cand_fmin <= nu <= cand_fmax):
            return
        if any(abs(nu - s) < float(min_sep_hz) for s in qpo_seeds):
            return
        qpo_seeds.append(nu)

    for nu in (forced_qpo_seeds or []):
        _add_seed(nu)
    if seed_peak_hz is not None:
        _add_seed(seed_peak_hz)

    cf     = np.asarray(cand_freq  if cand_freq  is not None else f, float)
    cp     = np.asarray(cand_power if cand_power is not None else p, float)
    m_cand = _safe_scalar_m(cand_m_eff if cand_m_eff is not None else m)
    for cand in find_qpo_candidates(
        cf, cp, cand_fmin=cand_fmin, cand_fmax=cand_fmax,
        smooth_scales=(list(smooth_scales) if smooth_scales else [float(smooth_hz)]),
        smooth_hz=float(smooth_hz), prominence=float(prominence),
        min_sep_hz=float(min_sep_hz), max_candidates=int(max_candidates),
        m_eff=m_cand, require_ksigma=cand_require_ksigma,
        cand_sigma_mode=str(cand_sigma_mode),
    ):
        _add_seed(cand.get("nu_hz", np.nan))

    n_forced = len([s for s in (forced_qpo_seeds or [])
                    if np.isfinite(s) and cand_fmin <= float(s) <= cand_fmax])
    qpo_seeds = qpo_seeds[:max(int(max_qpos), n_forced)]

    # ── 6. Build model configurations ─────────────────────────────────────
    # Shared args for _build_qpo_comp
    _bqc_kw = dict(
        f=f, p=p, lowf=lowf, eps_amp=eps_amp,
        qpo_amp_cap=qpo_amp_cap, qpo_fwhm_lim=qpo_fwhm_lim,
        qpo_detect_qmin=float(qpo_detect_qmin),
        qpo_fwhm_min=float(qpo_fwhm_min), qpo_fwhm_frac=float(qpo_fwhm_frac),
        cand_fmin=float(cand_fmin), cand_fmax=float(cand_fmax),
    )

    c2_comps = [cont1, cont2]
    c2_x0    = [x0_narrow, x0_wide]
    c2_fw    = [cont_fwhm_lim, cont_fwhm_lim]
    c2_ac    = [cont_amp_cap, cont_amp_cap]

    c3_comps = [cont1, cont2, cont3]
    c3_x0    = [x0_narrow, x0_wide, x0_free]
    c3_fw    = [cont_fwhm_lim] * 3
    c3_ac    = [cont_amp_cap]  * 3

    # (comps, x0_lims, fwhm_lims, amp_caps, comp_types, tag)
    configs = [
        (c2_comps, c2_x0, c2_fw, c2_ac, ["cont", "cont"],        "cont2"),
        (c3_comps, c3_x0, c3_fw, c3_ac, ["cont", "cont", "cont"], "cont3"),
    ]

    # Single-QPO configurations — one per seed
    for nu0 in qpo_seeds:
        qq, xq, qfw = _build_qpo_comp(nu0=nu0, **_bqc_kw)
        configs.append((
            c2_comps + [qq], c2_x0 + [xq], c2_fw + [qfw],
            c2_ac + [qpo_amp_cap], ["cont", "cont", "qpo"],
            f"cont2+qpo@{nu0:.3g}Hz",
        ))
        configs.append((
            c3_comps + [qq], c3_x0 + [xq], c3_fw + [qfw],
            c3_ac + [qpo_amp_cap], ["cont", "cont", "cont", "qpo"],
            f"cont3+qpo@{nu0:.3g}Hz",
        ))

    # Simultaneous 2-QPO configurations — one per unique seed pair
    # Only attempted when max_qpos >= 2 AND at least 2 seeds are available.
    if int(max_qpos) >= 2 and len(qpo_seeds) >= 2:
        for i, nu1 in enumerate(qpo_seeds):
            for nu2 in qpo_seeds[i + 1:]:
                q1, xq1, qfw1 = _build_qpo_comp(nu0=nu1, **_bqc_kw)
                q2, xq2, qfw2 = _build_qpo_comp(nu0=nu2, **_bqc_kw)
                pair_tag = f"qpo@{nu1:.3g}+{nu2:.3g}Hz"
                configs.append((
                    c2_comps + [q1, q2],
                    c2_x0   + [xq1, xq2],
                    c2_fw   + [qfw1, qfw2],
                    c2_ac   + [qpo_amp_cap, qpo_amp_cap],
                    ["cont", "cont", "qpo", "qpo"],
                    f"cont2+{pair_tag}",
                ))
                configs.append((
                    c3_comps + [q1, q2],
                    c3_x0   + [xq1, xq2],
                    c3_fw   + [qfw1, qfw2],
                    c3_ac   + [qpo_amp_cap, qpo_amp_cap],
                    ["cont", "cont", "cont", "qpo", "qpo"],
                    f"cont3+{pair_tag}",
                ))

    # ── 7. Run each configuration ─────────────────────────────────────────
    guard_kw = dict(
        guard_enable=bool(guard_enable),
        guard_overshoot_ksigma=float(guard_overshoot_ksigma),
        guard_overshoot_max_run_bins=int(guard_overshoot_max_run_bins),
        guard_overshoot_max_frac=float(guard_overshoot_max_frac),
        guard_comp_local_amp_factor=float(guard_comp_local_amp_factor),
    )

    all_fits: List[FitResult] = []
    for comps, x0_lims, fwhm_lims, amp_caps, comp_types, tag in configs:
        nlor = len(comps)
        t0   = [v for amp_s, x0_s, fwhm_s in comps
                  for v in (float(amp_s), float(x0_s), float(fwhm_s))]
        if include_const:
            t0.append(float(c0))
        fit = _run_triplea_stage(
            f, p, e, m_arr,
            nlor=nlor, t0=t0, x0_lims=x0_lims, fwhm_lims=fwhm_lims,
            amp_caps=amp_caps, comp_types=comp_types,
            include_const=include_const, const_cap=const_cap, eps_amp=eps_amp,
            qpo_detect_qmin=float(qpo_detect_qmin), df=df, tag=tag, **guard_kw,
        )
        if fit is not None:
            all_fits.append(fit)

    if not all_fits:
        return _make_failure("TripleA: all configurations failed guardrails", f)

    # ── 8. IC selection ───────────────────────────────────────────────────
    def _nq(fit: FitResult) -> int:
        return sum(1 for t in fit.comp_types if t == "qpo")

    # Best fit per (n_qpo, n_cont) bucket
    best_by_key: Dict[Tuple[int, int], FitResult] = {}
    for fit in all_fits:
        nq  = _nq(fit)
        key = (nq, fit.nlor - nq)
        ic  = _ic_value(fit, qpo_ic_criterion)
        if key not in best_by_key or ic < _ic_value(best_by_key[key], qpo_ic_criterion):
            best_by_key[key] = fit

    # Continuum upgrade: cont2 → cont3 (BIC gate)
    fit_c2, fit_c3 = best_by_key.get((0, 2)), best_by_key.get((0, 3))
    if fit_c2 is None:
        fit_0q = fit_c3
    elif fit_c3 is None:
        fit_0q = fit_c2
    else:
        fit_0q = (fit_c3 if _accept_upgrade_ic(fit_c2, fit_c3,
                  criterion=cont_ic_criterion, delta_min=float(cont_ic_delta_min)) else fit_c2)

    # Best 1-QPO fit (across all continuum orders)
    fit_1q: Optional[FitResult] = None
    for nc in range(2, 6):
        cand = best_by_key.get((1, nc))
        if cand is None:
            continue
        if fit_1q is None or _ic_value(cand, qpo_ic_criterion) < _ic_value(fit_1q, qpo_ic_criterion):
            fit_1q = cand

    # Best 2-QPO fit (across all continuum orders)
    fit_2q: Optional[FitResult] = None
    if int(max_qpos) >= 2:
        for nc in range(2, 6):
            cand = best_by_key.get((2, nc))
            if cand is None:
                continue
            if fit_2q is None or _ic_value(cand, qpo_ic_criterion) < _ic_value(fit_2q, qpo_ic_criterion):
                fit_2q = cand

    def _rchi_improves(ref: Optional[FitResult], challenger: FitResult) -> bool:
        r0, r1 = _rchi2_of(ref), _rchi2_of(challenger)
        return np.isfinite(r0) and np.isfinite(r1) and r1 < r0 - float(qpo_rchi_improve_min)

    # 0-QPO → 1-QPO gate (AIC + rchi2 fallback)
    if fit_0q is None:
        best = fit_1q
    elif fit_1q is None:
        best = fit_0q
    else:
        ic_ok   = _accept_upgrade_ic(fit_0q, fit_1q, criterion=qpo_ic_criterion,
                                     delta_min=float(qpo_ic_delta_min))
        best = fit_1q if (ic_ok or _rchi_improves(fit_0q, fit_1q)) else fit_0q

    # 1-QPO → 2-QPO gate (multi_qpo_ic_delta_min + rchi2 fallback)
    if fit_2q is not None:
        ref = best  # compare against current winner (may be 0 or 1 QPO)
        if ref is None:
            best = fit_2q
        else:
            ic_ok = _accept_upgrade_ic(ref, fit_2q, criterion=qpo_ic_criterion,
                                       delta_min=_multi_delta)
            if ic_ok or _rchi_improves(ref, fit_2q):
                best = fit_2q

    return best if best is not None else all_fits[0]


# ---------------------------------------------------------------------------
# Stingray / Powell compatibility stubs
# ---------------------------------------------------------------------------
# The interactive fitter (QPO_interactive.py) still uses the Stingray Powell
# path for its `fit` command.  These three functions are imported from here
# so that QPO_interactive.py does not need its own copies.
#
# None of these are called by fit_lorentzian_family or the TripleA path.

def _half_uniform(upper: float, eps: float = 1e-30):
    """
    Uniform prior on [eps, upper] for positive-definite parameters
    (amplitudes, white-noise constant).

    Returns a scipy.stats.uniform distribution, which has a .logpdf method
    accepted by Stingray's fit_lorentzians(priors=...) interface.
    """
    from scipy.stats import uniform as _u
    lo = float(eps)
    hi = max(float(upper), lo + 1e-30)
    return _u(loc=lo, scale=hi - lo)


def _hard_trunc_uniform(lo: float, hi: float):
    """
    Uniform prior on [lo, hi] for bounded parameters (nu0, fwhm).

    Returns a scipy.stats.uniform distribution accepted by Stingray's
    fit_lorentzians(priors=...) interface.
    """
    from scipy.stats import uniform as _u
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1e-30
    return _u(loc=lo, scale=hi - lo)


def _repair_params(
    t0: np.ndarray,
    *,
    nlor: int,
    include_const: bool,
    x0_lims: list,
    fwhm_lims: list,
    const_max: float,
    eps_amp: float = 1e-30,
) -> np.ndarray:
    """
    Clip a Stingray-order parameter vector t0 to its bounds.

    Stingray order: [amp_0, x0_0, fwhm_0, ..., const]

    Used by the interactive fitter before calling fit_lorentzians to
    ensure the starting point is strictly interior to the prior support.
    """
    t = np.asarray(t0, float).copy()
    for i in range(nlor):
        t[3 * i]     = float(max(float(eps_amp), t[3 * i]))
        lo_x, hi_x   = x0_lims[i]
        t[3 * i + 1] = float(np.clip(t[3 * i + 1], float(lo_x), float(hi_x)))
        flo, fhi      = fwhm_lims[i]
        t[3 * i + 2] = float(np.clip(t[3 * i + 2], float(flo), float(fhi)))
    if include_const:
        t[-1] = float(np.clip(t[-1], float(eps_amp), float(const_max)))
    return t
