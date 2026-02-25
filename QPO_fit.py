#!/usr/bin/env python3
# QPO_fit.py
#
#Requires stingray and numba

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from stingray import Powerspectrum
from stingray.modeling.scripts import fit_lorentzians

import scipy.signal


# -----------------------
# utilities expected by QPO_main.py
# -----------------------

def lorentz(freq: np.ndarray, nu0: float, fwhm: float, amp: float) -> np.ndarray:
    
    f = np.asarray(freq, float)
    g = 0.5 * float(fwhm)
    return float(amp) * (g * g) / ((f - float(nu0)) ** 2 + g * g)


def component_power_integral(freq: np.ndarray, comp: np.ndarray, fmin: float, fmax: float) -> float:
    f = np.asarray(freq, float)
    y = np.asarray(comp, float)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(f) & np.isfinite(y)
    if np.sum(m) < 2:
        return 0.0
    return float(np.trapz(y[m], f[m]))


def extract_qpo_params(fitres, *, qpo_fmin: float, qpo_fmax: float, qmin: float = 3.0):
    
    if fitres is None or not getattr(fitres, "ok", False):
        return None

    pars = np.asarray(getattr(fitres, "pars", []), float)
    if pars.size == 0:
        return None

    freq = np.asarray(getattr(fitres, "freq", []), float)
    if freq.size < 2:
        return None

    best = None
    for i, (nu0, fwhm, amp) in enumerate(pars):
        if not (np.isfinite(nu0) and np.isfinite(fwhm) and np.isfinite(amp)):
            continue
        if fwhm <= 0:
            continue
        if not (qpo_fmin <= nu0 <= qpo_fmax):
            continue

        Q = nu0 / fwhm
        if Q < qmin:
            continue

        area = float(np.trapz(lorentz(freq, nu0, fwhm, amp), freq))
        cand = dict(
            qpo_index=int(i),
            qpo_nu0_hz=float(nu0),
            qpo_fwhm_hz=float(fwhm),
            qpo_Q=float(Q),
            qpo_area=float(area),
        )
        if best is None or cand["qpo_area"] > best["qpo_area"]:
            best = cand
    return best


# -----------------------
# FitResult container
# -----------------------

@dataclass
class FitResult:
    ok: bool
    message: str
    nlor: int
    pars: np.ndarray   # (nlor, 3) => (nu0, fwhm, amp)
    const: float
    freq: np.ndarray
    model: np.ndarray
    aic: float
    bic: float
    deviance: float
    p_opt: np.ndarray
    p_err: Optional[np.ndarray] = None
    rchi2: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------
# internal helpers
# -----------------------

def _make_ps(freq, power, power_err=None, m=1, norm="frac") -> Powerspectrum:
    ps = Powerspectrum()
    ps.freq = np.asarray(freq, float)
    ps.power = np.asarray(power, float)
    ps.power_err = None if power_err is None else np.asarray(power_err, float)
    ps.df = float(np.median(np.diff(ps.freq)))
    ps.m = m  # can be scalar or array-like
    ps.norm = norm
    return ps


def _safe_scalar_m(m) -> int:
    try:
        arr = np.asarray(m, float)
        v = float(arr) if arr.ndim == 0 else float(np.nanmedian(arr))
        if (not np.isfinite(v)) or v < 1:
            return 1
        return int(v)
    except Exception:
        return 1


def _rolling_median(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    w = int(w)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    k = w // 2
    ypad = np.pad(y, (k, k), mode="edge")
    return np.array([np.median(ypad[i:i + w]) for i in range(len(y))], float)


def _seed_const(freq, power, const_seed_fmin):
    m = (freq >= const_seed_fmin) & np.isfinite(power) & (power > 0)
    if np.sum(m) < 10:
        mm = np.isfinite(power) & (power > 0)
        return float(np.nanmedian(power[mm])) if np.any(mm) else 0.0
    return float(np.nanmedian(power[m]))


def _seed_amp_at(freq, power, nu0):
    i = int(np.argmin(np.abs(np.asarray(freq) - float(nu0))))
    return float(np.asarray(power)[i])


def _seed_lowf_level(freq, power, fmax=2.0):
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    m = (f > 0) & (f <= float(fmax)) & np.isfinite(p) & (p > 0)
    if np.sum(m) < 10:
        mm = np.isfinite(p) & (p > 0)
        return float(np.nanmedian(p[mm])) if np.any(mm) else 0.0
    return float(np.nanmedian(p[m]))


def _compute_rchi2(power, model, power_err, m_avg, npar):
    
    p = np.asarray(power, float)
    mod = np.asarray(model, float)

    if power_err is None:
        m_eff = max(1, int(m_avg))
        err = p / np.sqrt(m_eff)
    else:
        err = np.asarray(power_err, float)

    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(err) & (err > 0)
    dof = int(np.sum(good) - int(npar))
    if np.sum(good) < 20 or dof <= 0:
        return np.nan
    chi2 = np.sum(((p[good] - mod[good]) / err[good]) ** 2)
    return float(chi2 / dof)


# -----------------------
# Candidate finder (sigma-gated) + helpers
# -----------------------

def _estimate_sigma_local(*, cont: np.ndarray, p: np.ndarray, m_eff: int, mode: str = "cont") -> np.ndarray:
    
    m_eff = max(1, int(m_eff))
    mode = str(mode).strip().lower()
    base = np.asarray(cont if mode == "cont" else p, float)
    base = np.where(np.isfinite(base) & (base > 0), base, np.nan)
    med = float(np.nanmedian(base)) if np.any(np.isfinite(base)) else 1.0
    base = np.where(np.isfinite(base), base, med)
    return base / np.sqrt(float(m_eff))


def find_qpo_candidates(
    freq,
    power,
    *,
    cand_fmin,
    cand_fmax,
    smooth_hz=0.5,
    prominence=0.12,
    min_sep_hz=0.15,
    max_candidates=5,
    min_width_bins=7,
    # --- sigma gate knobs ---
    m_eff: int = 1,
    require_ksigma: Optional[float] = None,
    cand_sigma_mode: str = "cont",
) -> List[Dict[str, float]]:
    """
    Find candidate QPO peaks using rolling-median whitening (ratio = P/cont),
    then optionally sigma-gate by excess above cont.

    Returns list sorted by prominence desc:
      {"nu_hz", "prominence", "excess_sigma", "ratio_peak"}
    """
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    m = (f >= cand_fmin) & (f <= cand_fmax) & np.isfinite(p) & (p > 0)
    f = f[m]
    p = p[m]
    if f.size < 25:
        return []

    df = np.median(np.diff(f))
    if not np.isfinite(df) or df <= 0:
        return []

    w = int(np.round(float(smooth_hz) / float(df)))
    w = max(int(min_width_bins), w)
    if w % 2 == 0:
        w += 1
    if w >= len(p):
        w = len(p) - 1 if (len(p) % 2 == 0) else len(p)
        if w < 3:
            return []

    cont = _rolling_median(p, w)
    good = np.isfinite(cont) & (cont > 0)
    if not np.any(good):
        return []
    cont = np.where(good, cont, np.nanmedian(cont[good]))

    ratio = p / cont

    distance = int(max(1, np.round(float(min_sep_hz) / float(df))))
    peaks, props = scipy.signal.find_peaks(ratio, prominence=float(prominence), distance=distance)
    if peaks.size == 0:
        return []

    prom = np.asarray(props.get("prominences", np.zeros_like(peaks, float)), float)

    if require_ksigma is not None and np.isfinite(require_ksigma) and float(require_ksigma) > 0:
        sigma = _estimate_sigma_local(cont=cont, p=p, m_eff=int(m_eff), mode=str(cand_sigma_mode))
        excess_sigma_all = (p - cont) / sigma
        keep = (excess_sigma_all[peaks] >= float(require_ksigma))
        peaks = peaks[keep]
        prom = prom[keep]
        if peaks.size == 0:
            return []
    else:
        excess_sigma_all = np.full_like(p, np.nan, dtype=float)

    order = np.argsort(prom)[::-1]
    peaks = peaks[order][: int(max_candidates)]
    prom = prom[order][: int(max_candidates)]

    out: List[Dict[str, float]] = []
    for idx, pr in zip(peaks, prom):
        idx = int(idx)
        out.append(
            dict(
                nu_hz=float(f[idx]),
                prominence=float(pr),
                excess_sigma=float(excess_sigma_all[idx]) if np.isfinite(excess_sigma_all[idx]) else float("nan"),
                ratio_peak=float(ratio[idx]),
            )
        )
    return out


def _nearest_candidate_metrics(cands: List[Dict[str, float]], nu: float, *, tol_hz: float) -> Tuple[float, float]:
    """
    Return (prominence, excess_sigma) of nearest candidate within tol_hz; else (nan, nan).
    """
    if not cands:
        return (float("nan"), float("nan"))
    nu = float(nu)
    best = None
    best_d = np.inf
    for c in cands:
        x = float(c.get("nu_hz", np.nan))
        if not np.isfinite(x):
            continue
        d = abs(x - nu)
        if d < best_d:
            best_d = d
            best = c
    if best is None or best_d > float(tol_hz):
        return (float("nan"), float("nan"))
    return (float(best.get("prominence", np.nan)), float(best.get("excess_sigma", np.nan)))


# -----------------------
# HARD-truncated priors (uniform on bounded intervals)
# -----------------------

def _hard_trunc_uniform_pdf(x, lo, hi):
    if not np.isfinite(x):
        return 0.0
    if x < lo or x > hi:
        return 0.0
    w = max(float(hi - lo), 1e-300)
    return 1.0 / w


def _half_uniform_pdf(hi):
    hi = float(hi)
    if (not np.isfinite(hi)) or hi <= 0:
        hi = 1.0

    def prior(x):
        return _hard_trunc_uniform_pdf(float(x), 0.0, hi)

    return prior


def _uniform_pdf(lo, hi):
    lo = float(lo)
    hi = float(hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0

    def prior(x):
        return _hard_trunc_uniform_pdf(float(x), lo, hi)

    return prior


def _build_priors(
    nlor: int,
    *,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    amp_max_list: List[float],
    include_const: bool,
    const_max: float,
) -> Dict[str, Any]:
    priors: Dict[str, Any] = {}
    for i in range(nlor):
        priors[f"amplitude_{i}"] = _half_uniform_pdf(amp_max_list[i])
        priors[f"x_0_{i}"] = _uniform_pdf(*x0_lims[i])
        priors[f"fwhm_{i}"] = _uniform_pdf(*fwhm_lims[i])
    if include_const:
        priors[f"amplitude_{nlor}"] = _half_uniform_pdf(const_max)
    return priors


def _pack_t0(comps_amp_x0_fwhm: List[Tuple[float, float, float]], const: Optional[float]):
    t0 = []
    for amp, x0, fwhm in comps_amp_x0_fwhm:
        t0 += [float(amp), float(x0), float(fwhm)]
    if const is not None:
        t0 += [float(const)]
    return t0


# -----------------------
# guardrails
# -----------------------

def _local_median_around(f, p, nu0, width_hz=0.5):
    f = np.asarray(f, float)
    p = np.asarray(p, float)
    if not (np.isfinite(nu0) and np.isfinite(width_hz) and width_hz > 0):
        mm = np.isfinite(p) & (p > 0)
        return float(np.nanmedian(p[mm])) if np.any(mm) else np.nan
    m = (f >= (nu0 - width_hz)) & (f <= (nu0 + width_hz)) & np.isfinite(p) & (p > 0)
    if np.sum(m) < 5:
        mm = np.isfinite(p) & (p > 0)
        return float(np.nanmedian(p[mm])) if np.any(mm) else np.nan
    return float(np.nanmedian(p[m]))


def _guardrail_overshoot(f, data_p, model_p, power_err, m_eff,
                         *, ksigma=3.0, max_run_bins=6, max_frac=0.08):
    p = np.asarray(data_p, float)
    mod = np.asarray(model_p, float)

    if power_err is None:
        m0 = max(1, int(m_eff))
        sig = p / np.sqrt(m0)
    else:
        sig = np.asarray(power_err, float)

    good = np.isfinite(p) & np.isfinite(mod) & np.isfinite(sig) & (sig > 0)
    if np.sum(good) < 30:
        return True, ""

    resid = mod[good] - p[good]
    thr = float(ksigma) * sig[good]
    bad = resid > thr

    if not np.any(bad):
        return True, ""

    frac = float(np.mean(bad))

    run = 0
    maxrun = 0
    for b in bad.astype(int):
        if b:
            run += 1
            maxrun = max(maxrun, run)
        else:
            run = 0

    if (maxrun > int(max_run_bins)) or (frac > float(max_frac)):
        return False, f"Overshoot FAIL run={maxrun} frac={frac:.4g}"
    return True, ""


def _guardrail_component_local_amp(f, p, fit_pars, *, local_amp_factor=5.0, local_width_hz=0.5):
    for (nu0, fwhm, amp) in np.asarray(fit_pars, float):
        if not (np.isfinite(nu0) and np.isfinite(amp)):
            continue
        if amp <= 0:
            continue
        local = _local_median_around(f, p, nu0, width_hz=local_width_hz)
        if not np.isfinite(local) or local <= 0:
            continue
        ratio = float(amp / local)
        if ratio > float(local_amp_factor):
            return False, f"CompAmp FAIL at {nu0:.3g}Hz ratio={ratio:.3g}"
    return True, ""


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
):
    p_opt = np.asarray(p_opt, float)
    x0_tol = float(max(2.0 * float(df), 1e-6))

    for i in range(nlor):
        amp_i = float(p_opt[3*i + 0])
        x0_i  = float(p_opt[3*i + 1])
        fwhm_i= float(p_opt[3*i + 2])

        if not np.isfinite(amp_i) or not np.isfinite(x0_i) or not np.isfinite(fwhm_i):
            return False, f"PARAM_FAIL: non-finite param in component {i}"
        if fwhm_i <= 0:
            return False, f"PARAM_FAIL: non-positive fwhm in component {i}"
        if amp_i < 0:
            return False, f"PARAM_FAIL: Negative amplitude in component {i}"

        lo, hi = x0_lims[i]
        if x0_i < (float(lo) - x0_tol) or x0_i > (float(hi) + x0_tol):
            return False, f"PARAM_FAIL: x0 out of bounds in component {i}: {x0_i}"

        flo, fhi = fwhm_lims[i]
        if fwhm_i < float(flo) or fwhm_i > float(fhi):
            return False, f"PARAM_FAIL: fwhm out of bounds in component {i}: {fwhm_i}"

        amax = float(amp_max_list[i])
        if amp_i > (amax * 1.05):
            return False, f"PARAM_FAIL: amp above cap in component {i}: {amp_i} > {amax}"

    if include_const:
        c = float(p_opt[-1])
        if not np.isfinite(c):
            return False, "PARAM_FAIL: non-finite const"
        if c < 0:
            return False, "PARAM_FAIL: Negative const"
        if c > (float(const_max) * 1.05):
            return False, f"PARAM_FAIL: const above cap: {c} > {const_max}"

    return True, ""


# -----------------------
# fit wrappers
# -----------------------

def _try_fit_lorentzians_once(
    ps: Powerspectrum,
    *,
    nlor: int,
    t0: np.ndarray,
    priors: Dict[str, Any],
    include_const: bool,
    fitmethod: str,
):
    try:
        parest, res = fit_lorentzians(
            ps,
            nlor,
            np.asarray(t0, float).tolist(),
            fit_whitenoise=include_const,
            max_post=True,
            priors=priors,
            fitmethod=str(fitmethod),
        )
        return res, None
    except Exception as e:
        return None, repr(e)


def _repair_clamp_params(
    t: np.ndarray,
    *,
    nlor: int,
    include_const: bool,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    const_max: float,
    eps_amp: float,
):
    out = np.asarray(t, float).copy()

    for i in range(nlor):
        amp_idx = 3*i + 0
        x0_idx  = 3*i + 1
        w_idx   = 3*i + 2

        if not np.isfinite(out[amp_idx]) or out[amp_idx] < eps_amp:
            out[amp_idx] = float(eps_amp)

        lo, hi = x0_lims[i]
        if not np.isfinite(out[x0_idx]):
            out[x0_idx] = float(np.clip(0.0, lo, hi))
        else:
            out[x0_idx] = float(np.clip(out[x0_idx], lo, hi))

        flo, fhi = fwhm_lims[i]
        if (not np.isfinite(out[w_idx])) or (out[w_idx] <= 0):
            out[w_idx] = float(max(flo, 1e-6))
        else:
            out[w_idx] = float(np.clip(out[w_idx], flo, fhi))

    if include_const and out.size >= (3*nlor + 1):
        if (not np.isfinite(out[-1])) or (out[-1] < eps_amp):
            out[-1] = float(eps_amp)
        out[-1] = float(np.clip(out[-1], 0.0, float(const_max)))

    return out


def _fit_stage(
    ps: Powerspectrum,
    *,
    comps: List[Tuple[float, float, float]],  # (amp, x0, fwhm)
    include_const: bool,
    const_seed: float,
    const_max: float,
    x0_lims: List[Tuple[float, float]],
    fwhm_lims: List[Tuple[float, float]],
    amp_max_list: List[float],
    fitmethod: str,
    n_starts: int,
    jitter_frac: float,
    rng: np.random.Generator,
    eps_amp: float = 1e-30,
):
    nlor = len(comps)
    priors = _build_priors(
        nlor,
        x0_lims=x0_lims,
        fwhm_lims=fwhm_lims,
        amp_max_list=amp_max_list,
        include_const=include_const,
        const_max=const_max,
    )

    base_t0 = np.array(_pack_t0(comps, const_seed if include_const else None), float)

    best_res = None
    last_exc = None

    for _ in range(max(1, int(n_starts))):
        t0 = base_t0.copy()
        nj = 3 * nlor
        if nj > 0 and jitter_frac > 0:
            jit = 1.0 + float(jitter_frac) * rng.normal(size=nj)
            jit = np.clip(jit, 0.5, 2.0)
            t0[:nj] *= jit

        t0 = _repair_clamp_params(
            t0,
            nlor=nlor,
            include_const=include_const,
            x0_lims=x0_lims,
            fwhm_lims=fwhm_lims,
            const_max=const_max,
            eps_amp=eps_amp,
        )

        res, exc = _try_fit_lorentzians_once(
            ps,
            nlor=nlor,
            t0=t0,
            priors=priors,
            include_const=include_const,
            fitmethod=fitmethod,
        )
        if res is None:
            last_exc = exc
            continue

        p0 = np.asarray(res.p_opt, float)
        t1 = _repair_clamp_params(
            p0,
            nlor=nlor,
            include_const=include_const,
            x0_lims=x0_lims,
            fwhm_lims=fwhm_lims,
            const_max=const_max,
            eps_amp=eps_amp,
        )
        if np.any(t1 != p0):
            res2, _ = _try_fit_lorentzians_once(
                ps,
                nlor=nlor,
                t0=t1,
                priors=priors,
                include_const=include_const,
                fitmethod=fitmethod,
            )
            if res2 is not None:
                res = res2

        if best_res is None or getattr(res, "aic", np.inf) < getattr(best_res, "aic", np.inf):
            best_res = res

    return best_res, last_exc


# -----------------------
# IC helpers
# -----------------------

def _ic_value(fr: FitResult, criterion: str) -> float:
    c = str(criterion).strip().lower()
    if c == "bic":
        return float(fr.bic) if np.isfinite(fr.bic) else np.inf
    return float(fr.aic) if np.isfinite(fr.aic) else np.inf


def _accept_upgrade_ic(simple: FitResult, complex_: FitResult, *, criterion: str, delta_min: float) -> bool:
    d = _ic_value(simple, criterion) - _ic_value(complex_, criterion)
    return np.isfinite(d) and (d >= float(delta_min))

def fit_lorentzian_family(
    freq,
    power,
    power_err=None,
    *,
    # candidate grid (light rebin) for peak finding + reseed tests
    cand_freq: Optional[np.ndarray] = None,
    cand_power: Optional[np.ndarray] = None,
    cand_m_eff: Optional[int] = None,
    m=1,
    fit_fmin=0.05,
    fit_fmax=64.0,
    cand_fmin=0.05,
    cand_fmax=10.0,
    const_seed_fmin=30.0,
    include_const=True,

    # --- compatibility knobs (ignored for now) ---
    include_harmonic: bool = False,          # QPO_main passes this
    harm_fwhm_lim=(0.03, 8.0),               # QPO_main passes this
    harm_amp_factor: float = 5.0,            # QPO_main passes this

    # -------- candidate finder knobs --------
    smooth_hz=0.5,
    prominence=0.12,
    min_sep_hz=0.15,
    max_candidates=6,

    # -------- detection-first sigma gate --------
    cand_require_ksigma: Optional[float] = None,
    cand_sigma_mode: str = "cont",

    # -------- multi-seed stage1 --------
    stage1_n_seeds: int = 3,

    # -------- conditional reseed --------
    reseed_enable: bool = True,
    reseed_rchi_bad: float = 1.8,
    reseed_edge_frac: float = 0.08,
    reseed_area_min: float = 0.0,
    reseed_exclude_hz_min: float = 0.5,
    reseed_exclude_df_mult: float = 10.0,
    reseed_prom_factor: float = 1.25,
    reseed_sigma_factor: float = 1.10,

    random_seed=42,
    fitmethod="Powell",
    rchi_target=1.3,

    # separate IC criteria for continuum vs QPO decisions
    cont_ic_criterion: str = "bic",
    cont_ic_delta_min: float = 10.0,
    qpo_ic_criterion: str = "aic",
    qpo_ic_delta_min: float = 2.0,

    # strict constraints
    cont_x0_max_hz: Optional[float] = None,   # user will widen this from parameter file
    cont_fwhm_lim=(0.3, 64.0),
    qpo_fwhm_lim=(0.03, 5.0),
    cont_amp_factor=10.0,
    qpo_amp_factor=5.0,
    qpo_fwhm_frac=0.06,
    qpo_fwhm_min=0.05,
    n_starts=6,
    jitter_frac=0.18,
    seed_peak_hz: Optional[float] = None,

    # gating knobs
    qpo_detect_qmin: float = 3.0,

    # guardrails
    guard_enable: bool = True,
    guard_overshoot_ksigma: float = 3.0,
    guard_overshoot_max_run_bins: int = 6,
    guard_overshoot_max_frac: float = 0.08,
    guard_comp_local_amp_factor: float = 5.0,

    # retry ladder
    max_retries: int = 5,

    # repair
    eps_amp: float = 1e-30,

    # -----------------------
    # post-QPO cont3 rescue knobs
    # -----------------------
    postqpo_cont3_enable: bool = True,
    postqpo_cont3_trigger_rchi: float = 1.6,
    postqpo_cont3_rchi_improve_min: float = 0.06,
    postqpo_cont3_ic_delta_min: float = 0.0,
    postqpo_cont3_rchi_not_worse_tol: float = 0.02,

    # -----------------------
    # NEW: hard override mode
    # if rchi is above your external criteria, always try cont3+qpo
    # -----------------------
    rchi_override_enable: bool = True,
    rchi_override_threshold: Optional[float] = None,  # if None, use postqpo_cont3_trigger_rchi
    rchi_override_min_improve: float = 0.02,          # accept if improves by this much
    **_ignored_kwargs,  # swallow any extra kwargs passed by QPO_main
) -> FitResult:
   

    # explicitly ignore harmonic knobs for now (compatibility only)
    _ = include_harmonic
    _ = harm_fwhm_lim
    _ = harm_amp_factor

    # ---- data selection ----
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    e = None if power_err is None else np.asarray(power_err, float)

    sel = (f >= fit_fmin) & (f <= fit_fmax) & np.isfinite(p) & (p > 0)
    f = f[sel]
    p = p[sel]
    e = None if e is None else e[sel]

    if f.size < 30:
        return FitResult(
            ok=False,
            message="Too few bins in fit band",
            nlor=0,
            pars=np.empty((0, 3)),
            const=0.0,
            freq=f,
            model=np.full_like(f, np.nan),
            aic=np.nan,
            bic=np.nan,
            deviance=np.nan,
            p_opt=np.array([]),
            p_err=None,
            rchi2=np.nan,
            meta={},
        )

    # ---- candidate grid fallback ----
    if cand_freq is None or cand_power is None:
        cand_freq = f
        cand_power = p
    cf = np.asarray(cand_freq, float)
    cp = np.asarray(cand_power, float)

    ps = _make_ps(f, p, e, m=m, norm="frac")
    df = float(ps.df)

    # m used for fitting diagnostics/guards
    m_fit = _safe_scalar_m(m)
    # m used for candidate sigma stats
    m_cand = _safe_scalar_m(cand_m_eff if cand_m_eff is not None else m)

    # Allow continuum centroid drift: cont_x0_max_hz
    if cont_x0_max_hz is None or (not np.isfinite(cont_x0_max_hz)) or cont_x0_max_hz <= 0:
        cont_x0_max_hz = 2.0 * df  # default legacy behavior
    cont_x0_lims = (-float(cont_x0_max_hz), float(cont_x0_max_hz))

    rng = np.random.default_rng(int(random_seed) if random_seed is not None else 12345)

    c0 = _seed_const(f, p, const_seed_fmin)
    lowf = _seed_lowf_level(f, p, fmax=min(2.0, fit_fmax))
    lowf = max(lowf, float(eps_amp))

    cont_amp_cap = float(cont_amp_factor * lowf)
    qpo_amp_cap = float(qpo_amp_factor * lowf)
    const_cap = float(max(eps_amp, 2.0 * max(c0, eps_amp)))

    # ---- base continuum components ----
    cont1 = (min(cont_amp_cap, lowf), 0.0, float(np.clip(2.0, cont_fwhm_lim[0], cont_fwhm_lim[1])))
    cont2 = (min(cont_amp_cap, lowf), 0.0, float(np.clip(10.0, cont_fwhm_lim[0], cont_fwhm_lim[1])))
    cont3_default = (min(cont_amp_cap, lowf), 0.0, float(np.clip(30.0, cont_fwhm_lim[0], cont_fwhm_lim[1])))

    # ---- wrap stingray result -> FitResult ----
    def _wrap(res, nlor, include_const_flag, stage_name, meta_extra=None):
        p_opt = np.asarray(res.p_opt, float)

        pars_out = []
        for i in range(nlor):
            amp_i = p_opt[3 * i + 0]
            x0_i = p_opt[3 * i + 1]
            fwhm_i = p_opt[3 * i + 2]
            pars_out.append((x0_i, fwhm_i, amp_i))
        pars_out = np.array(pars_out, float)

        const_val = float(p_opt[-1]) if include_const_flag else 0.0
        model = np.asarray(res.mfit, float)

        rchi2 = _compute_rchi2(p, model, e, m_avg=m_fit, npar=p_opt.size)

        meta = {
            "stage": stage_name,
            "df": df,
            "m_fit": int(m_fit),
            "m_cand": int(m_cand),
            "cont_ic_criterion": str(cont_ic_criterion).lower(),
            "cont_ic_delta_min": float(cont_ic_delta_min),
            "qpo_ic_criterion": str(qpo_ic_criterion).lower(),
            "qpo_ic_delta_min": float(qpo_ic_delta_min),
            "qpo_detect_qmin": float(qpo_detect_qmin),
            "cand_require_ksigma": (float(cand_require_ksigma) if cand_require_ksigma is not None else None),
            "cand_sigma_mode": str(cand_sigma_mode),
            "stage1_n_seeds": int(stage1_n_seeds),
            "cont_x0_max_hz": float(cont_x0_max_hz),
            "postqpo_cont3_enable": bool(postqpo_cont3_enable),
            "postqpo_cont3_trigger_rchi": float(postqpo_cont3_trigger_rchi),
            "postqpo_cont3_rchi_improve_min": float(postqpo_cont3_rchi_improve_min),
            "postqpo_cont3_ic_delta_min": float(postqpo_cont3_ic_delta_min),
            "rchi_override_enable": bool(rchi_override_enable),
            "rchi_override_threshold": (float(rchi_override_threshold) if rchi_override_threshold is not None else None),
            "rchi_override_min_improve": float(rchi_override_min_improve),
        }
        if isinstance(meta_extra, dict):
            meta.update(meta_extra)

        return FitResult(
            ok=True,
            message=f"OK ({stage_name})",
            nlor=int(nlor),
            pars=pars_out,
            const=const_val,
            freq=f,
            model=model,
            aic=float(getattr(res, "aic", np.nan)),
            bic=float(getattr(res, "bic", np.nan)),
            deviance=float(getattr(res, "deviance", np.nan)),
            p_opt=p_opt,
            p_err=(np.asarray(res.err, float) if hasattr(res, "err") else None),
            rchi2=rchi2,
            meta=meta,
        )

    # ---- guardrail evaluation ----
    def _evaluate_guardrails(
        fit: FitResult,
        *,
        x0_lims: List[Tuple[float, float]],
        fwhm_lims: List[Tuple[float, float]],
        amp_max_list: List[float],
        include_const_flag: bool,
        const_max: float,
        tag: str,
    ) -> Tuple[bool, str]:
        okp, msgp = _param_sanity_check(
            fit.p_opt,
            nlor=fit.nlor,
            include_const=include_const_flag,
            x0_lims=x0_lims,
            fwhm_lims=fwhm_lims,
            amp_max_list=amp_max_list,
            const_max=const_max,
            df=df,
        )
        if not okp:
            return False, f"{tag}: {msgp}"

        if not guard_enable:
            return True, ""

        oko, msgo = _guardrail_overshoot(
            f, p, fit.model, e, m_fit,
            ksigma=guard_overshoot_ksigma,
            max_run_bins=guard_overshoot_max_run_bins,
            max_frac=guard_overshoot_max_frac,
        )
        if not oko:
            return False, f"{tag}: {msgo}"

        okc, msgc = _guardrail_component_local_amp(
            f, p, fit.pars,
            local_amp_factor=guard_comp_local_amp_factor,
            local_width_hz=max(0.5, 6.0 * df),
        )
        if not okc:
            return False, f"{tag}: {msgc}"

        return True, ""

    # -----------------------
    # IC helpers
    # -----------------------
    def _ic_value(fr: FitResult, criterion: str) -> float:
        c = str(criterion).strip().lower()
        if c == "bic":
            return float(fr.bic) if np.isfinite(fr.bic) else np.inf
        return float(fr.aic) if np.isfinite(fr.aic) else np.inf

    def _accept_upgrade_ic(simple: FitResult, complex_: FitResult, *, criterion: str, delta_min: float) -> bool:
        d = _ic_value(simple, criterion) - _ic_value(complex_, criterion)
        return np.isfinite(d) and (d >= float(delta_min))

    # -----------------------
    # stages
    # -----------------------
    def _stage_cont2(attempt_fitmethod: str, attempt_jitter: float, attempt_n_starts: int, tag: str):
        comps0 = [cont1, cont2]
        x0_lims0 = [cont_x0_lims, cont_x0_lims]
        fwhm_lims0 = [cont_fwhm_lim, cont_fwhm_lim]
        amp_caps0 = [cont_amp_cap, cont_amp_cap]

        res0, exc0 = _fit_stage(
            ps,
            comps=comps0,
            include_const=include_const,
            const_seed=c0,
            const_max=const_cap,
            x0_lims=x0_lims0,
            fwhm_lims=fwhm_lims0,
            amp_max_list=amp_caps0,
            fitmethod=attempt_fitmethod,
            n_starts=attempt_n_starts,
            jitter_frac=attempt_jitter,
            rng=rng,
            eps_amp=eps_amp,
        )
        if res0 is None:
            return None, f"{tag} failed: {exc0}"
        fit0 = _wrap(res0, len(comps0), include_const, tag)
        okg, msgg = _evaluate_guardrails(
            fit0,
            x0_lims=x0_lims0,
            fwhm_lims=fwhm_lims0,
            amp_max_list=amp_caps0,
            include_const_flag=include_const,
            const_max=const_cap,
            tag=tag,
        )
        if not okg:
            return None, f"GUARDRAIL_FAIL ({tag}): {msgg} rchi2={fit0.rchi2}"
        return fit0, ""

    def _seed_cont3_from_residual(fit_base: FitResult) -> Tuple[float, float, float]:
        
        try:
            f0 = np.asarray(fit_base.freq, float)
            mod0 = np.asarray(fit_base.model, float)
            p0 = np.asarray(p, float)

            mband = (f0 >= fit_fmin) & (f0 <= min(2.0, float(fit_fmax))) & np.isfinite(mod0) & np.isfinite(p0)
            if np.sum(mband) < 10:
                return cont3_default

            resid = (p0[mband] - mod0[mband])
            ff = f0[mband]
            resid_pos = np.where(np.isfinite(resid) & (resid > 0), resid, 0.0)
            if np.sum(resid_pos) <= 0:
                return cont3_default

            cen = float(np.sum(ff * resid_pos) / np.sum(resid_pos))
            cen = float(np.clip(cen, cont_x0_lims[0], cont_x0_lims[1]))
            amp = float(min(cont_amp_cap, max(lowf, eps_amp)))
            fwhm = float(np.clip(20.0, cont_fwhm_lim[0], cont_fwhm_lim[1]))
            return (amp, cen, fwhm)
        except Exception:
            return cont3_default

    def _stage_cont3(
        attempt_fitmethod: str,
        attempt_jitter: float,
        attempt_n_starts: int,
        tag: str,
        *,
        seed_from: Optional[FitResult] = None,
    ):
        cont3_seed = _seed_cont3_from_residual(seed_from) if seed_from is not None else cont3_default

        comps0 = [cont1, cont2, cont3_seed]
        x0_lims0 = [cont_x0_lims, cont_x0_lims, cont_x0_lims]
        fwhm_lims0 = [cont_fwhm_lim, cont_fwhm_lim, cont_fwhm_lim]
        amp_caps0 = [cont_amp_cap, cont_amp_cap, cont_amp_cap]

        res0, exc0 = _fit_stage(
            ps,
            comps=comps0,
            include_const=include_const,
            const_seed=c0,
            const_max=const_cap,
            x0_lims=x0_lims0,
            fwhm_lims=fwhm_lims0,
            amp_max_list=amp_caps0,
            fitmethod=attempt_fitmethod,
            n_starts=attempt_n_starts,
            jitter_frac=attempt_jitter,
            rng=rng,
            eps_amp=eps_amp,
        )
        if res0 is None:
            return None, f"{tag} failed: {exc0}"
        fit0 = _wrap(res0, len(comps0), include_const, tag, meta_extra={"cont3_seed_x0": float(cont3_seed[1])})
        okg, msgg = _evaluate_guardrails(
            fit0,
            x0_lims=x0_lims0,
            fwhm_lims=fwhm_lims0,
            amp_max_list=amp_caps0,
            include_const_flag=include_const,
            const_max=const_cap,
            tag=tag,
        )
        if not okg:
            return None, f"GUARDRAIL_FAIL ({tag}): {msgg} rchi2={fit0.rchi2}"
        return fit0, ""

    def _stage_cont2_plus_qpo(
        qseed: float,
        attempt_fitmethod: str,
        attempt_jitter: float,
        attempt_n_starts: int,
        tag: str,
    ):
        qseed = float(qseed)
        qpo_amp = float(min(qpo_amp_cap, _seed_amp_at(f, p, qseed)))
        qpo_fwhm = float(max(qpo_fwhm_min, qpo_fwhm_frac * qseed))
        qpo_fwhm = float(np.clip(qpo_fwhm, qpo_fwhm_lim[0], qpo_fwhm_lim[1]))
        qpo_comp = (qpo_amp, qseed, qpo_fwhm)

        comps1 = [cont1, cont2, qpo_comp]
        x0_lims1 = [cont_x0_lims, cont_x0_lims, (cand_fmin, cand_fmax)]
        fwhm_lims1 = [cont_fwhm_lim, cont_fwhm_lim, qpo_fwhm_lim]
        amp_caps1 = [cont_amp_cap, cont_amp_cap, qpo_amp_cap]

        res1, exc1 = _fit_stage(
            ps,
            comps=comps1,
            include_const=include_const,
            const_seed=c0,
            const_max=const_cap,
            x0_lims=x0_lims1,
            fwhm_lims=fwhm_lims1,
            amp_max_list=amp_caps1,
            fitmethod=attempt_fitmethod,
            n_starts=attempt_n_starts,
            jitter_frac=attempt_jitter,
            rng=rng,
            eps_amp=eps_amp,
        )
        if res1 is None:
            return None, f"{tag} failed: {exc1}"
        fit1 = _wrap(res1, len(comps1), include_const, tag, meta_extra={"seed_hz": qseed})
        okg, msgg = _evaluate_guardrails(
            fit1,
            x0_lims=x0_lims1,
            fwhm_lims=fwhm_lims1,
            amp_max_list=amp_caps1,
            include_const_flag=include_const,
            const_max=const_cap,
            tag=tag,
        )
        if not okg:
            return None, f"GUARDRAIL_FAIL ({tag}): {msgg} rchi2={fit1.rchi2}"
        return fit1, ""

    def _stage_cont3_plus_qpo_from_fit(
        *,
        fit_seed: FitResult,
        qpo_nu0: float,
        qpo_fwhm: float,
        qpo_amp: float,
        attempt_fitmethod: str,
        attempt_jitter: float,
        attempt_n_starts: int,
        tag: str,
        meta_extra: Optional[Dict[str, Any]] = None,
    ):
        cont3_seed = _seed_cont3_from_residual(fit_seed)

        qpo_nu0 = float(qpo_nu0)
        qpo_fwhm = float(np.clip(float(qpo_fwhm), qpo_fwhm_lim[0], qpo_fwhm_lim[1]))
        qpo_amp = float(np.clip(float(qpo_amp), float(eps_amp), float(qpo_amp_cap)))
        qpo_comp = (qpo_amp, qpo_nu0, qpo_fwhm)

        comps = [cont1, cont2, cont3_seed, qpo_comp]
        x0_lims = [cont_x0_lims, cont_x0_lims, cont_x0_lims, (cand_fmin, cand_fmax)]
        fwhm_lims = [cont_fwhm_lim, cont_fwhm_lim, cont_fwhm_lim, qpo_fwhm_lim]
        amp_caps = [cont_amp_cap, cont_amp_cap, cont_amp_cap, qpo_amp_cap]

        res, exc = _fit_stage(
            ps,
            comps=comps,
            include_const=include_const,
            const_seed=c0,
            const_max=const_cap,
            x0_lims=x0_lims,
            fwhm_lims=fwhm_lims,
            amp_max_list=amp_caps,
            fitmethod=attempt_fitmethod,
            n_starts=attempt_n_starts,
            jitter_frac=attempt_jitter,
            rng=rng,
            eps_amp=eps_amp,
        )
        if res is None:
            return None, f"{tag} failed: {exc}"

        mex = {
            "qpo_seed_from_fit": True,
            "qpo_seed_hz": float(qpo_nu0),
            "cont3_seed_x0": float(cont3_seed[1]),
        }
        if isinstance(meta_extra, dict):
            mex.update(meta_extra)

        fitx = _wrap(res, len(comps), include_const, tag, meta_extra=mex)
        okg, msgg = _evaluate_guardrails(
            fitx,
            x0_lims=x0_lims,
            fwhm_lims=fwhm_lims,
            amp_max_list=amp_caps,
            include_const_flag=include_const,
            const_max=const_cap,
            tag=tag,
        )
        if not okg:
            return None, f"GUARDRAIL_FAIL ({tag}): {msgg} rchi2={fitx.rchi2}"
        return fitx, ""

    # -----------------------
    # QPO acceptance rule
    # -----------------------
    def _detect_real_qpo(fit1: FitResult) -> Optional[Dict[str, Any]]:
        return extract_qpo_params(
            fit1,
            qpo_fmin=float(cand_fmin),
            qpo_fmax=float(cand_fmax),
            qmin=float(qpo_detect_qmin),
        )

    def _accept_qpo_model(
        *,
        fit_cont_best: FitResult,
        fit_qpo: FitResult,
        cands: List[Dict[str, float]],
        qpo_real: Dict[str, Any],
        require_ksigma: Optional[float],
        cand_tol_hz: float,
        override_relax_ic: bool,
    ) -> Tuple[bool, str]:
        if qpo_real is None:
            return False, "reject: no QPO-like component (Q gate failed)"

        # IC gate (relaxed in override mode)
        if not override_relax_ic:
            if not _accept_upgrade_ic(
                fit_cont_best,
                fit_qpo,
                criterion=str(qpo_ic_criterion),
                delta_min=float(qpo_ic_delta_min),
            ):
                return False, f"reject: ?{qpo_ic_criterion.upper()} vs continuum below threshold"

        # sigma gate (optional)
        if require_ksigma is not None and np.isfinite(require_ksigma) and float(require_ksigma) > 0:
            prom, exsig = _nearest_candidate_metrics(cands, float(qpo_real["qpo_nu0_hz"]), tol_hz=cand_tol_hz)
            if not np.isfinite(exsig) or exsig < float(require_ksigma):
                return False, "reject: sigma gate failed at fitted QPO freq"

        return True, ""

    # -----------------------
    # candidate + seed building (used in initial and reseed passes)
    # -----------------------
    def _build_candidates_and_seeds(
        *,
        prom_use: float,
        require_ksigma_use: Optional[float],
        exclude_center_hz: Optional[float],
        exclude_halfwidth_hz: float,
        max_cands_use: int,
        seed_peak: Optional[float],
        stage1_k: int,
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        cands0 = find_qpo_candidates(
            cf, cp,
            cand_fmin=cand_fmin,
            cand_fmax=cand_fmax,
            smooth_hz=smooth_hz,
            prominence=float(prom_use),
            min_sep_hz=min_sep_hz,
            max_candidates=int(max_cands_use),
            m_eff=m_cand,
            require_ksigma=require_ksigma_use,
            cand_sigma_mode=cand_sigma_mode,
        )

        # apply exclusion window (for reseed)
        cands = []
        if exclude_center_hz is not None and np.isfinite(exclude_center_hz):
            lo = float(exclude_center_hz) - float(exclude_halfwidth_hz)
            hi = float(exclude_center_hz) + float(exclude_halfwidth_hz)
            for c in cands0:
                nu = float(c.get("nu_hz", np.nan))
                if np.isfinite(nu) and (lo <= nu <= hi):
                    continue
                cands.append(c)
        else:
            cands = cands0

        seeds: List[float] = []
        if seed_peak is not None and np.isfinite(seed_peak):
            sp = float(seed_peak)
            if cand_fmin <= sp <= cand_fmax:
                seeds.append(sp)

        for c in cands[: max(1, int(stage1_k))]:
            nu = float(c.get("nu_hz", np.nan))
            if not np.isfinite(nu):
                continue
            if any(abs(nu - s) <= max(0.5 * df, 1e-6) for s in seeds):
                continue
            seeds.append(nu)

        return cands, seeds

    def _is_edge(freq_hz: float) -> bool:
        if not np.isfinite(freq_hz):
            return False
        w = float(reseed_edge_frac) * float(cand_fmax - cand_fmin)
        w = max(w, 2.0 * df)
        return (freq_hz <= float(cand_fmin) + w) or (freq_hz >= float(cand_fmax) - w)

    # -----------------------
    # attempt runner
    # -----------------------
    def _run_one_attempt(
        *,
        attempt_fitmethod: str,
        attempt_jitter: float,
        attempt_n_starts: int,
        label: str,
    ) -> Tuple[Optional[FitResult], str]:

        # thresholds
        thr = float(rchi_override_threshold) if (rchi_override_threshold is not None and np.isfinite(rchi_override_threshold)) else float(postqpo_cont3_trigger_rchi)

        # Stage0 cont2
        fit0, fail0 = _stage_cont2(attempt_fitmethod, attempt_jitter, attempt_n_starts, "stage0 cont2")
        if fit0 is None:
            return None, fail0

        # Stage0b cont3 (baseline continuum order selection by IC, but we'll still keep cont3 around for later)
        fit0b, _ = _stage_cont3(attempt_fitmethod, attempt_jitter, attempt_n_starts, "stage0b cont3", seed_from=fit0)

        fit_cont_best = fit0
        if fit0b is not None:
            if _accept_upgrade_ic(fit0, fit0b, criterion=str(cont_ic_criterion), delta_min=float(cont_ic_delta_min)):
                fit_cont_best = fit0b
                fit_cont_best.message = f"OK (stage0b cont3 accepted by ?{str(cont_ic_criterion).upper()})"

        fit_cont_best.meta = fit_cont_best.meta or {}
        fit_cont_best.meta.update({
            "retry_label": label,
            "optimizer": attempt_fitmethod,
            "jitter": attempt_jitter,
            "n_starts": attempt_n_starts,
            "include_harmonic": bool(include_harmonic),
        })

        # Decide if we are already in override territory based on best continuum-only
        cont_bad = np.isfinite(fit_cont_best.rchi2) and (float(fit_cont_best.rchi2) > thr)
        override_active = bool(rchi_override_enable) and cont_bad
        fit_cont_best.meta["override_active_at_continuum"] = bool(override_active)
        fit_cont_best.meta["override_thr"] = float(thr)

        # Pass A: candidates + seeds (normal)
        cands, seeds = _build_candidates_and_seeds(
            prom_use=float(prominence),
            require_ksigma_use=cand_require_ksigma,
            exclude_center_hz=None,
            exclude_halfwidth_hz=0.0,
            max_cands_use=int(max_candidates),
            seed_peak=seed_peak_hz,
            stage1_k=int(stage1_n_seeds),
        )

        if len(seeds) == 0:
            # No candidates. If continuum is bad, try again with relaxed peak requirements (meaningful ladder)
            if override_active or (reseed_enable and np.isfinite(fit_cont_best.rchi2) and float(fit_cont_best.rchi2) > float(reseed_rchi_bad)):
                cands2, seeds2 = _build_candidates_and_seeds(
                    prom_use=float(prominence) / max(1e-6, float(reseed_prom_factor)),
                    require_ksigma_use=(None if cand_require_ksigma is None else max(0.0, float(cand_require_ksigma) / max(1e-6, float(reseed_sigma_factor)))),
                    exclude_center_hz=None,
                    exclude_halfwidth_hz=0.0,
                    max_cands_use=int(max(max_candidates, 2 * stage1_n_seeds)),
                    seed_peak=seed_peak_hz,
                    stage1_k=int(max(stage1_n_seeds, 2)),
                )
                if len(seeds2) > 0:
                    cands, seeds = cands2, seeds2

            if len(seeds) == 0:
                fit_cont_best.message = "OK (continuum-only; no candidate seeds)"
                fit_cont_best.meta["cands"] = cands
                return fit_cont_best, ""

        cand_tol_hz = max(0.5 * float(df), 0.05)

        # Stage1 multi-seed cont2+qpo
        best_qpo_fit: Optional[FitResult] = None
        best_qpo_real: Optional[Dict[str, Any]] = None
        best_seed = None
        best_reject = ""

        # accept policy: normal vs override
        # - in override mode, we relax the IC requirement here (still Q-gated)
        override_relax_ic = bool(override_active)

        for s in seeds:
            fit1, fail1 = _stage_cont2_plus_qpo(float(s), attempt_fitmethod, attempt_jitter, attempt_n_starts, "stage1 cont+qpo")
            if fit1 is None:
                best_reject = best_reject or f"seed {s:.4g}: {fail1}"
                continue

            qpo_real = _detect_real_qpo(fit1)
            ok_qpo, why = _accept_qpo_model(
                fit_cont_best=fit_cont_best,
                fit_qpo=fit1,
                cands=cands,
                qpo_real=qpo_real,
                require_ksigma=cand_require_ksigma,
                cand_tol_hz=cand_tol_hz,
                override_relax_ic=override_relax_ic,
            )

            fit1.meta = fit1.meta or {}
            fit1.meta.update({"cands": cands, "seed_hz": float(s), "has_real_qpo": bool(qpo_real is not None), "override_relax_ic": bool(override_relax_ic)})
            if qpo_real is not None:
                fit1.meta["qpo_real"] = dict(qpo_real)

            if not ok_qpo:
                best_reject = best_reject or f"seed {s:.4g}: {why}"
                continue

            if best_qpo_fit is None:
                best_qpo_fit = fit1
                best_qpo_real = qpo_real
                best_seed = float(s)
            else:
                # choose best by IC in normal mode; by rchi in override mode
                if not override_relax_ic:
                    if _ic_value(fit1, str(qpo_ic_criterion)) < _ic_value(best_qpo_fit, str(qpo_ic_criterion)):
                        best_qpo_fit = fit1
                        best_qpo_real = qpo_real
                        best_seed = float(s)
                else:
                    r1 = best_qpo_fit.rchi2 if np.isfinite(best_qpo_fit.rchi2) else np.inf
                    r2 = fit1.rchi2 if np.isfinite(fit1.rchi2) else np.inf
                    if r2 < r1 - 0.01:
                        best_qpo_fit = fit1
                        best_qpo_real = qpo_real
                        best_seed = float(s)

        # If Stage1 rejected: in normal mode we return continuum-only,
        # BUT if override_active we keep trying (meaningful ladder).
        if best_qpo_fit is None or best_qpo_real is None:
            if not override_active and not (reseed_enable and np.isfinite(fit_cont_best.rchi2) and float(fit_cont_best.rchi2) > float(reseed_rchi_bad)):
                fit_cont_best.message = "OK (continuum-only; Stage1 rejected QPO)"
                fit_cont_best.meta["cands"] = cands
                fit_cont_best.meta["stage1_reject"] = best_reject
                return fit_cont_best, ""

            # reseed pass (even if stage1 rejected): rebuild candidates with stronger ranking
            # and try again, relaxing IC only if override active.
            cands_r, seeds_r = _build_candidates_and_seeds(
                prom_use=float(prominence) / max(1e-6, float(reseed_prom_factor)),
                require_ksigma_use=(None if cand_require_ksigma is None else max(0.0, float(cand_require_ksigma) / max(1e-6, float(reseed_sigma_factor)))),
                exclude_center_hz=None,
                exclude_halfwidth_hz=0.0,
                max_cands_use=int(max(max_candidates, 2 * stage1_n_seeds)),
                seed_peak=seed_peak_hz,
                stage1_k=int(max(stage1_n_seeds, 2)),
            )

            best_qpo_fit = None
            best_qpo_real = None
            best_seed = None
            best_reject2 = ""

            for s in seeds_r:
                fit1, fail1 = _stage_cont2_plus_qpo(float(s), attempt_fitmethod, attempt_jitter, attempt_n_starts, "stage1r cont+qpo (reseed)")
                if fit1 is None:
                    best_reject2 = best_reject2 or f"seed {s:.4g}: {fail1}"
                    continue
                qpo_real = _detect_real_qpo(fit1)
                ok_qpo, why = _accept_qpo_model(
                    fit_cont_best=fit_cont_best,
                    fit_qpo=fit1,
                    cands=cands_r,
                    qpo_real=qpo_real,
                    require_ksigma=cand_require_ksigma,
                    cand_tol_hz=cand_tol_hz,
                    override_relax_ic=bool(override_active),
                )
                fit1.meta = fit1.meta or {}
                fit1.meta.update({"cands": cands_r, "seed_hz": float(s), "has_real_qpo": bool(qpo_real is not None), "reseed_pass": True})
                if qpo_real is not None:
                    fit1.meta["qpo_real"] = dict(qpo_real)

                if not ok_qpo:
                    best_reject2 = best_reject2 or f"seed {s:.4g}: {why}"
                    continue

                if best_qpo_fit is None:
                    best_qpo_fit = fit1
                    best_qpo_real = qpo_real
                    best_seed = float(s)
                else:
                    r1 = best_qpo_fit.rchi2 if np.isfinite(best_qpo_fit.rchi2) else np.inf
                    r2 = fit1.rchi2 if np.isfinite(fit1.rchi2) else np.inf
                    if r2 < r1 - 0.01:
                        best_qpo_fit = fit1
                        best_qpo_real = qpo_real
                        best_seed = float(s)

            if best_qpo_fit is None or best_qpo_real is None:
                # still nothing: return best continuum (but include diagnostics)
                fit_cont_best.message = "OK (continuum-only; Stage1+reseed rejected QPO)"
                fit_cont_best.meta["cands"] = cands
                fit_cont_best.meta["cands_reseed"] = cands_r
                fit_cont_best.meta["stage1_reject"] = best_reject
                fit_cont_best.meta["stage1_reject_reseed"] = best_reject2
                return fit_cont_best, ""

            # otherwise accept reseed best
            cands, seeds = cands_r, seeds_r

        fit_best = best_qpo_fit
        fit_best.meta = fit_best.meta or {}
        fit_best.meta["cands"] = cands
        fit_best.meta["seed_hz"] = float(best_seed)
        fit_best.meta["override_active"] = bool(override_active)

        # -----------------------
        # Reseed ladder when fit is bad/edgey/tiny-area
        # -----------------------
        if reseed_enable and np.isfinite(fit_best.rchi2):
            rbad = float(fit_best.rchi2) > float(reseed_rchi_bad)
            qnu = float(best_qpo_real.get("qpo_nu0_hz", np.nan)) if isinstance(best_qpo_real, dict) else np.nan
            edgey = _is_edge(qnu)

            tiny_area = False
            if reseed_area_min is not None and np.isfinite(reseed_area_min) and float(reseed_area_min) > 0:
                try:
                    qidx = int(best_qpo_real.get("qpo_index", 2))
                    qidx = int(np.clip(qidx, 0, max(0, fit_best.pars.shape[0] - 1)))
                    comp = lorentz(fit_best.freq, fit_best.pars[qidx, 0], fit_best.pars[qidx, 1], fit_best.pars[qidx, 2])
                    area = component_power_integral(fit_best.freq, comp, float(cand_fmin), float(cand_fmax))
                    tiny_area = np.isfinite(area) and (float(area) < float(reseed_area_min))
                except Exception:
                    tiny_area = False

            if rbad or edgey or tiny_area:
                exclude_hw = max(float(reseed_exclude_hz_min), float(reseed_exclude_df_mult) * float(df))
                cands_r, seeds_r = _build_candidates_and_seeds(
                    prom_use=float(prominence) * float(reseed_prom_factor),
                    require_ksigma_use=(None if cand_require_ksigma is None else float(cand_require_ksigma) * float(reseed_sigma_factor)),
                    exclude_center_hz=(qnu if np.isfinite(qnu) else None),
                    exclude_halfwidth_hz=exclude_hw,
                    max_cands_use=int(max(max_candidates, 2 * stage1_n_seeds)),
                    seed_peak=seed_peak_hz,
                    stage1_k=int(max(stage1_n_seeds, 2)),
                )

                best2 = fit_best
                best2_real = best_qpo_real
                best2_seed = best_seed

                for s in seeds_r:
                    fit2, fail2 = _stage_cont2_plus_qpo(float(s), attempt_fitmethod, attempt_jitter, attempt_n_starts, "stage2 cont+qpo (reseed-exclude)")
                    if fit2 is None:
                        continue
                    qpo_real2 = _detect_real_qpo(fit2)
                    ok2, _why2 = _accept_qpo_model(
                        fit_cont_best=fit_cont_best,
                        fit_qpo=fit2,
                        cands=cands_r,
                        qpo_real=qpo_real2,
                        require_ksigma=cand_require_ksigma,
                        cand_tol_hz=cand_tol_hz,
                        override_relax_ic=bool(override_active),
                    )
                    if not ok2:
                        continue
                    r1 = best2.rchi2 if np.isfinite(best2.rchi2) else np.inf
                    r2 = fit2.rchi2 if np.isfinite(fit2.rchi2) else np.inf
                    if r2 < r1 - 0.01:
                        best2 = fit2
                        best2_real = qpo_real2
                        best2_seed = float(s)

                fit_best = best2
                best_qpo_real = best2_real
                best_seed = best2_seed
                fit_best.meta = fit_best.meta or {}
                fit_best.meta.update({
                    "reseed_triggered": True,
                    "reseed_reason_rchi_bad": bool(rbad),
                    "reseed_reason_edge": bool(edgey),
                    "reseed_reason_tiny_area": bool(tiny_area),
                    "reseed_exclude_hw_hz": float(exclude_hw),
                    "cands_reseed_exclude": cands_r,
                    "seed_hz": float(best_seed) if best_seed is not None else None,
                })

        # -----------------------
        # FORCE cont3+qpo when rchi above criteria (override),
        # or when postqpo rescue condition holds.
        # -----------------------
        thr = float(rchi_override_threshold) if (rchi_override_threshold is not None and np.isfinite(rchi_override_threshold)) else float(postqpo_cont3_trigger_rchi)
        do_override = bool(rchi_override_enable) and np.isfinite(fit_best.rchi2) and float(fit_best.rchi2) > thr

        do_postqpo = (
            bool(postqpo_cont3_enable)
            and np.isfinite(fit_best.rchi2)
            and float(fit_best.rchi2) > float(postqpo_cont3_trigger_rchi)
            and int(fit_best.nlor) == 3
        )

        try_rescue = (int(fit_best.nlor) == 3) and (do_override or do_postqpo)

        if try_rescue and isinstance(best_qpo_real, dict):
            qidx = int(best_qpo_real.get("qpo_index", 2))
            qidx = int(np.clip(qidx, 0, max(0, fit_best.pars.shape[0] - 1)))
            q_nu0 = float(fit_best.pars[qidx, 0])
            q_fwhm = float(fit_best.pars[qidx, 1])
            q_amp = float(fit_best.pars[qidx, 2])

            fit_up, _ = _stage_cont3_plus_qpo_from_fit(
                fit_seed=fit_best,
                qpo_nu0=q_nu0,
                qpo_fwhm=q_fwhm,
                qpo_amp=q_amp,
                attempt_fitmethod=attempt_fitmethod,
                attempt_jitter=attempt_jitter,
                attempt_n_starts=attempt_n_starts,
                tag="stage3 cont3+qpo (rescue/override)",
                meta_extra={"from_stage": str(fit_best.meta.get("stage", "stage1")), "override": bool(do_override)},
            )

            if fit_up is not None and np.isfinite(fit_up.rchi2) and np.isfinite(fit_best.rchi2):
                r0 = float(fit_best.rchi2)
                r1 = float(fit_up.rchi2)
                dr = r0 - r1  # positive = improvement
                dic = _ic_value(fit_best, str(qpo_ic_criterion)) - _ic_value(fit_up, str(qpo_ic_criterion))
                rchi_not_worse = (r1 <= r0 + float(postqpo_cont3_rchi_not_worse_tol))

                # Override: accept if rchi improves by >= min_improve (IC not required)
                accept_override = bool(do_override) and np.isfinite(dr) and (dr >= float(rchi_override_min_improve)) and rchi_not_worse

                # PostQPO: accept by previous rule OR IC+not-worse
                accept_postqpo = (np.isfinite(dr) and (dr >= float(postqpo_cont3_rchi_improve_min))) or (
                    (np.isfinite(dic) and (dic >= float(postqpo_cont3_ic_delta_min)) and rchi_not_worse)
                )

                fit_best.meta = fit_best.meta or {}
                fit_best.meta.update({
                    "cont3qpo_attempted": True,
                    "cont3qpo_override": bool(do_override),
                    "cont3qpo_rchi_before": r0,
                    "cont3qpo_rchi_after": r1,
                    "cont3qpo_rchi_delta": dr,
                    "cont3qpo_ic_delta": float(dic),
                    "cont3qpo_ic_criterion": str(qpo_ic_criterion).lower(),
                })

                if accept_override or accept_postqpo:
                    fit_best = fit_up
                    fit_best.message = "OK (cont3+qpo accepted by rescue/override)"
                    fit_best.meta = fit_best.meta or {}
                    fit_best.meta["cont3qpo_accepted"] = True
                else:
                    fit_best.meta["cont3qpo_accepted"] = False

        # If rchi already good, stop
        if np.isfinite(fit_best.rchi2) and fit_best.rchi2 <= float(rchi_target):
            fit_best.message = "OK (rchi target reached)"
            return fit_best, ""

        fit_best.message = "OK (best model selected)"
        return fit_best, ""

    # -----------------------
    # retry ladder (meaningful: optimizer + jitter + n_starts changes)
    # -----------------------
    ladder = [
        dict(fitmethod=str(fitmethod), jitter=float(jitter_frac), n_starts=int(n_starts), label="attempt0"),
        dict(fitmethod=str(fitmethod), jitter=float(max(0.10, 0.60 * float(jitter_frac))), n_starts=int(max(6, n_starts)), label="attempt1_lessjitter_morestarts"),
        dict(fitmethod="Nelder-Mead", jitter=float(max(0.10, 0.70 * float(jitter_frac))), n_starts=int(max(6, n_starts)), label="attempt2_nm"),
        dict(fitmethod="Powell", jitter=float(max(0.16, 1.20 * float(jitter_frac))), n_starts=int(max(10, n_starts)), label="attempt3_powell_morejitter_morestarts"),
        dict(fitmethod="Nelder-Mead", jitter=float(max(0.20, 1.40 * float(jitter_frac))), n_starts=int(max(14, n_starts)), label="attempt4_nm_heavy"),
    ]
    ladder = ladder[: int(max(1, max_retries))]

    last_fail = ""
    for att in ladder:
        fit_out, fail = _run_one_attempt(
            attempt_fitmethod=att["fitmethod"],
            attempt_jitter=att["jitter"],
            attempt_n_starts=att["n_starts"],
            label=att["label"],
        )
        if fit_out is not None:
            fit_out.meta = fit_out.meta or {}
            fit_out.meta.update({"cand_grid_used": (cand_freq is not None and cand_power is not None)})
            return fit_out
        last_fail = fail

    return FitResult(
        ok=False,
        message=f"No successful fits after retries. Last fail: {last_fail}",
        nlor=0,
        pars=np.empty((0, 3)),
        const=0.0,
        freq=f,
        model=np.full_like(f, np.nan),
        aic=np.nan,
        bic=np.nan,
        deviance=np.nan,
        p_opt=np.array([]),
        p_err=None,
        rchi2=np.nan,
        meta={
            "stage": "retry_ladder",
            "last_fail": last_fail,
            "df": df,
            "m_fit": int(m_fit),
            "m_cand": int(m_cand),
            "cand_grid_used": (cand_freq is not None and cand_power is not None),
            "cont_ic_criterion": str(cont_ic_criterion).lower(),
            "cont_ic_delta_min": float(cont_ic_delta_min),
            "qpo_ic_criterion": str(qpo_ic_criterion).lower(),
            "qpo_ic_delta_min": float(qpo_ic_delta_min),
            "qpo_detect_qmin": float(qpo_detect_qmin),
            "cand_require_ksigma": (float(cand_require_ksigma) if cand_require_ksigma is not None else None),
            "cand_sigma_mode": str(cand_sigma_mode),
            "stage1_n_seeds": int(stage1_n_seeds),
            "postqpo_cont3_enable": bool(postqpo_cont3_enable),
            "postqpo_cont3_trigger_rchi": float(postqpo_cont3_trigger_rchi),
            "postqpo_cont3_rchi_improve_min": float(postqpo_cont3_rchi_improve_min),
            "postqpo_cont3_ic_delta_min": float(postqpo_cont3_ic_delta_min),
            "rchi_override_enable": bool(rchi_override_enable),
            "rchi_override_threshold": (float(rchi_override_threshold) if rchi_override_threshold is not None else None),
            "rchi_override_min_improve": float(rchi_override_min_improve),
        },
    )
