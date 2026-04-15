#!/usr/bin/env python3
"""
QPO_main.py  (v3 — TripleA-centred)
=====================================
Per-obsid analysis driver for the QPO fitting pipeline.

Changes from v2
---------------
- _fit_one_band rebuilt around TripleA:
    * All Powell-era retry/reseed/override kwargs removed.
    * fit_lorentzian_family routes to fit_lorentzian_family_triplea when
      FIT_METHOD == "TripleA" (dispatched inside QPO_fit, not here).
    * kwargs dict is ~half the size and clearly commented.
- Progress is always printed before each obsid (outside suppress context)
  so a hang is immediately visible even with QUIET=True.
- _suppress_context still suppresses noisy Stingray / optimizer output,
  but errors raised inside are re-raised and logged as FAIL.
- Cross-band reseeding kept intact — it provides the forced_qpo_seeds
  that TripleA uses as high-priority starting frequencies.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import warnings

import numpy as np
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))  # NumPy >=2.0 compat


# Pure-numpy helper — defined here so it is available at module level
# without importing QPO_utils (which imports stingray at its module level).
def _safe_m_avg_from_pds(pds) -> int:
    """Robust scalar effective-m from a Stingray PDS object."""
    try:
        m   = getattr(pds, "m", 1)
        arr = np.asarray(m, float)
        v   = float(arr) if arr.ndim == 0 else float(np.nanmedian(arr))
        return int(v) if (np.isfinite(v) and v >= 1) else 1
    except Exception:
        return 1
import scipy.signal
from scipy.ndimage import median_filter

from astropy.io import fits
from astropy.time import Time


from QPO_fit import (
    FitResult,
    _extract_component_cov,
    _nu_max,
    _nu_max_err_from_cov,
    _q_err_from_cov,
    _rms2_err_from_cov,
    component_power_integral,
    extract_qpo_params,
    extract_qpo_params_list,
    fit_lorentzian_family,
    lorentz,
)

import QPO_Parameter as P

# Force non-interactive Agg backend before any matplotlib import.
import matplotlib
matplotlib.use("Agg")


# Heavy imports (stingray, QPO_utils, QPO_plot) are deferred to the
# functions that need them.  This keeps worker-process startup clean:
# with spawn-context parallelism, 32 workers importing stingray
# simultaneously causes filesystem-lock contention and hangs.
print("[INFO] Using QPO_Parameter from:", getattr(P, "__file__", "UNKNOWN"))
print("[INFO] FIT_METHOD:", getattr(P, "FIT_METHOD", "Powell"))


# ============================================================================
# Logging suppression
# ============================================================================

def _suppress_context(enabled: bool):
    """
    Suppress stdout/stderr when QUIET=True.

    Errors raised inside propagate normally — they are never swallowed.
    Only verbose informational output (Stingray fit logs, optimizer messages)
    is redirected to /dev/null.
    """
    if not enabled:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                yield

    return _ctx()


# ============================================================================
# One-line summary helpers
# ============================================================================

def _src_label() -> str:
    return str(getattr(P, "SOURCE", "SOURCE")).strip() or "SOURCE"


def _count_continuum_components(fitres) -> int:
    if fitres is None or not getattr(fitres, "ok", False):
        return 0
    ctypes = getattr(fitres, "comp_types", None)
    if ctypes:
        return sum(1 for t in ctypes if str(t) == "cont")
    stage = str(getattr(fitres, "meta", {}).get("stage", "")).lower() if fitres.meta else ""
    if "cont4" in stage:
        return 4
    return 3 if "cont3" in stage else 2


def _qpos_from_fitres_by_type(fitres) -> list[dict]:
    """
    Extract QPO Lorentzians using comp_types — the optimiser's own classification.

    Authoritative for plot annotations, CSV output, and struct consistency.
    A component must satisfy ALL of:
      - comp_types entry is "qpo"
      - nu0 within [QPO_FMIN, QPO_FMAX]
      - Q = nu0/fwhm >= QPO_MIN_Q
    """
    if fitres is None or not getattr(fitres, "ok", False):
        return []
    pars       = np.asarray(getattr(fitres, "pars", []), float)
    comp_types = list(getattr(fitres, "comp_types", []))
    freq       = np.asarray(getattr(fitres, "freq", []), float)
    if pars.ndim != 2 or pars.shape[1] != 3 or pars.size == 0 or freq.size < 2:
        return []

    qpo_fmin = float(getattr(P, "QPO_FMIN",   0.1))
    qpo_fmax = float(getattr(P, "QPO_FMAX",  10.0))
    qmin     = float(getattr(P, "QPO_MIN_Q",   3.0))

    out = []
    for i, (nu0, fwhm, amp) in enumerate(pars):
        if i >= len(comp_types) or str(comp_types[i]) != "qpo":
            continue
        if not (np.isfinite(nu0) and np.isfinite(fwhm) and np.isfinite(amp)):
            continue
        if fwhm <= 0:
            continue
        Q = nu0 / fwhm
        if Q < qmin or not (qpo_fmin <= nu0 <= qpo_fmax):
            continue
        area = float(_trapz(lorentz(freq, nu0, fwhm, amp), freq))
        out.append(dict(
            qpo_index=int(i), qpo_nu0_hz=float(nu0),
            qpo_fwhm_hz=float(fwhm), qpo_Q=float(Q), qpo_area=float(area),
        ))

    out.sort(key=lambda d: (-d["qpo_area"], d["qpo_nu0_hz"]))
    return out


def _qpo_list_from_fit(fitres) -> list[dict]:
    if fitres is None or not getattr(fitres, "ok", False):
        return []
    try:
        return extract_qpo_params_list(
            fitres,
            qpo_fmin=getattr(P, "QPO_FMIN", 0.1),
            qpo_fmax=getattr(P, "QPO_FMAX", 10.0),
            qmin=getattr(P, "QPO_MIN_Q", 3.0),
            sort_by=getattr(P, "QPO_SORT_BY", "area"),
        )
    except Exception:
        return []


def _fmt_band(label: str, fitres) -> str:
    cont_n = _count_continuum_components(fitres)
    qpos   = _qpos_from_fitres_by_type(fitres)
    found  = len(qpos) > 0
    nu     = float(qpos[0]["qpo_nu0_hz"]) if found else np.nan
    Q      = float(qpos[0]["qpo_Q"])      if found else np.nan

    rchi   = getattr(fitres, "rchi2", np.nan) if fitres is not None else np.nan
    rchi_s = f"{rchi:.2f}" if np.isfinite(rchi) else "nan"
    flag   = ""
    rchi_max = getattr(P, "FIT_RCHI_MAX", None)
    if rchi_max is not None and np.isfinite(rchi) and rchi > float(rchi_max):
        flag = " (rchi above criteria)"

    if found and getattr(P, "SHOW_QPO_DETAILS", True):
        return f"{label}:cont={cont_n} Nqpo={len(qpos)} rchi={rchi_s} nu={nu:.3f} Q={Q:.2f}{flag}"
    return f"{label}:cont={cont_n} QPO={'Y' if found else 'N'} rchi={rchi_s}{flag}"


def _print_one_line_summary(obsid: str, row: dict) -> None:
    status = row.get("status", "")
    err    = row.get("error", "")
    mjd    = row.get("mjd_mid", "")
    try:
        mjd_s = f"{float(mjd):.5f}" if isinstance(mjd, (int, float)) and np.isfinite(mjd) else ""
    except Exception:
        mjd_s = ""

    head = obsid + (f"  mjd={mjd_s}" if mjd_s else "")

    if status and status != "OK":
        print(f"{head}  {status}  {err}" if err else f"{head}  {status}")
        return

    parts = [head, _fmt_band("F", row.get("_fit_full_res_obj"))]
    if getattr(P, "DO_ENERGY_BANDS", False):
        parts.append(_fmt_band("S", row.get("_fit_soft_res_obj")))
        parts.append(_fmt_band("H", row.get("_fit_hard_res_obj")))
    print("  |  ".join(parts))


# ============================================================================
# Filesystem helpers
# ============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def obsid_outdir(obsid: str) -> str:
    return ensure_dir(os.path.join(P.OUTDIR_BASE, obsid))


def common_outdir() -> str:
    return ensure_dir(os.path.join(P.OUTDIR_BASE, P.COMMON_DIRNAME))


def read_obsids(txt_path: str) -> list[str]:
    obsids = []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                obsids.append(s)
    return obsids


# ============================================================================
# Rebinning helpers
# ============================================================================


# ============================================================================
# MJD helper
# ============================================================================

def _evt_time_mjd_mid(evt_path: str) -> tuple[float | None, str | None]:
    try:
        with fits.open(evt_path, memmap=True) as hdul:
            h        = hdul[1].header
            tstart   = float(h.get("TSTART"))
            tstop    = float(h.get("TSTOP"))
            tmid     = 0.5 * (tstart + tstop)
            mjdrefi  = float(h.get("MJDREFI", 0.0))
            mjdreff  = float(h.get("MJDREFF", 0.0))
            mjdref   = mjdrefi + mjdreff
            timezero = float(h.get("TIMEZERO", 0.0))
            tmid    += timezero
            mjd_mid  = float(mjdref + tmid / 86400.0)
            iso_mid  = Time(mjd_mid, format="mjd", scale="tt").isot
            return mjd_mid, iso_mid
    except Exception:
        return None, None


# ============================================================================
# Diagnostic whitened peak finder
# ============================================================================

def _rolling_median_fast(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    w = int(w)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    return median_filter(y, size=w, mode="nearest").astype(float)


def _estimate_sigma_local_diag(
    cont: np.ndarray, p: np.ndarray, m_eff: int, mode: str = "cont"
) -> np.ndarray:
    me   = max(1, int(m_eff))
    base = np.asarray(cont if mode.lower().strip() == "cont" else p, float)
    base = np.where(np.isfinite(base) & (base > 0), base, np.nan)
    med  = float(np.nanmedian(base[np.isfinite(base)])) if np.any(np.isfinite(base)) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    base = np.where(np.isfinite(base) & (base > 0), base, med)
    return base / np.sqrt(float(me))


def find_qpo_peak_whitened(
    freq, power,
    fmin=0.05, fmax=10.0, smooth_hz=0.5, ignore_below=0.1,
    min_width_bins=7, m_eff: int = 1, require_ksigma=None,
    prominence_sigma: float = 1.0, min_sep_hz: float = 0.15,
    sigma_mode: str = "cont", prefer: str = "prominence",
) -> float:
    f = np.asarray(freq, float)
    p = np.asarray(power, float)
    m = np.isfinite(f) & np.isfinite(p) & (p > 0) & (f >= fmin) & (f <= fmax)
    if ignore_below is not None:
        m &= (f >= float(ignore_below))
    f, p = f[m], p[m]

    if f.size < 10:
        return float(f[np.argmax(p)]) if f.size else np.nan

    df = float(np.median(np.diff(f)))
    if not np.isfinite(df) or df <= 0:
        return float(f[np.argmax(p)])

    w = max(int(min_width_bins), int(np.round(smooth_hz / df)))
    if w % 2 == 0:
        w += 1
    if w >= p.size:
        w = p.size - 2 if p.size > 2 else 3
        if w % 2 == 0:
            w -= 1
    if w < 3 or w >= p.size:
        return float(f[np.argmax(p)])

    cont = _rolling_median_fast(p, w)
    good = np.isfinite(cont) & (cont > 0)
    if not np.any(good):
        return float(f[np.argmax(p)])
    cont = np.where(good, cont, float(np.nanmedian(cont[good])))

    sigma = _estimate_sigma_local_diag(cont, p, m_eff=int(m_eff), mode=str(sigma_mode))
    z     = (p - cont) / sigma

    distance = int(max(1, np.round(min_sep_hz / df)))
    height   = float(require_ksigma) if (require_ksigma is not None and np.isfinite(require_ksigma)) else None
    peaks, props = scipy.signal.find_peaks(
        z, height=height, prominence=float(prominence_sigma), distance=distance
    )
    if peaks.size == 0:
        j = int(np.nanargmax(z))
        return float(f[j]) if np.isfinite(z[j]) else float(f[np.argmax(p)])

    z_prom   = np.asarray(props.get("prominences", np.zeros(len(peaks))), float)
    z_height = np.asarray(props.get("peak_heights", z[peaks]), float)
    score    = z_prom if prefer.lower().strip() != "height" else z_height
    return float(f[int(peaks[np.argmax(score)])])


# ============================================================================
# Plotting
# ============================================================================

def save_threeband_fit_overlay_plot(
    obsid: str, outdir_obsid: str, band_items: list[dict],
) -> None:
    from QPO_plot import save_threeband_plot, fitresult_to_band_block
    outpath = os.path.join(outdir_obsid, f"{obsid}_fits_full_soft_hard.png")
    plot_items = []
    for item in band_items:
        pds    = item.get("pds_fit")
        fitres = item.get("fitres")
        plot_items.append({
            "label":      item.get("label", "Band"),
            "freq":       (None if pds is None else np.asarray(pds.freq,  float)),
            "power":      (None if pds is None else np.asarray(pds.power, float)),
            "power_err":  (None if pds is None or getattr(pds, "power_err", None) is None
                           else np.asarray(pds.power_err, float)),
            "band_block": fitresult_to_band_block(fitres),
        })
    save_threeband_plot(obsid, plot_items, outpath, clobber=False)


# ============================================================================
# RMS helpers
# ============================================================================

def _compute_rms_metrics(pds_fit, peak_hz) -> tuple[float, float, float, float]:
    broad_band = getattr(P, "BROAD_RMS_BAND", (0.1, 30.0))
    broad_rms, broad_err = pds_fit.compute_rms(*broad_band)

    if np.isfinite(peak_hz) and peak_hz > 0:
        eta  = getattr(P, "QPO_BW_FRAC", 0.10)
        bw   = float(np.clip(eta * peak_hz, getattr(P, "QPO_BW_MIN", 0.10), getattr(P, "QPO_BW_MAX", 2.00)))
        lo   = max(0.01, peak_hz - bw)
        hi   = peak_hz + bw
        qpo_rms, qpo_err = pds_fit.compute_rms(lo, hi)
    else:
        qpo_rms, qpo_err = np.nan, np.nan

    return broad_rms, broad_err, qpo_rms, qpo_err


def _stringify_pars_list(pars: np.ndarray) -> tuple[str, str, str]:
    try:
        arr = np.asarray(pars, float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            return "", "", ""
        return (
            ";".join(f"{x:.8g}" for x in arr[:, 0]),
            ";".join(f"{x:.8g}" for x in arr[:, 1]),
            ";".join(f"{x:.8g}" for x in arr[:, 2]),
        )
    except Exception:
        return "", "", ""


# ============================================================================
# Per-band fitting  (TripleA-centred)
# ============================================================================

def _fit_one_band(
    *,
    obsid: str,
    band_label: str,
    pds_fit: AveragedPowerspectrum,
    pds_cand: AveragedPowerspectrum,
    seed_peak_hz,
    forced_qpo_seeds: list | None = None,
):
    """
    Fit one energy band with TripleA (or Powell if FIT_METHOD != "TripleA").

    Seed priority
    -------------
    1. forced_qpo_seeds  — cross-band QPO frequencies from other bands.
    2. seed_peak_hz      — diagnostic whitened-peak frequency for this band.
    3. find_qpo_candidates inside fit_lorentzian_family_triplea.

    All three are passed to fit_lorentzian_family; TripleA uses them as
    high-priority starting frequencies for its model configurations.
    The fit itself is entirely handled by QPO_fit.fit_lorentzian_family.
    """
    empty = {
        f"{band_label}_fit_ok":          False,
        f"{band_label}_fit_nlor":        "",
        f"{band_label}_fit_const":       "",
        f"{band_label}_fit_rchi2":       "",
        f"{band_label}_fit_red_dev":     "",
        f"{band_label}_fit_msg":         "",
        f"{band_label}_fit_qpo_nu0_hz":  "",
        f"{band_label}_fit_qpo_fwhm_hz": "",
        f"{band_label}_fit_qpo_Q":       "",
        f"{band_label}_fit_qpo_rms2":    "",
        f"{band_label}_fit_qpo_rms":     "",
        f"{band_label}_fit_nqpo":        "",
        f"{band_label}_fit_qpo_nu0s_hz": "",
        f"{band_label}_fit_qpo_fwhms_hz":"",
        f"{band_label}_fit_qpo_Qs":      "",
        f"{band_label}_fit_qpo_rms2s":   "",
        f"{band_label}_fit_qpo_rmss":    "",
        # parameter errors (from Hessian inversion at the Whittle MLE)
        f"{band_label}_fit_qpo_nu0_err":   "",
        f"{band_label}_fit_qpo_fwhm_err":  "",
        f"{band_label}_fit_qpo_Q_err":     "",
        f"{band_label}_fit_qpo_rms2_err":  "",
        f"{band_label}_fit_qpo_rms_err":   "",
        f"{band_label}_fit_qpo_nu0_errs":  "",
        f"{band_label}_fit_qpo_fwhm_errs": "",
        f"{band_label}_fit_qpo_Q_errs":    "",
        f"{band_label}_fit_qpo_rms_errs":  "",
        f"{band_label}_fit_qpo_nu_max_hz":  "",
        f"{band_label}_fit_qpo_nu_max_err": "",
        f"{band_label}_fit_qpo_nu_maxs_hz": "",
        f"{band_label}_comp_nu0s":          "",
        f"{band_label}_comp_fwhms":      "",
        f"{band_label}_comp_amps":       "",
    }

    if pds_fit is None or pds_cand is None:
        empty[f"{band_label}_fit_msg"] = "No PDS"
        return None, empty

    out = dict(empty)

    m_avg_fit  = _safe_m_avg_from_pds(pds_fit)
    m_avg_cand = _safe_m_avg_from_pds(pds_cand)

    # ── kwargs shared by both TripleA and Powell ──────────────────────────
    fit_kwargs: dict = dict(
        # Data
        m          = getattr(pds_fit, "m", m_avg_fit),
        cand_m_eff = m_avg_cand,
        cand_freq  = pds_cand.freq,
        cand_power = pds_cand.power,

        # Frequency bands
        fit_fmin  = getattr(P, "FIT_FMIN",  0.05),
        fit_fmax  = getattr(P, "FIT_FMAX",  64.0),
        cand_fmin = getattr(P, "CAND_FMIN", 0.05),
        cand_fmax = getattr(P, "CAND_FMAX", 10.0),

        # Constant / white-noise floor
        include_const    = getattr(P, "FIT_INCLUDE_CONST",    True),
        const_seed_fmin  = getattr(P, "FIT_CONST_SEED_FMIN",  40.0),
        const_cap_factor = getattr(P, "FIT_CONST_CAP_FACTOR",  5.0),

        # Candidate peak finding (feeds QPO seeds)
        smooth_scales       = getattr(P, "PEAK_SMOOTH_SCALES", None),
        smooth_hz           = getattr(P, "PEAK_SMOOTH_HZ",     0.5),
        prominence          = getattr(P, "PEAK_PROMINENCE",    0.5),
        min_sep_hz          = getattr(P, "PEAK_MIN_SEP_HZ",    0.15),
        max_candidates      = getattr(P, "PEAK_MAX_CANDIDATES",  8),
        cand_require_ksigma = getattr(P, "PEAK_REQUIRE_KSIGMA", None),
        cand_sigma_mode     = getattr(P, "PEAK_SIGMA_MODE",    "cont"),

        # Multi-QPO
        max_qpos               = getattr(P, "FIT_MAX_QPOS",              1),
        multi_qpo_ic_delta_min = getattr(P, "FIT_MULTI_QPO_IC_DELTA_MIN", None),

        # Component bounds
        cont_x0_narrow_hz = getattr(P, "FIT_CONT_X0_NARROW_HZ", 0.3),
        cont_x0_wide_hz   = getattr(P, "FIT_CONT_X0_WIDE_HZ",   3.0),
        cont_x0_free_hz   = getattr(P, "FIT_CONT_X0_FREE_HZ",   8.0),
        cont_fwhm_lim = (getattr(P, "FIT_CONT_FWHM_MIN", 0.30),
                         getattr(P, "FIT_CONT_FWHM_MAX", 64.0)),
        qpo_fwhm_lim  = (getattr(P, "FIT_QPO_FWHM_MIN",  0.08),
                         getattr(P, "FIT_QPO_FWHM_MAX",   5.0)),
        cont_amp_factor = getattr(P, "FIT_CONT_AMP_FACTOR", 12.0),
        qpo_amp_factor  = getattr(P, "FIT_QPO_AMP_FACTOR",  12.0),
        qpo_fwhm_frac   = getattr(P, "FIT_QPO_FWHM_FRAC",   0.03),
        qpo_fwhm_min    = getattr(P, "FIT_QPO_FWHM_MIN_ABS", 0.08),

        # Model-order selection (IC criteria)
        cont_ic_criterion = getattr(P, "CONT_IC_CRITERION", "bic"),
        cont_ic_delta_min = getattr(P, "CONT_IC_DELTA_MIN", 15.0),
        qpo_ic_criterion     = getattr(P, "QPO_IC_CRITERION",        "aic"),
        qpo_ic_delta_min     = getattr(P, "QPO_IC_DELTA_MIN",        10.0),
        qpo_detect_qmin      = getattr(P, "QPO_MIN_Q",                3.0),
        # rchi2 fallback: accept QPO when IC gate fails but fit genuinely improves.
        # Set FIT_QPO_RCHI_IMPROVE_MIN = 0.0 to rely on the strict IC gate alone.
        qpo_rchi_improve_min = getattr(P, "FIT_QPO_RCHI_IMPROVE_MIN", 0.05),

        # Optimizer
        fitmethod    = getattr(P, "FIT_METHOD",      "TripleA"),
        random_seed  = getattr(P, "FIT_RANDOM_SEED",       42),

        # QPO seeds  (these are the primary signal to the fitter)
        forced_qpo_seeds = list(forced_qpo_seeds) if forced_qpo_seeds else None,

        # Powell-only params (ignored by TripleA, harmless to pass)
        n_starts     = getattr(P, "FIT_N_STARTS",    3) if getattr(P, "FIT_MULTI_START", True) else 1,
        jitter_frac  = getattr(P, "FIT_JITTER_FRAC", 0.12),
        max_retries  = getattr(P, "FIT_MAX_RETRIES",  3),
        guard_enable                  = getattr(P, "FIT_GUARD_ENABLE",                 True),
        guard_overshoot_ksigma        = getattr(P, "FIT_GUARD_OVERSHOOT_KSIGMA",        4.0),
        guard_overshoot_max_run_bins  = getattr(P, "FIT_GUARD_OVERSHOOT_MAX_RUN_BINS",    6),
        guard_overshoot_max_frac      = getattr(P, "FIT_GUARD_OVERSHOOT_MAX_FRAC",      0.10),
        guard_comp_local_amp_factor   = getattr(P, "FIT_GUARD_COMP_LOCAL_AMP_FACTOR",    6.0),
        cont4_enable       = getattr(P, "FIT_CONT4_ENABLE",       False),
        cont4_trigger_rchi = getattr(P, "FIT_CONT4_TRIGGER_RCHI",  1.5),
        cont4_ic_criterion = getattr(P, "FIT_CONT4_IC_CRITERION", "bic"),
        cont4_ic_delta_min = getattr(P, "FIT_CONT4_IC_DELTA_MIN", 30.0),
    )

    # seed_peak_hz is optional — only pass it if valid
    if seed_peak_hz is not None and np.isfinite(seed_peak_hz) and seed_peak_hz > 0:
        fit_kwargs["seed_peak_hz"] = float(seed_peak_hz)

    # ── Run the fitter ────────────────────────────────────────────────────
    fitres = fit_lorentzian_family(
        pds_fit.freq, pds_fit.power, pds_fit.power_err,
        **fit_kwargs,
    )

    # ── Extract results ───────────────────────────────────────────────────
    rchi_max = getattr(P, "FIT_RCHI_MAX", None)

    out[f"{band_label}_fit_ok"]    = bool(getattr(fitres, "ok", False))
    out[f"{band_label}_fit_nlor"]  = int(getattr(fitres, "nlor", 0)) or ""
    out[f"{band_label}_fit_msg"]   = str(getattr(fitres, "message", ""))
    cval = getattr(fitres, "const", 0.0)
    out[f"{band_label}_fit_const"] = float(cval) if np.isfinite(cval) else ""

    rchi = getattr(fitres, "rchi2", np.nan)
    out[f"{band_label}_fit_rchi2"] = float(rchi) if np.isfinite(rchi) else ""

    red_dev = getattr(fitres, "red_deviance", np.nan)
    out[f"{band_label}_fit_red_dev"] = float(red_dev) if np.isfinite(red_dev) else ""

    nu0s, fwhms, amps = _stringify_pars_list(getattr(fitres, "pars", np.empty((0, 3))))
    out[f"{band_label}_comp_nu0s"]  = nu0s
    out[f"{band_label}_comp_fwhms"] = fwhms
    out[f"{band_label}_comp_amps"]  = amps

    if rchi_max is not None and np.isfinite(rchi):
        out[f"{band_label}_fit_ok"] = bool(out[f"{band_label}_fit_ok"]) and (rchi <= float(rchi_max))

    qpos = _qpos_from_fitres_by_type(fitres)
    out[f"{band_label}_fit_nqpo"] = int(len(qpos)) if qpos else 0

    if qpos:
        qpo = qpos[0]
        out[f"{band_label}_fit_qpo_nu0_hz"]  = qpo["qpo_nu0_hz"]
        out[f"{band_label}_fit_qpo_fwhm_hz"] = qpo["qpo_fwhm_hz"]
        out[f"{band_label}_fit_qpo_Q"]        = qpo["qpo_Q"]

        fmin_int = getattr(P, "FIT_FMIN", 0.05)
        fmax_int = getattr(P, "FIT_FMAX", 64.0)
        p_cov    = getattr(fitres, "p_err", None)  # full covariance matrix

        rms2s, rmss, rms2_errs, rms_errs = [], [], [], []
        nu0_errs, fwhm_errs, Q_errs = [], [], []
        nu_maxs, nu_max_errs = [], []

        for q in qpos:
            try:
                idx            = int(q["qpo_index"])
                nu0_i, fwhm_i, amp_i = fitres.pars[idx]

                # Exact Lorentzian integral + error propagation
                cov3 = _extract_component_cov(p_cov, idx)
                rms2, rms2_err, rms, rms_err = _rms2_err_from_cov(
                    nu0_i, fwhm_i, amp_i, cov3 if cov3 is not None
                    else np.full((3,3), np.nan),
                    fmin_int, fmax_int,
                )

                # Per-parameter errors from diagonal of covariance
                if cov3 is not None:
                    nu0_err  = float(np.sqrt(max(float(cov3[1, 1]), 0.0)))
                    fwhm_err = float(np.sqrt(max(float(cov3[2, 2]), 0.0)))
                    cov_nu0_fwhm = cov3[np.ix_([1, 2], [1, 2])]
                    Q_err = _q_err_from_cov(nu0_i, fwhm_i, cov_nu0_fwhm)
                else:
                    nu0_err = fwhm_err = Q_err = np.nan

                # Characteristic frequency and its error
                nu_max_val = _nu_max(nu0_i, fwhm_i)
                nu_max_err_val = _nu_max_err_from_cov(nu0_i, fwhm_i, cov3) if cov3 is not None else np.nan

            except Exception:
                rms2 = rms = rms2_err = rms_err = np.nan
                nu0_err = fwhm_err = Q_err = np.nan
                nu_max_val = nu_max_err_val = np.nan

            rms2s.append(rms2);  rmss.append(rms)
            rms2_errs.append(rms2_err); rms_errs.append(rms_err)
            nu0_errs.append(nu0_err); fwhm_errs.append(fwhm_err)
            Q_errs.append(Q_err)
            nu_maxs.append(nu_max_val); nu_max_errs.append(nu_max_err_val)

        def _fmt(vals):
            return ";".join("" if not np.isfinite(x) else f"{float(x):.6g}" for x in vals)

        out[f"{band_label}_fit_qpo_nu0s_hz"]   = ";".join(f"{float(q['qpo_nu0_hz']):.6g}"  for q in qpos)
        out[f"{band_label}_fit_qpo_fwhms_hz"]  = ";".join(f"{float(q['qpo_fwhm_hz']):.6g}" for q in qpos)
        out[f"{band_label}_fit_qpo_Qs"]         = ";".join(f"{float(q['qpo_Q']):.6g}"       for q in qpos)
        out[f"{band_label}_fit_qpo_rms2s"]      = _fmt(rms2s)
        out[f"{band_label}_fit_qpo_rmss"]       = _fmt(rmss)
        out[f"{band_label}_fit_qpo_nu0_errs"]   = _fmt(nu0_errs)
        out[f"{band_label}_fit_qpo_fwhm_errs"]  = _fmt(fwhm_errs)
        out[f"{band_label}_fit_qpo_Q_errs"]     = _fmt(Q_errs)
        out[f"{band_label}_fit_qpo_rms_errs"]   = _fmt(rms_errs)
        out[f"{band_label}_fit_qpo_nu_maxs_hz"] = _fmt(nu_maxs)

        if np.isfinite(nu_maxs[0]):
            out[f"{band_label}_fit_qpo_nu_max_hz"]  = float(nu_maxs[0])
        if np.isfinite(nu_max_errs[0]):
            out[f"{band_label}_fit_qpo_nu_max_err"] = float(nu_max_errs[0])

        if np.isfinite(rms2s[0]):
            out[f"{band_label}_fit_qpo_rms2"]     = float(rms2s[0])
            out[f"{band_label}_fit_qpo_rms"]      = float(rmss[0])
        if np.isfinite(nu0_errs[0]):
            out[f"{band_label}_fit_qpo_nu0_err"]  = float(nu0_errs[0])
        if np.isfinite(fwhm_errs[0]):
            out[f"{band_label}_fit_qpo_fwhm_err"] = float(fwhm_errs[0])
        if np.isfinite(Q_errs[0]):
            out[f"{band_label}_fit_qpo_Q_err"]    = float(Q_errs[0])
        if np.isfinite(rms_errs[0]):
            out[f"{band_label}_fit_qpo_rms_err"]  = float(rms_errs[0])
        if np.isfinite(rms2_errs[0]):
            out[f"{band_label}_fit_qpo_rms2_err"] = float(rms2_errs[0])

    return fitres, out


# ============================================================================
# Main per-obsid analysis
# ============================================================================

def analyze_obsid(obsid: str, evt_path: str) -> dict:
    # Heavy imports deferred here so spawn workers import this
    # module with only numpy/scipy at startup (no stingray lock
    # contention across 32 simultaneous worker processes).
    from stingray.events import EventList
    from stingray.powerspectrum import AveragedPowerspectrum
    from QPO_utils import (
        filter_events_by_energy,
        maybe_rebin_pds_fit,
        maybe_rebin_pds_candidate,
    )
    from QPO_struct import save_fit_struct
    outdir_this = obsid_outdir(obsid)
    mjd_mid, iso_mid = _evt_time_mjd_mid(evt_path)

    ev  = EventList.read(evt_path)
    dt  = getattr(P, "DT",           0.00390625)
    seg = getattr(P, "SEGMENT_SIZE", 64)

    def _make_pds(ev_band: EventList):
        lc      = ev_band.to_lc(dt=dt)
        pds_raw = AveragedPowerspectrum(lc, segment_size=seg, norm="frac")
        return lc, pds_raw, maybe_rebin_pds_candidate(pds_raw), maybe_rebin_pds_fit(pds_raw)

    lc_full, pds_full_raw, pds_full_cand, pds_full_fit = _make_pds(ev)

    soft_ok = hard_ok = False
    lc_soft = pds_soft_raw = pds_soft_cand = pds_soft_fit = None
    lc_hard = pds_hard_raw = pds_hard_cand = pds_hard_fit = None

    if getattr(P, "DO_ENERGY_BANDS", False):
        for band_attr, band_kev in [("soft", getattr(P, "SOFT_BAND_KEV", (0.3, 2.0))),
                                    ("hard", getattr(P, "HARD_BAND_KEV", (2.0, 10.0)))]:
            try:
                ev_b = filter_events_by_energy(ev, band_kev)
                lc_b, raw_b, cand_b, fit_b = _make_pds(ev_b)
                if band_attr == "soft":
                    lc_soft, pds_soft_raw, pds_soft_cand, pds_soft_fit = lc_b, raw_b, cand_b, fit_b
                    soft_ok = True
                else:
                    lc_hard, pds_hard_raw, pds_hard_cand, pds_hard_fit = lc_b, raw_b, cand_b, fit_b
                    hard_ok = True
            except Exception:
                pass

    # ── Diagnostic peak finder ────────────────────────────────────────────
    def _peak(pds_cand, m_cand):
        return find_qpo_peak_whitened(
            pds_cand.freq, pds_cand.power,
            fmin=getattr(P, "FMIN",               0.05),
            fmax=getattr(P, "FMAX",               10.0),
            smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
            ignore_below=getattr(P, "PEAK_IGNORE_BELOW", 0.1),
            min_width_bins=7,
            m_eff=m_cand,
            require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
            prominence_sigma=float(getattr(P, "PEAK_PROMINENCE_SIGMA", 1.0)),
            min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
            sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),
            prefer=getattr(P, "PEAK_RANK_BY", "prominence"),
        )

    peak_full = _peak(pds_full_cand, _safe_m_avg_from_pds(pds_full_cand))
    peak_soft = _peak(pds_soft_cand, _safe_m_avg_from_pds(pds_soft_cand)) if soft_ok else None
    peak_hard = _peak(pds_hard_cand, _safe_m_avg_from_pds(pds_hard_cand)) if hard_ok else None

    # ── RMS metrics ───────────────────────────────────────────────────────
    broad_rms_full, broad_err_full, qpo_rms_full, qpo_err_full = _compute_rms_metrics(pds_full_fit, peak_full)
    broad_rms_soft = broad_err_soft = qpo_rms_soft = qpo_err_soft = np.nan
    broad_rms_hard = broad_err_hard = qpo_rms_hard = qpo_err_hard = np.nan
    if soft_ok:
        broad_rms_soft, broad_err_soft, qpo_rms_soft, qpo_err_soft = _compute_rms_metrics(pds_soft_fit, peak_soft)
    if hard_ok:
        broad_rms_hard, broad_err_hard, qpo_rms_hard, qpo_err_hard = _compute_rms_metrics(pds_hard_fit, peak_hard)

    # ── Fitting ───────────────────────────────────────────────────────────
    fit_full_res = fit_soft_res = fit_hard_res = None
    fit_full_out = fit_soft_out = fit_hard_out = {}
    seed_s = seed_h = None

    if getattr(P, "DO_FIT", False):
        shared_seed = float(peak_full) if (peak_full is not None and np.isfinite(peak_full) and peak_full > 0) else None

        # ── Pass 1: independent per-band fits ─────────────────────────────
        fit_full_res, fit_full_out = _fit_one_band(
            obsid=obsid, band_label="full",
            pds_fit=pds_full_fit, pds_cand=pds_full_cand,
            seed_peak_hz=shared_seed,
        )
        if soft_ok:
            seed_s = shared_seed or (float(peak_soft) if peak_soft and np.isfinite(peak_soft) else None)
            fit_soft_res, fit_soft_out = _fit_one_band(
                obsid=obsid, band_label="soft",
                pds_fit=pds_soft_fit, pds_cand=pds_soft_cand,
                seed_peak_hz=seed_s,
            )
        if hard_ok:
            seed_h = shared_seed or (float(peak_hard) if peak_hard and np.isfinite(peak_hard) else None)
            fit_hard_res, fit_hard_out = _fit_one_band(
                obsid=obsid, band_label="hard",
                pds_fit=pds_hard_fit, pds_cand=pds_hard_cand,
                seed_peak_hz=seed_h,
            )

        # ── Pass 2: cross-band reseeding ──────────────────────────────────
        # When a QPO is found in one band, inject its frequency as a
        # forced seed in bands that missed it.  This is the primary
        # mechanism for consistent multi-band QPO detection.
        if getattr(P, "DO_CROSS_BAND_RESEED", True):
            xb_thr = float(getattr(P, "CROSS_BAND_RESEED_RCHI_BAD",
                                   getattr(P, "FIT_RCHI_MAX", 1.5)))
            QPO_AREA_QUALITY = float(getattr(P, "CROSS_BAND_QPO_AREA_MIN", 1e-4))

            def _band_rchi(fres) -> float:
                r = getattr(fres, "rchi2", np.nan) if (fres is not None and getattr(fres, "ok", False)) else np.nan
                return float(r) if np.isfinite(r) else np.inf

            def _collect_qpo_freqs(fres) -> list[float]:
                if fres is None or not getattr(fres, "ok", False):
                    return []
                out = []
                for q in _qpo_list_from_fit(fres):
                    if float(q.get("qpo_area", 0)) >= QPO_AREA_QUALITY:
                        nu = float(q.get("qpo_nu0_hz", np.nan))
                        if np.isfinite(nu):
                            out.append(nu)
                return out

            # Collect all well-constrained QPO frequencies across all bands
            cross_seeds: list[float] = []
            seen: set[float] = set()
            for nu in (_collect_qpo_freqs(fit_full_res)
                       + _collect_qpo_freqs(fit_soft_res)
                       + _collect_qpo_freqs(fit_hard_res)):
                if any(abs(nu - s) < 0.05 for s in seen):
                    continue
                seen.add(nu)
                cross_seeds.append(nu)

            # Fallback: use diagnostic peaks when no QPO found anywhere
            if not cross_seeds and getattr(P, "CROSS_BAND_USE_DIAG_PEAKS_FALLBACK", True):
                diag_freqs = []
                for pk in [peak_full, peak_soft, peak_hard]:
                    if pk is not None and np.isfinite(pk) and pk > 0:
                        if not any(abs(pk - s) < 0.1 for s in diag_freqs):
                            diag_freqs.append(float(pk))
                if diag_freqs:
                    cross_seeds = diag_freqs

            if cross_seeds:
                band_specs = [
                    ("full", fit_full_res, fit_full_out, pds_full_fit, pds_full_cand, shared_seed),
                    ("soft", fit_soft_res, fit_soft_out, pds_soft_fit,  pds_soft_cand,  seed_s if soft_ok else None),
                    ("hard", fit_hard_res, fit_hard_out, pds_hard_fit,  pds_hard_cand,  seed_h if hard_ok else None),
                ]

                new_full = new_soft = new_hard = None

                for label, fres_cur, out_cur, pds_f, pds_c, spk in band_specs:
                    if pds_f is None or pds_c is None:
                        continue

                    cur_freqs = [float(q["qpo_nu0_hz"]) for q in _qpo_list_from_fit(fres_cur)]
                    cur_rchi  = _band_rchi(fres_cur)

                    # Seeds missing from this band
                    missing = [s for s in cross_seeds
                               if not any(abs(s - g) < 0.05 for g in cur_freqs)]
                    if not missing:
                        continue
                    # Trigger: bad rchi2 OR missing a QPO found elsewhere
                    if cur_rchi <= xb_thr and not missing:
                        continue

                    fres_new, out_new = _fit_one_band(
                        obsid=obsid, band_label=label,
                        pds_fit=pds_f, pds_cand=pds_c,
                        seed_peak_hz=spk,
                        forced_qpo_seeds=missing,
                    )
                    if fres_new is None:
                        continue

                    new_freqs = [float(q["qpo_nu0_hz"]) for q in _qpo_list_from_fit(fres_new)]
                    new_rchi  = _band_rchi(fres_new)

                    # Accept if rchi2 improved OR gained a QPO without getting worse
                    gained = any(any(abs(s - nf) < 0.05 for nf in new_freqs) for s in missing)
                    if new_rchi < cur_rchi - 0.01 or (gained and new_rchi <= cur_rchi + 0.10):
                        if label == "full":   new_full = (fres_new, out_new)
                        elif label == "soft": new_soft = (fres_new, out_new)
                        else:                 new_hard = (fres_new, out_new)

                if new_full: fit_full_res, fit_full_out = new_full; fit_full_out["full_cross_reseeded"] = True
                if new_soft: fit_soft_res, fit_soft_out = new_soft; fit_soft_out["soft_cross_reseeded"] = True
                if new_hard: fit_hard_res, fit_hard_out = new_hard; fit_hard_out["hard_cross_reseeded"] = True

        # ── Save structs ───────────────────────────────────────────────────
        save_fit_struct(fit_full_res, obsid, "full", mjd=mjd_mid,
                        peak_hz=peak_full if (peak_full is not None and np.isfinite(peak_full)) else None)
        save_fit_struct(fit_soft_res, obsid, "soft", mjd=mjd_mid,
                        peak_hz=peak_soft if (peak_soft is not None and np.isfinite(peak_soft)) else None)
        save_fit_struct(fit_hard_res, obsid, "hard", mjd=mjd_mid,
                        peak_hz=peak_hard if (peak_hard is not None and np.isfinite(peak_hard)) else None)

        if getattr(P, "SAVE_FIT_PLOTS", True):
            band_items = [
                {"label": "Full",           "pds_fit": pds_full_fit, "fitres": fit_full_res,
                 "seed_peak": shared_seed},
                {"label": "Soft 0.3-2 keV", "pds_fit": pds_soft_fit, "fitres": fit_soft_res,
                 "seed_peak": shared_seed or (float(peak_soft) if peak_soft and np.isfinite(peak_soft) else np.nan)},
                {"label": "Hard 2-10 keV",  "pds_fit": pds_hard_fit, "fitres": fit_hard_res,
                 "seed_peak": shared_seed or (float(peak_hard) if peak_hard and np.isfinite(peak_hard) else np.nan)},
            ]
            save_threeband_fit_overlay_plot(obsid, outdir_this, band_items)

    out = {
        "mjd_mid": float(mjd_mid)  if (mjd_mid  is not None and np.isfinite(mjd_mid))  else "",
        "iso_mid": iso_mid          if iso_mid    is not None                            else "",
        "tseg_s":          float(lc_full.tseg),
        "dt_s":            float(lc_full.dt),
        "mean_rate_cps":   float(lc_full.meanrate),
        "segment_size_s":  float(seg),
        "peak_f_hz_full":  float(peak_full) if np.isfinite(peak_full) else "",
        "peak_f_hz_soft":  float(peak_soft) if (peak_soft is not None and np.isfinite(peak_soft)) else "",
        "peak_f_hz_hard":  float(peak_hard) if (peak_hard is not None and np.isfinite(peak_hard)) else "",
        "broad_rms_0p1_30_full": float(broad_rms_full) if np.isfinite(broad_rms_full) else "",
        "broad_rms_err_full":    float(broad_err_full)  if np.isfinite(broad_err_full)  else "",
        "qpo_rms_full":          float(qpo_rms_full)    if np.isfinite(qpo_rms_full)    else "",
        "qpo_rms_err_full":      float(qpo_err_full)    if np.isfinite(qpo_err_full)    else "",
        "broad_rms_0p1_30_soft": float(broad_rms_soft) if np.isfinite(broad_rms_soft) else "",
        "broad_rms_err_soft":    float(broad_err_soft)  if np.isfinite(broad_err_soft)  else "",
        "qpo_rms_soft":          float(qpo_rms_soft)    if np.isfinite(qpo_rms_soft)    else "",
        "qpo_rms_err_soft":      float(qpo_err_soft)    if np.isfinite(qpo_err_soft)    else "",
        "broad_rms_0p1_30_hard": float(broad_rms_hard) if np.isfinite(broad_rms_hard) else "",
        "broad_rms_err_hard":    float(broad_err_hard)  if np.isfinite(broad_err_hard)  else "",
        "qpo_rms_hard":          float(qpo_rms_hard)    if np.isfinite(qpo_rms_hard)    else "",
        "qpo_rms_err_hard":      float(qpo_err_hard)    if np.isfinite(qpo_err_hard)    else "",
    }
    out.update(fit_full_out or {})
    out.update(fit_soft_out or {})
    out.update(fit_hard_out or {})
    out["_fit_full_res_obj"] = fit_full_res
    out["_fit_soft_res_obj"] = fit_soft_res
    out["_fit_hard_res_obj"] = fit_hard_res
    return out


# ============================================================================
# Driver per obsid
# ============================================================================

def _process_one_obsid(obsid: str) -> dict:
    from QPO_utils import build_evt_path
    base_dir = getattr(P, "BASE_DIR", ".")
    source   = getattr(P, "SOURCE",   "")
    evt_path = build_evt_path(base_dir, source, obsid)

    suppress = bool(getattr(P, "QUIET", False))

    row = {"obsid": obsid, "evt_path": evt_path, "status": "", "error": ""}

    # Always print which obsid is being processed — even in quiet mode.
    # This makes hangs immediately visible.
    print(f"  → {obsid}", end=" ", flush=True)

    try:
        if not os.path.exists(evt_path):
            raise FileNotFoundError(f"Missing event file: {evt_path}")

        with _suppress_context(suppress):
            res = analyze_obsid(obsid, evt_path)

        row.update(res)
        row["status"] = "OK"
        row["error"]  = ""

    except Exception as exc:
        row["status"] = "FAIL"
        row["error"]  = str(exc)

    # Print result (replaces the "→ obsid" line with the full summary)
    if getattr(P, "ONE_LINE_SUMMARY", True):
        # Overwrite the "→ obsid" prefix by printing the full summary
        print("\r", end="")   # carriage-return to overwrite
        _print_one_line_summary(obsid, row)
    else:
        if row["status"] != "OK":
            print(f"FAIL  {row['error']}")
        else:
            print("OK")

    return row


# ============================================================================
# Main
# ============================================================================

def main():
    obsids = read_obsids(getattr(P, "OBSIDS_TXT", "obsids.txt"))
    if not obsids:
        raise SystemExit(f"No ObsIDs in {getattr(P, 'OBSIDS_TXT', 'obsids.txt')}")

    fieldnames = [
        "obsid", "evt_path", "status", "error",
        "mjd_mid", "iso_mid",
        "tseg_s", "dt_s", "mean_rate_cps", "segment_size_s",
        "peak_f_hz_full", "peak_f_hz_soft", "peak_f_hz_hard",
        "broad_rms_0p1_30_full", "broad_rms_err_full", "qpo_rms_full", "qpo_rms_err_full",
        "broad_rms_0p1_30_soft", "broad_rms_err_soft", "qpo_rms_soft", "qpo_rms_err_soft",
        "broad_rms_0p1_30_hard", "broad_rms_err_hard", "qpo_rms_hard", "qpo_rms_err_hard",
        "full_fit_ok", "full_fit_nlor", "full_fit_const", "full_fit_rchi2", "full_fit_red_dev", "full_fit_msg",
        "full_fit_qpo_nu0_hz", "full_fit_qpo_fwhm_hz", "full_fit_qpo_Q", "full_fit_qpo_rms2", "full_fit_qpo_rms",
        "full_fit_qpo_nu_max_hz", "full_fit_qpo_nu_max_err",
        "full_fit_qpo_nu0_err", "full_fit_qpo_fwhm_err", "full_fit_qpo_Q_err", "full_fit_qpo_rms2_err", "full_fit_qpo_rms_err",
        "full_comp_nu0s", "full_comp_fwhms", "full_comp_amps",
        "soft_fit_ok", "soft_fit_nlor", "soft_fit_const", "soft_fit_rchi2", "soft_fit_red_dev", "soft_fit_msg",
        "soft_fit_qpo_nu0_hz", "soft_fit_qpo_fwhm_hz", "soft_fit_qpo_Q", "soft_fit_qpo_rms2", "soft_fit_qpo_rms",
        "soft_fit_qpo_nu_max_hz", "soft_fit_qpo_nu_max_err",
        "soft_fit_qpo_nu0_err", "soft_fit_qpo_fwhm_err", "soft_fit_qpo_Q_err", "soft_fit_qpo_rms2_err", "soft_fit_qpo_rms_err",
        "soft_comp_nu0s", "soft_comp_fwhms", "soft_comp_amps",
        "hard_fit_ok", "hard_fit_nlor", "hard_fit_const", "hard_fit_rchi2", "hard_fit_red_dev", "hard_fit_msg",
        "hard_fit_qpo_nu0_hz", "hard_fit_qpo_fwhm_hz", "hard_fit_qpo_Q", "hard_fit_qpo_rms2", "hard_fit_qpo_rms",
        "hard_fit_qpo_nu_max_hz", "hard_fit_qpo_nu_max_err",
        "hard_fit_qpo_nu0_err", "hard_fit_qpo_fwhm_err", "hard_fit_qpo_Q_err", "hard_fit_qpo_rms2_err", "hard_fit_qpo_rms_err",
        "hard_comp_nu0s", "hard_comp_fwhms", "hard_comp_amps",
    ]

    parallel    = bool(getattr(P, "PARALLEL_ENABLE", False))
    n_workers   = int(getattr(P, "N_WORKERS", 1) or 1)
    start_meth  = str(getattr(P, "PARALLEL_START_METHOD", "spawn")).lower()
    rows: list[dict] = []

    if parallel and n_workers > 1 and len(obsids) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        ctx = mp.get_context(start_meth)
        print(f"[INFO] Parallel: {n_workers} workers | method={start_meth} | n_obsids={len(obsids)}")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_process_one_obsid, o): o for o in obsids}
            for fut in as_completed(futs):
                obsid = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    row = {k: "" for k in fieldnames}
                    row.update(obsid=obsid,
                               evt_path=build_evt_path(getattr(P, "BASE_DIR", "."),
                                                       getattr(P, "SOURCE",   ""), obsid),
                               status="FAIL", error=f"Worker crashed: {exc}")
                    if getattr(P, "ONE_LINE_SUMMARY", True):
                        _print_one_line_summary(obsid, row)
                    else:
                        print(f"[FAIL] {obsid}  Worker crashed: {exc}")
                    rows.append(row)

        order = {o: i for i, o in enumerate(obsids)}
        rows.sort(key=lambda r: order.get(r.get("obsid", ""), 10**9))

    else:
        if parallel:
            print("[INFO] Parallel requested but running serial (n_workers<=1 or single obsid).")
        else:
            print("[INFO] Running serial.")
        for obsid in obsids:
            rows.append(_process_one_obsid(obsid))

    csv_path = os.path.join(common_outdir(), getattr(P, "OUT_CSV_NAME", "qpo_summary.csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote: {csv_path}")
    print(f"Per-ObsID outputs in: {getattr(P, 'OUTDIR_BASE', '.')}/<obsid>/")


if __name__ == "__main__":
    main()
