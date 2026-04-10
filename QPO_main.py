#!/usr/bin/env python3
"""
QPO_main.py  (v2)
=================
Per-obsid analysis driver for the QPO fitting pipeline.


"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import warnings

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter

from astropy.io import fits
from astropy.time import Time

from stingray.events import EventList
from stingray.powerspectrum import AveragedPowerspectrum

from QPO_fit import (
    FitResult,
    component_power_integral,
    extract_qpo_params,
    extract_qpo_params_list,
    fit_lorentzian_family,
    lorentz,
)

import QPO_Parameter as P
from QPO_struct import save_fit_struct

# Force non-interactive Agg backend before any matplotlib import.
# Must be set here, before QPO_plot (which imports pyplot) is loaded.
import matplotlib
matplotlib.use("Agg")

from QPO_utils import (
    safe_m_from_pds,
    kev_to_pi,
    filter_events_by_energy,
    make_averaged_pds,
    rebin_pds as _rebin_pds,
    maybe_rebin_pds_fit,
    maybe_rebin_pds_candidate,
    build_evt_path,
)
from QPO_plot import save_threeband_plot, fitresult_to_band_block

print("[INFO] Using QPO_Parameter from:", getattr(P, "__file__", "UNKNOWN"))


# ============================================================================
# Logging suppression
# ============================================================================

def _suppress_context(enabled: bool):
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


def _qpo_found_from_fit(fitres) -> tuple[bool, float, float]:
    if fitres is None or not getattr(fitres, "ok", False):
        return False, np.nan, np.nan
    try:
        qpo = extract_qpo_params(
            fitres,
            qpo_fmin=getattr(P, "QPO_FMIN", 0.1),
            qpo_fmax=getattr(P, "QPO_FMAX", 10.0),
            qmin=getattr(P, "QPO_MIN_Q", 3.0),
        )
        if qpo is None:
            return False, np.nan, np.nan
        return True, float(qpo["qpo_nu0_hz"]), float(qpo["qpo_Q"])
    except Exception:
        return False, np.nan, np.nan


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


def _qpos_from_fitres_by_type(fitres) -> list[dict]:
    """
    Extract QPO Lorentzians using comp_types — the optimizer's own classification.

    This is the authoritative QPO list for plot annotations, CSV output, and
    struct consistency.  A component must satisfy ALL of:
      - comp_types entry is "qpo"
      - nu0 within [QPO_FMIN, QPO_FMAX]
      - Q = nu0/fwhm >= QPO_MIN_Q

    Using comp_types avoids the Q-threshold ambiguity where a broad continuum
    component near the QPO frequency could be misidentified as a QPO, and also
    ensures agreement with what is stored in the struct.
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
        if Q < qmin:
            continue
        if not (qpo_fmin <= nu0 <= qpo_fmax):
            continue
        area = float(np.trapz(lorentz(freq, nu0, fwhm, amp), freq))
        out.append(dict(
            qpo_index=int(i),
            qpo_nu0_hz=float(nu0),
            qpo_fwhm_hz=float(fwhm),
            qpo_Q=float(Q),
            qpo_area=float(area),
        ))

    out.sort(key=lambda d: (-d["qpo_area"], d["qpo_nu0_hz"]))
    return out


def _fmt_band(label: str, fitres) -> str:
    cont_n = _count_continuum_components(fitres)
    # Use the same comp_types-aware extraction as the CSV writer so the
    # one-line summary and the CSV always report the same Nqpo.
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

# Backward-compatible alias used within this module
_safe_m_avg_from_pds = safe_m_from_pds


# ============================================================================
# MJD helper
# ============================================================================

def _evt_time_mjd_mid(evt_path: str) -> tuple[float | None, str | None]:
    try:
        with fits.open(evt_path, memmap=True) as hdul:
            h       = hdul[1].header
            tstart  = float(h.get("TSTART"))
            tstop   = float(h.get("TSTOP"))
            tmid    = 0.5 * (tstart + tstop)
            mjdrefi = float(h.get("MJDREFI", 0.0))
            mjdreff = float(h.get("MJDREFF", 0.0))
            mjdref  = mjdrefi + mjdreff
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
    fmin=0.05, fmax=10.0,
    smooth_hz=0.5,
    ignore_below=0.1,
    min_width_bins=7,
    m_eff: int = 1,
    require_ksigma=None,
    prominence_sigma: float = 1.0,
    min_sep_hz: float = 0.15,
    sigma_mode: str = "cont",
    prefer: str = "prominence",
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
# Plotting  (thin wrapper — rendering logic lives in QPO_plot)
# ============================================================================

def save_threeband_fit_overlay_plot(
    obsid:        str,
    outdir_obsid: str,
    band_items:   list[dict],
) -> None:
    """
    Save the three-band fit overlay PNG.

    Converts the internal band_items format (pds_fit + FitResult objects) into
    the array + band_block format expected by QPO_plot.save_threeband_plot,
    then delegates all rendering there.

    Parameters
    ----------
    obsid        : NICER observation ID
    outdir_obsid : per-obsid output directory (used to build the default path)
    band_items   : list of dicts with keys
                   {label, pds_fit, fitres, seed_peak}
    """
    outpath = os.path.join(outdir_obsid, f"{obsid}_fits_full_soft_hard.png")

    plot_items = []
    for item in band_items:
        pds    = item.get("pds_fit")
        fitres = item.get("fitres")
        plot_items.append({
            "label":      item.get("label", "Band"),
            "freq":       (None if pds is None
                           else np.asarray(pds.freq,  float)),
            "power":      (None if pds is None
                           else np.asarray(pds.power, float)),
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
# Per-band fitting wrapper
# ============================================================================

def _fit_one_band(
    *,
    obsid: str,
    band_label: str,
    pds_fit: AveragedPowerspectrum,
    pds_cand: AveragedPowerspectrum,
    seed_peak_hz,
    obsid_seed_offset: int = 0,
    forced_qpo_seeds: list | None = None,
):
    empty = {
        f"{band_label}_fit_ok":       False,
        f"{band_label}_fit_nlor":     "",
        f"{band_label}_fit_const":    "",
        f"{band_label}_fit_rchi2":    "",
        f"{band_label}_fit_red_dev":  "",
        f"{band_label}_fit_msg":      "",
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
        f"{band_label}_comp_nu0s":    "",
        f"{band_label}_comp_fwhms":   "",
        f"{band_label}_comp_amps":    "",
    }

    if pds_fit is None or pds_cand is None:
        empty[f"{band_label}_fit_msg"] = "No PDS"
        return None, empty

    out = dict(empty)

    m_avg_fit  = _safe_m_avg_from_pds(pds_fit)
    m_avg_cand = _safe_m_avg_from_pds(pds_cand)

    rchi_max = getattr(P, "FIT_RCHI_MAX", None)
    rchi_override_threshold = float(rchi_max) if (rchi_max is not None and np.isfinite(rchi_max)) else None

    fit_kwargs = dict(
        # Pass the full per-bin m array so rchi2 is computed correctly
        m=getattr(pds_fit, "m", m_avg_fit),
        cand_m_eff=m_avg_cand,

        fit_fmin=getattr(P, "FIT_FMIN", 0.05),
        fit_fmax=getattr(P, "FIT_FMAX", 64.0),
        cand_fmin=getattr(P, "CAND_FMIN", 0.05),
        cand_fmax=getattr(P, "CAND_FMAX", 10.0),
        include_const=getattr(P, "FIT_INCLUDE_CONST", True),
        const_seed_fmin=getattr(P, "FIT_CONST_SEED_FMIN", 30.0),
        const_cap_factor=getattr(P, "FIT_CONST_CAP_FACTOR", 5.0),

        include_harmonic=getattr(P, "DO_HARMONIC_SEARCH", False),

        # Multi-scale smoothing
        smooth_scales=getattr(P, "PEAK_SMOOTH_SCALES", None),
        smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
        prominence=getattr(P, "PEAK_PROMINENCE", 0.5),
        min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
        max_candidates=getattr(P, "PEAK_MAX_CANDIDATES", 8),

        cand_require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
        cand_sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),

        stage1_n_seeds=getattr(P, "FIT_STAGE1_N_SEEDS", 3),
        max_qpos=getattr(P, "FIT_MAX_QPOS", 1),
        multi_qpo_ic_delta_min=getattr(P, "FIT_MULTI_QPO_IC_DELTA_MIN", None),
        multi_qpo_require_improvement=getattr(P, "FIT_MULTI_QPO_REQUIRE_IMPROVEMENT", True),

        reseed_enable=getattr(P, "FIT_RESEED_ENABLE", True),
        reseed_rchi_bad=getattr(P, "FIT_RESEED_RCHI_BAD", 1.8),
        reseed_edge_frac=getattr(P, "FIT_RESEED_EDGE_FRAC", 0.08),
        reseed_area_min=getattr(P, "FIT_RESEED_AREA_MIN", 0.0),
        reseed_exclude_hz_min=getattr(P, "FIT_RESEED_EXCLUDE_HZ_MIN", 0.5),
        reseed_exclude_df_mult=getattr(P, "FIT_RESEED_EXCLUDE_DF_MULT", 10.0),
        reseed_prom_factor=getattr(P, "FIT_RESEED_PROM_FACTOR", 1.25),
        reseed_sigma_factor=getattr(P, "FIT_RESEED_SIGMA_FACTOR", 1.10),

        n_starts=getattr(P, "FIT_N_STARTS", 6) if getattr(P, "FIT_MULTI_START", True) else 1,
        jitter_frac=getattr(P, "FIT_JITTER_FRAC", 0.18),
        random_seed=getattr(P, "FIT_RANDOM_SEED", 42),
        obsid_seed_offset=obsid_seed_offset,

        fitmethod=getattr(P, "FIT_METHOD", "Powell"),
        rchi_target=getattr(P, "FIT_RCHI_TARGET", 1.3),

        # Per-component centroid limits
        cont_x0_narrow_hz=getattr(P, "FIT_CONT_X0_NARROW_HZ", 0.3),
        cont_x0_wide_hz=getattr(P, "FIT_CONT_X0_WIDE_HZ", 3.0),
        cont_x0_free_hz=getattr(P, "FIT_CONT_X0_FREE_HZ", 8.0),
        cont_fwhm_lim=(getattr(P, "FIT_CONT_FWHM_MIN", 0.3), getattr(P, "FIT_CONT_FWHM_MAX", 64.0)),
        qpo_fwhm_lim=(getattr(P, "FIT_QPO_FWHM_MIN", 0.03), getattr(P, "FIT_QPO_FWHM_MAX", 5.0)),
        harm_fwhm_lim=(getattr(P, "FIT_HARM_FWHM_MIN", 0.03), getattr(P, "FIT_HARM_FWHM_MAX", 8.0)),

        cont_amp_factor=getattr(P, "FIT_CONT_AMP_FACTOR", 12.0),
        qpo_amp_factor=getattr(P, "FIT_QPO_AMP_FACTOR", 12.0),
        harm_amp_factor=getattr(P, "FIT_HARM_AMP_FACTOR", 5.0),

        qpo_fwhm_frac=getattr(P, "FIT_QPO_FWHM_FRAC", 0.06),
        qpo_fwhm_min=getattr(P, "FIT_QPO_FWHM_MIN_ABS", 0.05),

        guard_enable=getattr(P, "FIT_GUARD_ENABLE", True),
        guard_overshoot_ksigma=getattr(P, "FIT_GUARD_OVERSHOOT_KSIGMA", 4.0),
        guard_overshoot_max_run_bins=getattr(P, "FIT_GUARD_OVERSHOOT_MAX_RUN_BINS", 6),
        guard_overshoot_max_frac=getattr(P, "FIT_GUARD_OVERSHOOT_MAX_FRAC", 0.10),
        guard_comp_local_amp_factor=getattr(P, "FIT_GUARD_COMP_LOCAL_AMP_FACTOR", 6.0),

        max_retries=getattr(P, "FIT_MAX_RETRIES", 5),

        cont_ic_criterion=getattr(P, "CONT_IC_CRITERION", "bic"),
        cont_ic_delta_min=getattr(P, "CONT_IC_DELTA_MIN", 5.0),
        qpo_ic_criterion=getattr(P, "QPO_IC_CRITERION", "aic"),
        qpo_ic_delta_min=getattr(P, "QPO_IC_DELTA_MIN", 2.0),

        force_cont3_rchi=float(getattr(P, "FIT_CONT_RCHI_FORCE_CONT3", np.inf)),

        postqpo_cont3_enable=True,
        postqpo_cont3_trigger_rchi=float(getattr(P, "FIT_POSTQPO_CONT3_TRIGGER_RCHI", 1.5)),
        postqpo_cont3_rchi_improve_min=float(getattr(P, "FIT_POSTQPO_CONT3_RCHI_IMPROVE_MIN", 0.05)),
        postqpo_cont3_ic_delta_min=float(getattr(P, "FIT_POSTQPO_CONT3_IC_DELTA_MIN", 0.0)),

        rchi_override_enable=True,
        rchi_override_threshold=rchi_override_threshold,

        # cont4 fallback
        cont4_enable=getattr(P, "FIT_CONT4_ENABLE", True),
        cont4_trigger_rchi=float(getattr(P, "FIT_CONT4_TRIGGER_RCHI", 1.5)),
        cont4_ic_criterion=getattr(P, "FIT_CONT4_IC_CRITERION", "bic"),
        cont4_ic_delta_min=float(getattr(P, "FIT_CONT4_IC_DELTA_MIN", 3.0)),

        forced_qpo_seeds=list(forced_qpo_seeds) if forced_qpo_seeds else None,
    )

    if seed_peak_hz is not None and np.isfinite(seed_peak_hz) and seed_peak_hz > 0:
        fit_kwargs["seed_peak_hz"] = float(seed_peak_hz)

    fitres = fit_lorentzian_family(
        pds_fit.freq, pds_fit.power, pds_fit.power_err,
        cand_freq=pds_cand.freq, cand_power=pds_cand.power,
        **fit_kwargs,
    )

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

    # Use comp_types-aware extraction so CSV matches what is stored in the struct.
    qpos = _qpos_from_fitres_by_type(fitres)
    out[f"{band_label}_fit_nqpo"] = int(len(qpos)) if qpos else 0

    if qpos:
        qpo = qpos[0]
        out[f"{band_label}_fit_qpo_nu0_hz"]  = qpo["qpo_nu0_hz"]
        out[f"{band_label}_fit_qpo_fwhm_hz"] = qpo["qpo_fwhm_hz"]
        out[f"{band_label}_fit_qpo_Q"]        = qpo["qpo_Q"]

        rms2s, rmss = [], []
        for q in qpos:
            try:
                idx  = int(q["qpo_index"])
                nu0_i, fwhm_i, amp_i = fitres.pars[idx]
                comp = lorentz(fitres.freq, nu0_i, fwhm_i, amp_i)
                rms2 = component_power_integral(fitres.freq, comp,
                                                getattr(P, "FIT_FMIN", 0.05),
                                                getattr(P, "FIT_FMAX", 64.0))
                rms  = float(np.sqrt(max(rms2, 0.0)))
            except Exception:
                rms2 = rms = np.nan
            rms2s.append(rms2)
            rmss.append(rms)

        out[f"{band_label}_fit_qpo_nu0s_hz"]  = ";".join(f"{float(q['qpo_nu0_hz']):.6g}"  for q in qpos)
        out[f"{band_label}_fit_qpo_fwhms_hz"] = ";".join(f"{float(q['qpo_fwhm_hz']):.6g}" for q in qpos)
        out[f"{band_label}_fit_qpo_Qs"]        = ";".join(f"{float(q['qpo_Q']):.6g}"       for q in qpos)
        out[f"{band_label}_fit_qpo_rms2s"]     = ";".join("" if not np.isfinite(x) else f"{float(x):.6g}" for x in rms2s)
        out[f"{band_label}_fit_qpo_rmss"]      = ";".join("" if not np.isfinite(x) else f"{float(x):.6g}" for x in rmss)

        if np.isfinite(rms2s[0]):
            out[f"{band_label}_fit_qpo_rms2"] = float(rms2s[0])
            out[f"{band_label}_fit_qpo_rms"]  = float(rmss[0])

    return fitres, out


# ============================================================================
# Main per-obsid analysis
# ============================================================================

def analyze_obsid(obsid: str, evt_path: str, obsid_seed_offset: int = 0) -> dict:
    outdir_this = obsid_outdir(obsid)
    mjd_mid, iso_mid = _evt_time_mjd_mid(evt_path)

    ev  = EventList.read(evt_path)
    dt  = getattr(P, "DT", 0.0078125)
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

    # ---- Diagnostic peak finder ----
    def _peak(pds_cand, m_cand):
        return find_qpo_peak_whitened(
            pds_cand.freq, pds_cand.power,
            fmin=getattr(P, "FMIN", 0.05),
            fmax=getattr(P, "FMAX", 10.0),
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

    # ---- RMS metrics ----
    broad_rms_full, broad_err_full, qpo_rms_full, qpo_err_full = _compute_rms_metrics(pds_full_fit, peak_full)
    broad_rms_soft = broad_err_soft = qpo_rms_soft = qpo_err_soft = np.nan
    broad_rms_hard = broad_err_hard = qpo_rms_hard = qpo_err_hard = np.nan
    if soft_ok:
        broad_rms_soft, broad_err_soft, qpo_rms_soft, qpo_err_soft = _compute_rms_metrics(pds_soft_fit, peak_soft)
    if hard_ok:
        broad_rms_hard, broad_err_hard, qpo_rms_hard, qpo_err_hard = _compute_rms_metrics(pds_hard_fit, peak_hard)

    # ---- Fitting ----
    fit_full_res = fit_soft_res = fit_hard_res = None
    fit_full_out = fit_soft_out = fit_hard_out = {}

    # Seed variables for cross-band (need to exist even if bands not computed)
    seed_s = seed_h = None

    if getattr(P, "DO_FIT", False):
        shared_seed = float(peak_full) if (peak_full is not None and np.isfinite(peak_full) and peak_full > 0) else None

        # ---- Pass 1: fit each band independently ----
        fit_full_res, fit_full_out = _fit_one_band(
            obsid=obsid, band_label="full",
            pds_fit=pds_full_fit, pds_cand=pds_full_cand,
            seed_peak_hz=shared_seed, obsid_seed_offset=obsid_seed_offset,
        )
        if soft_ok:
            seed_s = shared_seed or (float(peak_soft) if peak_soft and np.isfinite(peak_soft) else None)
            fit_soft_res, fit_soft_out = _fit_one_band(
                obsid=obsid, band_label="soft",
                pds_fit=pds_soft_fit, pds_cand=pds_soft_cand,
                seed_peak_hz=seed_s, obsid_seed_offset=obsid_seed_offset,
            )
        if hard_ok:
            seed_h = shared_seed or (float(peak_hard) if peak_hard and np.isfinite(peak_hard) else None)
            fit_hard_res, fit_hard_out = _fit_one_band(
                obsid=obsid, band_label="hard",
                pds_fit=pds_hard_fit, pds_cand=pds_hard_cand,
                seed_peak_hz=seed_h, obsid_seed_offset=obsid_seed_offset,
            )

        # ---- Pass 2: cross-band reseeding ----
        if getattr(P, "DO_CROSS_BAND_RESEED", True):
            xb_thr = float(getattr(P, "CROSS_BAND_RESEED_RCHI_BAD",
                                   getattr(P, "FIT_RCHI_MAX", 1.5)))

            def _band_rchi(fres) -> float:
                if fres is None or not getattr(fres, "ok", False):
                    return np.inf
                r = getattr(fres, "rchi2", np.nan)
                return float(r) if np.isfinite(r) else np.inf

            # ---- Per-QPO quality collection ----
            # A QPO found in a band with bad overall rchi2 may still be a real
            # detection (one bad continuum component is dragging the rchi2).
            # Accept any QPO that already passed the QPO_MIN_Q gate AND has
            # significant integrated power (area > 1e-4 frac-rms² ~ 1% rms).
            QPO_AREA_QUALITY = float(getattr(P, "CROSS_BAND_QPO_AREA_MIN", 1e-4))

            def _collect_quality_qpos(fres) -> list[float]:
                """Collect QPO frequencies that are individually well-constrained."""
                if fres is None or not getattr(fres, "ok", False):
                    return []
                # qpo_list_from_fit already filters by QPO_MIN_Q via _qpo_list_from_fit
                out = []
                for q in _qpo_list_from_fit(fres):
                    area = float(q.get("qpo_area", 0))
                    nu   = float(q.get("qpo_nu0_hz", np.nan))
                    if np.isfinite(nu) and area >= QPO_AREA_QUALITY:
                        out.append(nu)
                return out

            cross_seeds: list[float] = []
            seen: set[float] = set()
            for nu in (_collect_quality_qpos(fit_full_res)
                       + _collect_quality_qpos(fit_soft_res)
                       + _collect_quality_qpos(fit_hard_res)):
                if any(abs(nu - s) < 0.05 for s in seen):
                    continue
                seen.add(nu)
                cross_seeds.append(nu)

            # ---- FALLBACK: when no quality QPOs anywhere, use diagnostic peaks ----
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
                    ("full", fit_full_res, fit_full_out,
                     pds_full_fit,  pds_full_cand,  shared_seed),
                    ("soft", fit_soft_res, fit_soft_out,
                     pds_soft_fit,  pds_soft_cand,  seed_s if soft_ok else None),
                    ("hard", fit_hard_res, fit_hard_out,
                     pds_hard_fit,  pds_hard_cand,  seed_h if hard_ok else None),
                ]

                new_full = new_soft = new_hard = None

                for label, fres_cur, out_cur, pds_f, pds_c, spk in band_specs:
                    if pds_f is None or pds_c is None:
                        continue

                    cur_qpos = _qpo_list_from_fit(fres_cur)
                    cur_freqs = [float(q["qpo_nu0_hz"]) for q in cur_qpos]
                    cur_n     = len(cur_qpos)
                    cur_rchi  = _band_rchi(fres_cur)

                    # Determine missing seeds — frequencies in the cross-band
                    # collection that this band has not found locally.
                    missing = [
                        s for s in cross_seeds
                        if not any(abs(s - g) < 0.05 for g in cur_freqs)
                    ]

                    # ---- Reseed trigger ----
                    # 1. Bad rchi2 — original criterion
                    # 2. QPO-count disagreement — even if rchi2 is acceptable,
                    #    re-fit if another band found a QPO this one missed.
                    rchi_bad     = cur_rchi > xb_thr
                    count_short  = len(missing) > 0
                    if not (rchi_bad or count_short):
                        continue
                    if not missing:
                        continue   # nothing to inject

                    fres_new, out_new = _fit_one_band(
                        obsid=obsid,
                        band_label=label,
                        pds_fit=pds_f,
                        pds_cand=pds_c,
                        seed_peak_hz=spk,
                        obsid_seed_offset=obsid_seed_offset,
                        forced_qpo_seeds=missing,
                    )

                    if fres_new is None:
                        continue

                    new_qpos = _qpo_list_from_fit(fres_new)
                    new_freqs = [float(q["qpo_nu0_hz"]) for q in new_qpos]
                    new_n   = len(new_qpos)
                    new_rchi = _band_rchi(fres_new)

                    # ---- Acceptance criteria ----
                    # Accept if EITHER:
                    #   (a) rchi2 strictly improves (original criterion), OR
                    #   (b) gained ≥1 missing QPO without significantly worse rchi2.
                    # The (b) clause handles the case where the band already
                    # had acceptable rchi2 but was missing a real QPO that
                    # another band found.
                    gained_any = any(
                        any(abs(s - nf) < 0.05 for nf in new_freqs)
                        for s in missing
                    )
                    rchi_improved = new_rchi < cur_rchi - 0.01
                    rchi_not_worse = new_rchi <= max(cur_rchi + 0.10, xb_thr + 0.10)

                    accept = rchi_improved or (gained_any and rchi_not_worse)

                    if accept:
                        if label == "full":
                            new_full = (fres_new, out_new)
                        elif label == "soft":
                            new_soft = (fres_new, out_new)
                        else:
                            new_hard = (fres_new, out_new)

                if new_full:
                    fit_full_res, fit_full_out = new_full
                    fit_full_out["full_cross_reseeded"] = True
                if new_soft:
                    fit_soft_res, fit_soft_out = new_soft
                    fit_soft_out["soft_cross_reseeded"] = True
                if new_hard:
                    fit_hard_res, fit_hard_out = new_hard
                    fit_hard_out["hard_cross_reseeded"] = True

        # Save final fit results to struct.  Placed here — after cross-band
        # reseeding — so the struct always reflects the same fitres objects
        # that are used for the plot and CSV output.
        save_fit_struct(fit_full_res, obsid, "full", mjd=mjd_mid)
        save_fit_struct(fit_soft_res, obsid, "soft", mjd=mjd_mid)
        save_fit_struct(fit_hard_res, obsid, "hard", mjd=mjd_mid)

        if getattr(P, "SAVE_FIT_PLOTS", True):
            band_items = [
                {"label": "Full",         "pds_fit": pds_full_fit, "fitres": fit_full_res, "seed_peak": shared_seed},
                {"label": "Soft 0.3-2 keV", "pds_fit": pds_soft_fit, "fitres": fit_soft_res,
                 "seed_peak": shared_seed or (float(peak_soft) if peak_soft and np.isfinite(peak_soft) else np.nan)},
                {"label": "Hard 2-10 keV",  "pds_fit": pds_hard_fit, "fitres": fit_hard_res,
                 "seed_peak": shared_seed or (float(peak_hard) if peak_hard and np.isfinite(peak_hard) else np.nan)},
            ]
            save_threeband_fit_overlay_plot(obsid, outdir_this, band_items)

    out = {
        "mjd_mid": float(mjd_mid)  if (mjd_mid  is not None and np.isfinite(mjd_mid))  else "",
        "iso_mid": iso_mid          if iso_mid    is not None                            else "",
        "tseg_s":       float(lc_full.tseg),
        "dt_s":         float(lc_full.dt),
        "mean_rate_cps":float(lc_full.meanrate),
        "segment_size_s":float(seg),
        "peak_f_hz_full": float(peak_full) if np.isfinite(peak_full) else "",
        "peak_f_hz_soft": float(peak_soft) if (peak_soft is not None and np.isfinite(peak_soft)) else "",
        "peak_f_hz_hard": float(peak_hard) if (peak_hard is not None and np.isfinite(peak_hard)) else "",
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
    base_dir = getattr(P, "BASE_DIR", ".")
    source   = getattr(P, "SOURCE",   "")
    evt_path = build_evt_path(base_dir, source, obsid)

    suppress = bool(getattr(P, "QUIET", False))
    seed_offset = hash(obsid) & 0xFFFF

    row = {"obsid": obsid, "evt_path": evt_path, "status": "", "error": ""}

    try:
        if not os.path.exists(evt_path):
            raise FileNotFoundError(f"Missing event file: {evt_path}")

        with _suppress_context(suppress):
            res = analyze_obsid(obsid, evt_path, obsid_seed_offset=seed_offset)

        row.update(res)
        row["status"] = "OK"
        row["error"]  = ""

    except Exception as exc:
        row["status"] = "FAIL"
        row["error"]  = str(exc)

    if getattr(P, "ONE_LINE_SUMMARY", True):
        _print_one_line_summary(obsid, row)
    elif row["status"] != "OK":
        print(f"[FAIL] {obsid}  {row['error']}")

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
        "full_comp_nu0s", "full_comp_fwhms", "full_comp_amps",
        "soft_fit_ok", "soft_fit_nlor", "soft_fit_const", "soft_fit_rchi2", "soft_fit_red_dev", "soft_fit_msg",
        "soft_fit_qpo_nu0_hz", "soft_fit_qpo_fwhm_hz", "soft_fit_qpo_Q", "soft_fit_qpo_rms2", "soft_fit_qpo_rms",
        "soft_comp_nu0s", "soft_comp_fwhms", "soft_comp_amps",
        "hard_fit_ok", "hard_fit_nlor", "hard_fit_const", "hard_fit_rchi2", "hard_fit_red_dev", "hard_fit_msg",
        "hard_fit_qpo_nu0_hz", "hard_fit_qpo_fwhm_hz", "hard_fit_qpo_Q", "hard_fit_qpo_rms2", "hard_fit_qpo_rms",
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
                                evt_path=build_evt_path(getattr(P, "BASE_DIR", "."), getattr(P, "SOURCE", ""), obsid),
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
