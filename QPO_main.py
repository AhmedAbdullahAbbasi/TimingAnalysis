#!/usr/bin/env python3
# QPO_main.py


import os
import csv
import numpy as np
import warnings
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stingray.events import EventList
from stingray.powerspectrum import AveragedPowerspectrum

from QPO_fit import (
    fit_lorentzian_family,
    extract_qpo_params,
    component_power_integral,
    lorentz,
)

import QPO_Parameter as P

import scipy.signal
from scipy.ndimage import median_filter

from astropy.io import fits
from astropy.time import Time

print("[INFO] Using QPO_Parameter from:", getattr(P, "__file__", "UNKNOWN"))


# ============================================================================
# Logging / suppression
# ============================================================================

def _suppress_everything(enabled: bool):

    if not enabled:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
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
    stage = ""
    try:
        stage = str(getattr(fitres, "meta", {}).get("stage", "")).lower()
    except Exception:
        stage = ""
    return 3 if "cont3" in stage else 2


def _qpo_found_from_fit(fitres) -> tuple[bool, float, float]:
    if fitres is None or not getattr(fitres, "ok", False):
        return (False, np.nan, np.nan)
    try:
        qpo = extract_qpo_params(
            fitres,
            qpo_fmin=getattr(P, "QPO_FMIN", 0.1),
            qpo_fmax=getattr(P, "QPO_FMAX", 10.0),
            qmin=getattr(P, "QPO_MIN_Q", 3.0),
        )
        if qpo is None:
            return (False, np.nan, np.nan)
        return (True, float(qpo["qpo_nu0_hz"]), float(qpo["qpo_Q"]))
    except Exception:
        return (False, np.nan, np.nan)


def _fmt_band(label: str, fitres) -> str:
    cont_n = _count_continuum_components(fitres)
    found, nu, Q = _qpo_found_from_fit(fitres)

    rchi = getattr(fitres, "rchi2", np.nan) if fitres is not None else np.nan
    rchi_s = f"{rchi:.2f}" if np.isfinite(rchi) else "nan"

    flag = ""
    rchi_max = getattr(P, "FIT_RCHI_MAX", None)
    if rchi_max is not None and np.isfinite(rchi) and (rchi > float(rchi_max)):
        flag = " (rchi above criteria)"

    if found and getattr(P, "SHOW_QPO_DETAILS", True):
        return f"{label}:cont={cont_n} QPO=Y rchi={rchi_s} nu={nu:.3f} Q={Q:.2f}{flag}"
    return f"{label}:cont={cont_n} QPO={'Y' if found else 'N'} rchi={rchi_s}{flag}"


def _print_one_line_summary(obsid: str, row: dict):
    status = row.get("status", "")
    err = row.get("error", "")
    mjd = row.get("mjd_mid", "")
    mjd_s = ""
    try:
        if isinstance(mjd, (int, float)) and np.isfinite(mjd):
            mjd_s = f"{float(mjd):.5f}"
    except Exception:
        mjd_s = ""

    head = f"{obsid}"
    if mjd_s:
        head += f"  mjd={mjd_s}"

    if status and status != "OK":
        if err:
            print(f"{head}  {status}  {err}")
        else:
            print(f"{head}  {status}")
        return

    fit_full = row.get("_fit_full_res_obj", None)
    fit_soft = row.get("_fit_soft_res_obj", None)
    fit_hard = row.get("_fit_hard_res_obj", None)

    parts = [head, _fmt_band("F", fit_full)]
    if getattr(P, "DO_ENERGY_BANDS", False):
        parts.append(_fmt_band("S", fit_soft))
        parts.append(_fmt_band("H", fit_hard))

    print("  |  ".join(parts))


# ============================================================================
# Basic filesystem / IO
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
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            obsids.append(s)
    return obsids


def build_evt_path(base_dir: str, source: str, obsid: str) -> str:
    return os.path.join(base_dir, source, obsid, f"ni{obsid}_0mpu7_cl.evt")


# ============================================================================
# Energy filtering
# ============================================================================

def kev_to_pi(kev: float) -> int:
    return int(np.round(kev * getattr(P, "PI_PER_KEV", 100.0)))


def filter_events_by_energy(ev: EventList, band_kev: tuple[float, float]) -> EventList:
    emin, emax = band_kev
    pi_min = kev_to_pi(emin)
    pi_max = kev_to_pi(emax)

    if ("pi" not in ev.array_attrs()) and (not hasattr(ev, "pi")):
        raise ValueError("EventList missing PI column/attribute; cannot energy-filter.")

    pi = ev.pi
    m = (pi >= pi_min) & (pi < pi_max)

    if np.sum(m) < 10:
        raise ValueError(f"Too few events in band {band_kev} keV (PI {pi_min}-{pi_max}).")

    gti = getattr(ev, "gti", None) if getattr(P, "USE_GTIS", True) else None
    ev_filt = EventList(time=ev.time[m], gti=gti)
    ev_filt.pi = pi[m]
    return ev_filt


# ============================================================================
# Rebinning helpers
# ============================================================================

def _safe_m_avg_from_pds(pds) -> int:
    m_attr = getattr(pds, "m", 1)
    try:
        arr = np.asarray(m_attr, float)
        v = float(arr) if arr.ndim == 0 else float(np.nanmedian(arr))
        if (not np.isfinite(v)) or v < 1:
            return 1
        return int(v)
    except Exception:
        return 1


def _rebin_pds(pds: AveragedPowerspectrum, mode: str, log_f: float, factor: float, df_hz):
    mode = str(mode).lower()
    if mode == "log":
        return pds.rebin_log(f=float(log_f))
    if df_hz is not None:
        return pds.rebin(df=float(df_hz))
    return pds.rebin(f=float(factor))


def maybe_rebin_pds_fit(pds: AveragedPowerspectrum) -> AveragedPowerspectrum:
    if not getattr(P, "DO_REBIN", False):
        return pds
    return _rebin_pds(
        pds,
        mode=getattr(P, "REBIN_MODE", "log"),
        log_f=getattr(P, "REBIN_LOG_F", 0.02),
        factor=getattr(P, "REBIN_FACTOR", 4.0),
        df_hz=getattr(P, "REBIN_DF_HZ", None) if hasattr(P, "REBIN_DF_HZ") else None,
    )


def maybe_rebin_pds_candidate(pds: AveragedPowerspectrum) -> AveragedPowerspectrum:
    if not getattr(P, "DO_CANDIDATE_LIGHT_REBIN", True):
        return maybe_rebin_pds_fit(pds)

    if (not getattr(P, "DO_REBIN", False)) and (not getattr(P, "DO_REBIN_CAND_WHEN_FIT_OFF", True)):
        return pds

    return _rebin_pds(
        pds,
        mode=getattr(P, "CAND_REBIN_MODE", "log"),
        log_f=getattr(P, "CAND_REBIN_LOG_F", 0.008),
        factor=getattr(P, "CAND_REBIN_FACTOR", 2.0),
        df_hz=getattr(P, "CAND_REBIN_DF_HZ", None) if hasattr(P, "CAND_REBIN_DF_HZ") else None,
    )


# ============================================================================
# Fast event time stamp from FITS header
# ============================================================================

def _evt_time_mjd_mid(evt_path: str) -> tuple[float | None, str | None]:
    try:
        with fits.open(evt_path, memmap=True) as hdul:
            h = hdul[1].header
            tstart = float(h.get("TSTART"))
            tstop = float(h.get("TSTOP"))
            tmid = 0.5 * (tstart + tstop)

            mjdrefi = float(h.get("MJDREFI", 0.0))
            mjdreff = float(h.get("MJDREFF", 0.0))
            mjdref = mjdrefi + mjdreff

            timezero = float(h.get("TIMEZERO", 0.0))
            tmid = tmid + timezero

            mjd_mid = float(mjdref + (tmid / 86400.0))
            iso_mid = Time(mjd_mid, format="mjd", scale="tt").isot
            return mjd_mid, iso_mid
    except Exception:
        return None, None


# ============================================================================
# A2 sigma-excess peak finder
# ============================================================================

def _rolling_median_fast(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    w = int(w)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    return median_filter(y, size=w, mode="nearest")


def _estimate_sigma_local(cont: np.ndarray, p: np.ndarray, m_eff: int, mode: str = "cont") -> np.ndarray:
    me = max(1, int(m_eff))
    cont = np.asarray(cont, float)
    p = np.asarray(p, float)

    if str(mode).lower().strip() == "p":
        base = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    else:
        base = np.where(np.isfinite(cont) & (cont > 0), cont, np.nan)

    med = np.nanmedian(base[np.isfinite(base)])
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    base = np.where(np.isfinite(base) & (base > 0), base, med)
    sigma = base / np.sqrt(float(me))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, med / np.sqrt(float(me)))
    return sigma


def find_qpo_peak_whitened(
    freq, power,
    fmin=0.05, fmax=10.0,
    smooth_hz=0.5,
    ignore_below=0.1,
    min_width_bins=7,
    m_eff: int = 1,
    require_ksigma: float | None = 5.0,
    prominence_sigma: float = 1.0,
    min_sep_hz: float = 0.15,
    sigma_mode: str = "cont",
    prefer: str = "prominence",
):
    """
    A2 diagnostic peak finder:
      - cont = rolling median
      - sigma_local ~ cont/sqrt(m_eff)
      - find peaks in z = (P - cont)/sigma_local using SciPy
      - return best peak frequency
    """
    f = np.asarray(freq, float)
    p = np.asarray(power, float)

    m = np.isfinite(f) & np.isfinite(p) & (p > 0) & (f >= fmin) & (f <= fmax)
    if ignore_below is not None:
        m &= (f >= ignore_below)

    f = f[m]
    p = p[m]

    if f.size < 10:
        return float(f[np.argmax(p)]) if f.size else np.nan

    df = float(np.median(np.diff(f)))
    if (not np.isfinite(df)) or df <= 0:
        return float(f[np.argmax(p)])

    w = int(np.round(float(smooth_hz) / df))
    w = max(int(min_width_bins), w)
    if w % 2 == 0:
        w += 1
    if w >= p.size:
        w = p.size - 1 if (p.size % 2 == 0) else p.size
        if w < 3:
            return float(f[np.argmax(p)])

    cont = _rolling_median_fast(p, w)
    good = np.isfinite(cont) & (cont > 0)
    if not np.any(good):
        return float(f[np.argmax(p)])
    cont = np.where(good, cont, np.nanmedian(cont[good]))

    sigma = _estimate_sigma_local(cont, p, m_eff=int(m_eff), mode=str(sigma_mode))
    z = (p - cont) / sigma

    distance = int(max(1, np.round(float(min_sep_hz) / df)))

    height = None
    if require_ksigma is not None and np.isfinite(require_ksigma) and float(require_ksigma) > 0:
        height = float(require_ksigma)

    peaks, props = scipy.signal.find_peaks(
        z,
        height=height,
        prominence=float(prominence_sigma),
        distance=distance,
    )
    if peaks.size == 0:
        j = int(np.nanargmax(z))
        return float(f[j]) if np.isfinite(z[j]) else float(f[np.argmax(p)])

    z_prom = np.asarray(props.get("prominences", np.zeros_like(peaks, float)), float)
    z_height = np.asarray(props.get("peak_heights", z[peaks]), float)

    score = z_prom if str(prefer).lower().strip() != "height" else z_height
    best = int(peaks[int(np.argmax(score))])
    return float(f[best])


# ============================================================================
# Plot y-limits helpers
# ============================================================================

def _auto_ylim_from_arrays(
    arrays,
    *,
    pad_top=1.35,
    pad_bottom=0.85,
    ymin_floor=1e-6,
    include_zero=False,
):
    vals = []
    for a in arrays:
        if a is None:
            continue
        x = np.asarray(a, float).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return None

    v = np.concatenate(vals)
    vpos = v[v > 0]
    if vpos.size > 0:
        lo = float(np.nanmin(vpos))
        hi = float(np.nanmax(vpos))
        lo = max(lo, float(ymin_floor))
        if hi <= lo:
            hi = lo * 10.0
        lo = max(float(ymin_floor), lo * float(pad_bottom))
        hi = hi * float(pad_top)
        return lo, hi

    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    if include_zero:
        lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo = lo - 0.05 * span
    hi = hi + 0.10 * span
    return lo, hi


def _ylim_cap_by_model(
    data_arr,
    model_arr,
    *,
    ymin_floor=1e-6,
    cap_factor=2.0,
    pad_bottom=0.85,
):
    if model_arr is None:
        return None
    m = np.asarray(model_arr, float).ravel()
    m = m[np.isfinite(m) & (m > 0)]
    if m.size == 0:
        return None

    ymax = float(cap_factor * np.nanmax(m))
    ymax = max(ymax, 10.0 * ymin_floor)

    vals = []
    for a in (data_arr, model_arr):
        if a is None:
            continue
        x = np.asarray(a, float).ravel()
        x = x[np.isfinite(x) & (x > 0)]
        if x.size:
            vals.append(x)
    if not vals:
        ymin = ymin_floor
    else:
        ymin = float(np.nanmin(np.concatenate(vals)))
        ymin = max(ymin_floor, ymin * float(pad_bottom))

    if ymax <= ymin:
        ymax = ymin * 10.0

    return ymin, ymax


# ============================================================================
# Plotting: 3-band overlay + residuals
# ============================================================================

def save_threeband_fit_overlay_plot(obsid: str, outdir_obsid: str, band_items: list[dict]):
    outpath = os.path.join(outdir_obsid, f"{obsid}_fits_full_soft_hard.png")
    if os.path.exists(outpath) and (not getattr(P, "CLOBBER", False)):
        return

    fit_fmin = float(getattr(P, "FIT_FMIN", 0.05))
    fit_fmax = float(getattr(P, "FIT_FMAX", 64.0))
    ymin_floor = float(getattr(P, "PLOT_YMIN", 1e-6))
    dpi = int(getattr(P, "PLOT_DPI", 150))

    def _interp_to(x_src, y_src, x_tgt):
        x_src = np.asarray(x_src, float)
        y_src = np.asarray(y_src, float)
        x_tgt = np.asarray(x_tgt, float)
        m = np.isfinite(x_src) & np.isfinite(y_src)
        if np.sum(m) < 2:
            return np.full_like(x_tgt, np.nan, dtype=float)
        xs = x_src[m]
        ys = y_src[m]
        o = np.argsort(xs)
        xs = xs[o]
        ys = ys[o]
        return np.interp(x_tgt, xs, ys, left=np.nan, right=np.nan)

    def _resid_ylim(resid):
        r = np.asarray(resid, float)
        r = r[np.isfinite(r)]
        if r.size < 5:
            return (-5.0, 5.0)
        p1, p99 = np.nanpercentile(r, [1, 99])
        lo = float(min(p1, -3.0))
        hi = float(max(p99, 3.0))
        if hi - lo < 2.0:
            lo, hi = (-2.5, 2.5)
        lo = max(lo, -10.0)
        hi = min(hi, 10.0)
        return (lo, hi)

    fig, axes = plt.subplots(
        nrows=6, ncols=1,
        figsize=(9.5, 14.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 3.0, 1.2, 3.0, 1.2]},
    )

    for i, item in enumerate(band_items):
        ax_top = axes[2 * i]
        ax_res = axes[2 * i + 1]

        label = item.get("label", "Band")
        pds_fit = item.get("pds_fit", None)
        fitres = item.get("fitres", None)

        if pds_fit is None:
            ax_top.text(0.5, 0.5, f"{label}: No data", ha="center", va="center")
            ax_top.set_axis_off()
            ax_res.set_axis_off()
            continue

        m0 = (pds_fit.freq >= fit_fmin) & (pds_fit.freq <= fit_fmax)
        f0 = np.asarray(pds_fit.freq[m0], float)
        p0 = np.asarray(pds_fit.power[m0], float)
        e0 = np.asarray(getattr(pds_fit, "power_err", np.full_like(p0, np.nan))[m0], float)

        ax_top.loglog(f0, p0, lw=1, label="PDS")

        model0 = None
        qpo_nu = None
        qpo_Q = None
        qpo_fwhm = None

        if fitres is not None and bool(getattr(fitres, "ok", False)):
            ff = np.asarray(getattr(fitres, "freq", f0), float)
            model = np.asarray(getattr(fitres, "model", None), float) if getattr(fitres, "model", None) is not None else None

            if model is not None and np.size(model) == np.size(ff):
                model0 = _interp_to(ff, model, f0)
                rchi = getattr(fitres, "rchi2", np.nan)
                nlor = getattr(fitres, "nlor", "")
                cval = float(getattr(fitres, "const", 0.0) or 0.0)

                ax_top.loglog(f0, model0, lw=2, label="Model")

                if cval > 0 and np.isfinite(cval):
                    ax_top.hlines(
                        cval, xmin=fit_fmin, xmax=fit_fmax,
                        linestyles="--", linewidth=1.2, alpha=0.7,
                        label="Const",
                    )

                try:
                    for (nu0_i, fwhm_i, amp_i) in getattr(fitres, "pars", []):
                        comp = lorentz(f0, nu0_i, fwhm_i, amp_i)
                        ax_top.loglog(f0, comp, lw=1, alpha=0.5)
                except Exception:
                    pass

                try:
                    qpo = extract_qpo_params(
                        fitres,
                        qpo_fmin=getattr(P, "QPO_FMIN", 0.1),
                        qpo_fmax=getattr(P, "QPO_FMAX", 10.0),
                        qmin=getattr(P, "QPO_MIN_Q", 3.0),
                    )
                    if qpo is not None:
                        qpo_nu = float(qpo["qpo_nu0_hz"])
                        qpo_fwhm = float(qpo["qpo_fwhm_hz"])
                        qpo_Q = float(qpo["qpo_Q"])
                        if np.isfinite(qpo_nu):
                            ax_top.axvline(qpo_nu, linestyle=":", alpha=0.9, label=f"QPO {qpo_nu:.3f} Hz")
                            if getattr(P, "DO_HARMONIC_SEARCH", True):
                                harm = 2.0 * qpo_nu
                                if fit_fmin <= harm <= fit_fmax:
                                    ax_top.axvline(harm, linestyle="--", alpha=0.6, label=f"2× {harm:.3f} Hz")
                except Exception:
                    pass

                ann = [f"{label}", f"Nlor={nlor}", f"rχ²={rchi:.2f}", f"C={cval:.3g}"]
                if qpo_nu is not None and np.isfinite(qpo_nu):
                    ann += [f"QPO={qpo_nu:.3f} Hz", f"Q={qpo_Q:.2f}"]
                ax_top.text(0.02, 0.95, " | ".join(ann), transform=ax_top.transAxes,
                            va="top", ha="left", fontsize=9)

        else:
            ax_top.text(0.02, 0.95, label, transform=ax_top.transAxes, va="top", ha="left", fontsize=9)

        yl = _ylim_cap_by_model(p0, model0, cap_factor=2.0, ymin_floor=ymin_floor)
        if yl is None:
            yl = _auto_ylim_from_arrays(
                [p0],
                pad_top=float(getattr(P, "PLOT_YMAX_PAD", 1.35)),
                ymin_floor=ymin_floor,
            )
        if yl is not None:
            ax_top.set_ylim(*yl)

        ax_top.set_ylabel("Power (frac-rms$^2$/Hz)")
        ax_top.legend(fontsize=8, loc="best")

        ax_res.set_xscale("log")
        ax_res.axhline(0.0, lw=1, alpha=0.6)
        ax_res.axhline(+3.0, lw=0.8, alpha=0.35, linestyle="--")
        ax_res.axhline(-3.0, lw=0.8, alpha=0.35, linestyle="--")

        if model0 is not None and np.any(np.isfinite(model0)):
            good = np.isfinite(p0) & np.isfinite(model0) & np.isfinite(e0) & (e0 > 0)
            if np.sum(good) >= 5:
                resid = np.full_like(p0, np.nan, dtype=float)
                resid[good] = (p0[good] - model0[good]) / e0[good]
                ax_res.plot(f0, resid, lw=0.9)
                ax_res.set_ylabel("(P−M)/σ", fontsize=9)
                ax_res.set_ylim(*_resid_ylim(resid))
            else:
                good2 = np.isfinite(p0) & np.isfinite(model0) & (model0 > 0)
                resid = np.full_like(p0, np.nan, dtype=float)
                if np.sum(good2) >= 5:
                    resid[good2] = (p0[good2] / model0[good2]) - 1.0
                    ax_res.plot(f0, resid, lw=0.9)
                    ax_res.set_ylabel("P/M−1", fontsize=9)
                    ax_res.set_ylim(*_resid_ylim(resid))
                else:
                    ax_res.text(0.5, 0.5, "No residuals (bad model)", ha="center", va="center")
                    ax_res.set_ylim(-5, 5)
        else:
            ax_res.text(0.5, 0.5, "No residuals (no fit)", ha="center", va="center")
            ax_res.set_ylim(-5, 5)

        if qpo_nu is not None and np.isfinite(qpo_nu):
            ax_res.axvline(qpo_nu, linestyle=":", alpha=0.9)

        ax_res.set_xlabel("Frequency (Hz)")

    fig.suptitle(f"{_src_label()} {obsid} | Fits + Residuals (Full/Soft/Hard)", fontsize=12)
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


# ============================================================================
# RMS + fitting wrappers
# ============================================================================

def _compute_rms_metrics(pds_fit, peak_hz):
    broad_band = getattr(P, "BROAD_RMS_BAND", (0.1, 30.0))
    broad_rms, broad_err = pds_fit.compute_rms(*broad_band)

    eta = getattr(P, "QPO_BW_FRAC", 0.10)
    bw_min = getattr(P, "QPO_BW_MIN", 0.10)
    bw_max = getattr(P, "QPO_BW_MAX", 2.00)

    if np.isfinite(peak_hz) and peak_hz > 0:
        bw = float(np.clip(eta * peak_hz, bw_min, bw_max))
        lo = max(0.01, peak_hz - bw)
        hi = peak_hz + bw
        qpo_rms, qpo_err = pds_fit.compute_rms(lo, hi)
    else:
        qpo_rms, qpo_err = (np.nan, np.nan)

    return broad_rms, broad_err, qpo_rms, qpo_err


def _stringify_pars_list(pars: np.ndarray) -> tuple[str, str, str]:
    try:
        arr = np.asarray(pars, float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            return "", "", ""
        nu0s = ";".join([f"{x:.8g}" for x in arr[:, 0]])
        fwhms = ";".join([f"{x:.8g}" for x in arr[:, 1]])
        amps = ";".join([f"{x:.8g}" for x in arr[:, 2]])
        return nu0s, fwhms, amps
    except Exception:
        return "", "", ""


def _fit_one_band(
    *,
    obsid: str,
    band_label: str,
    pds_fit: AveragedPowerspectrum,
    pds_cand: AveragedPowerspectrum,
    seed_peak_hz: float | None,
):
    out = {
        f"{band_label}_fit_ok": False,
        f"{band_label}_fit_nlor": "",
        f"{band_label}_fit_const": "",
        f"{band_label}_fit_rchi2": "",
        f"{band_label}_fit_msg": "",
        f"{band_label}_fit_qpo_nu0_hz": "",
        f"{band_label}_fit_qpo_fwhm_hz": "",
        f"{band_label}_fit_qpo_Q": "",
        f"{band_label}_fit_qpo_rms2": "",
        f"{band_label}_fit_qpo_rms": "",
        f"{band_label}_comp_nu0s": "",
        f"{band_label}_comp_fwhms": "",
        f"{band_label}_comp_amps": "",
    }

    if pds_fit is None or pds_cand is None:
        out[f"{band_label}_fit_msg"] = "No PDS"
        return None, out

    m_avg_fit = _safe_m_avg_from_pds(pds_fit)
    m_avg_cand = _safe_m_avg_from_pds(pds_cand)  # candidate m

    # rchi "criteria" (used both for (a) CSV fit_ok flag and (b) cont3+qpo override trigger)
    rchi_max = getattr(P, "FIT_RCHI_MAX", None)
    rchi_override_threshold = float(rchi_max) if (rchi_max is not None and np.isfinite(rchi_max)) else None

    fit_kwargs = dict(
        m=m_avg_fit,
        cand_m_eff=m_avg_cand,

        fit_fmin=getattr(P, "FIT_FMIN", 0.05),
        fit_fmax=getattr(P, "FIT_FMAX", 64.0),
        cand_fmin=getattr(P, "CAND_FMIN", 0.05),
        cand_fmax=getattr(P, "CAND_FMAX", 10.0),

        include_const=getattr(P, "FIT_INCLUDE_CONST", True),
        const_seed_fmin=getattr(P, "FIT_CONST_SEED_FMIN", 30.0),

        # pass through (QPO_fit ignores harmonics for now, but accepts kwargs)
        include_harmonic=getattr(P, "DO_HARMONIC_SEARCH", True),

        smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
        prominence=getattr(P, "PEAK_PROMINENCE", 0.12),
        min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
        max_candidates=getattr(P, "PEAK_MAX_CANDIDATES", 4),

        cand_require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
        cand_sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),

        stage1_n_seeds=getattr(P, "FIT_STAGE1_N_SEEDS", 3),

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

        fitmethod=getattr(P, "FIT_METHOD", "Powell"),
        rchi_target=getattr(P, "FIT_RCHI_TARGET", 1.3),

        # continuum drift
        cont_x0_max_hz=getattr(P, "FIT_CONT_X0_MAX_HZ", None),
        cont_fwhm_lim=(getattr(P, "FIT_CONT_FWHM_MIN", 0.3), getattr(P, "FIT_CONT_FWHM_MAX", 64.0)),
        qpo_fwhm_lim=(getattr(P, "FIT_QPO_FWHM_MIN", 0.03), getattr(P, "FIT_QPO_FWHM_MAX", 5.0)),
        harm_fwhm_lim=(getattr(P, "FIT_HARM_FWHM_MIN", 0.03), getattr(P, "FIT_HARM_FWHM_MAX", 8.0)),

        cont_amp_factor=getattr(P, "FIT_CONT_AMP_FACTOR", 10.0),
        qpo_amp_factor=getattr(P, "FIT_QPO_AMP_FACTOR", 5.0),
        harm_amp_factor=getattr(P, "FIT_HARM_AMP_FACTOR", 5.0),

        qpo_fwhm_frac=getattr(P, "FIT_QPO_FWHM_FRAC", 0.06),
        qpo_fwhm_min=getattr(P, "FIT_QPO_FWHM_MIN_ABS", 0.05),

        guard_enable=getattr(P, "FIT_GUARD_ENABLE", True),
        guard_overshoot_ksigma=getattr(P, "FIT_GUARD_OVERSHOOT_KSIGMA", 3.0),
        guard_overshoot_max_run_bins=getattr(P, "FIT_GUARD_OVERSHOOT_MAX_RUN_BINS", 6),
        guard_overshoot_max_frac=getattr(P, "FIT_GUARD_OVERSHOOT_MAX_FRAC", 0.08),
        guard_comp_local_amp_factor=getattr(P, "FIT_GUARD_COMP_LOCAL_AMP_FACTOR", 5.0),

        max_retries=getattr(P, "FIT_MAX_RETRIES", 5),

        cont_ic_criterion=getattr(P, "CONT_IC_CRITERION", "bic"),
        cont_ic_delta_min=getattr(P, "CONT_IC_DELTA_MIN", 10.0),
        qpo_ic_criterion=getattr(P, "QPO_IC_CRITERION", "aic"),
        qpo_ic_delta_min=getattr(P, "QPO_IC_DELTA_MIN", 2.0),

       
        rchi_override_enable=True,
        rchi_override_threshold=rchi_override_threshold,
    )

    if seed_peak_hz is not None and np.isfinite(seed_peak_hz) and seed_peak_hz > 0:
        fit_kwargs["seed_peak_hz"] = float(seed_peak_hz)

    fitres = fit_lorentzian_family(
        pds_fit.freq, pds_fit.power, pds_fit.power_err,
        cand_freq=pds_cand.freq,
        cand_power=pds_cand.power,
        **fit_kwargs,
    )

    out[f"{band_label}_fit_ok"] = bool(getattr(fitres, "ok", False))
    out[f"{band_label}_fit_nlor"] = int(getattr(fitres, "nlor", 0)) if getattr(fitres, "nlor", 0) else ""
    out[f"{band_label}_fit_const"] = float(getattr(fitres, "const", 0.0)) if np.isfinite(getattr(fitres, "const", 0.0)) else ""
    out[f"{band_label}_fit_msg"] = str(getattr(fitres, "message", ""))

    rchi = getattr(fitres, "rchi2", np.nan)
    out[f"{band_label}_fit_rchi2"] = float(rchi) if np.isfinite(rchi) else ""

    nu0s, fwhms, amps = _stringify_pars_list(getattr(fitres, "pars", np.empty((0, 3))))
    out[f"{band_label}_comp_nu0s"] = nu0s
    out[f"{band_label}_comp_fwhms"] = fwhms
    out[f"{band_label}_comp_amps"] = amps

    if hasattr(P, "FIT_RCHI_MAX") and np.isfinite(rchi):
        out[f"{band_label}_fit_ok"] = bool(out[f"{band_label}_fit_ok"]) and (rchi <= getattr(P, "FIT_RCHI_MAX"))

    try:
        qpo = extract_qpo_params(
            fitres,
            qpo_fmin=getattr(P, "QPO_FMIN", 0.1),
            qpo_fmax=getattr(P, "QPO_FMAX", 10.0),
            qmin=getattr(P, "QPO_MIN_Q", 3.0),
        )
    except Exception:
        qpo = None

    if qpo is not None:
        out[f"{band_label}_fit_qpo_nu0_hz"] = qpo["qpo_nu0_hz"]
        out[f"{band_label}_fit_qpo_fwhm_hz"] = qpo["qpo_fwhm_hz"]
        out[f"{band_label}_fit_qpo_Q"] = qpo["qpo_Q"]

        try:
            idx = qpo["qpo_index"]
            nu0_i, fwhm_i, amp_i = fitres.pars[idx]
            comp = lorentz(fitres.freq, nu0_i, fwhm_i, amp_i)
            rms2 = component_power_integral(
                fitres.freq, comp,
                getattr(P, "FIT_FMIN", 0.05),
                getattr(P, "FIT_FMAX", 64.0),
            )
            out[f"{band_label}_fit_qpo_rms2"] = float(rms2)
            out[f"{band_label}_fit_qpo_rms"] = float(np.sqrt(max(rms2, 0.0)))
        except Exception:
            pass

    return fitres, out


# ============================================================================
# Main per-obsid analysis
# ============================================================================

def analyze_obsid(obsid: str, evt_path: str) -> dict:
    outdir_this = obsid_outdir(obsid)

    mjd_mid, iso_mid = _evt_time_mjd_mid(evt_path)

    ev = EventList.read(evt_path)

    dt = getattr(P, "DT", 0.0078125)
    seg = getattr(P, "SEGMENT_SIZE", 64)

    def _make_pds_for_eventlist(ev_band: EventList):
        lc = ev_band.to_lc(dt=dt)
        pds_raw = AveragedPowerspectrum(lc, segment_size=seg, norm="frac")
        pds_cand = maybe_rebin_pds_candidate(pds_raw)
        pds_fit = maybe_rebin_pds_fit(pds_raw)
        return lc, pds_raw, pds_cand, pds_fit

    lc_full, pds_full_raw, pds_full_cand, pds_full_fit = _make_pds_for_eventlist(ev)

    soft_ok = hard_ok = False
    lc_soft = pds_soft_raw = pds_soft_cand = pds_soft_fit = None
    lc_hard = pds_hard_raw = pds_hard_cand = pds_hard_fit = None

    if getattr(P, "DO_ENERGY_BANDS", False):
        try:
            ev_soft = filter_events_by_energy(ev, getattr(P, "SOFT_BAND_KEV", (0.3, 2.0)))
            lc_soft, pds_soft_raw, pds_soft_cand, pds_soft_fit = _make_pds_for_eventlist(ev_soft)
            soft_ok = True
        except Exception:
            soft_ok = False

        try:
            ev_hard = filter_events_by_energy(ev, getattr(P, "HARD_BAND_KEV", (2.0, 10.0)))
            lc_hard, pds_hard_raw, pds_hard_cand, pds_hard_fit = _make_pds_for_eventlist(ev_hard)
            hard_ok = True
        except Exception:
            hard_ok = False

    m_cand_full = _safe_m_avg_from_pds(pds_full_cand)
    peak_full = find_qpo_peak_whitened(
        pds_full_cand.freq, pds_full_cand.power,
        fmin=getattr(P, "FMIN", 0.05),
        fmax=getattr(P, "FMAX", 10.0),
        smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
        ignore_below=getattr(P, "PEAK_IGNORE_BELOW", 0.1),
        min_width_bins=7,
        m_eff=m_cand_full,
        require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
        prominence_sigma=float(getattr(P, "PEAK_PROMINENCE_SIGMA", 1.0)),
        min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
        sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),
        prefer=getattr(P, "PEAK_RANK_BY", "prominence"),
    )

    peak_soft = peak_hard = None
    if soft_ok:
        m_cand_soft = _safe_m_avg_from_pds(pds_soft_cand)
        peak_soft = find_qpo_peak_whitened(
            pds_soft_cand.freq, pds_soft_cand.power,
            fmin=getattr(P, "FMIN", 0.05),
            fmax=getattr(P, "FMAX", 10.0),
            smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
            ignore_below=getattr(P, "PEAK_IGNORE_BELOW", 0.1),
            m_eff=m_cand_soft,
            require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
            prominence_sigma=float(getattr(P, "PEAK_PROMINENCE_SIGMA", 1.0)),
            min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
            sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),
            prefer=getattr(P, "PEAK_RANK_BY", "prominence"),
        )
    if hard_ok:
        m_cand_hard = _safe_m_avg_from_pds(pds_hard_cand)
        peak_hard = find_qpo_peak_whitened(
            pds_hard_cand.freq, pds_hard_cand.power,
            fmin=getattr(P, "FMIN", 0.05),
            fmax=getattr(P, "FMAX", 10.0),
            smooth_hz=getattr(P, "PEAK_SMOOTH_HZ", 0.5),
            ignore_below=getattr(P, "PEAK_IGNORE_BELOW", 0.1),
            m_eff=m_cand_hard,
            require_ksigma=getattr(P, "PEAK_REQUIRE_KSIGMA", None),
            prominence_sigma=float(getattr(P, "PEAK_PROMINENCE_SIGMA", 1.0)),
            min_sep_hz=getattr(P, "PEAK_MIN_SEP_HZ", 0.15),
            sigma_mode=getattr(P, "PEAK_SIGMA_MODE", "cont"),
            prefer=getattr(P, "PEAK_RANK_BY", "prominence"),
        )

    broad_rms_full, broad_err_full, qpo_rms_full, qpo_err_full = _compute_rms_metrics(pds_full_fit, peak_full)

    broad_rms_soft = broad_err_soft = qpo_rms_soft = qpo_err_soft = np.nan
    broad_rms_hard = broad_err_hard = qpo_rms_hard = qpo_err_hard = np.nan

    if soft_ok:
        broad_rms_soft, broad_err_soft, qpo_rms_soft, qpo_err_soft = _compute_rms_metrics(pds_soft_fit, peak_soft)
    if hard_ok:
        broad_rms_hard, broad_err_hard, qpo_rms_hard, qpo_err_hard = _compute_rms_metrics(pds_hard_fit, peak_hard)

    fit_full_res = fit_soft_res = fit_hard_res = None
    fit_full_out = fit_soft_out = fit_hard_out = {}

    if getattr(P, "DO_FIT", False):
        shared_seed = float(peak_full) if (peak_full is not None and np.isfinite(peak_full) and peak_full > 0) else None

        fit_full_res, fit_full_out = _fit_one_band(
            obsid=obsid,
            band_label="full",
            pds_fit=pds_full_fit,
            pds_cand=pds_full_cand,
            seed_peak_hz=shared_seed,
        )

        if soft_ok:
            seed_soft = shared_seed
            if seed_soft is None and peak_soft is not None and np.isfinite(peak_soft) and peak_soft > 0:
                seed_soft = float(peak_soft)
            fit_soft_res, fit_soft_out = _fit_one_band(
                obsid=obsid,
                band_label="soft",
                pds_fit=pds_soft_fit,
                pds_cand=pds_soft_cand,
                seed_peak_hz=seed_soft,
            )

        if hard_ok:
            seed_hard = shared_seed
            if seed_hard is None and peak_hard is not None and np.isfinite(peak_hard) and peak_hard > 0:
                seed_hard = float(peak_hard)
            fit_hard_res, fit_hard_out = _fit_one_band(
                obsid=obsid,
                band_label="hard",
                pds_fit=pds_hard_fit,
                pds_cand=pds_hard_cand,
                seed_peak_hz=seed_hard,
            )

        if getattr(P, "SAVE_FIT_PLOTS", True):
            band_items = [
                {"label": "Full", "pds_fit": pds_full_fit, "fitres": fit_full_res, "seed_peak": shared_seed},
                {"label": "Soft 0.3-2 keV", "pds_fit": pds_soft_fit, "fitres": fit_soft_res,
                 "seed_peak": shared_seed if shared_seed is not None else (float(peak_soft) if (peak_soft is not None and np.isfinite(peak_soft)) else np.nan)},
                {"label": "Hard 2-10 keV", "pds_fit": pds_hard_fit, "fitres": fit_hard_res,
                 "seed_peak": shared_seed if shared_seed is not None else (float(peak_hard) if (peak_hard is not None and np.isfinite(peak_hard)) else np.nan)},
            ]
            save_threeband_fit_overlay_plot(obsid, outdir_this, band_items)

    out = {
        "mjd_mid": float(mjd_mid) if (mjd_mid is not None and np.isfinite(mjd_mid)) else "",
        "iso_mid": iso_mid if iso_mid is not None else "",

        "tseg_s": float(lc_full.tseg),
        "dt_s": float(lc_full.dt),
        "mean_rate_cps": float(lc_full.meanrate),
        "segment_size_s": float(seg),

        "peak_f_hz_full": float(peak_full) if np.isfinite(peak_full) else "",
        "peak_f_hz_soft": float(peak_soft) if (peak_soft is not None and np.isfinite(peak_soft)) else "",
        "peak_f_hz_hard": float(peak_hard) if (peak_hard is not None and np.isfinite(peak_hard)) else "",

        "broad_rms_0p1_30_full": float(broad_rms_full) if np.isfinite(broad_rms_full) else "",
        "broad_rms_err_full": float(broad_err_full) if np.isfinite(broad_err_full) else "",
        "qpo_rms_full": float(qpo_rms_full) if np.isfinite(qpo_rms_full) else "",
        "qpo_rms_err_full": float(qpo_err_full) if np.isfinite(qpo_err_full) else "",

        "broad_rms_0p1_30_soft": float(broad_rms_soft) if np.isfinite(broad_rms_soft) else "",
        "broad_rms_err_soft": float(broad_err_soft) if np.isfinite(broad_err_soft) else "",
        "qpo_rms_soft": float(qpo_rms_soft) if np.isfinite(qpo_rms_soft) else "",
        "qpo_rms_err_soft": float(qpo_err_soft) if np.isfinite(qpo_err_soft) else "",

        "broad_rms_0p1_30_hard": float(broad_rms_hard) if np.isfinite(broad_rms_hard) else "",
        "broad_rms_err_hard": float(broad_err_hard) if np.isfinite(broad_err_hard) else "",
        "qpo_rms_hard": float(qpo_rms_hard) if np.isfinite(qpo_rms_hard) else "",
        "qpo_rms_err_hard": float(qpo_err_hard) if np.isfinite(qpo_err_hard) else "",
    }

    out.update(fit_full_out if fit_full_out else {})
    out.update(fit_soft_out if fit_soft_out else {})
    out.update(fit_hard_out if fit_hard_out else {})

    out["_fit_full_res_obj"] = fit_full_res
    out["_fit_soft_res_obj"] = fit_soft_res
    out["_fit_hard_res_obj"] = fit_hard_res

    return out


# ============================================================================
# Driver per obsid
# ============================================================================

def _process_one_obsid(obsid: str) -> dict:
    base_dir = getattr(P, "BASE_DIR", ".")
    source = getattr(P, "SOURCE", "")
    evt_path = build_evt_path(base_dir, source, obsid)

    row = {"obsid": obsid, "evt_path": evt_path, "status": "", "error": ""}

    try:
        if not os.path.exists(evt_path):
            raise FileNotFoundError(f"Missing event file: {evt_path}")

        suppress = bool(getattr(P, "SUPPRESS_NOISY_OUTPUTS", False))
        with _suppress_everything(suppress):
            res = analyze_obsid(obsid, evt_path)

        row.update(res)
        row["status"] = "OK"
        row["error"] = ""

    except Exception as e:
        row["status"] = "FAIL"
        row["error"] = str(e)

    if getattr(P, "ONE_LINE_SUMMARY", True):
        _print_one_line_summary(obsid, row)
    else:
        if row["status"] != "OK":
            print(f"[FAIL] {obsid}  {row['error']}")

    return row


# ============================================================================
# Main
# ============================================================================

def main():
    obsids = read_obsids(getattr(P, "OBSIDS_TXT", "obsids.txt"))
    if not obsids:
        raise SystemExit(f"No ObsIDs found in {getattr(P, 'OBSIDS_TXT', 'obsids.txt')}")

    fieldnames = [
        "obsid", "evt_path", "status", "error",
        "mjd_mid", "iso_mid",
        "tseg_s", "dt_s", "mean_rate_cps", "segment_size_s",
        "peak_f_hz_full", "peak_f_hz_soft", "peak_f_hz_hard",
        "broad_rms_0p1_30_full", "broad_rms_err_full", "qpo_rms_full", "qpo_rms_err_full",
        "broad_rms_0p1_30_soft", "broad_rms_err_soft", "qpo_rms_soft", "qpo_rms_err_soft",
        "broad_rms_0p1_30_hard", "broad_rms_err_hard", "qpo_rms_hard", "qpo_rms_err_hard",

        "full_fit_ok", "full_fit_nlor", "full_fit_const", "full_fit_rchi2", "full_fit_msg",
        "full_fit_qpo_nu0_hz", "full_fit_qpo_fwhm_hz", "full_fit_qpo_Q", "full_fit_qpo_rms2", "full_fit_qpo_rms",
        "full_comp_nu0s", "full_comp_fwhms", "full_comp_amps",

        "soft_fit_ok", "soft_fit_nlor", "soft_fit_const", "soft_fit_rchi2", "soft_fit_msg",
        "soft_fit_qpo_nu0_hz", "soft_fit_qpo_fwhm_hz", "soft_fit_qpo_Q", "soft_fit_qpo_rms2", "soft_fit_qpo_rms",
        "soft_comp_nu0s", "soft_comp_fwhms", "soft_comp_amps",

        "hard_fit_ok", "hard_fit_nlor", "hard_fit_const", "hard_fit_rchi2", "hard_fit_msg",
        "hard_fit_qpo_nu0_hz", "hard_fit_qpo_fwhm_hz", "hard_fit_qpo_Q", "hard_fit_qpo_rms2", "hard_fit_qpo_rms",
        "hard_comp_nu0s", "hard_comp_fwhms", "hard_comp_amps",
    ]

    parallel = bool(getattr(P, "PARALLEL_ENABLE", False))
    n_workers = int(getattr(P, "N_WORKERS", 1) or 1)
    start_method = str(getattr(P, "PARALLEL_START_METHOD", "spawn")).lower()

    rows: list[dict] = []

    if parallel and n_workers > 1 and len(obsids) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        ctx = mp.get_context(start_method)
        print(f"[INFO] Parallel enabled: {n_workers} workers | start_method={start_method} | n_obsids={len(obsids)}")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_process_one_obsid, obsid): obsid for obsid in obsids}
            for fut in as_completed(futs):
                obsid = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {k: "" for k in fieldnames}
                    row["obsid"] = obsid
                    row["evt_path"] = build_evt_path(getattr(P, "BASE_DIR", "."), getattr(P, "SOURCE", ""), obsid)
                    row["status"] = "FAIL"
                    row["error"] = f"Worker crashed: {e}"
                    if getattr(P, "ONE_LINE_SUMMARY", True):
                        _print_one_line_summary(obsid, row)
                    else:
                        print(f"[FAIL] {obsid}  Worker crashed: {e}")

                rows.append(row)

        order = {o: i for i, o in enumerate(obsids)}
        rows.sort(key=lambda r: order.get(r.get("obsid", ""), 10**9))

    else:
        if parallel and (n_workers <= 1 or len(obsids) <= 1):
            print("[INFO] Parallel requested but not used (n_workers<=1 or only one obsid). Running serial.")
        else:
            print("[INFO] Running serial.")

        for obsid in obsids:
            rows.append(_process_one_obsid(obsid))

    csv_path = os.path.join(common_outdir(), getattr(P, "OUT_CSV_NAME", "qpo_summary.csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)

    print(f"\nWrote: {csv_path}")
    print(f"Per-ObsID outputs in: {getattr(P, 'OUTDIR_BASE', '.')}/<obsid>/")


if __name__ == "__main__":
    main()