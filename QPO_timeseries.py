#!/usr/bin/env python3
"""
QPO_timeseries.py
=================
Batch time-series plotter for the QPO fitting pipeline.

Reads the summary CSV produced by QPO_main.py and generates a set of
publication-ready PNG figures in a dedicated timeseries/ sub-directory.

Can be run independently at any time after QPO_main has finished:

    python QPO_timeseries.py
    python QPO_timeseries.py --csv /path/to/custom_summary.csv
    python QPO_timeseries.py --outdir /path/to/output_dir

Plots generated
---------------
  qpo_nu_vs_time.png       QPO centroid frequency per band
  qpo_fwhm_vs_time.png     QPO FWHM per band
  qpo_Q_vs_time.png        QPO quality factor Q per band
  qpo_rms_vs_time.png      QPO fractional RMS amplitude per band
  qpo_rms2_vs_time.png     QPO integrated power (rms²) per band
  broad_rms_vs_time.png    Broadband (0.1–30 Hz) RMS per band
  count_rate_vs_time.png   Mean count rate (all bands combined, single panel)
  fit_rchi2_vs_time.png    Reduced chi-squared per band  (fit quality monitor)
  fit_nlor_vs_time.png     Number of Lorentzians per band
  fit_const_vs_time.png    White-noise constant per band
  peak_freq_vs_time.png    Diagnostic whitened-peak frequency per band
  summary.png              4-panel overview: count rate / ν_QPO / Q / RMS (full band)

Design notes
------------
- Bad fits (fit_ok=False) are plotted with open markers at reduced alpha,
  rather than silently dropped, so data-quality problems are visible.
- Consecutive epochs are joined by a thin line to reveal temporal evolution.
- Band labels are derived from QPO_Parameter at runtime so they stay
  consistent with the energy ranges actually used in the analysis.
- The script  skips any plot whose required columns are absent,
  printing a [WARN] line rather than crashing.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import QPO_Parameter as P


# ============================================================================
# Constants
# ============================================================================

BANDS: List[str] = ["full", "soft", "hard"]

# Colours and markers consistent across all plots
_BAND_COLOR  = {"full": "#1565C0", "soft": "#E65100", "hard": "#2E7D32"}
_BAND_MARKER = {"full": "o",       "soft": "s",        "hard": "^"}

# Marker style for observations where fit_ok=False
_BAD_MARKER_STYLE = dict(
    linestyle   = "None",
    markersize  = 5,
    fillstyle   = "none",   # open symbol
    alpha       = 0.4,
)
_GOOD_MARKER_STYLE = dict(
    linestyle   = "None",
    markersize  = 5,
    alpha       = 0.85,
)


# ============================================================================
# Band label helper
# ============================================================================

def _band_label(band: str) -> str:
    """Human-readable energy-band label derived from QPO_Parameter at runtime."""
    if band == "full":
        return "Full band"
    if band == "soft":
        lo, hi = getattr(P, "SOFT_BAND_KEV", (0.3, 2.0))
        return f"Soft  {lo}–{hi} keV"
    if band == "hard":
        lo, hi = getattr(P, "HARD_BAND_KEV", (2.0, 10.0))
        return f"Hard  {lo}–{hi} keV"
    return band.upper()


# ============================================================================
# DataFrame helpers
# ============================================================================

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _to_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    """Return True only if every column in cols exists in df."""
    return all(c in df.columns for c in cols)


def _has_any_col(df: pd.DataFrame, cols: List[str]) -> bool:
    """Return True if at least one column in cols exists in df."""
    return any(c in df.columns for c in cols)


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of the summary DataFrame:
    - Keep only status=OK rows.
    - Parse mjd_mid to float and drop rows with missing MJD.
    - Sort by MJD.
    """
    d = df.copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).str.upper().eq("OK")].copy()

    if "mjd_mid" not in d.columns:
        raise SystemExit("CSV missing required column: mjd_mid")

    d["mjd_mid"] = _to_num(d["mjd_mid"])
    d = d[d["mjd_mid"].notna()].copy()
    d = d.sort_values("mjd_mid").reset_index(drop=True)
    return d


def _mask_fit_ok(d: pd.DataFrame, band: str) -> pd.Series:
    """Boolean mask: True where the fit for this band was accepted."""
    col = f"{band}_fit_ok"
    if col not in d.columns:
        return pd.Series(True, index=d.index)
    ok = d[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return ok.fillna(False)


# ============================================================================
# Plot primitives
# ============================================================================

def _plot_one_band_series(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    ok_mask: np.ndarray,
    band: str,
    *,
    connect: bool = True,
    label: Optional[str] = None,
) -> None:
    """
    Plot good and bad observations separately on the same Axes.

    Good observations (ok_mask=True)  : filled markers, full alpha.
    Bad  observations (ok_mask=False) : open markers, reduced alpha.
    Thin line connects all valid (finite) good points in time order.
    """
    color  = _BAND_COLOR[band]
    marker = _BAND_MARKER[band]
    lbl    = label if label is not None else _band_label(band)

    finite = np.isfinite(x) & np.isfinite(y)

    # ---- connecting line (good, finite points only) ----
    if connect:
        good_finite = finite & ok_mask
        if good_finite.sum() >= 2:
            ax.plot(x[good_finite], y[good_finite],
                    color=color, lw=0.6, alpha=0.35, zorder=1)

    # ---- good points ----
    gf = finite & ok_mask
    if gf.sum() > 0:
        if yerr is not None and np.any(np.isfinite(yerr[gf])):
            ax.errorbar(x[gf], y[gf], yerr=yerr[gf],
                        fmt=marker, color=color, capsize=2,
                        label=lbl, **_GOOD_MARKER_STYLE, zorder=3)
        else:
            ax.plot(x[gf], y[gf],
                    marker=marker, color=color,
                    label=lbl, **_GOOD_MARKER_STYLE, zorder=3)

    # ---- bad / rejected points ----
    bf = finite & ~ok_mask
    if bf.sum() > 0:
        bad_lbl = f"{lbl} (bad fit)"
        if yerr is not None and np.any(np.isfinite(yerr[bf])):
            ax.errorbar(x[bf], y[bf], yerr=yerr[bf],
                        fmt=marker, color=color, capsize=2,
                        label=bad_lbl, **_BAD_MARKER_STYLE, zorder=2)
        else:
            ax.plot(x[bf], y[bf],
                    marker=marker, color=color,
                    label=bad_lbl, **_BAD_MARKER_STYLE, zorder=2)


def _plot_threepanel(
    d: pd.DataFrame,
    *,
    outpath:         str,
    title:           str,
    ylabel:          str,
    ycols_by_band:   Dict[str, str],
    yerrcols_by_band: Optional[Dict[str, str]] = None,
    require_fit_ok:  bool  = True,
    y_floor_positive: bool = False,
    connect:         bool  = True,
    yscale:          str   = "linear",
    hline:           Optional[float] = None,
    hline_label:     Optional[str]   = None,
) -> None:
    """
    Three-panel (one per energy band) time-series figure.

    Parameters
    ----------
    d               : prepared DataFrame (output of _prep_df)
    outpath         : full path to write the PNG
    title           : figure suptitle
    ylabel          : shared y-axis label
    ycols_by_band   : {band: column_name} mapping
    yerrcols_by_band: {band: error_column_name} or None
    require_fit_ok  : if True, bad-fit points are shown with open symbols
                      rather than plotted normally
    y_floor_positive: if True, non-positive y values are excluded (log-safe)
    connect         : draw a thin line connecting consecutive good points
    yscale          : 'linear' or 'log'
    hline           : draw a horizontal reference line at this y value
    hline_label     : legend label for hline
    """
    dpi = int(getattr(P, "PLOT_DPI", 200))
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True,
    )

    any_data = False

    for ax, band in zip(axes, BANDS):
        ycol    = ycols_by_band.get(band)
        yerrcol = (yerrcols_by_band.get(band) if yerrcols_by_band else None)

        ax.set_title(_band_label(band), loc="left", fontsize=10)
        ax.set_yscale(yscale)

        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.9, ls="--",
                       label=hline_label, alpha=0.7, zorder=0)

        if ycol is None or ycol not in d.columns:
            ax.text(0.5, 0.5,
                    f"Column absent:\n{ycol}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_ylabel(ylabel, fontsize=9)
            continue

        x   = d["mjd_mid"].to_numpy(float)
        y   = _to_num(d[ycol]).to_numpy(float)
        ok  = _mask_fit_ok(d, band).to_numpy(bool)

        yerr = None
        if yerrcol is not None and yerrcol in d.columns:
            yerr = _to_num(d[yerrcol]).to_numpy(float)

        # Floor filter (applied to all rows equally; bad-fit status unchanged)
        if y_floor_positive:
            floor_ok = (y > 0)
            y   = np.where(floor_ok, y,   np.nan)
            if yerr is not None:
                yerr = np.where(floor_ok, yerr, np.nan)

        # For log scale, also gate error bars so y-yerr > 0
        if yscale == "log" and yerr is not None:
            safe_err = (y - yerr) > 0
            yerr = np.where(safe_err, yerr, np.nan)

        _plot_one_band_series(
            ax, x, y, yerr,
            ok_mask = ok if require_fit_ok else np.ones(len(ok), bool),
            band    = band,
            connect = connect,
        )

        n_good = int(np.sum(np.isfinite(y) & ok))
        n_bad  = int(np.sum(np.isfinite(y) & ~ok))
        if n_good + n_bad > 0:
            any_data = True

        ax.set_ylabel(ylabel, fontsize=9)
        leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.75)
        if not leg.get_texts():
            leg.set_visible(False)

        # Annotate n_obs
        ax.text(0.01, 0.97,
                f"n={n_good}" + (f"  (+{n_bad} bad)" if n_bad else ""),
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left", color="gray")

    axes[-1].set_xlabel("MJD (mid)", fontsize=10)
    src = str(getattr(P, "SOURCE", "")).strip()
    fig.suptitle(f"{src}  —  {title}" if src else title, fontsize=12)

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    status = "OK" if any_data else "EMPTY"
    print(f"  [{status}] {os.path.basename(outpath)}")


def _plot_singlepanel(
    d: pd.DataFrame,
    *,
    outpath:  str,
    title:    str,
    ylabel:   str,
    ycol:     str,
    yerrcol:  Optional[str]  = None,
    yscale:   str            = "linear",
    y_floor_positive: bool   = False,
    connect:  bool           = True,
    hline:    Optional[float] = None,
    hline_label: Optional[str] = None,
) -> None:
    """
    Single-panel time-series for quantities that are not split by energy band.
    """
    dpi = int(getattr(P, "PLOT_DPI", 200))
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.0), constrained_layout=True)
    ax.set_yscale(yscale)

    if hline is not None:
        ax.axhline(hline, color="gray", lw=0.9, ls="--",
                   label=hline_label, alpha=0.7, zorder=0)

    if ycol not in d.columns:
        ax.text(0.5, 0.5, f"Column absent:\n{ycol}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    else:
        x    = d["mjd_mid"].to_numpy(float)
        y    = _to_num(d[ycol]).to_numpy(float)
        yerr = None
        if yerrcol is not None and yerrcol in d.columns:
            yerr = _to_num(d[yerrcol]).to_numpy(float)

        if y_floor_positive:
            floor_ok = y > 0
            y    = np.where(floor_ok, y,    np.nan)
            if yerr is not None:
                yerr = np.where(floor_ok, yerr, np.nan)

        finite = np.isfinite(x) & np.isfinite(y)
        if connect and finite.sum() >= 2:
            ax.plot(x[finite], y[finite],
                    color="#555555", lw=0.6, alpha=0.3, zorder=1)

        if yerr is not None and np.any(np.isfinite(yerr[finite])):
            ax.errorbar(x[finite], y[finite], yerr=yerr[finite],
                        fmt="o", color="#1565C0", capsize=2,
                        markersize=5, alpha=0.85, zorder=3)
        else:
            ax.plot(x[finite], y[finite],
                    "o", color="#1565C0",
                    markersize=5, alpha=0.85, zorder=3)

        n = int(finite.sum())
        ax.text(0.01, 0.97, f"n={n}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left", color="gray")

    ax.set_xlabel("MJD (mid)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.75)

    src = str(getattr(P, "SOURCE", "")).strip()
    fig.suptitle(f"{src}  —  {title}" if src else title, fontsize=12)

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {os.path.basename(outpath)}")


def _plot_summary(d: pd.DataFrame, *, outpath: str) -> None:
    """
    Four-panel overview figure using the full-band columns.

    Panels (top → bottom):
      1. Mean count rate
      2. QPO centroid frequency  ν₀
      3. QPO quality factor Q
      4. Broadband fractional RMS (0.1–30 Hz)

    Only OK observations are plotted in panels 2–4.
    """
    dpi = int(getattr(P, "PLOT_DPI", 200))

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 12), sharex=True, constrained_layout=True,
    )

    x = d["mjd_mid"].to_numpy(float)

    panel_cfg = [
        # (title,         ylabel,                 ycol,                       yerrcol,                         fit_ok_gate, y_floor)
        ("Count rate",    "Rate (ct s⁻¹)",        "mean_rate_cps",            None,                            False,       True),
        ("QPO ν₀ (full)", "ν₀ (Hz)",              "full_fit_qpo_nu0_hz",      None,                            True,        True),
        ("QPO Q (full)",  "Q = ν₀/FWHM",          "full_fit_qpo_Q",           None,                            True,        True),
        ("Broad RMS (full)", "Fractional RMS",    "broad_rms_0p1_30_full",    "broad_rms_err_full",            False,       True),
    ]

    for ax, (panel_title, ylabel, ycol, yerrcol, gate_ok, y_floor) in zip(axes, panel_cfg):
        ax.set_title(panel_title, loc="left", fontsize=10)

        if ycol not in d.columns:
            ax.text(0.5, 0.5, f"Column absent: {ycol}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_ylabel(ylabel, fontsize=9)
            continue

        y    = _to_num(d[ycol]).to_numpy(float)
        yerr = None
        if yerrcol is not None and yerrcol in d.columns:
            yerr = _to_num(d[yerrcol]).to_numpy(float)

        ok = _mask_fit_ok(d, "full").to_numpy(bool) if gate_ok else np.ones(len(d), bool)

        if y_floor:
            bad_floor = y <= 0
            y    = np.where(bad_floor, np.nan, y)
            if yerr is not None:
                yerr = np.where(bad_floor, np.nan, yerr)

        _plot_one_band_series(
            ax, x, y, yerr,
            ok_mask = ok,
            band    = "full",
            connect = True,
            label   = "Full band",
        )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.75)

    axes[-1].set_xlabel("MJD (mid)", fontsize=10)
    src = str(getattr(P, "SOURCE", "")).strip()
    fig.suptitle(
        f"{src}  —  Overview time series" if src else "Overview time series",
        fontsize=13,
    )

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {os.path.basename(outpath)}")


# ============================================================================
# Orchestrator
# ============================================================================

def make_timeseries_plots(df: pd.DataFrame, outdir: str) -> None:
    """
    Generate all time-series plots from a QPO summary DataFrame.

    Parameters
    ----------
    df     : raw DataFrame read directly from the summary CSV
    outdir : directory where PNGs are written (created if absent)
    """
    _ensure_dir(outdir)
    d = _prep_df(df)

    n_obs = len(d)
    if n_obs == 0:
        print("[WARN] No OK observations in CSV — nothing to plot.")
        return

    print(f"[INFO] {n_obs} OK observations spanning "
          f"MJD {d['mjd_mid'].min():.3f} – {d['mjd_mid'].max():.3f}")

    def path(name: str) -> str:
        return os.path.join(outdir, name)

    # ------------------------------------------------------------------
    # 1. QPO centroid frequency ν₀
    # ------------------------------------------------------------------
    nu_cols = {b: f"{b}_fit_qpo_nu0_hz" for b in BANDS}
    if _has_any_col(d, list(nu_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("qpo_nu_vs_time.png"),
            title           = "QPO centroid frequency ν₀ vs time",
            ylabel          = "ν₀ (Hz)",
            ycols_by_band   = nu_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] qpo_nu_vs_time.png — no QPO ν₀ columns found")

    # ------------------------------------------------------------------
    # 2. QPO FWHM
    # ------------------------------------------------------------------
    fwhm_cols = {b: f"{b}_fit_qpo_fwhm_hz" for b in BANDS}
    if _has_any_col(d, list(fwhm_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("qpo_fwhm_vs_time.png"),
            title           = "QPO FWHM vs time",
            ylabel          = "FWHM (Hz)",
            ycols_by_band   = fwhm_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] qpo_fwhm_vs_time.png — no QPO FWHM columns found")

    # ------------------------------------------------------------------
    # 3. QPO quality factor Q
    # ------------------------------------------------------------------
    q_cols = {b: f"{b}_fit_qpo_Q" for b in BANDS}
    if _has_any_col(d, list(q_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("qpo_Q_vs_time.png"),
            title           = "QPO quality factor Q vs time",
            ylabel          = "Q = ν₀ / FWHM",
            ycols_by_band   = q_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
            hline           = float(getattr(P, "QPO_MIN_Q", 3.0)),
            hline_label     = f"Q_min = {getattr(P, 'QPO_MIN_Q', 3.0):.0f}",
        )
    else:
        print("  [SKIP] qpo_Q_vs_time.png — no QPO Q columns found")

    # ------------------------------------------------------------------
    # 4. QPO fractional RMS  (fit-based preferred; diagnostic fallback)
    # ------------------------------------------------------------------
    rms_fit_cols     = {b: f"{b}_fit_qpo_rms"     for b in BANDS}
    rms_fit_err_cols = {b: f"{b}_fit_qpo_rms_err" for b in BANDS}
    rms_diag_cols    = {b: f"qpo_rms_{b}"         for b in BANDS}
    rms_diag_err_cols= {b: f"qpo_rms_err_{b}"     for b in BANDS}

    if _has_any_col(d, list(rms_fit_cols.values())):
        use_err = _has_any_col(d, list(rms_fit_err_cols.values()))
        _plot_threepanel(
            d,
            outpath         = path("qpo_rms_vs_time.png"),
            title           = "QPO fractional RMS vs time  (fit-based)",
            ylabel          = "QPO RMS",
            ycols_by_band   = rms_fit_cols,
            yerrcols_by_band = rms_fit_err_cols if use_err else None,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
        )
    elif _has_any_col(d, list(rms_diag_cols.values())):
        use_err = _has_any_col(d, list(rms_diag_err_cols.values()))
        _plot_threepanel(
            d,
            outpath         = path("qpo_rms_vs_time.png"),
            title           = "QPO-band RMS vs time  (diagnostic window — fallback)",
            ylabel          = "QPO-band RMS",
            ycols_by_band   = rms_diag_cols,
            yerrcols_by_band = rms_diag_err_cols if use_err else None,
            require_fit_ok  = False,
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] qpo_rms_vs_time.png — no QPO RMS columns found")

    # ------------------------------------------------------------------
    # 5. QPO integrated power rms²
    # ------------------------------------------------------------------
    rms2_cols = {b: f"{b}_fit_qpo_rms2" for b in BANDS}
    if _has_any_col(d, list(rms2_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("qpo_rms2_vs_time.png"),
            title           = "QPO integrated power (rms²) vs time",
            ylabel          = "QPO rms²  (frac-rms²)",
            ycols_by_band   = rms2_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] qpo_rms2_vs_time.png — no QPO rms² columns found")

    # ------------------------------------------------------------------
    # 6. Broadband fractional RMS  (0.1–30 Hz)
    # ------------------------------------------------------------------
    broad_cols     = {b: f"broad_rms_0p1_30_{b}" for b in BANDS}
    broad_err_cols = {b: f"broad_rms_err_{b}"    for b in BANDS}
    if _has_any_col(d, list(broad_cols.values())):
        use_err = _has_any_col(d, list(broad_err_cols.values()))
        lo, hi = getattr(P, "BROAD_RMS_BAND", (0.1, 30.0))
        _plot_threepanel(
            d,
            outpath          = path("broad_rms_vs_time.png"),
            title            = f"Broadband fractional RMS ({lo}–{hi} Hz) vs time",
            ylabel           = "Broadband RMS",
            ycols_by_band    = broad_cols,
            yerrcols_by_band = broad_err_cols if use_err else None,
            require_fit_ok   = False,
            y_floor_positive  = True,
            connect          = True,
        )
    else:
        print("  [SKIP] broad_rms_vs_time.png — no broadband RMS columns found")

    # ------------------------------------------------------------------
    # 7. Mean count rate  (single panel, not split by band)
    # ------------------------------------------------------------------
    if "mean_rate_cps" in d.columns:
        _plot_singlepanel(
            d,
            outpath          = path("count_rate_vs_time.png"),
            title            = "Mean count rate vs time",
            ylabel           = "Rate  (ct s⁻¹)",
            ycol             = "mean_rate_cps",
            y_floor_positive  = True,
            connect          = True,
        )
    else:
        print("  [SKIP] count_rate_vs_time.png — column mean_rate_cps absent")

    # ------------------------------------------------------------------
    # 8. Reduced chi-squared per band  (fit quality monitor)
    # ------------------------------------------------------------------
    rchi_cols = {b: f"{b}_fit_rchi2" for b in BANDS}
    if _has_any_col(d, list(rchi_cols.values())):
        rchi_max = float(getattr(P, "FIT_RCHI_MAX", 1.5))
        _plot_threepanel(
            d,
            outpath         = path("fit_rchi2_vs_time.png"),
            title           = "Fit reduced χ² vs time  (quality monitor)",
            ylabel          = "rχ²",
            ycols_by_band   = rchi_cols,
            require_fit_ok  = False,   # show all — this IS the quality diagnostic
            y_floor_positive = True,
            connect         = True,
            hline           = rchi_max,
            hline_label     = f"rχ²_max = {rchi_max:.1f}",
        )
    else:
        print("  [SKIP] fit_rchi2_vs_time.png — no rchi2 columns found")

    # ------------------------------------------------------------------
    # 9. Number of Lorentzians per band
    # ------------------------------------------------------------------
    nlor_cols = {b: f"{b}_fit_nlor" for b in BANDS}
    if _has_any_col(d, list(nlor_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("fit_nlor_vs_time.png"),
            title           = "Number of Lorentzians (model complexity) vs time",
            ylabel          = "N_Lor",
            ycols_by_band   = nlor_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = False,   # discrete steps mislead; scatter is clearer
        )
    else:
        print("  [SKIP] fit_nlor_vs_time.png — no nlor columns found")

    # ------------------------------------------------------------------
    # 10. White-noise constant per band
    # ------------------------------------------------------------------
    const_cols = {b: f"{b}_fit_const" for b in BANDS}
    if _has_any_col(d, list(const_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("fit_const_vs_time.png"),
            title           = "White-noise constant vs time",
            ylabel          = "Const  (frac-rms² Hz⁻¹)",
            ycols_by_band   = const_cols,
            require_fit_ok  = True,
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] fit_const_vs_time.png — no const columns found")

    # ------------------------------------------------------------------
    # 11. Diagnostic whitened-peak frequency per band
    # ------------------------------------------------------------------
    peak_cols = {
        "full": "peak_f_hz_full",
        "soft": "peak_f_hz_soft",
        "hard": "peak_f_hz_hard",
    }
    if _has_any_col(d, list(peak_cols.values())):
        _plot_threepanel(
            d,
            outpath         = path("peak_freq_vs_time.png"),
            title           = "Diagnostic whitened-peak frequency vs time",
            ylabel          = "Peak freq (Hz)",
            ycols_by_band   = peak_cols,
            require_fit_ok  = False,   # diagnostic is independent of fit_ok
            y_floor_positive = True,
            connect         = True,
        )
    else:
        print("  [SKIP] peak_freq_vs_time.png — no diagnostic peak columns found")

    # ------------------------------------------------------------------
    # 12. Summary overview figure
    # ------------------------------------------------------------------
    _plot_summary(d, outpath=path("summary.png"))


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog        = "QPO_timeseries.py",
        description = "Generate time-series plots from a QPO summary CSV.",
    )
    parser.add_argument(
        "--csv",
        default = None,
        metavar = "PATH",
        help    = "Path to the summary CSV.  "
                  "Defaults to <OUTDIR_BASE>/<COMMON_DIRNAME>/<OUT_CSV_NAME> "
                  "from QPO_Parameter.",
    )
    parser.add_argument(
        "--outdir",
        default = None,
        metavar = "DIR",
        help    = "Output directory for PNGs.  "
                  "Defaults to <OUTDIR_BASE>/<COMMON_DIRNAME>/timeseries/.",
    )
    args = parser.parse_args()

    # ---- Resolve CSV path ----
    if args.csv is not None:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            getattr(P, "OUTDIR_BASE",    "."),
            getattr(P, "COMMON_DIRNAME", "commonfiles"),
            getattr(P, "OUT_CSV_NAME",   "qpo_summary.csv"),
        )

    if not os.path.exists(csv_path):
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

    # ---- Resolve output directory ----
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = os.path.join(
            getattr(P, "OUTDIR_BASE",    "."),
            getattr(P, "COMMON_DIRNAME", "commonfiles"),
            "timeseries",
        )

    print(f"[INFO] Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] {len(df)} rows total  →  writing to: {outdir}")

    make_timeseries_plots(df, outdir)

    print(f"\n[DONE] Time-series plots written to: {outdir}")


if __name__ == "__main__":
    main()
