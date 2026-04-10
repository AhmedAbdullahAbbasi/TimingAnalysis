#!/usr/bin/env python3
"""
QPO_plot.py
===========
PDS fit plots driven entirely by struct band-block dicts.

The module has no dependency on FitResult objects or any Stingray fitting
machinery.  Its only currency is:
  - Raw arrays  : (freq, power, power_err) supplied by the caller
  - Band blocks : the dict format already stored in every _fitresult.json file
                  {ok, pars, comp_types, const, rchi2, nlor}

This makes it safe to import from both QPO_main and QPO_interactive without
creating circular dependencies.

Dependency graph
----------------
  QPO_Parameter   (read-only config)
  QPO_fit.lorentz (Lorentzian evaluator — no fitting code)
  numpy / matplotlib

Public API
----------
fitresult_to_band_block(fitres) -> dict
    Thin adapter: convert a FitResult to the band_block dict format.
    Used by QPO_main to call save_threeband_plot without touching FitResult
    objects directly.

plot_band(ax_top, ax_res, freq, power, power_err, band_block, **kwargs)
    Atomic renderer.  Draw one band onto caller-supplied axes.
    Returns the model array (or None when no valid fit exists).

save_band_plot(obsid, band_label, freq, power, power_err, band_block,
               outpath=None, *, dpi=150) -> str
    Single-band 2-panel PNG (PDS+model  /  residuals).
    Default outpath: <OUTDIR_BASE>/<obsid>/<obsid>_fit_<band_label>.png

save_threeband_plot(obsid, band_items, outpath=None, *,
                    dpi=150, clobber=False) -> str
    6-panel PNG  (full / soft / hard, each with PDS+model and residuals).
    band_items: list of dicts — {label, freq, power, power_err, band_block}
    Default outpath: <OUTDIR_BASE>/<obsid>/<obsid>_fits_full_soft_hard.png
    Respects P.CLOBBER when clobber is False (batch default).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import QPO_Parameter as P
from QPO_fit import lorentz


# ============================================================================
# Colour palettes  (match QPO_interactive defaults)
# ============================================================================

_CONT_COLORS: List[str] = ["#1565C0", "#6A1B9A", "#00695C", "#E65100"]
_QPO_COLORS:  List[str] = ["#B71C1C", "#E53935", "#FF5722"]


# ============================================================================
# Internal helpers
# ============================================================================

def _src_label() -> str:
    return str(getattr(P, "SOURCE", "SOURCE")).strip() or "SOURCE"


def _auto_ylim(
    arrays,
    *,
    pad_top: float   = 1.35,
    pad_bottom: float = 0.85,
    ymin_floor: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """Return (ymin, ymax) computed from the union of all supplied arrays."""
    vals = [np.asarray(a, float).ravel() for a in arrays if a is not None]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    v    = np.concatenate(vals)
    vpos = v[v > 0]
    if vpos.size:
        lo = max(float(np.nanmin(vpos)), ymin_floor) * pad_bottom
        hi = float(np.nanmax(vpos)) * pad_top
        if hi <= lo:
            hi = lo * 10.0
        return lo, hi
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - 0.05 * span, hi + 0.10 * span


def _ylim_cap_by_model(
    data_arr,
    model_arr,
    *,
    ymin_floor: float = 1e-6,
    cap_factor: float = 2.0,
    pad_bottom: float = 0.85,
) -> Optional[Tuple[float, float]]:
    """Return (ymin, ymax) capping the top at cap_factor × max(model)."""
    if model_arr is None:
        return None
    m = np.asarray(model_arr, float).ravel()
    m = m[np.isfinite(m) & (m > 0)]
    if not m.size:
        return None
    ymax = max(float(cap_factor * np.nanmax(m)), 10.0 * ymin_floor)
    vals = []
    for a in (data_arr, model_arr):
        if a is None:
            continue
        x = np.asarray(a, float).ravel()
        x = x[np.isfinite(x) & (x > 0)]
        if x.size:
            vals.append(x)
    ymin = (max(ymin_floor, float(np.nanmin(np.concatenate(vals))) * pad_bottom)
            if vals else ymin_floor)
    if ymax <= ymin:
        ymax = ymin * 10.0
    return ymin, ymax


def _resid_ylim(resid: np.ndarray) -> Tuple[float, float]:
    """Symmetric residual y-limits with ±3σ reference lines always visible."""
    r = resid[np.isfinite(resid)]
    if r.size < 5:
        return -5.0, 5.0
    p1, p99 = np.nanpercentile(r, [1, 99])
    lo = max(float(min(p1, -3.0)), -10.0)
    hi = min(float(max(p99,  3.0)),  10.0)
    if hi - lo < 2.0:
        lo, hi = -2.5, 2.5
    return lo, hi


def _qpos_from_band_block(
    band_block: dict,
    *,
    qpo_fmin: float = 0.1,
    qpo_fmax: float = 10.0,
    qpo_min_q: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Extract QPO components from a band_block dict.

    Applies the same QPO_FMIN / QPO_FMAX / QPO_MIN_Q filters used by the
    batch fitter so that plot annotations are consistent with the struct.

    Returns a list of dicts: [{index, nu0, fwhm, Q}, …], sorted by nu0.
    """
    if not band_block.get("ok", False):
        return []

    pars_raw   = band_block.get("pars",       [])
    comp_types = band_block.get("comp_types", [])
    # Pull thresholds from P so interactive overrides are respected
    qpo_fmin   = float(getattr(P, "QPO_FMIN",  qpo_fmin))
    qpo_fmax   = float(getattr(P, "QPO_FMAX",  qpo_fmax))
    qpo_min_q  = float(getattr(P, "QPO_MIN_Q", qpo_min_q))

    out = []
    for i, (par, ctype) in enumerate(zip(pars_raw, comp_types)):
        if str(ctype) != "qpo":
            continue
        nu0, fwhm, amp = float(par[0]), float(par[1]), float(par[2])
        if fwhm <= 0 or not np.isfinite(nu0):
            continue
        Q = nu0 / fwhm
        if Q < qpo_min_q:
            continue
        if not (qpo_fmin <= nu0 <= qpo_fmax):
            continue
        out.append({"index": i, "nu0": nu0, "fwhm": fwhm, "amp": amp, "Q": Q})

    out.sort(key=lambda d: d["nu0"])
    return out


# ============================================================================
# Public adapter
# ============================================================================

def fitresult_to_band_block(fitres) -> Dict[str, Any]:
    """
    Convert a FitResult object to the band_block dict format.

    This is the only place in QPO_plot that touches a FitResult.  All other
    functions work purely with the dict so they can be called with struct data
    directly without reconstructing FitResult objects.

    Parameters
    ----------
    fitres : FitResult (or None / ok=False)

    Returns
    -------
    band_block dict  {ok, nlor, rchi2, const, comp_types, pars}
    """
    if fitres is None or not getattr(fitres, "ok", False):
        return {"ok": False}

    pars = getattr(fitres, "pars", None)
    return {
        "ok":         True,
        "nlor":       int(getattr(fitres, "nlor",        0)),
        "rchi2":      float(getattr(fitres, "rchi2",     np.nan)),
        "const":      float(getattr(fitres, "const",     0.0) or 0.0),
        "comp_types": list(getattr(fitres, "comp_types", [])),
        "pars": (
            pars.tolist()
            if (pars is not None and hasattr(pars, "tolist"))
            else []
        ),
    }


# ============================================================================
# Atomic band renderer
# ============================================================================

def plot_band(
    ax_top,
    ax_res,
    freq,
    power,
    power_err,
    band_block:  Dict[str, Any],
    *,
    band_label:  str   = "",
    fit_fmin:    float = 0.05,
    fit_fmax:    float = 64.0,
    ymin_floor:  float = 1e-6,
    cont_colors: Optional[List[str]] = None,
    qpo_colors:  Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Draw one PDS band onto a pair of caller-supplied matplotlib Axes.

    Axes layout expected
    --------------------
    ax_top : log-log PDS + model overlay + component traces
    ax_res : residuals ((P-M)/σ  or  P/M-1  as fallback)

    Parameters
    ----------
    ax_top, ax_res : matplotlib Axes
    freq, power, power_err : PDS arrays  (power_err may be None)
    band_block : struct-format dict  {ok, pars, comp_types, const, rchi2, nlor}
    band_label : string shown in the top-left annotation box
    fit_fmin, fit_fmax : frequency range to display (Hz)
    ymin_floor : floor for the y-axis lower bound
    cont_colors, qpo_colors : colour cycle overrides

    Returns
    -------
    model_arr (ndarray, shape matching the trimmed freq grid) or None
    """
    if cont_colors is None:
        cont_colors = _CONT_COLORS
    if qpo_colors is None:
        qpo_colors = _QPO_COLORS

    # ---- No data ----
    if freq is None or power is None:
        ax_top.text(0.5, 0.5, f"{band_label}\n(no data)",
                    ha="center", va="center", transform=ax_top.transAxes,
                    fontsize=10)
        ax_top.set_axis_off()
        ax_res.set_axis_off()
        return None

    f0 = np.asarray(freq,  float)
    p0 = np.asarray(power, float)
    e0 = (np.full_like(p0, np.nan) if power_err is None
          else np.asarray(power_err, float))

    # Trim to fit frequency window
    sel = (f0 >= fit_fmin) & (f0 <= fit_fmax)
    f0, p0, e0 = f0[sel], p0[sel], e0[sel]

    ax_top.loglog(f0, p0, color="black", lw=0.9, label="PDS", zorder=2)

    model0 = None

    if band_block.get("ok", False):
        pars_raw   = band_block.get("pars",       [])
        comp_types = band_block.get("comp_types", [])
        cval       = float(band_block.get("const", 0.0) or 0.0)
        rchi       = band_block.get("rchi2", np.nan)
        nlor       = band_block.get("nlor",  "?")

        pars_arr = np.asarray(pars_raw, float) if pars_raw else np.empty((0, 3))

        if pars_arr.ndim == 2 and pars_arr.shape[1] == 3 and f0.size > 0:
            model0 = np.full_like(f0, cval)
            cont_i = qpo_i = 0

            for j, (nu0_j, fwhm_j, amp_j) in enumerate(pars_arr):
                ctype = str(comp_types[j]) if j < len(comp_types) else "cont"
                comp  = lorentz(f0, nu0_j, fwhm_j, amp_j)
                model0 += comp

                if ctype == "qpo":
                    color = qpo_colors[qpo_i % len(qpo_colors)]
                    ls    = "-"
                    qpo_i += 1
                else:
                    color = cont_colors[cont_i % len(cont_colors)]
                    ls    = "--"
                    cont_i += 1

                Q_j    = nu0_j / max(fwhm_j, 1e-12)
                clabel = (
                    f"[{j}] {'Q' if ctype == 'qpo' else 'C'}  {nu0_j:.3f} Hz"
                    + (f"  Q={Q_j:.1f}" if ctype == "qpo" else "")
                )
                ax_top.loglog(f0, comp, lw=1.0, ls=ls, color=color,
                              alpha=0.75, label=clabel, zorder=3)
                if ctype == "qpo":
                    ax_top.axvline(nu0_j, lw=0.9, ls=":", color=color, alpha=0.7)
                    ax_res.axvline(nu0_j, lw=0.8, ls=":", color=color, alpha=0.55)

            ax_top.loglog(f0, model0, color="#D32F2F", lw=2.0,
                          label="Model", zorder=4)
            if cval > 0 and np.isfinite(cval):
                ax_top.hlines(cval, fit_fmin, fit_fmax,
                              colors="orange", linestyles="--",
                              lw=1.2, alpha=0.8, label="Const", zorder=3)

        # ---- Annotation box ----
        qpos   = _qpos_from_band_block(band_block)
        rchi_n = rchi if rchi is not None else np.nan
        rchi_s = f"{rchi_n:.2f}" if np.isfinite(rchi_n) else "nan"
        ann    = [band_label, f"Nlor={nlor}", f"rχ²={rchi_s}", f"C={cval:.3g}"]
        if qpos:
            ann += [
                f"Nqpo={len(qpos)}",
                f"QPO={qpos[0]['nu0']:.3f} Hz",
                f"Q={qpos[0]['Q']:.2f}",
            ]
        ax_top.text(
            0.02, 0.95, " | ".join(ann),
            transform=ax_top.transAxes,
            va="top", ha="left", fontsize=9,
        )
    else:
        ax_top.text(
            0.02, 0.95, band_label or "(no fit)",
            transform=ax_top.transAxes, va="top", ha="left", fontsize=9,
        )

    # ---- Y limits (PDS panel) ----
    yl = _ylim_cap_by_model(p0, model0, cap_factor=2.0, ymin_floor=ymin_floor)
    if yl is None:
        yl = _auto_ylim([p0], ymin_floor=ymin_floor)
    if yl:
        ax_top.set_ylim(*yl)
    ax_top.set_ylabel("Power (frac-rms²/Hz)")
    ax_top.legend(fontsize=7, loc="upper right")

    # ---- Residuals panel ----
    ax_res.set_xscale("log")
    ax_res.axhline(0.0,  lw=1.0, alpha=0.6)
    ax_res.axhline(+3.0, lw=0.8, alpha=0.35, ls="--")
    ax_res.axhline(-3.0, lw=0.8, alpha=0.35, ls="--")

    if model0 is not None and np.any(np.isfinite(model0)):
        good = (np.isfinite(p0) & np.isfinite(model0)
                & np.isfinite(e0) & (e0 > 0))
        if good.sum() >= 5:
            resid = np.full_like(p0, np.nan)
            resid[good] = (p0[good] - model0[good]) / e0[good]
            ax_res.plot(f0, resid, lw=0.9)
            ax_res.set_ylabel("(P−M)/σ", fontsize=9)
            ax_res.set_ylim(*_resid_ylim(resid))
        else:
            # Fallback: fractional residuals when no error array
            good2 = np.isfinite(p0) & np.isfinite(model0) & (model0 > 0)
            resid = np.full_like(p0, np.nan)
            if good2.sum() >= 5:
                resid[good2] = p0[good2] / model0[good2] - 1.0
                ax_res.plot(f0, resid, lw=0.9)
                ax_res.set_ylabel("P/M−1", fontsize=9)
                ax_res.set_ylim(*_resid_ylim(resid))
            else:
                ax_res.text(0.5, 0.5, "No residuals (bad model)",
                            ha="center", va="center")
                ax_res.set_ylim(-5, 5)
    else:
        ax_res.text(0.5, 0.5, "No residuals (no fit)",
                    ha="center", va="center")
        ax_res.set_ylim(-5, 5)

    ax_res.set_xlabel("Frequency (Hz)")
    return model0


# ============================================================================
# Single-band PNG
# ============================================================================

def save_band_plot(
    obsid:      str,
    band_label: str,
    freq,
    power,
    power_err,
    band_block: Dict[str, Any],
    outpath:    Optional[str] = None,
    *,
    dpi: int = 150,
) -> str:
    """
    Write a  2-panel PNG for one energy band.

    The plot is always written (no CLOBBER check) because this function is
    called on explicit user request from the interactive fitter.

    Parameters
    ----------
    obsid, band_label : used for the default filename and the figure title
    freq, power, power_err : PDS arrays
    band_block : struct-format dict
    outpath    : destination PNG path.  Defaults to
                 <OUTDIR_BASE>/<obsid>/<obsid>_fit_<band_label>.png
    dpi        : output resolution

    Returns
    -------
    The path actually written.
    """
    if outpath is None:
        outdir  = os.path.join(getattr(P, "OUTDIR_BASE", "."), str(obsid))
        outpath = os.path.join(outdir, f"{obsid}_fit_{band_label}.png")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)

    fit_fmin   = float(getattr(P, "FIT_FMIN",  0.05))
    fit_fmax   = float(getattr(P, "FIT_FMAX",  64.0))
    ymin_floor = float(getattr(P, "PLOT_YMIN", 1e-6))
    dpi        = int(getattr(P, "PLOT_DPI",    dpi))

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(9.5, 6.5), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )

    plot_band(
        axes[0], axes[1],
        freq, power, power_err, band_block,
        band_label  = band_label,
        fit_fmin    = fit_fmin,
        fit_fmax    = fit_fmax,
        ymin_floor  = ymin_floor,
    )

    fig.suptitle(
        f"{_src_label()} {obsid}  [{band_label}]  Fit + Residuals",
        fontsize=12,
    )
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


# ============================================================================
# Three-band PNG
# ============================================================================

def save_threeband_plot(
    obsid:      str,
    band_items: List[Dict[str, Any]],
    outpath:    Optional[str] = None,
    *,
    dpi:     int  = 150,
    clobber: bool = False,
) -> str:
    """
    Write a 6-panel PNG covering up to three energy bands.

    Parameters
    ----------
    obsid : observation ID (used for the default filename and figure title)
    band_items : list of up to 3 dicts, each with keys:
        label      : str   — e.g. 'Full', 'Soft 0.3-2 keV'
        freq       : ndarray | None
        power      : ndarray | None
        power_err  : ndarray | None
        band_block : dict  — struct-format band block
    outpath : destination PNG path.  Defaults to
              <OUTDIR_BASE>/<obsid>/<obsid>_fits_full_soft_hard.png
    dpi     : output resolution
    clobber : when False (batch default) the file is skipped if it already
              exists and P.CLOBBER is False.  Set to True for interactive
              calls where the user explicitly requested a re-render.

    Returns
    -------
    The path actually written (or the existing path when skipped).
    """
    if outpath is None:
        outdir  = os.path.join(getattr(P, "OUTDIR_BASE", "."), str(obsid))
        outpath = os.path.join(outdir, f"{obsid}_fits_full_soft_hard.png")

    if (not clobber
            and os.path.exists(outpath)
            and not getattr(P, "CLOBBER", False)):
        return outpath

    os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)

    fit_fmin   = float(getattr(P, "FIT_FMIN",  0.05))
    fit_fmax   = float(getattr(P, "FIT_FMAX",  64.0))
    ymin_floor = float(getattr(P, "PLOT_YMIN", 1e-6))
    dpi        = int(getattr(P, "PLOT_DPI",    dpi))

    n_bands = len(band_items)
    fig, axes = plt.subplots(
        nrows=n_bands * 2,
        ncols=1,
        figsize=(9.5, 4.8 * n_bands),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2] * n_bands},
    )
    # Ensure axes is always a flat array even for n_bands == 1
    axes = np.asarray(axes).ravel()

    for i, item in enumerate(band_items):
        ax_top = axes[2 * i]
        ax_res = axes[2 * i + 1]
        plot_band(
            ax_top, ax_res,
            item.get("freq"),
            item.get("power"),
            item.get("power_err"),
            item.get("band_block", {"ok": False}),
            band_label = item.get("label", f"Band {i+1}"),
            fit_fmin   = fit_fmin,
            fit_fmax   = fit_fmax,
            ymin_floor = ymin_floor,
        )

    band_names = " / ".join(it.get("label", "") for it in band_items)
    fig.suptitle(
        f"{_src_label()} {obsid}  |  Fits + Residuals  ({band_names})",
        fontsize=12,
    )
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath
