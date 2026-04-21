#!/usr/bin/env python3
"""
QPO_interactive.py
==================
Terminal-driven interactive Lorentzian fitter with a live matplotlib window.

The plot window shows the PDS, model overlay, individual components, and
residuals.  All interaction is typed in the terminal.

PATCH NOTES (vs. original)
--------------------------
* The interactive fitter now propagates the *true* effective number of
  averaged powers per bin (`m_eff`) from the rebinned Stingray PDS into
  the Powerspectrum object used by `fit_lorentzians`.  Hardcoding `ps.m=1`
  on a log-rebinned PDS made the Whittle likelihood ~m× too shallow,
  letting the priors drag known-good parameters off into the bounds.
* `_compute_rchi2` and `_sigma` now use the same `m_eff`.
* Continuum x0 bound tightened to FIT_CONT_X0_MAX_HZ (matches batch fitter).
* Amplitude cap is now `max(5*c['amp'], 5*low-f level)` instead of 50× current.
* `_load_pds`, `_rebuild_pds`, and `_apply_rebin` all recompute `m_eff`
  from the rebinned `pds.m` array.

TripleA integration (vs. previous patch)
-----------------------------------------
* When FIT_METHOD = "TripleA" (or fit_method is set to "TripleA" via
  setparam), _run_direct_fit bypasses the Stingray fit_lorentzians loop
  entirely and calls QPO_TripleA.tripleA_fit_once instead.
* Frozen parameters are implemented as tight L-BFGS-B box bounds rather
  than prior penalty windows — the semantics are identical but the
  optimiser respects them as hard constraints.
* The Stingray loop remains unchanged and is still used for any other
  fitmethod value (Powell, Nelder-Mead).

Commands
--------
  addcomp qpo / continuum  Add a component (prompts for parameters)
  removecomp <idx>         Remove a component by index
  editcomp   <idx>         Re-enter parameters for a component
  freeze  <idx> [nu0|fwhm|amp]   Fix a parameter so the optimiser cannot move it
  unfreeze <idx> [nu0|fwhm|amp]  Release a frozen parameter
  setconst   <value>       Set the white-noise floor (always held fixed during fit)
  list                     Print the component table with frozen status
  params                   Show all tunable parameters
  setparam   <n> <val>     Set a parameter (e.g. setparam dt 0.005)
  fit                      Run the optimiser on the current components
  status                   Print fit statistics
  saveresult [path.json]   Save fit result struct  (default: update the loaded struct)
  load <obsid> [band]      Load a saved fit struct as initial components
  load <path.json> [band]  Load from an explicit path
  save       [filename]    Save the plot to a PNG file
  rebin log <f> / linear <df> / none   Rebin the PDS
  zoom  <fmin> <fmax>      Set x-axis limits (Hz)
  clear                    Remove all components (keeps last fit)
  reset                    Clear components AND the last fit result
  help                     Show this list
  quit                     Exit

Usage
-----
  python QPO_interactive.py --obsid 1200120106 --band full

  from QPO_interactive import launch
  launch(obsid="1200120106", band_label="full")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from stingray import Powerspectrum
from stingray.events import EventList
from stingray.modeling.scripts import fit_lorentzians
from stingray.powerspectrum import AveragedPowerspectrum

import QPO_Parameter as P
from QPO_fit import (
    FitResult,
    _compute_rchi2,
    _half_uniform,
    _hard_trunc_uniform,
    _repair_params,
    component_power_integral,
    extract_qpo_params_list,
    lorentz,
)
from QPO_struct import (
    load_fit_struct,
    save_fit_struct,
    struct_path,
    struct_summary,
    struct_to_warm_comps,
)
from QPO_utils import (
    safe_m_from_pds,
    filter_events_by_energy,
    make_averaged_pds,
    maybe_rebin_pds_fit,
    load_pds_for_band,
)
from QPO_plot import save_band_plot, save_threeband_plot, fitresult_to_band_block
from QPO_TripleA import tripleA_fit_once as _tripleA_fit_once

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

def _ansi(c: str) -> str:
    return c if sys.stdout.isatty() else ""

_R  = _ansi("\033[0m")
_B  = _ansi("\033[1m")
_D  = _ansi("\033[2m")
_RD = _ansi("\033[31m")
_GR = _ansi("\033[32m")
_YL = _ansi("\033[33m")
_BL = _ansi("\033[34m")
_CY = _ansi("\033[36m")

_CONT_COLORS = ["#4878CF", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"]
_QPO_COLORS  = ["#EF5350", "#FF8F00", "#26A69A", "#AB47BC", "#EC407A"]



# ---------------------------------------------------------------------------
# TerminalFitter
# ---------------------------------------------------------------------------

class TerminalFitter:
    """Interactive Lorentzian fitter driven by typed terminal commands."""

    _HELP = """\

  addcomp qpo              Prompt for ν₀, FWHM, amplitude → add QPO
  addcomp continuum        Prompt for ν₀, FWHM, amplitude → add continuum
  removecomp <idx>         Remove component by index
  editcomp   <idx>         Re-enter parameters for a component
  freeze  <idx> [field]    Freeze a component field: nu0, fwhm, amp, or all
  unfreeze <idx> [field]   Unfreeze a component field (or all)
  setconst  <value>        Set the white-noise floor (always fixed during fit)
  list                     Print the component table with frozen status
  params                   Show all tunable parameters
  setparam  <n> <val>      Change a parameter (e.g. setparam dt 0.005)
  fit                      Run the optimiser on the current components
  status                   Print fit statistics
  saveresult [path.json]   Save fit result (default: update the loaded struct)
  load <obsid> [band]      Load a saved struct as initial components
  load <path.json> [band]  Load from an explicit file path
  plotresult [path.png]    Save a publication-quality PNG for this band
                           Uses current PDS + fit result (or component overlay
                           if no fit has been run yet).
                           Default: <OUTDIR_BASE>/<obsid>/<obsid>_fit_<band>.png
  plotall    [path.png]    Generate the full 3-band plot from the saved struct.
                           Reloads all energy bands from the event file.
                           Default: <OUTDIR_BASE>/<obsid>/<obsid>_fits_full_soft_hard.png
  save      [filename]     Save the live interactive canvas to PNG
  rebin log <f>            Log-rebin with fractional step f
  rebin linear <df>        Linear-rebin to bin width df Hz
  rebin none               Reset to original binning
  zoom  <fmin> <fmax>      Set x-axis limits (Hz)
  clear                    Remove all components (keeps last fit)
  reset                    Clear components AND last fit result
  help                     Show this list
  quit                     Exit
"""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        freq:      np.ndarray,
        power:     np.ndarray,
        power_err: Optional[np.ndarray] = None,
        *,
        obsid:            str = "",
        band_label:       str = "full",
        warm_start_comps: Optional[Dict[str, Any]] = None,
        fit_fmin:   Optional[float] = None,
        fit_fmax:   Optional[float] = None,
        cand_fmin:  Optional[float] = None,
        cand_fmax:  Optional[float] = None,
        evt_path:   Optional[str]   = None,
        band_kev:   Optional[Tuple[float, float]] = None,
        source_struct_path: Optional[str] = None,
        m_eff:      int  = 1,
    ) -> None:
        self.freq      = np.asarray(freq,  float)
        self.power     = np.asarray(power, float)
        self.power_err = None if power_err is None else np.asarray(power_err, float)

        # PATCH: store true effective m from the rebinned PDS.  This is the
        # number of original FFT powers averaged into each bin (segments ×
        # bins-per-log-bin).  Used by fit_lorentzians via ps.m, by sigma
        # estimates, and by rchi2.
        self._m_eff = max(1, int(m_eff))

        self.obsid      = str(obsid)
        self.band_label = str(band_label)
        self._evt_path  = evt_path
        self._band_kev  = band_kev

        # Path of the struct file this session was seeded from.
        self._source_struct_path: Optional[str] = source_struct_path

        self.fit_fmin  = float(fit_fmin  or getattr(P, "FIT_FMIN",  0.05))
        self.fit_fmax  = float(fit_fmax  or getattr(P, "FIT_FMAX",  64.0))
        self.cand_fmin = float(cand_fmin or getattr(P, "CAND_FMIN", 0.05))
        self.cand_fmax = float(cand_fmax or getattr(P, "CAND_FMAX", 10.0))

        # ---- Tunable parameters ----
        self._params: Dict[str, Any] = {
            "dt":              float(getattr(P, "DT",               0.0078125)),
            "segment_size":    float(getattr(P, "SEGMENT_SIZE",     64.0)),
            "fit_fmin":        self.fit_fmin,
            "fit_fmax":        self.fit_fmax,
            "cand_fmin":       self.cand_fmin,
            "cand_fmax":       self.cand_fmax,
            "qpo_fmin":        float(getattr(P, "QPO_FMIN",         0.10)),
            "qpo_fmax":        float(getattr(P, "QPO_FMAX",         10.0)),
            "qpo_min_q":       float(getattr(P, "QPO_MIN_Q",        3.0)),
            "fit_rchi_max":    float(getattr(P, "FIT_RCHI_MAX",     1.5)),
            "fit_method":      str(getattr(P,   "FIT_METHOD",       "TripleA")),
            "fit_n_starts":    int(getattr(P,   "FIT_N_STARTS",     6)),
            "fit_jitter_frac": float(getattr(P, "FIT_JITTER_FRAC",  0.18)),
            "fit_random_seed": int(getattr(P,   "FIT_RANDOM_SEED",  42)),
            "cont_fwhm_min":   float(getattr(P, "FIT_CONT_FWHM_MIN", 0.30)),
            "cont_fwhm_max":   float(getattr(P, "FIT_CONT_FWHM_MAX", 64.0)),
            "qpo_fwhm_min":    float(getattr(P, "FIT_QPO_FWHM_MIN",  0.01)),
            "qpo_fwhm_max":    float(getattr(P, "FIT_QPO_FWHM_MAX",  5.0)),
            "cont_x0_max_hz":  float(getattr(P, "FIT_CONT_X0_MAX_HZ", 0.2)),
            "cont_amp_factor": float(getattr(P, "FIT_CONT_AMP_FACTOR", 5.0)),
            "qpo_amp_factor":  float(getattr(P, "FIT_QPO_AMP_FACTOR",  8.0)),
        }

        # name → (type, description, needs_rebuild, needs_remask)
        self._param_meta: Dict[str, Tuple] = {
            "dt":              (float, "Time resolution (s)",              True,  False),
            "segment_size":    (float, "APS segment size (s)",             True,  False),
            "fit_fmin":        (float, "Fit band lower limit (Hz)",        False, True),
            "fit_fmax":        (float, "Fit band upper limit (Hz)",        False, True),
            "cand_fmin":       (float, "Candidate band lower limit (Hz)",  False, False),
            "cand_fmax":       (float, "Candidate band upper limit (Hz)",  False, False),
            "qpo_fmin":        (float, "QPO detect lower limit (Hz)",      False, False),
            "qpo_fmax":        (float, "QPO detect upper limit (Hz)",      False, False),
            "qpo_min_q":       (float, "Min Q for QPO detection",          False, False),
            "fit_rchi_max":    (float, "rchi2 threshold for plot colouring",False,False),
            "fit_method":      (str,   "Optimizer (TripleA / Powell / Nelder-Mead)", False, False),
            "fit_n_starts":    (int,   "Number of multi-starts (Stingray path only)", False, False),
            "fit_jitter_frac": (float, "Multi-start jitter fraction",      False, False),
            "fit_random_seed": (int,   "RNG seed",                         False, False),
            "cont_fwhm_min":   (float, "Continuum FWHM lower bound (Hz)",  False, False),
            "cont_fwhm_max":   (float, "Continuum FWHM upper bound (Hz)",  False, False),
            "qpo_fwhm_min":    (float, "QPO FWHM lower bound (Hz)",        False, False),
            "qpo_fwhm_max":    (float, "QPO FWHM upper bound (Hz)",        False, False),
            "cont_x0_max_hz":  (float, "Continuum x0 |max| (Hz)",          False, False),
            "cont_amp_factor": (float, "Cont amp cap × low-f level",       False, False),
            "qpo_amp_factor":  (float, "QPO amp cap × low-f level",        False, False),
        }

        # ---- Component list ----
        self.components: List[Dict[str, Any]] = []
        if warm_start_comps:
            for nu0, fwhm, amp in warm_start_comps.get("cont", []):
                self.components.append(
                    {"nu0": float(nu0), "fwhm": float(fwhm),
                     "amp": float(amp), "type": "cont", "frozen": set()}
                )
            for nu0, fwhm, amp in warm_start_comps.get("qpo", []):
                self.components.append(
                    {"nu0": float(nu0), "fwhm": float(fwhm),
                     "amp": float(amp), "type": "qpo", "frozen": set()}
                )
            if warm_start_comps.get("const") is not None:
                c = float(warm_start_comps["const"])
                self._ws_const = c if (np.isfinite(c) and c > 0) else None
            else:
                self._ws_const = None
        else:
            self._ws_const = None

        # White-noise floor
        hf = (self.freq >= 40.0) & np.isfinite(self.power) & (self.power > 0)
        self.const = float(np.nanmedian(self.power[hf])) if np.any(hf) else 1e-4
        if self._ws_const is not None:
            self.const = self._ws_const

        self.fit_result: Optional[FitResult] = None

        # Fit-band display arrays
        sel = (
            (self.freq >= self.fit_fmin) & (self.freq <= self.fit_fmax)
            & np.isfinite(self.power) & (self.power > 0)
        )
        self._f = self.freq[sel]
        self._p = self.power[sel]
        self._e = None if self.power_err is None else self.power_err[sel]

        # Low-frequency power level (for amp caps, like batch fitter)
        lowf_mask = (self._f > 0) & (self._f <= 2.0) & np.isfinite(self._p) & (self._p > 0)
        if np.any(lowf_mask):
            self._lowf = float(np.nanmedian(self._p[lowf_mask]))
        else:
            self._lowf = float(np.nanmedian(self._p[self._p > 0])) if np.any(self._p > 0) else 1e-3

        # Originals for rebin
        self._f_orig         = self._f.copy()
        self._p_orig         = self._p.copy()
        self._e_orig         = None if self._e is None else self._e.copy()
        self._freq_orig      = self.freq.copy()
        self._power_orig     = self.power.copy()
        self._power_err_orig = None if self.power_err is None else self.power_err.copy()
        self._m_eff_orig     = self._m_eff
        self._rebin_mode:  str            = "none"
        self._rebin_value: Optional[float] = None

        self._comp_artists: List = []
        self._resid_fill = None

        self._build_figure()

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        title = "QPO Interactive Fitter"
        if self.obsid:     title += f"  —  {self.obsid}"
        if self.band_label: title += f"  [{self.band_label}]"
        self.fig.canvas.manager.set_window_title(title)
        self.fig.suptitle(title, fontsize=11)

        gs = gridspec.GridSpec(
            2, 1, figure=self.fig, height_ratios=[3, 1], hspace=0.06,
            left=0.09, right=0.97, top=0.93, bottom=0.09,
        )
        self.ax_pds = self.fig.add_subplot(gs[0])
        self.ax_res = self.fig.add_subplot(gs[1], sharex=self.ax_pds)

        self.ax_pds.set_xscale("log"); self.ax_pds.set_yscale("log")
        self.ax_pds.set_ylabel("Power (frac-rms² Hz⁻¹)", fontsize=10)
        self.ax_pds.tick_params(labelbottom=False)

        self.ax_res.set_xscale("log")
        self.ax_res.set_ylabel("(P − M) / σ", fontsize=9)
        self.ax_res.set_xlabel("Frequency (Hz)", fontsize=10)
        for y, ls, a in [(0, "-", 0.7), (3, "--", 0.4), (-3, "--", 0.4)]:
            self.ax_res.axhline(y, lw=0.9, ls=ls, alpha=a, color="gray")

        self._data_line, = self.ax_pds.plot(
            self._f, self._p, lw=0.9, color="black", alpha=0.75, zorder=2, label="PDS data"
        )
        self._model_line, = self.ax_pds.plot(
            [], [], lw=2.2, color="#D32F2F", zorder=6, label="Model"
        )
        self._const_line = self.ax_pds.axhline(
            self.const, lw=1.0, ls="--", color="#FB8C00", alpha=0.8, label="Const (fixed)"
        )
        self._resid_line, = self.ax_res.plot([], [], lw=0.9, color="#1565C0", alpha=0.85)
        self._resid_fill = None

        self.ax_pds.legend(fontsize=8, loc="upper right", framealpha=0.85)

        self._fit_ann = self.ax_pds.text(
            0.02, 0.04, "", transform=self.ax_pds.transAxes,
            fontsize=9, va="bottom", ha="left", family="monospace", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75),
        )

        plt.show(block=False)
        plt.pause(0.05)

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        for art in self._comp_artists:
            try: art.remove()
            except Exception: pass
        self._comp_artists.clear()

        if self._resid_fill is not None:
            try: self._resid_fill.remove()
            except Exception: pass
            self._resid_fill = None

        model = self._compute_model()
        self._data_line.set_data(self._f, self._p)
        self._model_line.set_data(self._f, model)
        self._const_line.set_ydata([self.const, self.const])

        cont_i = qpo_i = 0
        for idx, comp in enumerate(self.components):
            ctype = comp["type"]
            if ctype == "qpo":
                color  = _QPO_COLORS[qpo_i % len(_QPO_COLORS)]; qpo_i += 1; ls = "-"
            else:
                color  = _CONT_COLORS[cont_i % len(_CONT_COLORS)]; cont_i += 1; ls = "--"

            y = lorentz(self._f, comp["nu0"], comp["fwhm"], comp["amp"])
            frz = comp.get("frozen", set())

            curve, = self.ax_pds.loglog(self._f, y, lw=1.4, ls=ls, color=color, alpha=0.75, zorder=4)
            vl_p = self.ax_pds.axvline(comp["nu0"], lw=0.9, ls=":", color=color, alpha=0.65, zorder=3)
            vl_r = self.ax_res.axvline(comp["nu0"], lw=0.8, ls=":", color=color, alpha=0.55)

            Q   = comp["nu0"] / max(comp["fwhm"], 1e-12)
            frz_s = f"  ❄{''.join(sorted(f[0] for f in frz))}" if frz else ""
            tag = (
                f"[{idx}] {ctype[0].upper()}  {comp['nu0']:.3f} Hz"
                + (f"  Q={Q:.1f}" if ctype == "qpo" else "")
                + frz_s
            )
            txt = self.ax_pds.text(
                comp["nu0"], float(comp["amp"]) * 2.8, tag,
                fontsize=7, ha="center", va="bottom",
                color=color, clip_on=True, zorder=7,
            )
            self._comp_artists.extend([curve, vl_p, vl_r, txt])

        sigma = self._sigma(model)
        good  = np.isfinite(self._p) & np.isfinite(model) & np.isfinite(sigma) & (sigma > 0)
        resid = np.full(len(self._f), np.nan)
        resid[good] = (self._p[good] - model[good]) / sigma[good]
        self._resid_line.set_data(self._f, resid)

        r_ok = resid[np.isfinite(resid)]
        if r_ok.size > 5:
            p1, p99 = np.nanpercentile(r_ok, [1, 99])
            margin  = max(1.0, 0.15 * (p99 - p1))
            self.ax_res.set_ylim(max(p1 - margin, -12.0), min(p99 + margin, 12.0))

        self._resid_fill = self.ax_res.fill_between(
            self._f, -1, 1, color="gray", alpha=0.07, zorder=0
        )

        pos = np.concatenate([self._p[self._p > 0], model[model > 0]])
        if pos.size:
            self.ax_pds.set_ylim(float(np.nanmin(pos)) * 0.35, float(np.nanmax(pos)) * 5.0)

        self._update_fit_ann()
        self.fig.canvas.draw_idle()
        plt.pause(0.02)

    # ------------------------------------------------------------------
    # Fit annotation
    # ------------------------------------------------------------------

    def _update_fit_ann(self) -> None:
        fr = self.fit_result
        if fr is None:
            self._fit_ann.set_text(""); return

        rchi     = getattr(fr, "rchi2", np.nan)
        rchi_max = float(self._params.get("fit_rchi_max", getattr(P, "FIT_RCHI_MAX", 1.5)))
        ok_fit   = np.isfinite(rchi) and rchi <= rchi_max
        rchi_s   = f"{rchi:.3f}" if np.isfinite(rchi) else "nan"

        lines = [f"nlor={fr.nlor}   rchi2={rchi_s}   m={self._m_eff}"]

        qpos = extract_qpo_params_list(
            fr,
            qpo_fmin=float(self._params["cand_fmin"]),
            qpo_fmax=float(self._params["cand_fmax"]),
            qmin=float(self._params["qpo_min_q"]),
        )
        for q in qpos:
            rms2 = np.nan
            try:
                nu0_q, fwhm_q, amp_q = fr.pars[int(q["qpo_index"])]
                rms2 = component_power_integral(
                    fr.freq, lorentz(fr.freq, nu0_q, fwhm_q, amp_q),
                    float(self._params["fit_fmin"]), float(self._params["fit_fmax"]),
                )
            except Exception:
                pass
            rms_s = f"   rms={np.sqrt(max(rms2,0)):.4f}" if np.isfinite(rms2) else ""
            lines.append(
                f"QPO  ν={q['qpo_nu0_hz']:.4f} Hz"
                f"   FWHM={q['qpo_fwhm_hz']:.4f} Hz"
                f"   Q={q['qpo_Q']:.1f}{rms_s}"
            )
        if not qpos:
            lines.append("No QPO detected")

        self._fit_ann.set_text("\n".join(lines))
        self._fit_ann.set_color("#1B5E20" if ok_fit else "#B71C1C")

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _compute_model(self) -> np.ndarray:
        mod = np.full(len(self._f), self.const)
        for c in self.components:
            mod += lorentz(self._f, c["nu0"], c["fwhm"], c["amp"])
        return mod

    def _sigma(self, model: np.ndarray) -> np.ndarray:
        """σ = model / sqrt(m_eff).  Uses the true rebinned m, not seg/(2 dt)."""
        if self._e is not None:
            return self._e.copy()
        m = max(1, int(self._m_eff))
        return np.where(model > 0, model / np.sqrt(m), np.nan)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ask(label: str, default=None, required: bool = False) -> Optional[str]:
        dflt = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"    {label}{dflt}: ").strip()
            if not raw:
                if default is not None: return str(default)
                if required: print(f"    {_RD}Required.{_R}"); continue
                return None
            return raw

    @classmethod
    def _ask_float(cls, label: str, default=None, required: bool = False,
                   lo: float = -np.inf, hi: float = np.inf) -> Optional[float]:
        dflt_s = f"{default:.4g}" if isinstance(default, float) else default
        while True:
            raw = cls._ask(label, default=dflt_s, required=required)
            if raw is None: return None
            try: v = float(raw)
            except ValueError: print(f"    {_RD}Not a number.{_R}"); continue
            if not (lo <= v <= hi): print(f"    {_RD}Must be in [{lo}, {hi}].{_R}"); continue
            return v

    def _local_amp(self, nu0: float) -> float:
        if self._f.size == 0: return 1e-3
        return float(self._p[int(np.argmin(np.abs(self._f - nu0)))])

    # ------------------------------------------------------------------
    # Commands — components
    # ------------------------------------------------------------------

    def _cmd_addcomp(self, args: List[str]) -> None:
        if not args:
            print(f"  {_YL}Usage: addcomp qpo | addcomp continuum{_R}"); return

        raw = args[0].lower()
        if raw == "qpo":
            ctype = "qpo"
        elif raw in ("continuum", "cont", "c"):
            ctype = "cont"
        else:
            print(f"  {_RD}Unknown type {raw!r}.  Use 'qpo' or 'continuum'.{_R}"); return

        print(f"\n  {_B}Add {ctype.upper()}{_R}  {_D}(press Enter to accept default){_R}")

        if ctype == "cont":
            nu0 = self._ask_float("Centre ν₀ (Hz)", default=0.0,
                                  lo=self.fit_fmin, hi=self.fit_fmax)
        else:
            nu0 = self._ask_float("Centre ν₀ (Hz)", required=True,
                                  lo=self.cand_fmin, hi=self.cand_fmax)
        if nu0 is None: print(f"  {_YL}Aborted.{_R}"); return

        if ctype == "qpo":
            dfwhm = round(max(0.06 * nu0, float(getattr(P, "FIT_QPO_FWHM_MIN", 0.03))), 4)
            flo, fhi = float(self._params["qpo_fwhm_min"]), float(self._params["qpo_fwhm_max"])
        else:
            dfwhm = 5.0
            flo, fhi = float(self._params["cont_fwhm_min"]), float(self._params["cont_fwhm_max"])

        fwhm = self._ask_float("FWHM (Hz)", default=dfwhm, lo=flo, hi=fhi)
        if fwhm is None: fwhm = dfwhm

        auto_amp = self._local_amp(nu0)
        amp = self._ask_float(f"Amplitude (auto={auto_amp:.3g})", default=auto_amp, lo=0.0)
        if amp is None: amp = auto_amp

        self.components.append(
            {"nu0": float(nu0), "fwhm": float(fwhm), "amp": float(amp),
             "type": ctype, "frozen": set()}
        )
        idx = len(self.components) - 1
        Q_s = f"  Q = {nu0/max(fwhm,1e-12):.1f}" if ctype == "qpo" else ""
        print(f"\n  {_GR}Added {ctype.upper()} [{idx}]{_R}"
              f"  ν = {nu0:.4f} Hz  FWHM = {fwhm:.4f} Hz  amp = {amp:.3g}{Q_s}")
        self._redraw()

    def _cmd_removecomp(self, args: List[str]) -> None:
        if not args: print(f"  {_YL}Usage: removecomp <index>{_R}"); return
        try: idx = int(args[0])
        except ValueError: print(f"  {_RD}Index must be an integer.{_R}"); return
        if not (0 <= idx < len(self.components)):
            print(f"  {_RD}No component [{idx}].{_R}"); return
        removed = self.components.pop(idx)
        print(f"  {_YL}Removed [{idx}] {removed['type'].upper()}  ν = {removed['nu0']:.4f} Hz{_R}")
        self._redraw()

    def _cmd_editcomp(self, args: List[str]) -> None:
        if not args: print(f"  {_YL}Usage: editcomp <index>{_R}"); return
        try: idx = int(args[0])
        except ValueError: print(f"  {_RD}Index must be an integer.{_R}"); return
        if not (0 <= idx < len(self.components)):
            print(f"  {_RD}No component [{idx}].{_R}"); return

        comp = self.components[idx]
        frz  = comp.get("frozen", set())
        print(f"\n  {_B}Edit [{idx}] {comp['type'].upper()}{_R}"
              f"  {_D}(Enter = keep current; frozen fields are marked ❄){_R}")

        for field, label, lo, hi in [
            ("nu0",  "Centre ν₀ (Hz)", -np.inf, np.inf),
            ("fwhm", "FWHM (Hz)",       0.0,    np.inf),
            ("amp",  "Amplitude",        0.0,    np.inf),
        ]:
            if field in frz:
                print(f"    {label}: {comp[field]:.4g}  {_D}[frozen]{_R}")
                continue
            v = self._ask_float(label, default=comp[field])
            if v is not None: comp[field] = v

        Q_s = f"  Q = {comp['nu0']/max(comp['fwhm'],1e-12):.1f}" if comp["type"]=="qpo" else ""
        print(f"\n  {_GR}Updated [{idx}]{_R}"
              f"  ν = {comp['nu0']:.4f} Hz  FWHM = {comp['fwhm']:.4f} Hz"
              f"  amp = {comp['amp']:.3g}{Q_s}")
        self._redraw()

    def _cmd_freeze(self, args: List[str]) -> None:
        if not args: print(f"  {_YL}Usage: freeze <idx> [nu0|fwhm|amp]{_R}"); return
        try: idx = int(args[0])
        except ValueError: print(f"  {_RD}Index must be an integer.{_R}"); return
        if not (0 <= idx < len(self.components)):
            print(f"  {_RD}No component [{idx}].{_R}"); return

        comp  = self.components[idx]
        frz   = comp.setdefault("frozen", set())
        valid = {"nu0", "fwhm", "amp"}
        fields = {args[1].lower()} if len(args) > 1 else valid

        bad = fields - valid
        if bad: print(f"  {_RD}Unknown field(s): {bad}.  Must be nu0, fwhm, amp.{_R}"); return

        frz |= fields
        print(f"  {_CY}[{idx}] frozen: {sorted(frz)}{_R}")
        self._redraw()

    def _cmd_unfreeze(self, args: List[str]) -> None:
        if not args: print(f"  {_YL}Usage: unfreeze <idx> [nu0|fwhm|amp]{_R}"); return
        try: idx = int(args[0])
        except ValueError: print(f"  {_RD}Index must be an integer.{_R}"); return
        if not (0 <= idx < len(self.components)):
            print(f"  {_RD}No component [{idx}].{_R}"); return

        comp  = self.components[idx]
        frz   = comp.setdefault("frozen", set())
        valid = {"nu0", "fwhm", "amp"}
        fields = {args[1].lower()} if len(args) > 1 else valid

        bad = fields - valid
        if bad: print(f"  {_RD}Unknown field(s): {bad}.{_R}"); return

        frz -= fields
        print(f"  {_GR}[{idx}] frozen: {sorted(frz) or 'none'}{_R}")
        self._redraw()

    def _cmd_setconst(self, args: List[str]) -> None:
        if not args: print(f"  {_YL}Usage: setconst <value>{_R}"); return
        try: val = float(args[0])
        except ValueError: print(f"  {_RD}Not a number.{_R}"); return
        if val <= 0: print(f"  {_RD}Must be positive.{_R}"); return
        self.const = val
        print(f"  Constant set to {val:.4g}  {_D}(held fixed during fit){_R}")
        self._redraw()

    def _cmd_list(self) -> None:
        sep = "─" * 72
        print(f"\n  {_B}{sep}{_R}")
        print(f"  {'Idx':>4}  {'Type':6}  {'ν₀ (Hz)':>10}  "
              f"{'FWHM (Hz)':>10}  {'Amplitude':>12}  {'Q':>7}  Frozen")
        print(f"  {sep}")
        if not self.components:
            print(f"  {_D}  (no components){_R}")
        for i, c in enumerate(self.components):
            Q   = c["nu0"] / max(c["fwhm"], 1e-12)
            Q_s = f"{Q:7.1f}" if c["type"] == "qpo" else "      —"
            frz = c.get("frozen", set())
            frz_s = ",".join(sorted(frz)) if frz else "—"
            print(f"  [{i:>2}]  {c['type']:6s}  "
                  f"{c['nu0']:>10.4f}  {c['fwhm']:>10.4f}  "
                  f"{c['amp']:>12.4g}  {Q_s}  {frz_s}")
        print(f"  {sep}")
        print(f"  White-noise const = {self.const:.4g}  {_D}(fixed){_R}")
        print(f"  m_eff (per bin)   = {self._m_eff}")
        print(f"  {sep}\n")

    # ------------------------------------------------------------------
    # Commands — fitting
    # ------------------------------------------------------------------

    def _cmd_fit(self) -> None:
        """Fit with the current component structure. const is always held fixed."""
        if not self.components:
            print(f"  {_YL}No components yet.  Use addcomp first.{_R}"); return

        frz_summary = []
        for i, c in enumerate(self.components):
            frz = c.get("frozen", set())
            if frz: frz_summary.append(f"[{i}]:{','.join(sorted(frz))}")
        if frz_summary:
            print(f"  {_D}Frozen: {' '.join(frz_summary)}{_R}")

        fmethod = str(self._params["fit_method"])
        print(f"  {_BL}Running optimiser ({fmethod})…  "
              f"(const={self.const:.4g} fixed, m_eff={self._m_eff}){_R}", flush=True)

        fitres = self._run_direct_fit()
        if fitres is None:
            print(f"  {_RD}All optimiser starts failed.{_R}"); return

        self.fit_result = fitres

        self.components.clear()
        pars       = np.asarray(fitres.pars, float)
        comp_types = list(getattr(fitres, "comp_types", []) or [])
        for i, (nu0_i, fwhm_i, amp_i) in enumerate(pars):
            ctype = comp_types[i] if i < len(comp_types) else "cont"
            self.components.append(
                {"nu0": float(nu0_i), "fwhm": float(fwhm_i),
                 "amp": float(amp_i), "type": str(ctype), "frozen": set()}
            )
        self.const = float(getattr(fitres, "const", self.const) or self.const)

        self._model_line.set_data(self._f, fitres.model)
        self._redraw()
        self._cmd_status()
        print(f"  {_D}Type 'saveresult' to save.{_R}")

    def _run_direct_fit(self) -> Optional[FitResult]:
            """
            Optimise directly with no IC, no guardrails.

            Supports two optimiser paths controlled by self._params["fit_method"]:

            TripleA path (default)
            ----------------------
            Calls QPO_TripleA.tripleA_fit_once once with the current component
            seed.  TripleA manages its own multi-start loop (AAA_N_STARTS restarts)
            internally — the outer Stingray loop is bypassed entirely.
            Frozen parameters are implemented as tight L-BFGS-B box bounds rather
            than prior penalty windows.

            Stingray path (Powell / Nelder-Mead / other)
            ---------------------------------------------
            Unchanged from the previous version: builds priors as half-uniform /
            hard-truncated-uniform penalty functions and calls fit_lorentzians in
            a jittered multi-start loop.

            Design notes (shared)
            ---------------------
            * ps.m is set to self._m_eff (the true number of averaged powers per
              rebinned bin), so the Whittle likelihood weight matches the batch
              fitter.
            * The white-noise const is NOT pinned.  It is given a half-uniform
              prior over [0, 2 × const_seed], identical to the batch fitter.
            * Per-component bounds are derived from the CURRENT value of each
              loaded/edited component, not from global limits.  This guarantees
              warm-start parameters are strictly interior to their priors, so
              the optimiser's first probe step never lands on a hard-truncation
              boundary (where prior = 0 → log-posterior = -inf → sentinel).
            * Frozen fields get a ±0.1 % pin window for the Stingray path and a
              tight box bound for the TripleA path — semantics are identical.
            """
            nlor = len(self.components)
            if nlor == 0:
                return None

            fmethod  = str(self._params["fit_method"])
            n_starts = int(self._params["fit_n_starts"])
            jitter   = float(self._params["fit_jitter_frac"])
            cfmin    = float(self._params["cand_fmin"])
            cfmax    = float(self._params["cand_fmax"])
            x0_max   = float(self._params["cont_x0_max_hz"])
            cont_af  = float(self._params["cont_amp_factor"])
            qpo_af   = float(self._params["qpo_amp_factor"])
            cfwhm_lo = float(self._params["cont_fwhm_min"])
            cfwhm_hi = float(self._params["cont_fwhm_max"])
            qfwhm_lo = float(self._params["qpo_fwhm_min"])
            qfwhm_hi = float(self._params["qpo_fwhm_max"])
            eps      = 1e-30

            # Build the Powerspectrum object.
            ps           = Powerspectrum()
            ps.freq      = self._f.copy()
            ps.power     = self._p.copy()
            ps.power_err = None if self._e is None else self._e.copy()
            ps.df        = float(np.median(np.diff(self._f))) if self._f.size > 1 else 1.0
            ps.m         = int(self._m_eff)
            ps.norm      = "frac"

            # Re-derive const seed from the data's high-frequency median.
            hf_mask = (self._f >= 30.0) & np.isfinite(self._p) & (self._p > 0)
            if np.any(hf_mask):
                const_seed = float(np.nanmedian(self._p[hf_mask]))
            else:
                const_seed = float(self.const)
            const_seed = max(const_seed, 1e-30)
            const_cap  = max(2.0 * const_seed, 1e-29)

            # Global amp caps from the low-frequency PDS level.
            cont_amp_cap_g = max(cont_af * self._lowf, 1e-10)
            qpo_amp_cap_g  = max(qpo_af  * self._lowf, 1e-10)

            # Per-component bounds.
            comp_types: List[str] = []
            x0_lims:    List[Tuple[float, float]] = []
            fwhm_lims:  List[Tuple[float, float]] = []
            amp_caps:   List[float] = []

            for c in self.components:
                ctype = str(c["type"])
                comp_types.append(ctype)

                cur_nu0  = float(c["nu0"])
                cur_fwhm = float(c["fwhm"])
                cur_amp  = float(c["amp"])

                if ctype == "qpo":
                    lo = max(cfmin, cur_nu0 - 1.0)
                    hi = min(cfmax, cur_nu0 + 1.0)
                    if cur_nu0 - lo < 1e-6: lo = max(cfmin, cur_nu0 - max(0.05, 0.05 * cur_nu0))
                    if hi - cur_nu0 < 1e-6: hi = min(cfmax, cur_nu0 + max(0.05, 0.05 * cur_nu0))
                    x0_lims.append((lo, hi))

                    f_lo = max(qfwhm_lo, cur_fwhm / 5.0)
                    f_hi = min(qfwhm_hi, cur_fwhm * 5.0)
                    if f_hi <= f_lo: f_lo, f_hi = qfwhm_lo, qfwhm_hi
                    fwhm_lims.append((f_lo, f_hi))

                    amp_caps.append(max(qpo_amp_cap_g, 20.0 * cur_amp, 1e-10))

                else:  # continuum
                    half = max(x0_max, abs(cur_nu0) + 0.5, 0.5)
                    x0_lims.append((-half, +half))

                    f_lo = max(cfwhm_lo, cur_fwhm / 5.0)
                    f_hi = min(cfwhm_hi, cur_fwhm * 5.0)
                    if f_hi <= f_lo: f_lo, f_hi = cfwhm_lo, cfwhm_hi
                    fwhm_lims.append((f_lo, f_hi))

                    amp_caps.append(max(cont_amp_cap_g, 20.0 * cur_amp, 1e-10))

            # Build t0 from current component values.
            base_t0 = np.array([
                val
                for c in self.components
                for val in (float(c["amp"]), float(c["nu0"]), float(c["fwhm"]))
            ] + [const_seed], dtype=float)

            # Sanity check: ensure every t0 component is strictly interior to its bounds.
            for i in range(nlor):
                a  = base_t0[3*i]
                x  = base_t0[3*i+1]
                w  = base_t0[3*i+2]
                lo_x, hi_x = x0_lims[i]
                f_lo, f_hi = fwhm_lims[i]
                if not (lo_x < x < hi_x):
                    print(f"  {_YL}[warn] comp [{i}] nu0={x:.4g} on x0 bound "
                          f"[{lo_x:.4g}, {hi_x:.4g}] — widening{_R}")
                if not (f_lo < w < f_hi):
                    print(f"  {_YL}[warn] comp [{i}] fwhm={w:.4g} on fwhm bound "
                          f"[{f_lo:.4g}, {f_hi:.4g}] — widening{_R}")
                if a >= amp_caps[i]:
                    print(f"  {_YL}[warn] comp [{i}] amp={a:.4g} ≥ cap={amp_caps[i]:.4g} "
                          f"— widening{_R}")

            rng      = np.random.default_rng(int(self._params.get("fit_random_seed", 42)))
            best_res = None
            best_aic = np.inf
            n_failed = 0

            # ---- TripleA short-circuit ------------------------------------
            # TripleA handles its own multi-start loop internally.
            # Frozen fields become tight box bounds; the Stingray prior-penalty
            # approach is not used.
            if fmethod.lower() == "triplea":
                # Build per-component amplitude lower bounds and tighten bounds
                # for any frozen fields.
                aaa_amp_lo = []
                aaa_x0     = list(x0_lims)
                aaa_fwhm   = list(fwhm_lims)
                aaa_acaps  = list(amp_caps)

                for i, c in enumerate(self.components):
                    frz = c.get("frozen", set())
                    v_a = max(float(c["amp"]),  eps)
                    v_x = float(c["nu0"])
                    v_f = max(float(c["fwhm"]), eps)

                    # Amplitude lower bound (always eps unless frozen)
                    if "amp" in frz:
                        w = max(v_a * 1e-3, 1e-12)
                        aaa_amp_lo.append(max(v_a - w, eps))
                        aaa_acaps[i] = v_a + w
                    else:
                        aaa_amp_lo.append(eps)

                    # ν₀ bounds (tight window if frozen)
                    if "nu0" in frz:
                        w = max(abs(v_x) * 1e-3, 1e-6)
                        aaa_x0[i] = (v_x - w, v_x + w)

                    # FWHM bounds (tight window if frozen)
                    if "fwhm" in frz:
                        w = max(v_f * 1e-3, 1e-6)
                        aaa_fwhm[i] = (v_f - w, v_f + w)

                best_res, _exc = _tripleA_fit_once(
                    ps,
                    nlor          = nlor,
                    t0            = base_t0.tolist(),
                    include_const = True,
                    x0_lims       = aaa_x0,
                    fwhm_lims     = aaa_fwhm,
                    amp_caps      = aaa_acaps,
                    amp_lo_list   = aaa_amp_lo,
                    const_max     = const_cap,
                )
                if best_res is None:
                    print(f"  {_RD}TripleA failed: {_exc}{_R}")
                    print(f"  {_D}Try: setparam fit_method Powell{_R}")
                    return None
                best_aic = float(best_res.aic)
                # n_failed stays 0 — TripleA handles retries internally.

            else:
                # ---- Stingray / Powell / Nelder-Mead loop (unchanged) ----

                # Build priors.  const is FREE (half-uniform), not pinned.
                priors: Dict[str, Any] = {}
                for i, c in enumerate(self.components):
                    frz = c.get("frozen", set())

                    if "amp" in frz:
                        v = max(float(c["amp"]), eps)
                        w = max(v * 1e-3, 1e-12)
                        priors[f"amplitude_{i}"] = _hard_trunc_uniform(v - w, v + w)
                    else:
                        priors[f"amplitude_{i}"] = _half_uniform(amp_caps[i])

                    if "nu0" in frz:
                        v = float(c["nu0"])
                        w = max(abs(v) * 1e-3, 1e-6)
                        priors[f"x_0_{i}"] = _hard_trunc_uniform(v - w, v + w)
                    else:
                        priors[f"x_0_{i}"] = _hard_trunc_uniform(*x0_lims[i])

                    if "fwhm" in frz:
                        v = max(float(c["fwhm"]), eps)
                        w = max(v * 1e-3, 1e-6)
                        priors[f"fwhm_{i}"] = _hard_trunc_uniform(v - w, v + w)
                    else:
                        priors[f"fwhm_{i}"] = _hard_trunc_uniform(*fwhm_lims[i])

                priors[f"amplitude_{nlor}"] = _half_uniform(const_cap)

                for start_idx in range(max(1, n_starts)):
                    t0 = base_t0.copy()

                    if start_idx > 0 and jitter > 0:
                        for i, c in enumerate(self.components):
                            frz   = c.get("frozen", set())
                            lo_x, hi_x = x0_lims[i]
                            flo, fhi   = fwhm_lims[i]

                            if "amp" not in frz:
                                t0[3*i]   = max(eps, t0[3*i] * (1.0 + jitter * rng.standard_normal()))
                            if "nu0" not in frz:
                                t0[3*i+1] = float(np.clip(
                                    t0[3*i+1] + jitter * (hi_x - lo_x) * 0.5 * rng.standard_normal(),
                                    lo_x + 1e-9 * (hi_x - lo_x),
                                    hi_x - 1e-9 * (hi_x - lo_x),
                                ))
                            if "fwhm" not in frz:
                                t0[3*i+2] = float(np.clip(
                                    t0[3*i+2] * max(0.5, 1.0 + jitter * rng.standard_normal()),
                                    flo + 1e-9 * (fhi - flo),
                                    fhi - 1e-9 * (fhi - flo),
                                ))
                        t0[-1] = float(np.clip(
                            const_seed * (1.0 + 0.05 * rng.standard_normal()),
                            1e-30, const_cap * 0.999,
                        ))

                    t0 = _repair_params(
                        t0, nlor=nlor, include_const=True,
                        x0_lims=x0_lims, fwhm_lims=fwhm_lims,
                        const_max=const_cap, eps_amp=eps,
                    )

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            _par, res = fit_lorentzians(
                                ps, nlor, t0.tolist(),
                                fit_whitenoise=True,
                                max_post=True,
                                priors=priors,
                                fitmethod=fmethod,
                            )
                    except Exception:
                        n_failed += 1
                        continue

                    if res is None:
                        n_failed += 1
                        continue

                    aic = float(getattr(res, "aic", np.inf))
                    if aic >= 1e15:
                        n_failed += 1
                        continue

                    if aic < best_aic:
                        best_aic = aic
                        best_res = res

                if best_res is None:
                    print(f"  {_RD}All {n_starts} optimizer starts failed "
                          f"({n_failed} returned sentinel/exception).{_R}")
                    print(f"  {_D}Try: setparam fit_method TripleA{_R}")
                    print(f"  {_D}Or:  edit a component slightly to nudge the start point{_R}")
                    return None

            # ---- Extract result (common to both paths) --------------------
            p_opt    = np.asarray(best_res.p_opt, float)
            pars_out = np.array(
                [(p_opt[3*i+1], p_opt[3*i+2], p_opt[3*i]) for i in range(nlor)], float
            )
            const_val = float(p_opt[-1])
            model     = np.asarray(best_res.mfit, float)
            rchi2     = _compute_rchi2(self._p, model, self._e,
                                        m_avg=int(self._m_eff), npar=p_opt.size)

            # Update self.const with the fitted value.
            self.const = const_val

            return FitResult(
                ok=True,
                message=(f"OK (direct  nlor={nlor}  m={self._m_eff}  "
                          f"const={const_val:.3g}  method={fmethod})"),
                nlor=nlor,
                pars=pars_out,
                comp_types=comp_types,
                const=const_val,
                freq=self._f,
                model=model,
                aic=float(getattr(best_res, "aic",     np.nan)),
                bic=float(getattr(best_res, "bic",     np.nan)),
                deviance=float(getattr(best_res, "deviance", np.nan)),
                rchi2=rchi2,
                stingray_p_opt=p_opt,
                meta={"direct_fit": True, "n_starts": n_starts,
                      "fitmethod": fmethod, "m_eff": int(self._m_eff),
                      "const_seed": const_seed, "n_failed_starts": n_failed},
            )

    # ------------------------------------------------------------------
    # Commands — status, save, load
    # ------------------------------------------------------------------

    def _cmd_status(self) -> None:
        if self.fit_result is None:
            print(f"  {_YL}No fit yet — run 'fit' first.{_R}"); return

        fr   = self.fit_result
        rchi = getattr(fr, "rchi2", np.nan)
        rchi_max = float(self._params.get("fit_rchi_max", getattr(P, "FIT_RCHI_MAX", 1.5)))
        rc   = _GR if (np.isfinite(rchi) and rchi <= rchi_max) else _RD

        print(f"\n  {_B}── Fit status ─────────────────────{_R}")
        print(f"  nlor  = {fr.nlor}")
        print(f"  rchi2 = {rc}{rchi:.3f}{_R}" if np.isfinite(rchi) else "  rchi2 = nan")
        print(f"  const = {self.const:.4g}  {_D}(fixed){_R}")
        print(f"  m_eff = {self._m_eff}")

        qpos = extract_qpo_params_list(
            fr, qpo_fmin=self.cand_fmin, qpo_fmax=self.cand_fmax,
            qmin=float(self._params["qpo_min_q"]),
        )
        if qpos:
            print(f"  {_B}── QPO(s) ──────────────────────────{_R}")
            for q in qpos:
                rms2 = np.nan
                try:
                    nu0_q, fwhm_q, amp_q = fr.pars[int(q["qpo_index"])]
                    rms2 = component_power_integral(
                        fr.freq, lorentz(fr.freq, nu0_q, fwhm_q, amp_q),
                        float(self._params["fit_fmin"]), float(self._params["fit_fmax"]),
                    )
                except Exception:
                    pass
                rms_s = f"  rms = {np.sqrt(max(rms2,0)):.4f}" if np.isfinite(rms2) else ""
                print(f"  {_GR}  ν = {q['qpo_nu0_hz']:.4f} Hz"
                      f"  FWHM = {q['qpo_fwhm_hz']:.4f} Hz"
                      f"  Q = {q['qpo_Q']:.2f}{rms_s}{_R}")
        else:
            print(f"  {_D}  No QPO detected (Q < {self._params['qpo_min_q']}){_R}")
        print()

    def _cmd_saveresult(self, args: List[str]) -> None:
        if self.fit_result is None:
            print(f"  {_YL}No fit result — run 'fit' first.{_R}"); return

        if args:
            target_path = args[0]
            self._write_struct_to_path(target_path)
        elif self._source_struct_path is not None:
            self._write_struct_to_path(self._source_struct_path)
        else:
            saved = save_fit_struct(self.fit_result, self.obsid, self.band_label)
            self._source_struct_path = saved
            s = load_fit_struct(self.obsid)
            if s:
                print(f"  {_GR}Saved.{_R}  {struct_summary(s)}")
            else:
                print(f"  {_GR}Saved: {saved}{_R}")

    def _write_struct_to_path(self, path: str) -> None:
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        from QPO_struct import _make_band_block, _STRUCT_VERSION, _BANDS

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        existing: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    existing = _json.load(fh)
            except Exception:
                pass

        struct = {
            "version":   _STRUCT_VERSION,
            "obsid":     self.obsid or existing.get("obsid", ""),
            "source":    getattr(P, "SOURCE", "") or existing.get("source", ""),
            "mjd_mid":   existing.get("mjd_mid"),
            "timestamp": _dt.now(_tz.utc).isoformat(),
        }
        for b in _BANDS:
            if b in existing and b != self.band_label:
                struct[b] = existing[b]
        struct[self.band_label] = _make_band_block(self.fit_result)

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(struct, fh, indent=2)
        os.replace(tmp, path)
        self._source_struct_path = path
        print(f"  {_GR}Saved: {path}{_R}")

    def _cmd_load(self, args: List[str]) -> None:
        if not args:
            print(f"  {_YL}Usage: load <obsid> [band]  |  load <path.json> [band]{_R}"); return

        first = args[0]
        band  = args[1] if len(args) > 1 else self.band_label

        if first.endswith(".json") or os.sep in first:
            path = first
            if not os.path.exists(path):
                print(f"  {_RD}File not found: {path}{_R}"); return
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    s = json.load(fh)
            except Exception as exc:
                print(f"  {_RD}Could not read {path}: {exc}{_R}"); return
        else:
            obsid = first
            s = load_fit_struct(obsid)
            path = struct_path(obsid)
            if s is None:
                print(f"  {_RD}No struct found for obsid {obsid!r}{_R}")
                print(f"  {_D}  Expected: {path}{_R}"); return

        print(f"\n  {_GR}Struct:{_R}  {struct_summary(s)}")

        from QPO_struct import _BANDS
        available = [b for b in _BANDS if s.get(b, {}).get("ok", False)]
        if not available:
            print(f"  {_RD}No successfully-fitted bands in this struct.{_R}"); return

        if band not in available:
            print(f"  {_YL}Band {band!r} not in struct.  Available: {available}{_R}")
            if len(available) == 1:
                band = available[0]
                print(f"  Using {band!r} instead.")
            else:
                return

        ws = struct_to_warm_comps(s, band)
        if ws is None:
            print(f"  {_RD}Could not extract components for band {band!r}.{_R}"); return

        self.components.clear()
        for nu0, fwhm, amp in ws.get("cont", []):
            self.components.append(
                {"nu0": float(nu0), "fwhm": float(fwhm), "amp": float(amp),
                 "type": "cont", "frozen": set()}
            )
        for nu0, fwhm, amp in ws.get("qpo", []):
            self.components.append(
                {"nu0": float(nu0), "fwhm": float(fwhm), "amp": float(amp),
                 "type": "qpo", "frozen": set()}
            )
        if ws.get("const") is not None:
            self.const = float(ws["const"])

        self._source_struct_path = path
        self.fit_result = None
        self._model_line.set_data([], [])

        print(f"  Loaded {len(self.components)} component(s) from [{band}].")
        self._cmd_list()
        self._redraw()

    def _cmd_save(self, args: List[str]) -> None:
        path = args[0] if args else (
            f"{self.obsid}_{self.band_label}_interactive.png"
            if self.obsid else "qpo_interactive.png"
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fig.savefig(path, dpi=int(getattr(P, "PLOT_DPI", 150)), bbox_inches="tight")
        print(f"  {_GR}Saved: {path}{_R}")

    def _cmd_zoom(self, args: List[str]) -> None:
        if len(args) < 2: print(f"  {_YL}Usage: zoom <fmin> <fmax>{_R}"); return
        try: fmin, fmax = float(args[0]), float(args[1])
        except ValueError: print(f"  {_RD}Both arguments must be numbers.{_R}"); return
        if fmin >= fmax or fmin <= 0: print(f"  {_RD}Need 0 < fmin < fmax.{_R}"); return
        self.ax_pds.set_xlim(fmin, fmax)
        self.fig.canvas.draw_idle(); plt.pause(0.02)

    # ------------------------------------------------------------------
    # Commands — rebin
    # ------------------------------------------------------------------

    def _apply_rebin(self, mode: str, value: Optional[float]) -> None:
        """
        Re-rebin the original PDS arrays.

        PATCH: also recompute self._m_eff from the rebinned pds.m so the
        likelihood weight stays correct after the user types `rebin`.
        """
        from stingray import Powerspectrum as _SP

        if mode == "none":
            self._f = self._f_orig.copy(); self._p = self._p_orig.copy()
            self._e = None if self._e_orig is None else self._e_orig.copy()
            self.freq = self._freq_orig.copy(); self.power = self._power_orig.copy()
            self.power_err = (None if self._power_err_orig is None
                              else self._power_err_orig.copy())
            self._m_eff = self._m_eff_orig
            return

        def _rb(f, p, e, m_in):
            ps = _SP(); ps.freq = f; ps.power = p
            ps.power_err = np.zeros_like(p) if e is None else e
            ps.df = float(np.median(np.diff(f))); ps.m = int(m_in); ps.norm = "frac"
            return (ps.rebin_log(f=float(value)) if mode == "log"
                    else ps.rebin(df=float(value)))

        rb = _rb(self._f_orig, self._p_orig,
                 np.zeros_like(self._p_orig) if self._e_orig is None else self._e_orig,
                 self._m_eff_orig)
        self._f = np.asarray(rb.freq, float); self._p = np.asarray(rb.power, float)
        self._e = np.asarray(rb.power_err, float) if rb.power_err is not None else None
        self._m_eff = safe_m_from_pds(rb)

        rb2 = _rb(self._freq_orig, self._power_orig,
                  np.zeros_like(self._power_orig) if self._power_err_orig is None
                  else self._power_err_orig,
                  self._m_eff_orig)
        self.freq = np.asarray(rb2.freq, float); self.power = np.asarray(rb2.power, float)
        self.power_err = np.asarray(rb2.power_err, float) if rb2.power_err is not None else None

    def _cmd_rebin(self, args: List[str]) -> None:
        if not args:
            label = (f"log f={self._rebin_value}" if self._rebin_mode == "log"
                     else f"linear df={self._rebin_value} Hz" if self._rebin_mode == "linear"
                     else "none (original)")
            print(f"  Rebin: {_CY}{label}{_R}  ({len(self._f)} bins, m_eff={self._m_eff})"); return

        mode = args[0].lower()
        if mode == "none":
            self._rebin_mode = "none"; self._rebin_value = None
            self._apply_rebin("none", None)
            print(f"  {_GR}Reset to original binning.{_R}  "
                  f"({len(self._f)} bins, m_eff={self._m_eff})")
            self._redraw(); return

        if mode not in ("log", "linear"):
            print(f"  {_RD}Use: rebin log <f> | rebin linear <df> | rebin none{_R}"); return
        if len(args) < 2:
            print(f"  {_YL}Usage: rebin {mode} <value>{_R}"); return
        try: value = float(args[1])
        except ValueError: print(f"  {_RD}Value must be a number.{_R}"); return
        if value <= 0: print(f"  {_RD}Must be positive.{_R}"); return
        if mode == "log" and value >= 1.0:
            print(f"  {_RD}Log factor must be < 1 (e.g. 0.02).{_R}"); return

        orig_n = len(self._f_orig)
        self._apply_rebin(mode, value)
        self._rebin_mode = mode; self._rebin_value = value
        label = f"log f={value}" if mode == "log" else f"linear df={value} Hz"
        print(f"  {_GR}Rebinned ({label}){_R}  {orig_n} → {len(self._f)} bins  "
              f"(m_eff={self._m_eff})")
        self._redraw()

    # ------------------------------------------------------------------
    # Commands — params
    # ------------------------------------------------------------------

    def _remask(self) -> None:
        fmin = self._params["fit_fmin"]; fmax = self._params["fit_fmax"]
        self.fit_fmin = fmin; self.fit_fmax = fmax
        sel = ((self._freq_orig >= fmin) & (self._freq_orig <= fmax)
               & np.isfinite(self._power_orig) & (self._power_orig > 0))
        self._f_orig = self._freq_orig[sel]; self._p_orig = self._power_orig[sel]
        self._e_orig = None if self._power_err_orig is None else self._power_err_orig[sel]
        if self._rebin_mode == "none":
            self._f = self._f_orig.copy(); self._p = self._p_orig.copy()
            self._e = None if self._e_orig is None else self._e_orig.copy()
        else:
            self._apply_rebin(self._rebin_mode, self._rebin_value)

    def _rebuild_pds(self) -> None:
        """
        Re-read the event file with the new dt / segment_size.

        Uses QPO_utils.filter_events_by_energy and make_averaged_pds so that
        the energy-filtering logic is not duplicated here.  Also re-derives
        self._m_eff (and self._m_eff_orig) from the rebinned pds.m so that
        subsequent fits use the correct likelihood weight.
        """
        if self._evt_path is None:
            print(f"  {_YL}PDS was supplied as arrays — cannot rebuild.{_R}"); return
        dt = self._params["dt"]; seg = self._params["segment_size"]
        print(f"  {_BL}Rebuilding PDS  (dt={dt}s  segment={seg}s)…{_R}", flush=True)
        ev = EventList.read(self._evt_path)
        if self._band_kev is not None:
            ev = filter_events_by_energy(ev, self._band_kev)
        pds_raw = make_averaged_pds(ev, dt=dt, segment_size=seg)
        pds = (pds_raw.rebin_log(f=float(getattr(P, "REBIN_LOG_F", 0.01)))
               if getattr(P, "DO_REBIN", False)
               else pds_raw.rebin_log(f=float(getattr(P, "CAND_REBIN_LOG_F", 0.008))))
        self.freq = np.asarray(pds.freq, float); self.power = np.asarray(pds.power, float)
        self.power_err = None if pds.power_err is None else np.asarray(pds.power_err, float)
        self._freq_orig = self.freq.copy(); self._power_orig = self.power.copy()
        self._power_err_orig = None if self.power_err is None else self.power_err.copy()
        self._m_eff      = safe_m_from_pds(pds)
        self._m_eff_orig = self._m_eff
        self._remask()
        hf = (self.freq >= 40.0) & np.isfinite(self.power) & (self.power > 0)
        if np.any(hf): self.const = float(np.nanmedian(self.power[hf]))
        lowf_mask = (self._f > 0) & (self._f <= 2.0) & np.isfinite(self._p) & (self._p > 0)
        if np.any(lowf_mask):
            self._lowf = float(np.nanmedian(self._p[lowf_mask]))
        print(f"  {_GR}PDS rebuilt.{_R}  {len(self._f)} bins, m_eff={self._m_eff}.")
        self._redraw()

    _PARAM_GROUPS = [
        ("PDS construction", ["dt", "segment_size"]),
        ("Frequency ranges", ["fit_fmin", "fit_fmax", "cand_fmin", "cand_fmax",
                               "qpo_fmin", "qpo_fmax", "qpo_min_q"]),
        ("Plot / display",   ["fit_rchi_max"]),
        ("Optimizer",        ["fit_method", "fit_n_starts", "fit_jitter_frac", "fit_random_seed"]),
        ("Component bounds", ["cont_fwhm_min", "cont_fwhm_max",
                               "qpo_fwhm_min",  "qpo_fwhm_max",
                               "cont_x0_max_hz", "cont_amp_factor", "qpo_amp_factor"]),
    ]

    def _cmd_params(self) -> None:
        print()
        for group, keys in self._PARAM_GROUPS:
            print(f"  {_B}── {group} {'─'*(44-len(group))}{_R}")
            for k in keys:
                if k not in self._params: continue
                val  = self._params[k]
                desc = self._param_meta[k][1]
                need = self._param_meta[k][2]
                tag  = f"  {_D}[rebuild]{_R}" if need else ""
                print(f"    {k:<18}  {str(val):<14}  {_D}{desc}{_R}{tag}")
        print(f"\n  {_D}[rebuild] = reloads event file.  Use setparam <n> <value> to change.{_R}")
        print(f"  {_D}TripleA knobs (AAA_*) are set in QPO_Parameter.py only.{_R}\n")

    def _cmd_setparam(self, args: List[str]) -> None:
        if len(args) < 2:
            print(f"  {_YL}Usage: setparam <n> <value>{_R}"); return

        name = args[0].lower()
        raw  = " ".join(args[1:])

        if name not in self._param_meta:
            close = [k for k in self._param_meta if k.startswith(name[:4])]
            hint  = f"  Did you mean: {', '.join(close)}?" if close else ""
            print(f"  {_RD}Unknown parameter {name!r}.{_R}{hint}  Run 'params'."); return

        type_fn, desc, needs_rebuild, needs_remask = self._param_meta[name]
        try:
            val = raw.lower() in ("true","1","yes","on") if type_fn is bool else type_fn(raw)
        except (ValueError, TypeError):
            print(f"  {_RD}{name} expects {type_fn.__name__}, got {raw!r}{_R}"); return

        if name == "fit_method" and val not in ("TripleA", "Powell", "Nelder-Mead", "L-BFGS-B"):
            print(f"  {_YL}Unusual optimizer {val!r} — proceeding.{_R}")
        if name == "fit_n_starts" and val < 1:
            print(f"  {_RD}Must be >= 1.{_R}"); return

        old = self._params[name]
        self._params[name] = val
        print(f"  {_GR}{name}{_R}: {old}  →  {val}")

        if needs_rebuild: self._rebuild_pds()
        elif needs_remask: self._remask(); self._redraw(); print(f"  {len(self._f)} bins.")

    # ------------------------------------------------------------------
    # Commands — plot export
    # ------------------------------------------------------------------

    def _cmd_plotresult(self, args: List[str]) -> None:
        """
        Save a publication-quality 2-panel PNG for the current band.

        Uses the current PDS arrays (self.freq / power / power_err) together
        with whichever of the following is available (in priority order):
          1. self.fit_result  — the last completed optimiser result
          2. self.components  — the current component overlay (no rchi2 shown)
          3. Neither          — the bare PDS is plotted with no model

        Default output path:
            <OUTDIR_BASE>/<obsid>/<obsid>_fit_<band_label>.png
        """
        if self.fit_result is not None:
            band_block = fitresult_to_band_block(self.fit_result)
        elif self.components:
            band_block = {
                "ok":         True,
                "nlor":       len(self.components),
                "rchi2":      float("nan"),
                "const":      self.const,
                "comp_types": [c["type"] for c in self.components],
                "pars":       [[c["nu0"], c["fwhm"], c["amp"]]
                               for c in self.components],
            }
        else:
            band_block = {"ok": False}

        if args:
            outpath = args[0]
        elif self.obsid:
            outdir  = os.path.join(getattr(P, "OUTDIR_BASE", "."), self.obsid)
            outpath = os.path.join(outdir,
                                   f"{self.obsid}_fit_{self.band_label}.png")
        else:
            outpath = f"qpo_fit_{self.band_label}.png"

        print(f"  {_BL}Saving single-band plot…{_R}", flush=True)
        try:
            saved = save_band_plot(
                self.obsid, self.band_label,
                self.freq, self.power, self.power_err,
                band_block, outpath,
            )
            print(f"  {_GR}Saved: {saved}{_R}")
        except Exception as exc:
            print(f"  {_RD}Plot failed: {exc}{_R}")

    def _cmd_plotall(self, args: List[str]) -> None:
        """
        Generate a 3-band publication-quality PNG using the saved struct.

        Workflow
        --------
        1. Load the struct for self.obsid.
        2. Read the event file once.
        3. For each band present and ok in the struct, filter the events and
           build a fit-quality rebinned PDS.
        4. Call save_threeband_plot with the PDS arrays + struct band_blocks.

        Default output path:
            <OUTDIR_BASE>/<obsid>/<obsid>_fits_full_soft_hard.png
        """
        if not self.obsid:
            print(f"  {_RD}No obsid set — cannot locate struct or output path.{_R}")
            return

        s = load_fit_struct(self.obsid)
        if s is None:
            print(f"  {_RD}No struct found for {self.obsid}.  "
                  f"Run the batch fitter or saveresult first.{_R}")
            return

        if self._evt_path is None or not os.path.exists(self._evt_path):
            print(f"  {_RD}Event file not available "
                  f"({self._evt_path or 'path unknown'}).{_R}")
            return

        BAND_CFG = {
            "full": (None,
                     "Full"),
            "soft": (getattr(P, "SOFT_BAND_KEV", (0.3, 2.0)),
                     "Soft 0.3-2 keV"),
            "hard": (getattr(P, "HARD_BAND_KEV", (2.0, 10.0)),
                     "Hard 2-10 keV"),
        }

        print(f"  {_BL}Loading event file for all bands…{_R}", flush=True)
        ev_full = EventList.read(self._evt_path)

        dt  = float(self._params["dt"])
        seg = float(self._params["segment_size"])

        band_items: List[Dict] = []

        for band_key in ("full", "soft", "hard"):
            band_kev, label = BAND_CFG[band_key]
            bd = s.get(band_key, {})

            if not bd.get("ok", False):
                band_items.append({
                    "label": label,
                    "freq": None, "power": None, "power_err": None,
                    "band_block": {"ok": False},
                })
                continue

            try:
                ev_band = (filter_events_by_energy(ev_full, band_kev)
                           if band_kev is not None else ev_full)
                pds_raw = make_averaged_pds(ev_band, dt=dt, segment_size=seg)
                pds     = maybe_rebin_pds_fit(pds_raw)

                band_items.append({
                    "label":      label,
                    "freq":       np.asarray(pds.freq,  float),
                    "power":      np.asarray(pds.power, float),
                    "power_err":  (None if pds.power_err is None
                                   else np.asarray(pds.power_err, float)),
                    "band_block": dict(bd),
                })
                print(f"    {_GR}{band_key}{_R}: {len(pds.freq)} bins")

            except Exception as exc:
                print(f"    {_YL}{band_key}: PDS load failed ({exc}){_R}")
                band_items.append({
                    "label": label,
                    "freq": None, "power": None, "power_err": None,
                    "band_block": {"ok": False},
                })

        if args:
            outpath = args[0]
        else:
            outdir  = os.path.join(getattr(P, "OUTDIR_BASE", "."), self.obsid)
            outpath = os.path.join(outdir,
                                   f"{self.obsid}_fits_full_soft_hard.png")

        print(f"  {_BL}Rendering 3-band plot…{_R}", flush=True)
        try:
            saved = save_threeband_plot(
                self.obsid, band_items, outpath,
                clobber=True,
            )
            print(f"  {_GR}Saved: {saved}{_R}")
        except Exception as exc:
            print(f"  {_RD}Plot failed: {exc}{_R}")

    # ------------------------------------------------------------------
    # Commands — clear / reset
    # ------------------------------------------------------------------

    def _cmd_clear(self) -> None:
        self.components.clear()
        print(f"  {_YL}Components cleared.{_R}")
        self._redraw()

    def _cmd_reset(self) -> None:
        self.components.clear(); self.fit_result = None
        self._model_line.set_data([], [])
        print(f"  {_YL}Reset.{_R}")
        self._redraw()

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, raw: str) -> bool:
        parts = raw.split()
        if not parts: return True
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("quit", "exit", "q"):           return False
        elif cmd in ("addcomp", "add"):            self._cmd_addcomp(args)
        elif cmd in ("removecomp", "remove", "rm"):self._cmd_removecomp(args)
        elif cmd in ("editcomp", "edit"):          self._cmd_editcomp(args)
        elif cmd in ("freeze",):                   self._cmd_freeze(args)
        elif cmd in ("unfreeze",):                 self._cmd_unfreeze(args)
        elif cmd == "setconst":                    self._cmd_setconst(args)
        elif cmd in ("list", "ls"):                self._cmd_list()
        elif cmd in ("params", "param"):           self._cmd_params()
        elif cmd in ("setparam", "set"):           self._cmd_setparam(args)
        elif cmd == "fit":                         self._cmd_fit()
        elif cmd in ("status", "stat"):            self._cmd_status()
        elif cmd in ("saveresult", "saveres"):     self._cmd_saveresult(args)
        elif cmd in ("load", "loadresult"):        self._cmd_load(args)
        elif cmd in ("plotresult", "plotres"):     self._cmd_plotresult(args)
        elif cmd in ("plotall",):                  self._cmd_plotall(args)
        elif cmd == "save":                        self._cmd_save(args)
        elif cmd == "rebin":                       self._cmd_rebin(args)
        elif cmd == "zoom":                        self._cmd_zoom(args)
        elif cmd == "clear":                       self._cmd_clear()
        elif cmd == "reset":                       self._cmd_reset()
        elif cmd in ("help", "?", "h"):            print(self._HELP)
        else: print(f"  {_RD}Unknown command: {cmd!r}{_R}  (type 'help')")
        return True

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def run(self) -> Optional[FitResult]:
        src = getattr(P, "SOURCE", "")
        header = f"{_B}QPO Interactive Fitter{_R}"
        if src or self.obsid: header += f"  —  {src} {self.obsid}".rstrip()
        if self.band_label:   header += f"  [{self.band_label}]"
        print(f"\n{header}")
        print(f"  {_D}Type 'help' for commands, 'quit' to exit.{_R}")
        print(f"  {_D}m_eff = {self._m_eff} (effective averaged powers per bin){_R}\n")

        if self.components:
            print(f"  {_CY}Loaded {len(self.components)} component(s) from struct:{_R}")
            self._cmd_list()

        self._redraw()

        while True:
            if not plt.fignum_exists(self.fig.number):
                print(f"\n  {_YL}Plot window closed — exiting.{_R}"); break
            try:
                raw = input(f"{_B}qpo>{_R} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not self._dispatch(raw): break

        plt.ioff()
        if plt.fignum_exists(self.fig.number):
            plt.show(block=True)

        return self.fit_result


# ============================================================================
# PDS loader and launcher
# ============================================================================

def _load_pds(
    obsid: str,
    band_kev: Optional[Tuple[float, float]] = None,
    *,
    dt:           Optional[float] = None,
    segment_size: Optional[float] = None,
) -> Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    Return (evt_path, freq, power, power_err, m_eff).

    Thin wrapper around QPO_utils.load_pds_for_band using the candidate
    rebin mode that the interactive fitter expects on startup.
    """
    return load_pds_for_band(
        obsid, band_kev,
        dt=dt,
        segment_size=segment_size,
        rebin_mode="cand",
    )


def launch(
    obsid:      str = "",
    band_label: str = "full",
    band_kev:   Optional[Tuple[float, float]] = None,
    *,
    freq:      Optional[np.ndarray] = None,
    power:     Optional[np.ndarray] = None,
    power_err: Optional[np.ndarray] = None,
    m_eff:     Optional[int] = None,
) -> Optional[FitResult]:
    """
    Open the interactive fitter for one band.

    If an existing fit struct is found for this obsid/band it is loaded
    automatically as warm-start components.

    PATCH: m_eff is plumbed through from the rebinned PDS so the Whittle
    likelihood inside fit_lorentzians has the correct weight.  When the
    user supplies (freq, power) directly without an event file, they
    should also pass m_eff (otherwise it defaults to 1, which is only
    correct for an unrebinned single-segment periodogram).
    """
    s = load_fit_struct(obsid) if obsid else None
    warm_comps: Optional[Dict[str, Any]] = None
    source_struct_path: Optional[str] = None

    if s is not None:
        warm_comps = struct_to_warm_comps(s, band_label)
        if warm_comps:
            source_struct_path = struct_path(obsid)
            print(f"Found struct for {obsid} — loading [{band_label}] parameters.")

    if freq is None or power is None:
        print(f"Loading PDS for {obsid} [{band_label}]…")
        evt_path, freq, power, power_err, m_eff_loaded = _load_pds(obsid, band_kev=band_kev)
        m_eff = m_eff_loaded
    else:
        evt_path = None
        if m_eff is None:
            warnings.warn(
                "launch(): freq/power supplied without m_eff — defaulting to 1. "
                "This is only correct for an unrebinned single-segment periodogram. "
                "For an averaged/rebinned PDS, pass the true m_eff.",
                RuntimeWarning, stacklevel=2,
            )
            m_eff = 1

    return TerminalFitter(
        freq, power, power_err,
        obsid=obsid,
        band_label=band_label,
        warm_start_comps=warm_comps,
        evt_path=evt_path,
        band_kev=band_kev,
        source_struct_path=source_struct_path,
        m_eff=int(m_eff),
    ).run()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        prog="QPO_interactive.py",
        description="Terminal-driven interactive Lorentzian PDS fitter.",
    )
    p.add_argument("--obsid", required=True, metavar="ID")
    p.add_argument("--band",  default="full", choices=["full", "soft", "hard"])
    args = p.parse_args()

    band_kev = None
    if args.band == "soft": band_kev = getattr(P, "SOFT_BAND_KEV", (0.3, 2.0))
    elif args.band == "hard": band_kev = getattr(P, "HARD_BAND_KEV", (2.0, 10.0))

    launch(obsid=args.obsid, band_label=args.band, band_kev=band_kev)


if __name__ == "__main__":
    main()
