#!/usr/bin/env python3
"""
QPO_utils.py
============
Shared helpers for event-file loading, energy filtering, and PDS construction.

Used by QPO_main.py, QPO_interactive.py, and QPO_plot.py so that the
event-file / PDS logic is defined in exactly one place.

Public API
----------
safe_m_from_pds(pds) -> int
    Robust scalar m from a Stingray Powerspectrum or rebinned PSD.

kev_to_pi(kev) -> int
    Convert energy in keV to NICER PI channel number.

filter_events_by_energy(ev, band_kev) -> EventList
    Return a new EventList filtered to the given (emin, emax) keV band.

make_averaged_pds(ev, *, dt, segment_size) -> AveragedPowerspectrum
    Compute a fractional-RMS AveragedPowerspectrum from an EventList.

rebin_pds(pds, mode, *, log_f, factor, df_hz) -> AveragedPowerspectrum
    Rebin a PDS by the chosen mode ('log' or 'linear').

maybe_rebin_pds_fit(pds) -> AveragedPowerspectrum
    Rebin using the fit-quality parameters from QPO_Parameter.

maybe_rebin_pds_candidate(pds) -> AveragedPowerspectrum
    Rebin using the candidate-search parameters from QPO_Parameter.

build_evt_path(base_dir, source, obsid) -> str
    Canonical cleaned event-file path for a NICER obsid.

load_pds_for_band(obsid, band_kev, *, dt, segment_size, rebin_mode)
    -> (evt_path, freq, power, power_err, m_eff)
    Load, filter, rebin, and return a PDS ready for fitting or plotting.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from stingray.events import EventList
from stingray.powerspectrum import AveragedPowerspectrum

import QPO_Parameter as P


# ============================================================================
# m helpers
# ============================================================================

def safe_m_from_pds(pds) -> int:
    """
    Return a robust scalar effective-m from a Stingray Powerspectrum.

    Handles both scalar and per-bin array m attributes, returning the median
    for array m.  Falls back to 1 for anything non-finite or < 1.
    """
    try:
        m   = getattr(pds, "m", 1)
        arr = np.asarray(m, float)
        v   = float(arr) if arr.ndim == 0 else float(np.nanmedian(arr))
        if not np.isfinite(v) or v < 1:
            return 1
        return int(round(v))
    except Exception:
        return 1


# ============================================================================
# Energy filtering
# ============================================================================

def kev_to_pi(kev: float) -> int:
    """Convert an energy in keV to a NICER PI channel number."""
    return int(np.round(kev * float(getattr(P, "PI_PER_KEV", 100.0))))


def filter_events_by_energy(
    ev: EventList,
    band_kev: Tuple[float, float],
) -> EventList:
    """
    Return a new EventList filtered to the given (emin, emax) keV band.

    Parameters
    ----------
    ev       : Stingray EventList with a PI column
    band_kev : (emin_keV, emax_keV) — half-open interval [emin, emax)

    Raises
    ------
    ValueError
        If the EventList has no PI column, or if fewer than 10 events
        remain after filtering.
    """
    emin, emax = band_kev
    pi_min, pi_max = kev_to_pi(emin), kev_to_pi(emax)

    if ("pi" not in ev.array_attrs()) and (not hasattr(ev, "pi")):
        raise ValueError("EventList missing PI column; cannot energy-filter.")

    pi = ev.pi
    m  = (pi >= pi_min) & (pi < pi_max)
    if np.sum(m) < 10:
        raise ValueError(
            f"Too few events in {band_kev} keV "
            f"(PI {pi_min}–{pi_max}; only {int(np.sum(m))} remain)."
        )

    gti        = getattr(ev, "gti", None) if getattr(P, "USE_GTIS", True) else None
    ev_filt    = EventList(time=ev.time[m], gti=gti)
    ev_filt.pi = pi[m]
    return ev_filt


# ============================================================================
# PDS construction
# ============================================================================

def make_averaged_pds(
    ev: EventList,
    *,
    dt:           float,
    segment_size: float,
) -> AveragedPowerspectrum:
    """
    Compute a fractional-RMS AveragedPowerspectrum from an EventList.

    Parameters
    ----------
    ev           : filtered (or full-band) EventList
    dt           : time resolution in seconds
    segment_size : segment length in seconds
    """
    lc = ev.to_lc(dt=dt)
    return AveragedPowerspectrum(lc, segment_size=segment_size, norm="frac")


def rebin_pds(
    pds:    AveragedPowerspectrum,
    mode:   str,
    *,
    log_f:  float,
    factor: float,
    df_hz:  Optional[float] = None,
) -> AveragedPowerspectrum:
    """
    Rebin a PDS.

    Parameters
    ----------
    pds    : input AveragedPowerspectrum
    mode   : 'log' or 'linear'
    log_f  : fractional step for log rebinning (e.g. 0.02)
    factor : linear rebin factor (used when df_hz is None and mode='linear')
    df_hz  : target bin width in Hz (takes priority over factor when not None)
    """
    if mode.lower() == "log":
        return pds.rebin_log(f=float(log_f))
    if df_hz is not None:
        return pds.rebin(df=float(df_hz))
    return pds.rebin(f=float(factor))


def maybe_rebin_pds_fit(pds: AveragedPowerspectrum) -> AveragedPowerspectrum:
    """
    Apply the fit-quality rebin specified in QPO_Parameter.

    Uses DO_REBIN / REBIN_MODE / REBIN_LOG_F / REBIN_FACTOR / REBIN_DF_HZ.
    Returns the input unchanged when DO_REBIN is False.
    """
    if not getattr(P, "DO_REBIN", False):
        return pds
    return rebin_pds(
        pds,
        mode   = getattr(P, "REBIN_MODE",   "log"),
        log_f  = getattr(P, "REBIN_LOG_F",  0.02),
        factor = getattr(P, "REBIN_FACTOR", 4.0),
        df_hz  = getattr(P, "REBIN_DF_HZ",  None) if hasattr(P, "REBIN_DF_HZ") else None,
    )


def maybe_rebin_pds_candidate(pds: AveragedPowerspectrum) -> AveragedPowerspectrum:
    """
    Apply the lighter candidate-search rebin specified in QPO_Parameter.

    Uses DO_CANDIDATE_LIGHT_REBIN / CAND_REBIN_* parameters.
    Falls back to maybe_rebin_pds_fit when the light-rebin flag is off.
    """
    if not getattr(P, "DO_CANDIDATE_LIGHT_REBIN", True):
        return maybe_rebin_pds_fit(pds)
    if (not getattr(P, "DO_REBIN", False)
            and not getattr(P, "DO_REBIN_CAND_WHEN_FIT_OFF", True)):
        return pds
    return rebin_pds(
        pds,
        mode   = getattr(P, "CAND_REBIN_MODE",   "log"),
        log_f  = getattr(P, "CAND_REBIN_LOG_F",  0.008),
        factor = getattr(P, "CAND_REBIN_FACTOR",  2.0),
        df_hz  = getattr(P, "CAND_REBIN_DF_HZ",  None) if hasattr(P, "CAND_REBIN_DF_HZ") else None,
    )


# ============================================================================
# Path helpers
# ============================================================================

def build_evt_path(base_dir: str, source: str, obsid: str) -> str:
    """Return the canonical path to a NICER cleaned event file for one obsid."""
    return os.path.join(base_dir, source, obsid, f"ni{obsid}_0mpu7_cl.evt")


# ============================================================================
# High-level loader
# ============================================================================

def load_pds_for_band(
    obsid:        str,
    band_kev:     Optional[Tuple[float, float]] = None,
    *,
    dt:           Optional[float] = None,
    segment_size: Optional[float] = None,
    rebin_mode:   str = "fit",
) -> Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    Load, filter, and rebin a PDS for one obsid / energy band.

    Single entry-point used by both the interactive fitter and the plotter so
    that event-file handling is not duplicated across modules.

    Parameters
    ----------
    obsid        : NICER observation ID
    band_kev     : (emin_keV, emax_keV) or None for the full band
    dt           : time resolution in seconds  (default: P.DT)
    segment_size : segment length in seconds   (default: P.SEGMENT_SIZE)
    rebin_mode   : 'fit'  → maybe_rebin_pds_fit
                   'cand' → maybe_rebin_pds_candidate
                   'none' → no rebinning

    Returns
    -------
    (evt_path, freq, power, power_err, m_eff)
        power_err is None when the PDS object carries no error array.

    Raises
    ------
    FileNotFoundError  if the event file does not exist
    ValueError         if energy filtering leaves too few events
    """
    evt_path = build_evt_path(
        getattr(P, "BASE_DIR", "."),
        getattr(P, "SOURCE",   ""),
        obsid,
    )
    if not os.path.exists(evt_path):
        raise FileNotFoundError(f"Event file not found: {evt_path}")

    ev = EventList.read(evt_path)
    if band_kev is not None:
        ev = filter_events_by_energy(ev, band_kev)

    _dt  = float(dt           or getattr(P, "DT",           0.0078125))
    _seg = float(segment_size or getattr(P, "SEGMENT_SIZE", 64.0))

    pds_raw = make_averaged_pds(ev, dt=_dt, segment_size=_seg)

    if rebin_mode == "fit":
        pds = maybe_rebin_pds_fit(pds_raw)
    elif rebin_mode == "cand":
        pds = maybe_rebin_pds_candidate(pds_raw)
    else:
        pds = pds_raw

    m_eff = safe_m_from_pds(pds)

    return (
        evt_path,
        np.asarray(pds.freq,  float),
        np.asarray(pds.power, float),
        (None if pds.power_err is None else np.asarray(pds.power_err, float)),
        m_eff,
    )
