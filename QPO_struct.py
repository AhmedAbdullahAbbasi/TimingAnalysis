#!/usr/bin/env python3
"""
QPO_struct.py  (v1.1)
======================
Per-obsid fit result structs.



One JSON file per obsid holds the results for all three energy bands.
Only fit quality metrics and Lorentzian parameters are stored — no
frequency or model arrays.  This keeps files small and human-readable,
and leaves room for a future source-level aggregation struct.

File layout
-----------
<OUTDIR_BASE>/<obsid>/<obsid>_fitresult.json

    {
      "version":   "1.0",
      "obsid":     "1200120106",
      "source":    "GX339",
      "mjd_mid":   59500.12,
      "timestamp": "2025-01-01T12:00:00+00:00",

      "full": {
        "ok":          true,
        "message":     "OK (cont2+qpo)",
        "nlor":        3,
        "rchi2":       1.08,
        "aic":        -312.4,
        "bic":        -298.1,
        "deviance":    450.2,
        "red_deviance": 1.12,
        "const":       0.00142,
        "comp_types":  ["cont", "cont", "qpo"],
        "pars":        [[nu0, fwhm, amp], ...]
      },

      "soft": { ... },   // absent if band was not fitted
      "hard": { ... }
    }

Public API
----------
save_fit_struct(fitres, obsid, band, *, mjd=None, outdir=None) -> str
    Add or update one band in the obsid struct.  Creates the file if
    it does not exist; merges into the existing file if it does.
    Returns the path written.

load_fit_struct(obsid, *, outdir=None) -> dict | None
    Load the full obsid struct.  Returns None if the file does not exist.

struct_path(obsid, outdir=None) -> str
    Canonical file path for an obsid.

struct_to_warm_comps(struct, band) -> dict | None
    Extract warm_start_comps for one band:
        {"cont": [(nu0,fwhm,amp),...], "qpo": [...], "const": float|None}
    Returns None if the band is absent or the fit was not ok.

struct_summary(struct, band=None) -> str
    One-line human-readable description.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import QPO_Parameter as P

_STRUCT_VERSION = "1.0"
_BANDS = ("full", "soft", "hard")


# ============================================================================
# Path
# ============================================================================

def struct_path(obsid: str, outdir: Optional[str] = None) -> str:
    """Canonical path: <outdir>/<obsid>_fitresult.json"""
    d = outdir if outdir is not None else os.path.join(
        getattr(P, "OUTDIR_BASE", "."), str(obsid)
    )
    return os.path.join(d, f"{obsid}_fitresult.json")


# ============================================================================
# Save  (merge into existing file)
# ============================================================================

def save_fit_struct(
    fitres,
    obsid:  str,
    band:   str,
    *,
    mjd:    Optional[float] = None,
    outdir: Optional[str]   = None,
) -> str:
    """
    Write one band's fit result into the obsid struct file.

    If the file already exists the other bands are preserved; only the
    specified band is updated.  A failed or None fitres is still written
    (as ok=false) so the file always reflects what was attempted.

    Returns the path written.
    """
    path = struct_path(obsid, outdir)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # Load existing struct (if any) so other bands are preserved
    existing: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}

    # Build / update top-level fields
    struct: Dict[str, Any] = {
        "version":   _STRUCT_VERSION,
        "obsid":     str(obsid),
        "source":    getattr(P, "SOURCE", ""),
        "mjd_mid":   _f(mjd) if mjd is not None else existing.get("mjd_mid"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Carry over existing bands that are NOT being updated
    for b in _BANDS:
        if b in existing and b != band:
            struct[b] = existing[b]

    # Build the band block
    struct[str(band)] = _make_band_block(fitres)

    # Atomic write
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(struct, fh, indent=2)
    os.replace(tmp, path)
    return path


def _make_band_block(fitres) -> Dict[str, Any]:
    """Serialize one FitResult into the band sub-dict."""
    if fitres is None or not getattr(fitres, "ok", False):
        return {"ok": False}

    pars = getattr(fitres, "pars", None)
    return {
        "ok":           True,
        "message":      str(getattr(fitres, "message",     "")),
        "nlor":         int(getattr(fitres, "nlor",        0)),
        "rchi2":        _f(getattr(fitres, "rchi2",        np.nan)),
        "aic":          _f(getattr(fitres, "aic",          np.nan)),
        "bic":          _f(getattr(fitres, "bic",          np.nan)),
        "deviance":     _f(getattr(fitres, "deviance",     np.nan)),
        "red_deviance": _f(getattr(fitres, "red_deviance", np.nan)),
        "const":        _f(getattr(fitres, "const",        0.0)),
        "comp_types":   list(getattr(fitres, "comp_types", [])),
        "pars": (
            pars.tolist()
            if (pars is not None and hasattr(pars, "tolist"))
            else []
        ),
    }


def _f(v) -> Optional[float]:
    """Float or None (JSON-safe; NaN/inf → None)."""
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ============================================================================
# Load
# ============================================================================

def load_fit_struct(
    obsid:  str,
    *,
    outdir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the full obsid struct.  Returns None if the file does not exist.
    """
    path = struct_path(obsid, outdir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        import warnings
        warnings.warn(f"QPO_struct: failed to load {path}: {exc}", RuntimeWarning)
        return None


# ============================================================================
# Conversions
# ============================================================================

def struct_to_warm_comps(
    struct: Dict[str, Any],
    band:   str,
) -> Optional[Dict[str, Any]]:
    """
    Extract warm_start_comps for one band from an obsid struct.

    Returns:
        {"cont": [(nu0,fwhm,amp),...], "qpo": [(nu0,fwhm,amp),...], "const": float|None}
    or None if the band is absent, ok=false, or has no pars.
    """
    band_data = struct.get(str(band))
    if not band_data or not band_data.get("ok", False):
        return None

    pars_raw   = band_data.get("pars",       [])
    comp_types = band_data.get("comp_types", [])
    const      = band_data.get("const",      None)

    if not pars_raw:
        return None

    cont: List[Tuple] = []
    qpo:  List[Tuple] = []
    for par, ctype in zip(pars_raw, comp_types):
        tup = tuple(float(x) for x in par)   # (nu0, fwhm, amp)
        if str(ctype) == "qpo":
            qpo.append(tup)
        else:
            cont.append(tup)

    return {
        "cont":  cont,
        "qpo":   qpo,
        "const": float(const) if (const is not None) else None,
    }


# ============================================================================
# Summary
# ============================================================================

def struct_summary(struct: Dict[str, Any], band: Optional[str] = None) -> str:
    """One-line summary of an obsid struct (all bands, or one band)."""
    obsid  = struct.get("obsid",   "?")
    source = struct.get("source",  "")
    mjd    = struct.get("mjd_mid")
    mjd_s  = f"  mjd={mjd:.4f}" if mjd is not None else ""
    prefix = f"{source} {obsid}{mjd_s}".strip()

    bands_to_show = [band] if band else [b for b in _BANDS if b in struct]

    parts = []
    for b in bands_to_show:
        bd = struct.get(b, {})
        if not bd:
            continue
        if not bd.get("ok", False):
            parts.append(f"{b}:FAIL")
            continue
        rchi   = bd.get("rchi2")
        nlor   = bd.get("nlor", "?")
        rchi_s = f"{rchi:.3f}" if rchi is not None else "nan"
        ctypes = bd.get("comp_types", [])
        pars   = bd.get("pars",       [])
        qmin   = float(getattr(P, "QPO_MIN_Q", 3.0))
        # Apply Q-filter so summary matches CSV and plot annotation
        qpos   = [
            p for p, t in zip(pars, ctypes)
            if t == "qpo"
            and len(p) >= 2
            and p[1] > 0
            and (p[0] / p[1]) >= qmin
        ]
        nu_s   = "+".join(f"{p[0]:.3f}Hz" for p in qpos) if qpos else "noQPO"
        parts.append(f"{b}:nlor={nlor} rchi2={rchi_s} {nu_s}")

    return prefix + "  |  " + "  |  ".join(parts) if parts else prefix + "  (no bands)"
