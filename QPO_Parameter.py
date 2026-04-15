#!/usr/bin/env python3
"""
QPO_Parameter.py
================
All user-configurable knobs for the QPO fitting pipeline.

Edit the values in this file directly.  A validation pass runs automatically
at import time; a clear error is raised if any value is out of range or has
the wrong type, so problems are caught before the first observation is touched.

Changes from v1
----------------
- Per-component continuum centroid limits (narrow / wide / free).
- Multi-scale smoothing for candidate finding.
- Higher amplitude and constant caps (guardrails catch pathological cases).
- cont4 fallback stage for complex states.
- Relaxed overshoot guardrail for log-rebinned data.
- Cross-band fallback uses diagnostic peaks when all bands are bad.

Changes from v2
---------------
- FIT_METHOD = "TripleA": custom Whittle-likelihood optimiser with
  analytical gradient and true L-BFGS-B box constraints.
  See QPO_TripleA.py for full documentation.
- AAA_* parameter block for TripleA tuning.
"""

from __future__ import annotations
import warnings

# ==========================
# INPUT DIRECTORY STRUCTURE
# ==========================
BASE_DIR   = "/data2/sena.ozgur/nicer/"
SOURCE     = "maxij1820+70"
OBSIDS_TXT = "MAXIJ1820_sample.txt"

# ==========================
# OUTPUT DIRECTORY STRUCTURE
# ==========================
OUTDIR_BASE    = "/data2/NICER_DATA/NICER/MAXIJ1820_timing_res/"
COMMON_DIRNAME = "commonfiles"
OUT_CSV_NAME   = "MAXIJ1820_qpo_summary.csv"

# ==========================
# TIMING PARAMETERS
# ==========================
DT           = 0.00390625   # seconds (time resolution)
SEGMENT_SIZE = 64           # seconds (AveragedPowerspectrum segment_size)

# ==========================
# DIAGNOSTIC PEAK SEARCH  (find_qpo_peak_whitened)
# ==========================
FMIN              = 0.1    # Hz — lower bound for diagnostic peak search
FMAX              = 10.0    # Hz — upper bound
PEAK_IGNORE_BELOW = 0.10    # Hz — ignore peaks below this (suppress red-noise artefacts)

# Whitening smooth scale (rolling-median continuum estimate)
PEAK_SMOOTH_HZ = 0.3        # Hz — used by the diagnostic peak finder

# Multi-scale smoothing for the candidate finder (union of peaks across scales).
# Ensures peaks that are invisible at one scale are found at another.
PEAK_SMOOTH_SCALES = [0.2, 0.5, 1.5]   # Hz — list of scales to try

# Sigma-excess gating
PEAK_REQUIRE_KSIGMA   = None    # float or None — None disables the sigma gate
PEAK_PROMINENCE_SIGMA = 0.5    # minimum peak prominence in sigma units
PEAK_SIGMA_MODE       = "cont"  # "cont": sigma ~ cont/sqrt(m);  "p": sigma ~ p/sqrt(m)
PEAK_RANK_BY          = "height"   # "prominence" or "height"

# Candidate separation / limit (shared with fitter candidate finder)
PEAK_MIN_SEP_HZ     = 0.12
PEAK_MAX_CANDIDATES = 8

# Candidate finder prominence threshold.
# Now operates in z-score space (sigma units) instead of ratio space.
PEAK_PROMINENCE = 0.5

# ==========================
# RMS INTEGRATION
# ==========================
BROAD_RMS_BAND = (0.1, 30.0)   # Hz  (broadband rms integration limits)

# Diagnostic QPO-band rms (around diagnostic peak, not fit-based)
QPO_BW_FRAC = 0.10   # fractional half-width around peak
QPO_BW_MIN  = 0.10   # Hz — minimum half-width
QPO_BW_MAX  = 2.00   # Hz — maximum half-width

# ==========================
# PLOT SETTINGS
# ==========================
PLOT_DPI  = 200
CLOBBER   = True       # overwrite existing output files

# ==========================
# ENERGY BANDS
# ==========================
DO_ENERGY_BANDS = True
SOFT_BAND_KEV   = (0.3, 2.0)
HARD_BAND_KEV   = (2.0, 10.0)
PI_PER_KEV      = 100.0
USE_GTIS        = True

# ==========================
# FITTING: MASTER SWITCHES
# ==========================
DO_FIT         = True
SAVE_FIT_PLOTS = True

# Fit band (Hz)
FIT_FMIN = 0.05
FIT_FMAX = 64.0

# Candidate band (Hz) — QPO seeds must lie in this range
CAND_FMIN = 0.05
CAND_FMAX = 10.0

# ==========================
# QPO SELECTION AFTER FITTING
# ==========================
QPO_FMIN   = 0.10
QPO_FMAX   = 10.0
QPO_MIN_Q  = 3

QPO_SORT_BY = "area"   # "area", "freq", or "q"
# NOTE: "area" (integrated power) is the physically preferred primary sort key.
# Sorting by "q" caused the highest-Q component — often a spurious narrow spike
# rather than a genuine QPO — to be written as the primary result in the CSV.

# ==========================
# OPTIMIZER + ACCEPTANCE
# ==========================
# Supported values: "TripleA" | "Powell" | "Nelder-Mead"
#
# "TripleA" (default): custom L-BFGS-B optimiser with analytical Whittle
#   gradient and true box constraints.  Faster convergence, more reliable
#   on ridges, and no prior-cliff problems.  See QPO_TripleA.py.
#
# "Powell" / "Nelder-Mead": Stingray's gradient-free path (legacy).
#   Use for comparison or if QPO_TripleA.py is not available.
FIT_METHOD      = "TripleA"
FIT_RCHI_MAX    = 1.5     # marks fit_ok=False in CSV if rchi2 > this
FIT_RCHI_TARGET = 1.3     # retry ladder stops early once rchi2 <= this

# ==========================
# FITTING: CONSTANT (WHITE-NOISE FLOOR)
# ==========================
FIT_INCLUDE_CONST   = True
FIT_CONST_SEED_FMIN = 40.0   # Hz — frequency range used to seed the white-noise floor
# Constant cap multiplier: const_cap = CONST_CAP_FACTOR * c0.
# v1 used a hard 2.0; raised to 5.0 so short/noisy observations don't clip.
FIT_CONST_CAP_FACTOR = 5.0

# ==========================
# FITTING: MULTI-START ROBUSTNESS
# ==========================
# Controls the outer _jittered_starts loop in _run_stage.
# When FIT_METHOD = "TripleA", the total independent starts per stage call
# is FIT_N_STARTS × AAA_N_STARTS (outer × inner).  The outer loop provides
# diversity across continuum + QPO joint configurations; the inner loop
# polishes each configuration with L-BFGS-B.
# In the interactive fitter the outer loop is bypassed, so only AAA_N_STARTS
# starts run.
FIT_MULTI_START = True
FIT_N_STARTS    = 3
FIT_JITTER_FRAC = 0.12
FIT_RANDOM_SEED = 42

# ==========================
# FITTING: COMPONENT-WISE BOUNDS / CAPS
# ==========================
# Per-component continuum centroid limits.
# - "Narrow": the broadest zero-centred Lorentzian (Lh).  Should stay near 0.
# - "Wide":   the medium component (Lb).  Can sit at 0.5–3 Hz in intermediate states.
# - "Free":   cont3/cont4 components.  Allowed anywhere in the low-frequency band.
FIT_CONT_X0_NARROW_HZ = 0.3    # broadest Lorentzian: ±0.3 Hz
FIT_CONT_X0_WIDE_HZ   = 3.0    # medium Lorentzian:   ±3.0 Hz
FIT_CONT_X0_FREE_HZ   = 8.0    # cont3/cont4:         ±8.0 Hz

FIT_CONT_FWHM_MIN = 0.30
FIT_CONT_FWHM_MAX = 64.0

FIT_QPO_FWHM_MIN  = 0.08
FIT_QPO_FWHM_MAX  = 5.0

FIT_HARM_FWHM_MIN = 0.03
FIT_HARM_FWHM_MAX = 8.0

# Amplitude caps as multiples of the low-frequency power level.
# Raised from 5/8 to 12/12 — guardrails prevent pathological fits.
FIT_CONT_AMP_FACTOR = 12.0
FIT_QPO_AMP_FACTOR  = 12.0
FIT_HARM_AMP_FACTOR = 5.0

# QPO seed width: initial FWHM = max(FWHM_MIN_ABS, FWHM_FRAC * nu0)
FIT_QPO_FWHM_FRAC    = 0.03
FIT_QPO_FWHM_MIN_ABS = 0.08

# Harmonic search (currently not used inside fitter; kept for API compatibility)
DO_HARMONIC_SEARCH = False

# ==========================
# FITTING: GUARDRAILS
# ==========================
FIT_GUARD_ENABLE                 = True
# Raised from 3.0 to 4.0: log-rebinned data has non-Gaussian tails that
# cause 3σ overshoot in valid fits.
FIT_GUARD_OVERSHOOT_KSIGMA       = 4.0
FIT_GUARD_OVERSHOOT_MAX_RUN_BINS = 6
FIT_GUARD_OVERSHOOT_MAX_FRAC     = 0.10
FIT_GUARD_COMP_LOCAL_AMP_FACTOR  = 6.0

# ==========================
# FITTING: RETRY LADDER
# ==========================
FIT_MAX_RETRIES = 3

# ==========================
# OPTIMIZER: TRIPLEA (AAA)
# ==========================
# These knobs are read by QPO_TripleA.tripleA_fit_once at call time.
# Changes take effect immediately without restarting the pipeline.
#
# AAA_N_STARTS
#   Independent L-BFGS-B starts per tripleA_fit_once call.
#   Each start costs ~50-200 gradient evaluations, compared to
#   ~1000-5000 function evaluations for a Powell attempt — so 5
#   TripleA starts cost less wall time than 1 Powell attempt.
#
# AAA_JITTER_STD_LOG
#   Standard deviation of the log-space perturbation applied to
#   amplitude and FWHM between starts.
#   0.30 → typical multiplier exp(±0.30) ≈ 0.74–1.35.
#   Increase for badly-conditioned fits; decrease for highly precise seeds
#   (e.g. loaded from a struct).
#
# AAA_JITTER_STD_NU0
#   Std of the ν₀ perturbation as a fraction of the x0 range.
#   0.10 → ±10 % of the allowed centroid window.
#
# AAA_FTOL
#   L-BFGS-B function-value convergence tolerance.
#   1e-11 is tighter than the scipy default (2.22e-9) and appropriate
#   given the analytical gradient.
#
# AAA_GTOL
#   Gradient-norm convergence tolerance.
#   The optimiser declares convergence when max|∇L| < AAA_GTOL.
#
# AAA_MAXITER
#   Maximum L-BFGS-B iterations per start.
#   1000 is generous; well-conditioned fits converge in 20-100 iterations.

AAA_N_STARTS       = 5
AAA_JITTER_STD_LOG = 0.30
AAA_JITTER_STD_NU0 = 0.10
AAA_FTOL           = 1e-11
AAA_GTOL           = 1e-7
AAA_MAXITER        = 1000

# ==========================
# MODEL ORDER SELECTION (IC)
# ==========================
CONT_IC_CRITERION = "aic"
CONT_IC_DELTA_MIN = 15.0

QPO_IC_CRITERION = "aic"
#If Qpos weak (as in the case of GX339-4) keep low (even 0). If qpos strong, can increase. 
QPO_IC_DELTA_MIN = 0

# ==========================
# FITTING: FORCED THIRD CONTINUUM
# ==========================
FIT_CONT_RCHI_FORCE_CONT3 = 1.5

# Post-QPO cont3 rescue trigger.
FIT_POSTQPO_CONT3_TRIGGER_RCHI      = 1.5
FIT_POSTQPO_CONT3_RCHI_IMPROVE_MIN  = 0.05
FIT_POSTQPO_CONT3_IC_DELTA_MIN      = 0.0

# ==========================
# FITTING: FOURTH CONTINUUM FALLBACK
# ==========================
# When cont3+QPO still yields rchi2 above threshold, try cont4.
# Gated by BIC to prevent overfitting.  Useful for hard-intermediate states
# (GX 339-4 Lh + Lb + Lu + QPO decomposition).
FIT_CONT4_ENABLE          = False
FIT_CONT4_TRIGGER_RCHI    = 1.5   # only try cont4 if current rchi2 > this
FIT_CONT4_IC_CRITERION    = "bic"  # strict criterion to prevent overfitting
FIT_CONT4_IC_DELTA_MIN    = 30.0

# ==========================
# CROSS-BAND RESEEDING
# ==========================
DO_CROSS_BAND_RESEED       = True
CROSS_BAND_RESEED_RCHI_BAD = 1.5
# When ALL bands are bad (no well-fitting band produces QPO seeds),
# fall back to using diagnostic whitened-peak frequencies as cross-seeds.
CROSS_BAND_USE_DIAG_PEAKS_FALLBACK = True

# Per-QPO quality threshold for cross-band donation.
# A QPO is eligible to seed other bands if its integrated area exceeds this
# value (frac-rms² units; ~1e-4 corresponds to ~1% rms).  This lets QPOs
# from a band with bad overall rchi2 still donate seeds, as long as the
# QPO itself is well-constrained.
CROSS_BAND_QPO_AREA_MIN = 1e-4

# Multi-seed Stage 1
FIT_STAGE1_N_SEEDS = 6

# Multi-QPO growth
FIT_MAX_QPOS                    = 3
# IC delta required to UPGRADE from 1-QPO to 2-QPO.
# Kept separate from QPO_IC_DELTA_MIN so primary QPO detection stays sensitive
# (delta=0) while secondary QPO detection is strictly gated.
# A value of 5 means the 2-QPO AIC must beat the 1-QPO AIC by ≥ 5 — roughly
# one additional free parameter must be fully justified by the data.
FIT_MULTI_QPO_IC_DELTA_MIN      = 5
FIT_MULTI_QPO_REQUIRE_IMPROVEMENT = True

# Conditional reseed
FIT_RESEED_ENABLE          = True
FIT_RESEED_RCHI_BAD        = 1.5
FIT_RESEED_EDGE_FRAC       = 0.08
FIT_RESEED_AREA_MIN        = 1e-3
FIT_RESEED_EXCLUDE_HZ_MIN  = 0.5
FIT_RESEED_EXCLUDE_DF_MULT = 10.0
FIT_RESEED_PROM_FACTOR     = 1.25
FIT_RESEED_SIGMA_FACTOR    = 1.10

# ==========================
# REBINNING (frequency domain)
# ==========================
DO_REBIN   = True
REBIN_MODE = "log"

REBIN_LOG_F = 0.01

REBIN_FACTOR = 4.0

# ==========================
# CANDIDATE REBINNING (lighter, for peak finding / seed grid)
# ==========================
DO_CANDIDATE_LIGHT_REBIN   = True
DO_REBIN_CAND_WHEN_FIT_OFF = True

CAND_REBIN_MODE  = "log"
CAND_REBIN_LOG_F = 0.008

CAND_REBIN_FACTOR = 2.0

# ==========================
# PARALLEL EXECUTION
# ==========================
PARALLEL_ENABLE       = False
N_WORKERS             = 32
PARALLEL_START_METHOD = "spawn"

# ==========================
# LOGGING / OUTPUT
# ==========================
QUIET            = True
ONE_LINE_SUMMARY = True
SHOW_QPO_DETAILS = True


# ============================================================================
# VALIDATION  (runs automatically at import)
# ============================================================================

def _validate_config() -> None:
    """Validate all parameters at import time.  Raises ValueError on error."""
    errors: list[str] = []

    def _chk_pos(name: str, val, allow_zero: bool = False) -> None:
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric, got {val!r}")
            return
        if allow_zero and v < 0:
            errors.append(f"{name} must be >= 0, got {v}")
        elif (not allow_zero) and v <= 0:
            errors.append(f"{name} must be > 0, got {v}")

    def _chk_range(name: str, val, lo: float, hi: float) -> None:
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric, got {val!r}")
            return
        if not (lo <= v <= hi):
            errors.append(f"{name}={v} is outside [{lo}, {hi}]")

    def _chk_band(name: str, band) -> None:
        if not (isinstance(band, (tuple, list)) and len(band) == 2):
            errors.append(f"{name} must be a 2-tuple (lo, hi), got {band!r}")
            return
        lo, hi = band
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            errors.append(f"{name} elements must be numeric")
            return
        if lo >= hi:
            errors.append(f"{name}: lo ({lo}) must be < hi ({hi})")

    # Timing
    _chk_pos("DT",           DT)
    _chk_pos("SEGMENT_SIZE", SEGMENT_SIZE)
    if SEGMENT_SIZE < DT * 8:
        errors.append(f"SEGMENT_SIZE ({SEGMENT_SIZE}s) is too small relative to DT ({DT}s)")

    # Frequency bands
    _chk_pos("FMIN",  FMIN)
    _chk_pos("FMAX",  FMAX)
    if FMIN >= FMAX:
        errors.append(f"FMIN ({FMIN}) must be < FMAX ({FMAX})")
    _chk_pos("CAND_FMIN", CAND_FMIN)
    _chk_pos("CAND_FMAX", CAND_FMAX)
    _chk_pos("FIT_FMIN",  FIT_FMIN)
    _chk_pos("FIT_FMAX",  FIT_FMAX)

    # QPO selection
    _chk_pos("QPO_MIN_Q",  QPO_MIN_Q)
    _chk_pos("QPO_FMIN",   QPO_FMIN)
    _chk_pos("QPO_FMAX",   QPO_FMAX)

    # Fit params
    _chk_pos("FIT_RCHI_MAX",    FIT_RCHI_MAX)
    _chk_pos("FIT_RCHI_TARGET", FIT_RCHI_TARGET)
    if FIT_RCHI_TARGET > FIT_RCHI_MAX:
        warnings.warn(
            f"FIT_RCHI_TARGET ({FIT_RCHI_TARGET}) > FIT_RCHI_MAX ({FIT_RCHI_MAX}): "
            "the early-stop target is looser than the acceptance threshold.",
            UserWarning, stacklevel=2,
        )
    _chk_pos("FIT_N_STARTS",    FIT_N_STARTS)
    _chk_range("FIT_JITTER_FRAC", FIT_JITTER_FRAC, 0.0, 1.0)
    _chk_pos("FIT_CONST_SEED_FMIN", FIT_CONST_SEED_FMIN)
    _chk_pos("FIT_CONST_CAP_FACTOR", FIT_CONST_CAP_FACTOR)
    _chk_pos("FIT_CONT_X0_NARROW_HZ", FIT_CONT_X0_NARROW_HZ)
    _chk_pos("FIT_CONT_X0_WIDE_HZ",   FIT_CONT_X0_WIDE_HZ)
    _chk_pos("FIT_CONT_X0_FREE_HZ",   FIT_CONT_X0_FREE_HZ)
    _chk_pos("FIT_CONT_FWHM_MIN",   FIT_CONT_FWHM_MIN)
    _chk_pos("FIT_CONT_FWHM_MAX",   FIT_CONT_FWHM_MAX)
    _chk_pos("FIT_QPO_FWHM_MIN",    FIT_QPO_FWHM_MIN)
    _chk_pos("FIT_QPO_FWHM_MAX",    FIT_QPO_FWHM_MAX)
    if FIT_CONT_FWHM_MIN >= FIT_CONT_FWHM_MAX:
        errors.append("FIT_CONT_FWHM_MIN must be < FIT_CONT_FWHM_MAX")
    if FIT_QPO_FWHM_MIN >= FIT_QPO_FWHM_MAX:
        errors.append("FIT_QPO_FWHM_MIN must be < FIT_QPO_FWHM_MAX")

    # Multi-scale smooth
    if not isinstance(PEAK_SMOOTH_SCALES, (list, tuple)) or len(PEAK_SMOOTH_SCALES) < 1:
        errors.append("PEAK_SMOOTH_SCALES must be a non-empty list of Hz values")
    else:
        for i, s in enumerate(PEAK_SMOOTH_SCALES):
            try:
                if float(s) <= 0:
                    errors.append(f"PEAK_SMOOTH_SCALES[{i}] must be > 0")
            except (TypeError, ValueError):
                errors.append(f"PEAK_SMOOTH_SCALES[{i}] must be numeric")

    # IC
    if str(CONT_IC_CRITERION).lower() not in ("aic", "bic"):
        errors.append(f"CONT_IC_CRITERION must be 'aic' or 'bic', got {CONT_IC_CRITERION!r}")
    if str(QPO_IC_CRITERION).lower() not in ("aic", "bic"):
        errors.append(f"QPO_IC_CRITERION must be 'aic' or 'bic', got {QPO_IC_CRITERION!r}")
    _chk_pos("CONT_IC_DELTA_MIN", CONT_IC_DELTA_MIN, allow_zero=True)
    _chk_pos("QPO_IC_DELTA_MIN",  QPO_IC_DELTA_MIN,  allow_zero=True)

    # Energy bands
    _chk_band("SOFT_BAND_KEV",  SOFT_BAND_KEV)
    _chk_band("HARD_BAND_KEV",  HARD_BAND_KEV)
    _chk_band("BROAD_RMS_BAND", BROAD_RMS_BAND)

    # Workers
    if int(N_WORKERS) < 1:
        errors.append(f"N_WORKERS must be >= 1, got {N_WORKERS}")
    if str(PARALLEL_START_METHOD).lower() not in ("spawn", "fork", "forkserver"):
        errors.append(f"PARALLEL_START_METHOD must be 'spawn', 'fork', or 'forkserver'")

    # TripleA knobs
    _chk_pos("AAA_N_STARTS",       AAA_N_STARTS)
    _chk_pos("AAA_JITTER_STD_LOG", AAA_JITTER_STD_LOG, allow_zero=True)
    _chk_pos("AAA_JITTER_STD_NU0", AAA_JITTER_STD_NU0, allow_zero=True)
    _chk_pos("AAA_FTOL",           AAA_FTOL)
    _chk_pos("AAA_GTOL",           AAA_GTOL)
    _chk_pos("AAA_MAXITER",        AAA_MAXITER)

    if errors:
        msg = "QPO_Parameter.py — validation errors:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ValueError(msg)


_validate_config()
