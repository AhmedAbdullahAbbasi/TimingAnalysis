

# ==========================
# INPUT DIRECTORY STRUCTURE
# ==========================
BASE_DIR    = "/data2/NICER_DATA/NICER/out_2"
SOURCE      = "GX339"
OBSIDS_TXT  = "gx339_obsids.txt"

# ==========================
# OUTPUT DIRECTORY STRUCTURE
# ==========================
OUTDIR_BASE     = "/data2/NICER_DATA/NICER/GX339_timing_res"
COMMON_DIRNAME  = "commonfiles"
OUT_CSV_NAME    = "gx339_qpo_summary.csv"

# ==========================
# TIMING PARAMETERS
# ==========================
DT            = 0.00390625   # seconds
SEGMENT_SIZE  = 64           # seconds (AveragedPowerspectrum segment_size)

# ==========================
# DIAGNOSTIC PEAK SEARCH (QPO_main.find_qpo_peak_whitened)
# ==========================
FMIN              = 0.05     # Hz (diagnostic plotting + peak-finder band)
FMAX              = 10.0     # Hz
PEAK_IGNORE_BELOW = 0.10     # Hz

# Whitening smooth scale (used by BOTH diagnostic peak finder and fitter candidate finder)
PEAK_SMOOTH_HZ    = 0.3     # Hz

# A2 sigma-excess gate (diagnostic peak finder + fitter detection-first gating)
PEAK_REQUIRE_KSIGMA   = None      # set None to disable sigma gate
PEAK_PROMINENCE_SIGMA = 0.75     # prominence threshold in z-space (diagnostic peak finder)
PEAK_SIGMA_MODE       = "cont"   # "cont" or "p"  (sigma ~ cont/sqrt(m) or p/sqrt(m))
PEAK_RANK_BY          = "prominence"  # "prominence" or "height" (diagnostic peak finder)

# Candidate separation/limit (used by BOTH diagnostic peak finder and fitter candidate finder)
PEAK_MIN_SEP_HZ     = 0.15
PEAK_MAX_CANDIDATES = 8

# Candidate finder prominence in ratio-space 
PEAK_PROMINENCE = 0.08

# ==========================
# RMS INTEGRATION
# ==========================
BROAD_RMS_BAND = (0.1, 30.0)   # Hz (broadband rms)

# "QPO-band" diagnostic RMS around the diagnostic peak_full (not fit-based)
QPO_BW_FRAC = 0.10   # fractional half-width = frac * peak
QPO_BW_MIN  = 0.10   # Hz
QPO_BW_MAX  = 2.00   # Hz

# ==========================
# PLOT SETTINGS
# ==========================
PLOT_DPI      = 200
PLOT_LOGLOG   = True
CLOBBER       = True



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

# Candidate band (Hz) for QPO seeds + candidate finder
CAND_FMIN = 0.05
CAND_FMAX = 10.0

# ==========================
# QPO SELECTION AFTER FITTING (extract_qpo_params in QPO_main overlay)
# ==========================
QPO_FMIN   = 0.10
QPO_FMAX   = 10.0
QPO_MIN_Q  = 3

# ==========================
# OPTIMIZER + ACCEPTANCE
# ==========================
FIT_METHOD     = "Powell"
FIT_RCHI_MAX   = 1.5     # QPO_main uses this to mark fit_ok False if rchi2 > max
FIT_RCHI_TARGET = 1.3    # passed into fitter (stops escalation early if reached)

# ==========================
# FITTING: CONSTANT (WHITE-NOISE FLOOR)
# ==========================
FIT_INCLUDE_CONST   = True
FIT_CONST_SEED_FMIN = 40.0   # Hz, used to seed the constant floor

# ==========================
# FITTING: MULTI-START ROBUSTNESS
# ==========================
FIT_MULTI_START = True
FIT_N_STARTS    = 6
FIT_JITTER_FRAC = 0.12
FIT_RANDOM_SEED = 42

# ==========================
# FITTING: COMPONENT-WISE BOUNDS / CAPS (passed by QPO_main)
# ==========================
# Continuum centroid limits; if None => fitter uses ~2*df internally 
#NOTE: If df is too small, I would advise giving thsi some wiggle room.
FIT_CONT_X0_MAX_HZ = 0.2

FIT_CONT_FWHM_MIN = 0.30
FIT_CONT_FWHM_MAX = 64.0

FIT_QPO_FWHM_MIN  = 0.01
FIT_QPO_FWHM_MAX  = 5.0

FIT_HARM_FWHM_MIN = 0.03
FIT_HARM_FWHM_MAX = 8.0

# Amplitude caps (multiples of low-frequency level estimate)
FIT_CONT_AMP_FACTOR = 5.0
FIT_QPO_AMP_FACTOR  = 8.0
FIT_HARM_AMP_FACTOR = 5.0

# QPO seed width controls
FIT_QPO_FWHM_FRAC    = 0.03
FIT_QPO_FWHM_MIN_ABS = 0.02

# Harmonic search toggle
DO_HARMONIC_SEARCH = False

# ==========================
# FITTING: GUARDRAILS (passed by QPO_main)
# ==========================
FIT_GUARD_ENABLE = True
FIT_GUARD_OVERSHOOT_KSIGMA       = 3.0
FIT_GUARD_OVERSHOOT_MAX_RUN_BINS = 6
FIT_GUARD_OVERSHOOT_MAX_FRAC     = 0.08
FIT_GUARD_COMP_LOCAL_AMP_FACTOR  = 5.0

# ==========================
# FITTING: RETRIES (QPO_main passes as max_retries)
# ==========================
FIT_MAX_RETRIES = 10


# ==========================
# MODEL ORDER SELECTION (IC)
# ==========================
# Continuum: stricter (prevents cont3 from eating missed QPO peaks)
CONT_IC_CRITERION = "aic"     # "bic" recommended for continuum order
CONT_IC_DELTA_MIN = 10.0      

# QPO: less strict (Q gate + sigma gate + guardrails do the heavy lifting)
QPO_IC_CRITERION = "aic"      # "aic" recommended for QPO acceptance
QPO_IC_DELTA_MIN = 1.0        # try 2–6




# Multi-seed Stage1: try top-K candidate seeds
FIT_STAGE1_N_SEEDS = 6

# Conditional reseed (conservative)
FIT_RESEED_ENABLE          = True
FIT_RESEED_RCHI_BAD        = 1.5     # only reseed if best stage1 looks bad
FIT_RESEED_EDGE_FRAC       = 0.08    # too close to edge of candidate band triggers reseed
FIT_RESEED_AREA_MIN        = 1e-3     # set >0 to add "tiny area" trigger
FIT_RESEED_EXCLUDE_HZ_MIN  = 0.5
FIT_RESEED_EXCLUDE_DF_MULT = 10.0
FIT_RESEED_PROM_FACTOR     = 1.25
FIT_RESEED_SIGMA_FACTOR    = 1.10

# ==========================
# REBINNING (frequency domain)
# ==========================
DO_REBIN   = True
REBIN_MODE = "log"     # "log" or "linear"

# If REBIN_MODE == "log":
REBIN_LOG_F = 0.01    # typical 0.01–0.05 (higher = more aggressive)

# If REBIN_MODE == "linear": choose ONE
REBIN_FACTOR = 4.0
# REBIN_DF_HZ = 0.05    # optional absolute bin width (Hz); if set overrides REBIN_FACTOR

# ==========================
# CANDIDATE REBINNING (light, for peak finding / seed grid)
# ==========================
DO_CANDIDATE_LIGHT_REBIN     = True
DO_REBIN_CAND_WHEN_FIT_OFF   = True

CAND_REBIN_MODE  = "log"
CAND_REBIN_LOG_F = 0.008

CAND_REBIN_FACTOR = 2.0
# CAND_REBIN_DF_HZ = 0.02

# ==========================
# PARALLEL EXECUTION
# ==========================
PARALLEL_ENABLE       = True
N_WORKERS             = 32
PARALLEL_START_METHOD = "spawn"  # "spawn" safest; "fork" often fine on Linux


# ==========================
# LOGGING / OUTPUT CLEANUP
# ==========================
QUIET = True              # suppress non-essential prints + warnings
ONE_LINE_SUMMARY = True   # print one line per obsid
SHOW_QPO_DETAILS = True   # print nu and Q when QPO found

