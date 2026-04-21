[README.md](https://github.com/user-attachments/files/26918983/README.md)
# QPO Fitting Pipeline

A Python pipeline for detecting and characterising quasi-periodic oscillations (QPOs) in NICER X-ray power density spectra. The pipeline fits sums of Lorentzians to averaged PSDs using the TripleA optimiser — a custom Whittle-likelihood minimiser with an analytical gradient and L-BFGS-B box constraints — and produces a per-observation summary CSV, per-obsid JSON structs, multi-band PNG plots, and a full set of time-series figures.

---

## Contents

- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Quick start](#quick-start)
- [Input layout](#input-layout)
- [Configuration](#configuration)
  - [I/O paths](#io-paths)
  - [Timing and PDS](#timing-and-pds)
  - [Energy bands](#energy-bands)
  - [Fitting](#fitting)
  - [TripleA optimiser](#triplea-optimiser)
  - [Model-order selection](#model-order-selection)
  - [Cross-band reseeding](#cross-band-reseeding)
  - [Rebinning](#rebinning)
  - [Parallel execution](#parallel-execution)
  - [Plotting](#plotting)
  - [Logging](#logging)
- [Batch pipeline (QPO_main.py)](#batch-pipeline-qpo_mainpy)
  - [Pipeline stages](#pipeline-stages)
  - [Output layout](#output-layout)
  - [CSV column reference](#csv-column-reference)
- [Interactive fitter (QPO_interactive.py)](#interactive-fitter-qpo_interactivepy)
  - [Commands](#commands)
  - [Workflow tips](#workflow-tips)
- [Time-series plots (QPO_timeseries.py)](#time-series-plots-qpo_timeseriespy)
- [TripleA optimiser details](#triplea-optimiser-details)
- [Fit struct format](#fit-struct-format)
- [Module reference](#module-reference)

---

## Architecture

```
QPO_Parameter.py     All user-configurable knobs (101 parameters, validated at import)
QPO_main.py          Batch driver — one obsid at a time or parallel across N workers
QPO_fit.py           Fitting engine: model configs, IC selection, QPO extraction, error propagation
QPO_TripleA.py       TripleA optimiser: Whittle NLL, analytical gradient, L-BFGS-B
QPO_interactive.py   Terminal-driven interactive fitter with a live matplotlib window
QPO_plot.py          Plot rendering from band-block dicts (no FitResult dependency)
QPO_struct.py        Per-obsid JSON fit-result structs (save / load / warm-start)
QPO_timeseries.py    Batch time-series figures from the summary CSV
QPO_utils.py         Shared helpers: event loading, energy filtering, PDS construction, rebinning
```

`QPO_Parameter.py` is the single source of truth for all configuration. Every other module reads from it via `getattr(P, 'NAME', default)` at call time, so changes take effect without restarting a running pipeline.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| Python ≥ 3.10 | Type-union syntax in annotations |
| NumPy | Array maths throughout |
| SciPy | L-BFGS-B optimiser (`scipy.optimize.minimize`), peak finder (`scipy.signal.find_peaks`), median filter |
| Matplotlib | All plots; batch uses `Agg` backend; interactive uses `TkAgg` |
| Astropy | FITS reading (`astropy.io.fits`), MJD → ISO conversion (`astropy.time.Time`) |
| Stingray | `EventList`, `AveragedPowerspectrum`; `fit_lorentzians` for the optional Powell/Nelder-Mead path in the interactive fitter |
| Pandas | Time-series plotter CSV ingestion |

Install with:

```bash
pip install numpy scipy matplotlib astropy stingray pandas
```

---

## Quick start

```bash
# 1. Edit QPO_Parameter.py — set BASE_DIR, SOURCE, OBSIDS_TXT, OUTDIR_BASE
# 2. Run the batch pipeline
python QPO_main.py

# 3. Inspect one obsid interactively
python QPO_interactive.py --obsid 1200120106 --band full

# 4. Generate time-series plots after the batch run
python QPO_timeseries.py
```

---

## Input layout

The pipeline expects NICER cleaned event files at:

```
<BASE_DIR>/<SOURCE>/<obsid>/ni<obsid>_0mpu7_cl.evt
```

For example, with `BASE_DIR = "/data/nicer/"` and `SOURCE = "maxij1820+70"`:

```
/data/nicer/maxij1820+70/1200120106/ni1200120106_0mpu7_cl.evt
```

The obsid list is a plain-text file (`OBSIDS_TXT`), one obsid per line, `#` for comments:

```
# MAXI J1820+070 — hard state sample
1200120106
1200120107
1200120115
```

---

## Configuration

All parameters live in `QPO_Parameter.py`. The file is validated at import time — a `ValueError` with a clear list of every problem is raised before any observation is touched.

### I/O paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_DIR` | `/data2/sena.ozgur/nicer/` | Root of the event-file tree |
| `SOURCE` | `maxij1820+70` | Source sub-directory name |
| `OBSIDS_TXT` | `MAXIJ1820_sample.txt` | Plain-text list of obsids to process |
| `OUTDIR_BASE` | `/data2/.../MAXIJ1820_timing_res/` | Root of all pipeline outputs |
| `COMMON_DIRNAME` | `commonfiles` | Sub-directory for the CSV and time-series plots |
| `OUT_CSV_NAME` | `MAXIJ1820_qpo_summary.csv` | Filename of the summary CSV |

### Timing and PDS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DT` | `0.00390625` s | Time resolution (1/256 s) |
| `SEGMENT_SIZE` | `64` s | Segment length for `AveragedPowerspectrum` |

The Nyquist frequency is `1/(2·DT)` = 128 Hz. The frequency resolution is `1/SEGMENT_SIZE` = 0.015625 Hz.

### Energy bands

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DO_ENERGY_BANDS` | `True` | Fit soft and hard bands in addition to the full band |
| `SOFT_BAND_KEV` | `(0.3, 2.0)` | Soft-band energy range in keV |
| `HARD_BAND_KEV` | `(2.0, 10.0)` | Hard-band energy range in keV |
| `PI_PER_KEV` | `100.0` | NICER PI-channel conversion (100 channels per keV) |
| `USE_GTIS` | `True` | Propagate GTIs from the event file into the filtered EventList |

### Fitting

#### Master switches

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DO_FIT` | `True` | Run Lorentzian fitting (disable for RMS-only runs) |
| `SAVE_FIT_PLOTS` | `True` | Write a 6-panel PNG after each obsid |
| `FIT_METHOD` | `"TripleA"` | Optimiser: `"TripleA"` (default), `"Powell"`, or `"Nelder-Mead"` |
| `FIT_RCHI_MAX` | `1.5` | rchi2 above this marks `fit_ok = False` in the CSV |

#### Frequency ranges

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_FMIN` | `0.05` Hz | Lower edge of the fit band |
| `FIT_FMAX` | `64.0` Hz | Upper edge of the fit band |
| `CAND_FMIN` | `0.05` Hz | Lower edge of the QPO-candidate search band |
| `CAND_FMAX` | `10.0` Hz | Upper edge of the QPO-candidate search band |
| `QPO_FMIN` | `0.10` Hz | Lower bound for QPO reporting (applied to CSV and plots) |
| `QPO_FMAX` | `10.0` Hz | Upper bound for QPO reporting |
| `QPO_MIN_Q` | `3` | Minimum Q = ν₀/FWHM for a component to be classified as a QPO |

#### Candidate peak finder

The pipeline runs a multi-scale z-score peak finder on the lightly rebinned PDS to generate QPO seed frequencies before the full fit. Each scale independently detects peaks; results are deduplicated by minimum separation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FMIN` / `FMAX` | `0.1` / `10.0` Hz | Search range for the diagnostic peak finder |
| `PEAK_SMOOTH_SCALES` | `[0.2, 0.5, 1.5]` Hz | Smoothing widths for multi-scale candidate search |
| `PEAK_PROMINENCE` | `0.4` | Minimum z-score prominence for a candidate peak |
| `PEAK_PROMINENCE_SIGMA` | `0.5` | Minimum absolute z-score height |
| `PEAK_MIN_SEP_HZ` | `0.10` Hz | Minimum separation between distinct candidates |
| `PEAK_MAX_CANDIDATES` | `8` | Maximum number of QPO seeds forwarded to the fitter |
| `PEAK_REQUIRE_KSIGMA` | `None` | If set, only peaks above this z-score are kept |
| `PEAK_SIGMA_MODE` | `"cont"` | Sigma estimate: `"cont"` (rolling median) or `"p"` (raw power) |
| `PEAK_RANK_BY` | `"height"` | Sort candidates by `"height"` or `"prominence"` |
| `PEAK_IGNORE_BELOW` | `0.10` Hz | Suppress peaks below this frequency (red-noise artefacts) |
| `PEAK_SMOOTH_HZ` | `0.3` Hz | Smoothing width for the single-scale diagnostic peak finder |

#### Component bounds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_CONT_X0_NARROW_HZ` | `0.3` Hz | Centroid window for the broadest continuum (±) |
| `FIT_CONT_X0_WIDE_HZ` | `3.0` Hz | Centroid window for the medium continuum (±) |
| `FIT_CONT_X0_FREE_HZ` | `8.0` Hz | Centroid window for the third continuum (±) |
| `FIT_CONT_FWHM_MIN` | `0.30` Hz | Minimum FWHM for continuum Lorentzians |
| `FIT_CONT_FWHM_MAX` | `64.0` Hz | Maximum FWHM for continuum Lorentzians |
| `FIT_QPO_FWHM_MIN` | `0.08` Hz | Minimum FWHM for QPO components |
| `FIT_QPO_FWHM_MAX` | `5.0` Hz | Maximum FWHM for QPO components |
| `FIT_CONT_AMP_FACTOR` | `12.0` | Continuum amplitude cap as a multiple of the low-frequency power level |
| `FIT_QPO_AMP_FACTOR` | `12.0` | QPO amplitude cap as a multiple of the low-frequency power level |
| `FIT_QPO_FWHM_FRAC` | `0.03` | QPO seed FWHM as a fraction of ν₀ |
| `FIT_QPO_FWHM_MIN_ABS` | `0.08` Hz | Absolute floor for the QPO seed FWHM |

#### White-noise floor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_INCLUDE_CONST` | `True` | Add a free white-noise constant to every model |
| `FIT_CONST_SEED_FMIN` | `40.0` Hz | Median PDS power above this frequency seeds the constant |
| `FIT_CONST_CAP_FACTOR` | `5.0` | Constant upper bound = cap × seed value |

#### Guardrails

Guardrails reject a fit before IC comparison if the model systematically overshoots the data or if any component is implausibly larger than its local PDS level.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_GUARD_ENABLE` | `True` | Enable guardrail checks |
| `FIT_GUARD_OVERSHOOT_KSIGMA` | `4.0` | Overshoot threshold in σ units (raised to 4 for log-rebinned data) |
| `FIT_GUARD_OVERSHOOT_MAX_RUN_BINS` | `6` | Maximum run of consecutive overshoot bins |
| `FIT_GUARD_OVERSHOOT_MAX_FRAC` | `0.10` | Maximum fraction of overshoot bins across the fit band |
| `FIT_GUARD_COMP_LOCAL_AMP_FACTOR` | `6.0` | Maximum ratio of component amplitude to local PDS level |

#### QPO acceptance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QPO_IC_CRITERION` | `"aic"` | IC used for the 0-QPO → 1-QPO upgrade gate |
| `QPO_IC_DELTA_MIN` | `0` | IC improvement required to accept a QPO (0 = any improvement) |
| `FIT_QPO_RCHI_IMPROVE_MIN` | `0.05` | Minimum rchi2 reduction to accept a QPO when the IC gate fails |
| `QPO_SORT_BY` | `"area"` | Primary sort key for multiple QPOs: `"area"`, `"freq"`, or `"q"` |

#### Multi-QPO detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_MAX_QPOS` | `3` | Maximum simultaneous QPO components |
| `FIT_MULTI_QPO_IC_DELTA_MIN` | `5` | IC improvement required for the 1-QPO → 2-QPO upgrade |

#### Continuum model-order selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONT_IC_CRITERION` | `"aic"` | IC used for the cont2 → cont3 upgrade gate |
| `CONT_IC_DELTA_MIN` | `12.0` | IC improvement required to prefer a 3-component continuum |

#### Multi-start (Stingray/Powell path only)

These parameters only affect the Stingray/Powell fallback path in the interactive fitter. They have no effect on the default TripleA batch pipeline, which manages its own restarts via `AAA_N_STARTS`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIT_N_STARTS` | `3` | Number of multi-start attempts |
| `FIT_JITTER_FRAC` | `0.12` | Jitter fraction applied to each subsequent start |
| `FIT_RANDOM_SEED` | `42` | RNG seed (shared with TripleA) |

### TripleA optimiser

All six `AAA_*` parameters are read by `QPO_TripleA.tripleA_fit_once` at call time, so they can be changed mid-run.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AAA_N_STARTS` | `5` | Independent L-BFGS-B restarts per model configuration |
| `AAA_JITTER_STD_LOG` | `0.30` | Log-space std for amplitude/FWHM jitter between restarts (≈ ±35%) |
| `AAA_JITTER_STD_NU0` | `0.10` | ν₀ jitter as a fraction of the centroid window width |
| `AAA_FTOL` | `1e-11` | L-BFGS-B function-value convergence tolerance |
| `AAA_GTOL` | `1e-7` | L-BFGS-B gradient-norm convergence tolerance |
| `AAA_MAXITER` | `1000` | Maximum L-BFGS-B iterations per restart |

### Cross-band reseeding

After independent per-band fits, QPO frequencies found in one band are injected as forced seeds into bands that did not detect them. This is the primary mechanism for consistent multi-band QPO detection.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DO_CROSS_BAND_RESEED` | `True` | Enable cross-band seed injection |
| `CROSS_BAND_RESEED_RCHI_BAD` | `1.5` | Trigger cross-seed refit if current rchi2 exceeds this |
| `CROSS_BAND_QPO_AREA_MIN` | `1e-4` | Minimum QPO integrated power (frac-rms²) for a QPO to donate seeds |
| `CROSS_BAND_USE_DIAG_PEAKS_FALLBACK` | `True` | When no band finds a QPO, use diagnostic peak frequencies as seeds |

### Rebinning

The pipeline maintains two separately rebinned PDS per band: one for the full fit, one for the lighter candidate search.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DO_REBIN` | `True` | Enable rebinning of the fit PDS |
| `REBIN_MODE` | `"log"` | `"log"` or `"linear"` |
| `REBIN_LOG_F` | `0.01` | Fractional log-bin step (1% per bin) |
| `REBIN_FACTOR` | `4.0` | Linear rebin factor (when `REBIN_MODE = "linear"`) |
| `REBIN_DF_HZ` | `None` | Fixed bin width in Hz for linear rebinning; overrides `REBIN_FACTOR` |
| `DO_CANDIDATE_LIGHT_REBIN` | `True` | Rebin the candidate PDS more lightly than the fit PDS |
| `DO_REBIN_CAND_WHEN_FIT_OFF` | `True` | Rebin the candidate PDS even when `DO_REBIN = False` |
| `CAND_REBIN_MODE` | `"log"` | Candidate rebinning mode |
| `CAND_REBIN_LOG_F` | `0.008` | Fractional log-bin step for the candidate PDS |
| `CAND_REBIN_FACTOR` | `2.0` | Linear factor for candidate PDS |
| `CAND_REBIN_DF_HZ` | `None` | Fixed Hz width for candidate linear rebinning |

### RMS integration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BROAD_RMS_BAND` | `(0.1, 30.0)` Hz | Integration limits for broadband fractional RMS |
| `QPO_BW_FRAC` | `0.10` | Fractional half-width around the diagnostic peak for QPO-band RMS |
| `QPO_BW_MIN` | `0.10` Hz | Minimum QPO-band half-width |
| `QPO_BW_MAX` | `2.00` Hz | Maximum QPO-band half-width |

### Parallel execution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PARALLEL_ENABLE` | `False` | Enable multi-process execution |
| `N_WORKERS` | `32` | Worker process count |
| `PARALLEL_START_METHOD` | `"spawn"` | `"spawn"`, `"fork"`, or `"forkserver"` — use `"spawn"` to avoid Stingray import lock contention |

### Plotting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLOT_DPI` | `200` | Output DPI for all PNG files |
| `PLOT_YMIN` | `1e-6` | Floor for the PDS y-axis lower bound |
| `CLOBBER` | `True` | Overwrite existing PNG files |

### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QUIET` | `True` | Suppress verbose Stingray/optimiser output during batch runs |
| `ONE_LINE_SUMMARY` | `True` | Print a one-line result summary per obsid |
| `SHOW_QPO_DETAILS` | `True` | Include ν₀ and Q in the one-line summary |

---

## Batch pipeline (`QPO_main.py`)

```bash
python QPO_main.py
```

### Pipeline stages

For each obsid the pipeline runs through these stages in order:

1. **Event loading** — reads `ni<obsid>_0mpu7_cl.evt` with Stingray `EventList.read`. The mid-observation MJD is extracted from the FITS header keywords `TSTART`, `TSTOP`, `MJDREFI`, `MJDREFF`, and `TIMEZERO`.

2. **PDS construction** — light curves are built with `EventList.to_lc(dt=DT)` and averaged spectra with `AveragedPowerspectrum(lc, segment_size=SEGMENT_SIZE, norm="frac")`. For energy-band runs, the event list is filtered to PI channels corresponding to `SOFT_BAND_KEV` and `HARD_BAND_KEV` before this step.

3. **Rebinning** — the raw PDS is rebinned twice: once lightly for the candidate finder and once more aggressively for the fit.

4. **Diagnostic peak finding** — a whitened (rolling-median continuum subtracted) peak finder runs on the candidate PDS to produce a reference QPO frequency. This is recorded in the CSV independently of whether a fit is performed.

5. **RMS metrics** — broadband fractional RMS (0.1–30 Hz) and a narrow-band QPO-region RMS (centred on the diagnostic peak) are computed directly from the PDS.

6. **Lorentzian fitting** — `fit_lorentzian_family` builds every candidate model configuration (cont2, cont3, cont2+QPO, cont3+QPO, and simultaneous 2-QPO pairs when `FIT_MAX_QPOS ≥ 2`), runs TripleA on each, and selects the best model via IC comparison. The fit result includes parameter covariance from Hessian inversion at the Whittle MLE, used to propagate errors on ν₀, FWHM, Q, ν_max, rms², and rms.

7. **Cross-band reseeding** — QPO frequencies found in any band are offered as forced seeds to bands that missed them, and those bands are re-fitted.

8. **Struct save** — each band's fit result is merged into `<OUTDIR_BASE>/<obsid>/<obsid>_fitresult.json`.

9. **Plot save** — a 6-panel PNG (full/soft/hard × PDS+model / residuals) is written when `SAVE_FIT_PLOTS = True`.

### Output layout

```
<OUTDIR_BASE>/
  commonfiles/
    <OUT_CSV_NAME>               Summary CSV (one row per obsid)
    timeseries/                  Time-series PNGs (written by QPO_timeseries.py)
  <obsid>/
    <obsid>_fitresult.json       Per-obsid fit result struct (v1.1)
    <obsid>_fits_full_soft_hard.png   6-panel fit overlay
```

### CSV column reference

The summary CSV has 88 columns. The most commonly used ones are:

| Column | Description |
|--------|-------------|
| `obsid` | Observation ID |
| `mjd_mid` | Mid-observation MJD |
| `mean_rate_cps` | Mean count rate (full band) |
| `peak_f_hz_full/soft/hard` | Diagnostic whitened-peak frequency per band |
| `broad_rms_0p1_30_<band>` | Broadband fractional RMS (0.1–30 Hz) |
| `broad_rms_err_<band>` | 1σ error on broadband RMS |
| `<band>_fit_ok` | `True` if fit converged with rchi2 ≤ `FIT_RCHI_MAX` |
| `<band>_fit_rchi2` | Reduced chi-squared |
| `<band>_fit_nlor` | Number of Lorentzian components |
| `<band>_fit_qpo_nu0_hz` | Primary QPO centroid frequency (Hz) |
| `<band>_fit_qpo_fwhm_hz` | Primary QPO FWHM (Hz) |
| `<band>_fit_qpo_Q` | Primary QPO quality factor Q = ν₀/FWHM |
| `<band>_fit_qpo_rms` | Primary QPO fractional RMS amplitude |
| `<band>_fit_qpo_rms2` | Primary QPO integrated power (frac-rms²) |
| `<band>_fit_qpo_nu_max_hz` | Primary QPO characteristic frequency ν_max = √(ν₀² + (FWHM/2)²) |
| `<band>_fit_qpo_nu0_err` | 1σ error on ν₀ (from Hessian covariance) |
| `<band>_fit_qpo_fwhm_err` | 1σ error on FWHM |
| `<band>_fit_qpo_Q_err` | 1σ error on Q |
| `<band>_fit_qpo_rms_err` | 1σ error on fractional RMS |
| `<band>_fit_qpo_nu_max_err` | 1σ error on ν_max |
| `<band>_comp_nu0s` | Semicolon-separated ν₀ for all Lorentzians |
| `<band>_comp_fwhms` | Semicolon-separated FWHM for all Lorentzians |
| `<band>_comp_amps` | Semicolon-separated amplitudes |

`<band>` is `full`, `soft`, or `hard`. Multi-QPO observations additionally populate `<band>_fit_qpo_nu0s_hz`, `<band>_fit_qpo_Qs`, `<band>_fit_qpo_rmss`, and their error counterparts as semicolon-separated lists.

---

## Interactive fitter (`QPO_interactive.py`)

```bash
python QPO_interactive.py --obsid 1200120106 --band full
python QPO_interactive.py --obsid 1200120106 --band soft
```

A live matplotlib window shows the PDS, the current model overlay, individual component traces, and a residual panel. All interaction is typed in the terminal. The fitter loads the event file directly from `BASE_DIR`/`SOURCE`, constructs and rebins the PDS using the same parameters as the batch pipeline, and can save results back to the same JSON struct that `QPO_main.py` writes.

### Commands

```
addcomp qpo              Prompt for ν₀, FWHM, amplitude → add QPO
addcomp continuum        Prompt for ν₀, FWHM, amplitude → add continuum
removecomp <idx>         Remove component by index
editcomp   <idx>         Re-enter parameters for a component
freeze  <idx> [field]    Freeze a component field: nu0, fwhm, amp, or all
unfreeze <idx> [field]   Unfreeze a component field (or all)
setconst  <value>        Set the white-noise floor (always fixed during fit)
list                     Print the component table with frozen status
params                   Show all tunable parameters
setparam  <n> <val>      Change a parameter (e.g. setparam fit_rchi_max 2.0)
fit                      Run the optimiser on the current components
status                   Print fit statistics and QPO parameters
saveresult [path.json]   Save fit result (default: update the loaded struct)
load <obsid> [band]      Load a saved struct as initial components
load <path.json> [band]  Load from an explicit file path
plotresult [path.png]    Save a publication-quality 2-panel PNG for this band
plotall    [path.png]    Generate the full 3-band plot from the saved struct
save      [filename]     Save the live interactive canvas to PNG
rebin log <f>            Log-rebin with fractional step f
rebin linear <df>        Linear-rebin to bin width df Hz
rebin none               Reset to original binning
zoom  <fmin> <fmax>      Set x-axis limits (Hz)
clear                    Remove all components (keeps last fit)
reset                    Clear components AND last fit result
help                     Show this list
quit                     Exit
```

### Workflow tips

**Starting from a batch result:** `load <obsid> full` seeds all continuum and QPO components from the saved struct. The white-noise constant is also loaded. Run `fit` to refine, then `saveresult` to write back.

**Freezing parameters:** Use `freeze <idx> nu0` to hold ν₀ fixed while letting FWHM and amplitude float — useful when the QPO frequency is well-known from another band. TripleA implements frozen fields as tight L-BFGS-B box bounds (±0.1% of the current value), which are respected as hard constraints.

**Switching optimiser:** `setparam fit_method Powell` switches to the Stingray Powell path for comparison. `setparam fit_method TripleA` switches back. The Powell path uses `FIT_N_STARTS` jittered restarts and Stingray's uniform-prior penalty approach.

**Rebinning:** `rebin log 0.02` gives a coarser log-rebin for visual inspection. The fit always runs on whatever binning is currently active. `rebin none` restores the original Stingray-averaged PDS.

**The interactive fitter uses a single continuum centroid window** (`FIT_CONT_X0_MAX_HZ = 0.2 Hz` by default) applied to all continuum components, rather than the three-level scheme used by the batch fitter. For sources where the medium continuum (Lb) sits well above 0.2 Hz, increase this via `setparam cont_x0_max_hz 3.0` before fitting.

---

## Time-series plots (`QPO_timeseries.py`)

```bash
python QPO_timeseries.py                          # uses paths from QPO_Parameter.py
python QPO_timeseries.py --csv /path/to/my.csv
python QPO_timeseries.py --outdir /path/to/output
```

Reads the summary CSV and writes 13 PNG figures to `<OUTDIR_BASE>/<COMMON_DIRNAME>/timeseries/`. Only rows with `status = OK` are plotted; bad-fit observations are shown with open markers at reduced opacity rather than dropped, so data-quality problems remain visible.

| Figure | Content |
|--------|---------|
| `qpo_nu_vs_time.png` | QPO centroid frequency ν₀ (per band) |
| `qpo_nu_max_vs_time.png` | Characteristic frequency ν_max = √(ν₀² + (FWHM/2)²) |
| `qpo_fwhm_vs_time.png` | QPO FWHM |
| `qpo_Q_vs_time.png` | Quality factor Q = ν₀/FWHM (reference line at `QPO_MIN_Q`) |
| `qpo_rms_vs_time.png` | QPO fractional RMS (fit-based preferred; diagnostic fallback) |
| `qpo_rms2_vs_time.png` | QPO integrated power rms² |
| `broad_rms_vs_time.png` | Broadband fractional RMS (0.1–30 Hz) |
| `count_rate_vs_time.png` | Mean count rate (single panel) |
| `fit_rchi2_vs_time.png` | Reduced chi-squared per band (fit quality monitor) |
| `fit_nlor_vs_time.png` | Number of Lorentzians (model complexity) |
| `fit_const_vs_time.png` | White-noise constant |
| `peak_freq_vs_time.png` | Diagnostic whitened-peak frequency |
| `summary.png` | 4-panel overview: count rate / ν_max / Q / broadband RMS (full band only) |

---

## TripleA optimiser details

TripleA solves the Whittle negative log-likelihood:

```
L(θ) = Σᵢ [ log M(fᵢ; θ)  +  P(fᵢ) / M(fᵢ; θ) ]
```

where `M(f; θ) = C + Σ_k A_k g_k² / ((f − ν₀_k)² + g_k²)` is the sum-of-Lorentzians model and `P(f)` is the observed power.

**Parameterisation.** Amplitudes and FWHMs are log-transformed: `θ[3k] = log A_k`, `θ[3k+1] = ν₀_k` (linear Hz), `θ[3k+2] = log Δ_k`. This makes positivity automatic, removes the need for prior-cliff penalties, and equalises parameter scales for the Hessian approximation.

**Analytical gradient.** All partial derivatives are derived in closed form as vectorised dot products, each O(N_freq). The total gradient evaluation is faster than a single finite-difference step with Powell.

**Parameter covariance.** After convergence the Hessian is estimated by central finite differences of the analytical gradient (2·npar gradient evaluations). The log-space covariance is converted to linear space via the delta method (Jacobian is diagonal: ∂A/∂log A = A, etc.) and stored in the fit struct for downstream error propagation on Q, ν_max, rms, and rms².

**Gradient check.** To verify the gradient numerically:

```python
from scipy.optimize import check_grad
from QPO_TripleA import _whittle_loss_and_grad, _pack_theta, _build_bounds

err = check_grad(
    lambda t: _whittle_loss_and_grad(t, freq, power, nlor, True)[0],
    lambda t: _whittle_loss_and_grad(t, freq, power, nlor, True)[1],
    theta0,
)
print(f"gradient error: {err:.3e}")   # typical: 1e-7 to 1e-9
```

---

## Fit struct format

Each obsid produces a JSON file `<obsid>_fitresult.json` (struct version 1.1). The file contains fit results for up to three energy bands, merged atomically so a partial run never corrupts earlier results.

```json
{
  "version":   "1.1",
  "obsid":     "1200120106",
  "source":    "maxij1820+70",
  "mjd_mid":   59500.123,
  "timestamp": "2025-01-01T12:00:00+00:00",

  "full": {
    "ok":           true,
    "message":      "OK (cont2+qpo@3.142Hz)",
    "nlor":         3,
    "rchi2":        1.08,
    "aic":         -312.4,
    "bic":         -298.1,
    "deviance":     450.2,
    "red_deviance": 1.12,
    "const":        0.00142,
    "const_err":    0.000012,
    "peak_hz":      3.14,
    "comp_types":   ["cont", "cont", "qpo"],
    "pars":         [[nu0, fwhm, amp], ...],
    "par_errors":   [[nu0_err, fwhm_err, amp_err], ...],
    "nu_max":       [{"nu_max": float, "nu_max_err": float}, ...]
  },
  "soft": { ... },
  "hard": { ... }
}
```

`par_errors` entries are `null` when the Hessian was singular at that component (active bounds, degenerate fit). The `nu_max` list is pre-computed as `√(ν₀² + (FWHM/2)²)` for each component so downstream tools do not need to recompute it.

Structs can be loaded as warm starts in either the batch pipeline (cross-band reseeding reads them) or the interactive fitter (`load <obsid> [band]`). The function `struct_to_warm_comps(struct, band)` in `QPO_struct.py` extracts a `{"cont": [...], "qpo": [...], "const": float}` dict suitable for seeding either path.

---

## Module reference

| Module | Public API |
|--------|-----------|
| `QPO_fit` | `fit_lorentzian_family`, `find_qpo_candidates`, `extract_qpo_params_list`, `component_power_integral`, `lorentz`, `FitResult` |
| `QPO_TripleA` | `tripleA_fit_once` |
| `QPO_struct` | `save_fit_struct`, `load_fit_struct`, `struct_path`, `struct_to_warm_comps`, `struct_summary` |
| `QPO_plot` | `save_threeband_plot`, `save_band_plot`, `fitresult_to_band_block`, `plot_band` |
| `QPO_utils` | `load_pds_for_band`, `make_averaged_pds`, `rebin_pds`, `maybe_rebin_pds_fit`, `maybe_rebin_pds_candidate`, `filter_events_by_energy`, `build_evt_path`, `safe_m_from_pds`, `kev_to_pi` |
| `QPO_timeseries` | `make_timeseries_plots` |
| `QPO_main` | `analyze_obsid`, `main`, `read_obsids` |
| `QPO_interactive` | `launch`, `TerminalFitter` |
