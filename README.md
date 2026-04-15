[README.md](https://github.com/user-attachments/files/26619135/README.md)
# QPO Fitting Pipeline

A Python pipeline for detecting and characterizing QPOs in NICER X-ray power density spectra. Fits multi-component Lorentzian models across three energy bands (full, soft, hard) per observation, using the Whittle maximum-likelihood criterion via Stingray.

> Tested on **GX 339-4** in the hard-intermediate spectral state.

---

## Architecture

```
obsids.txt → QPO_main.py (per-obsid loop, optionally parallel)
                ├── QPO_utils.py       event loading, energy filtering, PDS construction
                ├── QPO_fit.py         Lorentzian fitter, candidate finder, model selection
                ├── QPO_plot.py        three-band fit overlay plots
                ├── QPO_struct.py      per-obsid JSON fit result I/O
                └── QPO_Parameter.py   all configuration (validated at import)

QPO_timeseries.py    standalone; reads CSV → temporal evolution plots
QPO_interactive.py   standalone; terminal-driven single-obsid fitter
```

**Input:** `<BASE_DIR>/<SOURCE>/<obsid>/ni<obsid>_0mpu7_cl.evt`

**Output:**
```
<OUTDIR_BASE>/commonfiles/gx339_qpo_summary.csv
<OUTDIR_BASE>/<obsid>/<obsid>_fitresult.json
<OUTDIR_BASE>/<obsid>/<obsid>_threeband_fit.png
<OUTDIR_BASE>/commonfiles/timeseries/*.png   (12 temporal evolution figures)
```

---

## Installation

```bash
pip install numpy scipy astropy stingray matplotlib pandas
python -c "import QPO_Parameter"   # verify config loads cleanly
```

Stingray ≥ 2.0 required.

---

## Configuration

All parameters live in **`QPO_Parameter.py`** and are validated at import. Set these before the first run:

```python
# Paths
BASE_DIR   = "/path/to/event/files/"
SOURCE     = "GX339"
OBSIDS_TXT = "GX339_obs_hs.txt"
OUTDIR_BASE  = "/path/to/output/"

# Timing
DT           = 0.00390625   # seconds (1/256 s for NICER)
SEGMENT_SIZE = 64           # seconds

# QPO selection
QPO_FMIN  = 0.10    # Hz
QPO_FMAX  = 10.0    # Hz
QPO_MIN_Q = 3.0     # minimum Q = ν₀/FWHM

# Fit acceptance
FIT_RCHI_MAX = 1.5   # fits above this are flagged as bad

# Parallelism
PARALLEL_ENABLE = True
N_WORKERS       = 32
```

---

## Usage

**Batch pipeline:**
```bash
python QPO_main.py
```

**Temporal evolution plots** (after batch completes):
```bash
python QPO_timeseries.py
python QPO_timeseries.py --csv /path/to/summary.csv --outdir /path/to/output
```

**Interactive single-obsid inspection:**
```bash
python QPO_interactive.py --obsid 1200120106 --band full
```
Key interactive commands: `addcomp`, `removecomp`, `fit`, `freeze`, `rebin`, `saveresult`, `load`, `zoom`, `quit`. Type `help` for the full list.

---

## Key Design Notes

**Cross-band reseeding.** When one band finds a QPO but another does not, the pipeline donates the QPO frequency as a seed to the failing band and refits. When all bands fail, diagnostic whitened-peak frequencies are used as seeds (`CROSS_BAND_USE_DIAG_PEAKS_FALLBACK`).

**Per-bin m for log-rebinned PDS.** Logarithmic rebinning gives each bin a different effective averaging count. The per-bin `m` from Stingray is passed directly to the Whittle optimizer, ensuring correctly normalized rchi2 and IC values.

**`comp_types` as authoritative QPO label.** QPOs are identified by the optimizer's own classification, not post-hoc Q-threshold filtering, ensuring consistency across CSV, JSON struct, and plot annotations.

**IC gating.** Continuum order is selected by BIC (`CONT_IC_DELTA_MIN = 15`); QPO detection uses AIC with a lenient threshold (tunable to 0 for weak-QPO sources). Multi-QPO upgrades are gated separately by `FIT_MULTI_QPO_IC_DELTA_MIN`.

---

## Known Limitations

- **NICER-specific:** Event file paths assume `ni<obsid>_0mpu7_cl.evt`. Other instruments require changes to `build_evt_path()`.
- **Interactive mode** requires a display (TkAgg backend). Headless use needs X-forwarding or VNC.
- **`cont4` disabled by default.** Enable `FIT_CONT4_ENABLE = True` only for complex spectral states requiring a four-component continuum.

