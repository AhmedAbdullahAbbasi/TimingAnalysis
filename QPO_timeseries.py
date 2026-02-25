#!/usr/bin/env python3
# QPO_timeseries.py

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QPO_Parameter as P


BANDS = ["full", "soft", "hard"]


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _to_num(x):
    return pd.to_numeric(x, errors="coerce")


def _has_cols(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).str.upper().eq("OK")]

    if "mjd_mid" not in d.columns:
        raise SystemExit("CSV missing required column: mjd_mid")

    d["mjd_mid"] = _to_num(d["mjd_mid"])
    d = d[d["mjd_mid"].notna()]
    d = d.sort_values("mjd_mid")
    return d


def _mask_fit_ok(d: pd.DataFrame, band: str) -> pd.Series:
    col = f"{band}_fit_ok"
    if col not in d.columns:
        return pd.Series(True, index=d.index)
    ok = d[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return ok.fillna(False)


def _plot_threepanel(
    d: pd.DataFrame,
    *,
    outpath: str,
    title: str,
    ylabel: str,
    ycols_by_band: dict,
    yerrcols_by_band: dict | None = None,
    require_fit_ok: bool = True,
    y_floor_positive: bool = False,
):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)

    for ax, band in zip(axes, BANDS):
        ycol = ycols_by_band.get(band)
        yerrcol = (yerrcols_by_band.get(band) if yerrcols_by_band else None)

        ax.set_title(band.upper(), loc="left", fontsize=10)

        if ycol is None or ycol not in d.columns:
            ax.text(0.5, 0.5, f"Missing column: {ycol}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        mask = pd.Series(True, index=d.index)
        if require_fit_ok:
            mask &= _mask_fit_ok(d, band)

        x = d.loc[mask, "mjd_mid"].to_numpy(float)
        y = _to_num(d.loc[mask, ycol]).to_numpy(float)

        yerr = None
        if yerrcol is not None and (yerrcol in d.columns):
            yerr = _to_num(d.loc[mask, yerrcol]).to_numpy(float)

        good = np.isfinite(x) & np.isfinite(y)
        if yerr is not None:
            good &= np.isfinite(yerr) & (yerr >= 0)

        if y_floor_positive:
            good &= (y > 0)
            if yerr is not None:
                good &= ((y - yerr) > 0)  # keep log-safe if we decide to log later

        x = x[good]
        y = y[good]
        if yerr is not None:
            yerr = yerr[good]

        # Errorbar scatter
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle="None", capsize=2)
        else:
            ax.plot(x, y, marker="o", linestyle="None")

        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("MJD (mid)")
    fig.suptitle(title, fontsize=12)

    dpi = int(getattr(P, "PLOT_DPI", 200))
    plt.savefig(outpath, dpi=dpi)
    plt.close(fig)


def make_timeseries_plots(df: pd.DataFrame, outdir: str):
    d = _prep_df(df)

    # 1) ν_QPO vs time (fit-based)
    nu_cols = {band: f"{band}_fit_qpo_nu0_hz" for band in BANDS}
    if _has_cols(d, list(nu_cols.values())):
        _plot_threepanel(
            d,
            outpath=os.path.join(outdir, "qpo_nu_vs_time.png"),
            title="QPO centroid frequency vs time (fit-based)",
            ylabel="ν_QPO (Hz)",
            ycols_by_band=nu_cols,
            yerrcols_by_band=None,
            require_fit_ok=True,
            y_floor_positive=True,
        )
    else:
        print("[WARN] Missing one or more fit-based ν_QPO columns; skipping qpo_nu_vs_time.png")

    # 2) Q vs time (fit-based)
    q_cols = {band: f"{band}_fit_qpo_Q" for band in BANDS}
    if _has_cols(d, list(q_cols.values())):
        _plot_threepanel(
            d,
            outpath=os.path.join(outdir, "qpo_Q_vs_time.png"),
            title="QPO quality factor Q vs time (fit-based)",
            ylabel="Q = ν0/FWHM",
            ycols_by_band=q_cols,
            yerrcols_by_band=None,
            require_fit_ok=True,
            y_floor_positive=True,
        )
    else:
        print("[WARN] Missing one or more fit-based Q columns; skipping qpo_Q_vs_time.png")

    # 3) QPO RMS vs time
    
    qpo_rms_fit_cols = {band: f"{band}_fit_qpo_rms" for band in BANDS}
    qpo_rms_fit_err_cols = {band: f"{band}_fit_qpo_rms_err" for band in BANDS}

    qpo_rms_diag_cols = {band: f"qpo_rms_{band}" for band in BANDS}
    qpo_rms_diag_err_cols = {band: f"qpo_rms_err_{band}" for band in BANDS}

    if _has_cols(d, list(qpo_rms_fit_cols.values())):
        
        use_fit_err = _has_cols(d, list(qpo_rms_fit_err_cols.values()))
        _plot_threepanel(
            d,
            outpath=os.path.join(outdir, "qpo_rms_vs_time.png"),
            title=("QPO RMS vs time (fit-based)" + (" + errors" if use_fit_err else "")),
            ylabel="QPO RMS",
            ycols_by_band=qpo_rms_fit_cols,
            yerrcols_by_band=(qpo_rms_fit_err_cols if use_fit_err else None),
            require_fit_ok=True,
            y_floor_positive=True,
        )
    elif _has_cols(d, list(qpo_rms_diag_cols.values())):
        
        use_diag_err = _has_cols(d, list(qpo_rms_diag_err_cols.values()))
        _plot_threepanel(
            d,
            outpath=os.path.join(outdir, "qpo_rms_vs_time.png"),
            title="QPO RMS vs time (diagnostic window; fallback)",
            ylabel="QPO-band RMS",
            ycols_by_band=qpo_rms_diag_cols,
            yerrcols_by_band=(qpo_rms_diag_err_cols if use_diag_err else None),
            require_fit_ok=False,
            y_floor_positive=True,
        )
    else:
        print("[WARN] No QPO RMS columns found (fit-based or diagnostic); skipping qpo_rms_vs_time.png")

    
    broad_cols = {band: f"broad_rms_0p1_30_{band}" for band in BANDS}
    broad_err_cols = {band: f"broad_rms_err_{band}" for band in BANDS}
    if _has_cols(d, list(broad_cols.values())):
        use_broad_err = _has_cols(d, list(broad_err_cols.values()))
        _plot_threepanel(
            d,
            outpath=os.path.join(outdir, "broad_rms_vs_time.png"),
            title="Broadband RMS (0.1–30 Hz) vs time",
            ylabel="Broadband RMS",
            ycols_by_band=broad_cols,
            yerrcols_by_band=(broad_err_cols if use_broad_err else None),
            require_fit_ok=False,
            y_floor_positive=True,
        )
    else:
        print("[WARN] Missing one or more broadband RMS columns; skipping broad_rms_vs_time.png")


def main():
    csv_path = os.path.join(
        getattr(P, "OUTDIR_BASE", "."),
        getattr(P, "COMMON_DIRNAME", "commonfiles"),
        getattr(P, "OUT_CSV_NAME", "qpo_summary.csv"),
    )
    outdir = _ensure_dir(os.path.join(
        getattr(P, "OUTDIR_BASE", "."),
        getattr(P, "COMMON_DIRNAME", "commonfiles"),
        "timeseries",
    ))

    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    make_timeseries_plots(df, outdir)
    print(f"[OK] Wrote time-series plots to: {outdir}")


if __name__ == "__main__":
    main()
