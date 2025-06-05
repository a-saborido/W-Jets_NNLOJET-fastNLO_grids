#!/usr/bin/env python3
"""
Grid-closure plots for W + jet at √s = 8 TeV (ATLAS).

Outputs
-------

1.  Per-order figures
        figure_<ORDER>_<OBS>.png
    with three stacked panels:
        • differential distribution
        • Grid / NNLOJET ratio
        • Grid / HEPData ratio

2.  Combined canvas per observable
        figure_<OBS>_combined.png
    showing LO | NLO | NNLO columns side-by-side.
      • Legend placed in the NNLO column.
      • y-axis labels and tick numbers kept only on the LO column.

Central scale choice is muR = muF = mW.
"""

# --------------------------------------------------------------------------- #
#  Imports & matplotlib style                                                 #
# --------------------------------------------------------------------------- #
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({"font.size": 10})           # uniform font size

# --------------------------------------------------------------------------- #
#  Axis‑label dictionaries                                                    #
# --------------------------------------------------------------------------- #
X_LABEL: Dict[str, str] = {
    "ptj1":   r"$p_T^{j1}$  [GeV]",
    "ptw":    r"$p_T^{W}$   [GeV]",
    "abs_yj1": r"$|y^{j1}|$",
    "ht_full": r"$H_T$ [GeV]",
}

Y_LABEL: Dict[str, str] = {
    "ptj1":   r"$\mathrm{d}\sigma/\mathrm{d}p_T^{\,j1}$  [fb/GeV]",
    "ptw":    r"$\mathrm{d}\sigma/\mathrm{d}p_T^{\,W}$   [fb/GeV]",
    "abs_yj1": r"$\mathrm{d}\sigma/\mathrm{d}|y^{\,j1}|$  [fb]",
    "ht_full": r"$\mathrm{d}\sigma/\mathrm{d}H_T$         [fb/GeV]",
}

# --------------------------------------------------------------------------- #
#  File‑parsing helpers                                                       #
# --------------------------------------------------------------------------- #
def read_hepdata(csv_file: Path) -> Tuple[pd.Series, pd.Series, List[pd.Series]]:
    """Return bin centres, cross-sections, and ±errors from a HEPData CSV."""
    df = pd.read_csv(csv_file)
    centres = pd.to_numeric(df.iloc[:, 0])
    sigma   = pd.to_numeric(df.iloc[:, 3])
    err_up  = np.abs(pd.to_numeric(df.iloc[:, 4]))
    err_dn  = np.abs(pd.to_numeric(df.iloc[:, 5]))
    return centres, sigma, [err_dn, err_up]


def parse_grid_out(path: Path) -> pd.DataFrame:
    """Parse grid *.out file → dataframe of all bins x scale points."""
    rows = []
    xmur = xmuf = None

    pat_scale = re.compile(r"xmur, xmuf chosen here are:\s*([0-9.]+),\s*([0-9.]+)")
    pat_data  = re.compile(r"^\s*\d+\s")

    with open(path) as handle:
        for line in handle:
            if (m := pat_scale.search(line)):
                xmur, xmuf = map(float, m.groups())
                continue
            if pat_data.match(line):
                p = line.split()
                bin_min, bin_max = map(float, (p[3], p[4]))
                rows.append(
                    dict(
                        BinCenter=0.5 * (bin_min + bin_max),
                        BinMin=bin_min,
                        BinMax=bin_max,
                        xmur=xmur,
                        xmuf=xmuf,
                        LO=float(p[6]),
                        NLO=float(p[7]),
                        NNLO=float(p[8]),
                    )
                )
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def envelope(df: pd.DataFrame, order: str) -> pd.DataFrame:
    """Return dataframe with central sigma and scale envelope ±errors for <order>."""
    central = df[(df.xmur == 1.0) & (df.xmuf == 1.0)]
    central = (
        central.set_index("BinCenter")[[order, "BinMin", "BinMax"]]
        .rename(columns={order: "sigma"})
    )

    group = df.groupby("BinCenter")[order]
    env   = pd.DataFrame({"lo": group.min(), "hi": group.max()})

    merged = pd.concat([central, env], axis=1).dropna()
    merged["err_dn"] = merged["sigma"] - merged["lo"]
    merged["err_up"] = merged["hi"]    - merged["sigma"]
    merged["width"]  = merged["BinMax"] - merged["BinMin"]
    print("ENVELOPE")
    print(merged.reset_index())
    return merged.reset_index()

# --------------------------------------------------------------------------- #
#  Maths                                                                      #
# --------------------------------------------------------------------------- #
def ratio_and_err(a: pd.Series, a_err: pd.Series,
                  b: pd.Series, b_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Ratio a/b with propagated symmetric error."""
    ratio  = a / b
    rel_sq = (a_err / a) ** 2 + (b_err / b) ** 2
    return ratio, ratio * np.sqrt(rel_sq)

# --------------------------------------------------------------------------- #
#  Column‑drawing helper (one order)                                          #
# --------------------------------------------------------------------------- #
def _make_one_column(
    ax_main,
    ax_r1,
    ax_r2,
    *,
    order: str,
    observable: str,
    env_df: pd.DataFrame,
    var_df: pd.DataFrame,
    hep_tuple: Tuple[pd.Series, pd.Series, List[pd.Series]],
    show_legend: bool,
) -> None:
    """Draw distribution + two ratio panels for one perturbative order."""

    # --- MAIN PANEL (scale & stat bands) ----------------------------------- #
    first_scale, first_stat = True, True
    for _, row in env_df.iterrows():
        ax_main.fill_between(
            [row.BinMin, row.BinMax],
            [row.sigma - row.err_dn] * 2,
            [row.sigma + row.err_up] * 2,
            color="tab:blue",
            alpha=0.25,
            step="pre",
            label="scale unc." if first_scale else None,
        )
        first_scale = False

        ax_main.fill_between(
            [row.BinMin, row.BinMax],
            [row.sigma - row.stat_err] * 2,
            [row.sigma + row.stat_err] * 2,
            color="tab:orange",
            alpha=0.40,
            step="pre",
            label="stat. unc." if first_stat else None,
        )
        first_stat = False

    # grid points, NNLOJET prediction, HEPData
    ax_main.scatter(env_df["BinCenter"], env_df["sigma"],
                    s=20, color="tab:blue", zorder=3, label="Grid central")

    ax_main.errorbar(env_df["BinCenter"], env_df["sigma"],
                     xerr=env_df["width"] * 0.5, yerr=None,
                     ecolor="tab:blue", fmt="none", alpha=.7)

    ax_main.errorbar(var_df["BinCenter"], var_df["cs"], yerr=var_df["cs_err"],
                     fmt="s", color="red", ms=4, label="NNLOJET prediction")

    hep_x, hep_y, hep_err = hep_tuple
    ax_main.errorbar(hep_x, hep_y, yerr=hep_err,
                     fmt="^", color="k", ms=5, label="HEP data")

    ax_main.set_ylabel(Y_LABEL[observable])
    ax_main.grid(True, ls="--", alpha=.35)
    if show_legend:
        ax_main.legend(frameon=False, fontsize=8, loc="upper right")

    # --- RATIO 1 : Grid / NNLOJET ------------------------------------------ #
    merged1 = env_df.merge(
        var_df[["BinCenter", "cs", "cs_err"]], on="BinCenter", how="inner"
    )
    r1_y, r1_err = ratio_and_err(
        merged1["sigma"], merged1["stat_err"], merged1["cs"], merged1["cs_err"]
    )
    ax_r1.errorbar(merged1["BinCenter"], r1_y, yerr=r1_err,
                   fmt="s", color="red", ms=4)
    ax_r1.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_r1.set_ylabel("Grid/\nNNLOJET")
    ax_r1.grid(True, ls="--", alpha=.35)
    plt.setp(ax_r1.get_xticklabels(), visible=False)

    # --- RATIO 2 : Grid / HEPData ------------------------------------------ #
    hep_df = pd.DataFrame({
        "BinCenter": hep_x,
        "hep_sig":   hep_y,
        "hep_err":   0.5 * (hep_err[0] + hep_err[1]),   # symmetrise
    })
    merged2 = env_df.merge(hep_df, on="BinCenter", how="inner")
    r2_y, r2_err = ratio_and_err(
        merged2["sigma"], merged2["stat_err"], merged2["hep_sig"], merged2["hep_err"]
    )
    ax_r2.errorbar(merged2["BinCenter"], r2_y, yerr=r2_err,
                   fmt="^", color="k", ms=5)
    ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_r2.set_ylabel("Grid/\nHEPData")
    ax_r2.set_xlabel(X_LABEL[observable])
    ax_r2.grid(True, ls="--", alpha=.35)

# --------------------------------------------------------------------------- #
#  One figure per (order, observable)                                         #
# --------------------------------------------------------------------------- #
def plot_single_order(
    order: str,
    observable: str,
    nx_nodes: str,
    grid_out: Path,
    var_dat: Path,
    hep_tuple: Tuple[pd.Series, pd.Series, List[pd.Series]],
    out_name: Path,
) -> None:
    """Produce the classic three‑panel figure for a single order."""
    df_grid = parse_grid_out(grid_out)
    env_df  = envelope(df_grid, order)

    var_df = pd.read_csv(
        var_dat, sep=r"\s+", comment="#", header=None,
        names=["BinMin", "BinCenter", "BinMax",
               "cs", "cs_err",
               "tot02", "tot02_err", "tot03", "tot03_err"],
    )
    env_df["stat_err"] = env_df["BinCenter"].map(
        var_df.set_index("BinCenter")["cs_err"]
    )

    fig = plt.figure(figsize=(8, 8))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_r1   = fig.add_subplot(gs[1], sharex=ax_main)
    ax_r2   = fig.add_subplot(gs[2], sharex=ax_main)

    _make_one_column(
        ax_main, ax_r1, ax_r2,
        order=order,
        observable=observable,
        env_df=env_df,
        var_df=var_df,
        hep_tuple=hep_tuple,
        show_legend=True,           # legend ON in single‑order plots
    )

    ax_main.set_title(f"{order} closure test  -  {observable}  ({nx_nodes} x-nodes)")
    fig.tight_layout()
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_name} written")

# --------------------------------------------------------------------------- #
#  Combined canvas: LO | NLO | NNLO                                           #
# --------------------------------------------------------------------------- #
def plot_combined(
    observable: str,
    orders: List[str],
    nx_nodes: str,
    hep_dict: Dict[str, Tuple[pd.Series, pd.Series, List[pd.Series]]],
    width: int = 8,
    height: int = 8,
) -> None:
    """Make one wide canvas per observable with three columns."""
    fig = plt.figure(figsize=(width * 3, height))
    gs  = gridspec.GridSpec(
        3, 3,
        width_ratios=[1, 1, 1],
        height_ratios=[3, 1, 1],
        wspace=0.03,
        hspace=0.04,
    )

    # collect y‑limits so each row shares a common scale
    main_min = r1_min = r2_min = np.inf
    main_max = r1_max = r2_max = -np.inf
    axes_main, axes_r1, axes_r2 = [], [], []

    for col, order in enumerate(orders):
        # ---------- data for this column ----------------------------------- #
        grid_out = Path(f"NNLO.{observable}.tab.gz.txt.out")
        var_dat  = Path(f"{order}.{observable}_var.dat")

        df_grid = parse_grid_out(grid_out)
        env_df  = envelope(df_grid, order)

        var_df = pd.read_csv(
            var_dat, sep=r"\s+", comment="#", header=None,
            names=["BinMin", "BinCenter", "BinMax",
                   "cs", "cs_err",
                   "tot02", "tot02_err", "tot03", "tot03_err"],
        )
        env_df["stat_err"] = env_df["BinCenter"].map(
            var_df.set_index("BinCenter")["cs_err"]
        )

        # ---------- axes ---------------------------------------------------- #
        ax_main = fig.add_subplot(gs[0, col])
        ax_r1   = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_r2   = fig.add_subplot(gs[2, col], sharex=ax_main)

        axes_main.append(ax_main)
        axes_r1.append(ax_r1)
        axes_r2.append(ax_r2)

        _make_one_column(
            ax_main, ax_r1, ax_r2,
            order=order,
            observable=observable,
            env_df=env_df,
            var_df=var_df,
            hep_tuple=hep_dict[observable],
            show_legend=(order == "NNLO"),   # legend ONLY on NNLO column
        )

        ax_main.set_title(order, fontsize=12, pad=8, fontweight="bold")

        # remove y‑labels/ticks from NLO & NNLO columns (col > 0)
        if col > 0:
            for a in (ax_main, ax_r1, ax_r2):
                a.set_ylabel("")
                a.tick_params(labelleft=False)

        # track y‑ranges
        main_lo, main_hi = ax_main.get_ylim()
        r1_lo,   r1_hi   = ax_r1.get_ylim()
        r2_lo,   r2_hi   = ax_r2.get_ylim()

        main_min, main_max = min(main_min, main_lo), max(main_max, main_hi)
        r1_min,   r1_max   = min(r1_min,   r1_lo),   max(r1_max,   r1_hi)
        r2_min,   r2_max   = min(r2_min,   r2_lo),   max(r2_max,   r2_hi)

    # ---------- enforce identical y‑ranges row‑by‑row ----------------------- #
    def _apply_limits(ax_list, lo, hi):
        pad = 0.05 * (hi - lo)
        for a in ax_list:
            a.set_ylim(lo - pad, hi + pad)

    _apply_limits(axes_main, main_min, main_max)
    _apply_limits(axes_r1,   r1_min,   r1_max)
    _apply_limits(axes_r2,   r2_min,   r2_max)

    # ---------- global title & save ---------------------------------------- #
    fig.suptitle(
        f"{observable}  –  {nx_nodes} x-nodes  (muR= muF = mW)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_file = Path(f"figure_{observable}_combined.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_file} written")

# --------------------------------------------------------------------------- #
#  Main driver                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    OBSERVABLES = ["ptj1", "ptw", "abs_yj1", "ht_full"]
    ORDERS      = ["LO", "NLO", "NNLO"]
    NX_NODES    = "25"

    # --- HEPData cached once per observable -------------------------------- #
    HEP: Dict[str, Tuple[pd.Series, pd.Series, List[pd.Series]]] = {
        "ptj1":    read_hepdata("data/HEPData_WpJ_ptj1.csv"),
        "abs_yj1": read_hepdata("data/HEPData_WpJ_abs_yj1.csv"),
        "ht_full": read_hepdata("data/HEPData_WpJ_HT.csv"),
        "ptw":     read_hepdata("data/HEPData_WpJ_ptw.csv"),
    }

    # --- (1) individual (order, observable) figures ------------------------ #
    for order in ORDERS:
        for obs in OBSERVABLES:
            plot_single_order(
                order,
                obs,
                NX_NODES,
                grid_out=Path(f"NNLO.{obs}.tab.gz.txt.out"),
                var_dat=Path(f"{order}.{obs}_var.dat"),
                hep_tuple=HEP[obs],
                out_name=Path(f"figure_{order}_{obs}.png"),
            )
    print("\nAll per-order plots done.\n")

    # --- (2) combined canvases -------------------------------------------- #
    for obs in OBSERVABLES:
        plot_combined(obs, ORDERS, NX_NODES, HEP)

    print("All combined overview canvases done.")
