#!/usr/bin/env python3
"""
Grid-closure plots for W + jet at √s = 8 TeV (ATLAS).
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

ROUND_DECIMALS = 6              # decimals used when matching bin centres

# --------------------------------------------------------------------------- #
#  Order bookkeeping                                                          #
# --------------------------------------------------------------------------- #
BASE_ORDERS    = ["LO", "NLO", "NNLO"]
CONTRIB_ORDERS = {"R", "V", "RRa", "RRb", "RV", "VV"}

# --------------------------------------------------------------------------- #
#  File‑parsing helpers                                                       #
# --------------------------------------------------------------------------- #

def read_hepdata(csv_file: Path) -> Tuple[pd.Series, pd.Series, List[pd.Series]]:
    """
    Return bin centres, cross-sections and ±errors from a HEPData CSV.

    If the file (or the whole HEPdata/ directory) is missing, return three
    *empty* Series objects.  Down-stream code can simply test
    `if not centres.empty:` (or `if centres.size:`) to decide whether to draw
    the HEP-data layers and the Grid/HEP ratio panel.
    """
    csv_file = Path(csv_file)
    if not csv_file.is_file():
        empty = pd.Series(dtype=float)
        return empty, empty, [empty.copy(), empty.copy()]

    df       = pd.read_csv(csv_file)
    centres  = pd.to_numeric(df.iloc[:, 0])
    sigma    = pd.to_numeric(df.iloc[:, 3])
    err_up   = np.abs(pd.to_numeric(df.iloc[:, 4]))
    err_dn   = np.abs(pd.to_numeric(df.iloc[:, 5]))
    return centres, sigma, [err_dn, err_up]


def parse_grid_out(path: Path, order: str) -> pd.DataFrame:
    """Parse grid *.out → dataframe of all bins x scale points.
    * Contribution tables (single number per bin) are handled when *order*
      is one of {R,V,RRa,RRb,RV,VV}.  Cross-section columns that are not
      present are left as NaN so that the rest of the code can reference
      them safely.
    """
    rows = []
    xmur = xmuf = None

    pat_scale = re.compile(r"xmur, xmuf chosen here are:\s*([0-9.]+),\s*([0-9.]+)")
    pat_data  = re.compile(r"^\s*\d+\s")

    with open(path) as handle:
        for line in handle:
            if (m := pat_scale.search(line)):
                xmur, xmuf = map(float, m.groups())
                continue
            if not pat_data.match(line):
                continue

            p = line.split()
            bin_min, bin_max = map(float, (p[3], p[4]))

            row = dict(
                BinCenter=0.5 * (bin_min + bin_max),
                BinMin=bin_min,
                BinMax=bin_max,
                xmur=xmur,
                xmuf=xmuf,
            )

            if len(p) >= 9:                       # classic layout  (LO NLO NNLO)
                row.update(LO=float(p[6]), NLO=float(p[7]), NNLO=float(p[8]))
            else:                                 # contribution layout (single sigma)
                row[order] = float(p[6])

            rows.append(row)

    return pd.DataFrame(rows)

def read_nnlojet_dat (nnlojet_dat: Path, unitsfactor_grid_nnlojet: float = 1.) -> pd.DataFrame:
    var_df = pd.read_csv(
        nnlojet_dat, sep=r"\s+", comment="#", header=None,
        names=["BinMin", "BinCenter", "BinMax",
               "cs", "cs_err",
               "tot02", "tot02_err", "tot03", "tot03_err"],
    )
    # Scale all cross-section and uncertainty columns
    cols_to_scale = ["cs", "cs_err", "tot02", "tot02_err", "tot03", "tot03_err"]
    var_df[cols_to_scale] = var_df[cols_to_scale] / unitsfactor_grid_nnlojet
    return var_df


# --------------------------------------------------------------------------- #
#  Maths & helpers                                                            #
# --------------------------------------------------------------------------- #

def ratio_and_err(a: pd.Series, a_err: pd.Series,
                  b: pd.Series, b_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Ratio a/b with propagated symmetric error."""
    ratio  = a / b
    rel_sq = (a_err / a) ** 2 + (b_err / b) ** 2
    return ratio, abs(ratio) * np.sqrt(rel_sq)


def envelope(df: pd.DataFrame, order: str, central_scale_factor: float) -> pd.DataFrame:
    """Return dataframe with central sigma and scale envelope ±errors for <order>."""
    central = df[(df.xmur == central_scale_factor) & (df.xmuf == central_scale_factor)]
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
    return merged.reset_index()


def _rounded(df: pd.DataFrame, col: str = "BinCenter") -> pd.DataFrame:
    """
    Return a copy of *df* with an auxiliary column ``__bc__`` that contains the
    values in *col* rounded to *ROUND_DECIMALS*.  Keeps the original data intact.
    """
    out = df.copy()
    out["__bc__"] = out[col].round(ROUND_DECIMALS)
    return out


def attach_stat_err(env_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'stat_err' column to *env_df*, matching rows on *rounded* BinCenter.
    Rows that find no match get a statistical error of 0 (should not happen)
    """
    env_r = _rounded(env_df)          # adds __bc__
    var_r = _rounded(var_df)

    # build a mapping (rounded bin centre → cs_err)
    stat_map = var_r.set_index("__bc__")["cs_err"]

    out = env_r.copy()
    out["stat_err"] = out["__bc__"].map(stat_map).fillna(0.0)

    return out.drop(columns="__bc__")  # keep the returned frame clean


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
    has_hep = not hep_tuple[0].empty        # check if hep data exist

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

    # HEPData points (skip for individual contributions)
    if has_hep and order not in CONTRIB_ORDERS:
        hep_x, hep_y, hep_err = hep_tuple
        ax_main.errorbar(hep_x, hep_y, yerr=hep_err,
                         fmt="^", color="k", ms=5, label="HEP data")

    ax_main.set_ylabel(Y_LABEL[observable])
    ax_main.grid(True, ls="--", alpha=.35)
    if show_legend:
        ax_main.legend(frameon=False, fontsize=8)

    # ---------- Robust merge (rounded centres) ---------------------------- #
    env_r = _rounded(env_df)
    var_r = _rounded(var_df)
    merged1 = env_r.merge(var_r[["__bc__", "cs", "cs_err"]], on="__bc__", how="inner")

    # ---------- RATIO 1 : Grid / NNLOJET ---------------------------------- #
    r1_y, r1_err = ratio_and_err(
        merged1["sigma"], merged1["stat_err"], merged1["cs"], merged1["cs_err"]
    )
    ax_r1.errorbar(merged1["BinCenter"], r1_y, yerr=r1_err,
                   fmt="s", color="red", ms=4)
    ax_r1.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_r1.set_ylabel("Grid/\nNNLOJET")
    ax_r1.grid(True, ls="--", alpha=.35)
    ax_r1.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax_r1.yaxis.get_offset_text().set_visible(False)    # hide "×10^n"
    plt.setp(ax_r1.get_xticklabels(), visible=False)

    # ---------- RATIO 2 : Grid / HEPData (omit panel for contributions) ---- #
    if has_hep and order not in CONTRIB_ORDERS:
        hep_x, hep_y, hep_err = hep_tuple
        hep_df = pd.DataFrame({
            "BinCenter": hep_x,
            "hep_sig":   hep_y,
            "hep_err":   0.5 * (hep_err[0] + hep_err[1]),   # symmetrise
        })
        hep_r = _rounded(hep_df)
        merged2 = env_r.merge(hep_r[["__bc__", "hep_sig", "hep_err"]],
                              on="__bc__", how="inner")

        r2_y, r2_err = ratio_and_err(
            merged2["sigma"], merged2["stat_err"],
            merged2["hep_sig"], merged2["hep_err"]
        )
        ax_r2.errorbar(merged2["BinCenter"], r2_y, yerr=r2_err,
                       fmt="^", color="k", ms=5)
        ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_r2.set_ylabel("Grid/HEPData")
        ax_r2.set_xlabel(X_LABEL[observable])
        ax_r2.grid(True, ls="--", alpha=.35)
        ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax_r2.yaxis.get_offset_text().set_visible(False)    # hide "×10^n"
        plt.setp(ax_r1.get_xticklabels(), visible=False)
    else:
        # Hide the unused bottom axis and move the x-label to the middle ratio panel
        ax_r2.set_visible(False)
        ax_r1.set_xlabel(X_LABEL[observable])

        # Explicitly show x-axis ticks and labels
        ax_r1.tick_params(axis='x', which='both', labelbottom=True)



# --------------------------------------------------------------------------- #
#  One figure per (order, observable)                                         #
# --------------------------------------------------------------------------- #

def plot_single_order(
    unitsfactor_grid_nnlojet: float,
    central_scale_factor: float,
    order: str,
    observable: str,
    grid_specs: str,
    grid_out: Path,
    nnlojet_dat: Path,
    hep_tuple: Tuple[pd.Series, pd.Series, List[pd.Series]],
    out_name: Path,
) -> None:
    """Produce the classic three-panel figure for a single order."""
    df_grid = parse_grid_out(grid_out, order)
    env_df  = envelope(df_grid, order, central_scale_factor)
    var_df = read_nnlojet_dat(nnlojet_dat, unitsfactor_grid_nnlojet)

    env_df  = attach_stat_err(env_df, var_df)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05, figure=fig)
    ax_main = fig.add_subplot(gs[0])
    #if order in BASE_ORDERS:
    #    ax_main.set_yscale("log")
    plt.setp(ax_main.get_xticklabels(), visible=False)
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

    ax_main.set_title(f"{order} closure test  -  {observable}. {grid_specs}")
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_name} written")


# --------------------------------------------------------------------------- #
#  Combined canvas: LO | NLO | NNLO                                           #
# --------------------------------------------------------------------------- #

def plot_combined(
    unitsfactor_grid_nnlojet: float,
    central_scale_factor: float,
    orders: List[str],
    observable: str,
    grid_specs: str,
    grid_out: Path,
    nnlojet_dat_folder: str,
    hep_tuple: Tuple[pd.Series, pd.Series, List[pd.Series]],
    out_name: Path,
    width: int = 8,
    height: int = 8,
) -> None:
    """Make one wide canvas per observable with three columns."""
    fig = plt.figure(figsize=(width * 3, height), constrained_layout=True)
    gs  = gridspec.GridSpec(
        3, 3,
        width_ratios=[1, 1, 1],
        height_ratios=[3, 1, 1],
        wspace=0.03,
        hspace=0.04,
        figure=fig,
    )

    # collect y‑limits so each row shares a common scale
    main_min = r1_min = r2_min = np.inf
    main_max = r1_max = r2_max = -np.inf
    axes_main, axes_r1, axes_r2 = [], [], []

    for col, order in enumerate(orders):

        nnlojet_dat=Path(f"{nnlojet_dat_folder}{order}.{observable}_var.dat")

        # ---------- data for this column ----------------------------------- #
        df_grid = parse_grid_out(grid_out, order)
        env_df  = envelope(df_grid, order, central_scale_factor)
        var_df = read_nnlojet_dat(nnlojet_dat, unitsfactor_grid_nnlojet)

        env_df = attach_stat_err(env_df, var_df)

        # ---------- axes ---------------------------------------------------- #
        ax_main = fig.add_subplot(gs[0, col])
        ax_main.set_yscale("log")
        plt.setp(ax_main.get_xticklabels(), visible=False)
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
            hep_tuple=hep_tuple,
            show_legend=(order == "NNLO"),   # legend ONLY on NNLO column
        )

        ax_main.set_title(order, fontsize=12, pad=8, fontweight="bold")

        if col > 0:   # remove y‑labels/ticks from NLO & NNLO columns
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

    # ---------- enforce identical y‑ranges row‑by‑row --------------------- #
    def _apply_limits(ax_list, lo, hi):
        for a in ax_list:
            if a.get_yscale() == "log":
                # avoid non-positive lower limit on log axis
                lo_pos = max(lo, 1e-6)
                a.set_ylim(lo_pos, hi * 1.1)
            else:
                pad = 0.05 * (hi - lo)
                a.set_ylim(lo - pad, hi + pad)


    _apply_limits(axes_main, main_min, main_max)
    _apply_limits(axes_r1,   r1_min,   r1_max)
    _apply_limits(axes_r2,   r2_min,   r2_max)

    # ---------- global title & save -------------------------------------- #
    fig.suptitle(
        f"{grid_specs}",
        fontsize=14,
        x=0.1,
        y=0.995,
    )

    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_name} written")


# --------------------------------------------------------------------------- #
#  Main driver                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    PLOT_DIR = Path("Analysis_plots")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    OBSERVABLES = ["ptj1", "ptw", "abs_yj1", "ht_full"]
    ORDERS      = ["LO", "NLO", "NNLO", "R", "V", "RRa", "RRb", "RV", "VV"]
    grid_specs    = "25"
    grid_specs = f"{grid_specs} x-nodes, μ$_R$= μ$_F$ = m$_W$"
    cs_units = "fb"
    central_scale_factor = 1.0

    '''
    Factor to take into account the difference in units between grid production
    and nnlojet .dat files.
    Example: If grid in pb, unitsfactor_grid_nnlojet = 10e3 (nnlojet .dat is always in fb)
    '''
    unitsfactor_grid_nnlojet = 1.


    # ---------- Axis‑label dictionaries --------------------------------- #
    X_LABEL: Dict[str, str] = {
        "ptj1":     fr"$p_T^{{j1}}$  [GeV]",
        "ptw":      fr"$p_T^{{W}}$   [GeV]",
        "abs_yj1": fr"$|y^{{j1}}|$",
        "abs_etaj1": fr"$|\eta^{{j1}}|$",
        "ht_full":  fr"$H_T$ [GeV]",
        "ht_jets":  fr"$H_T^{{jets}}$ [GeV]",
    }

    Y_LABEL: Dict[str, str] = {
        "ptj1":    fr"$\mathrm{{d}}\sigma/\mathrm{{d}}p_T^{{\,j1}}$  [{cs_units}/GeV]",
        "ptw":     fr"$\mathrm{{d}}\sigma/\mathrm{{d}}p_T^{{\,W}}$  [{cs_units}/GeV]",
        "abs_yj1": fr"$\mathrm{{d}}\sigma/\mathrm{{d}}|y^{{\,j1}}|$  [{cs_units}]",
        "abs_etaj1": fr"$\mathrm{{d}}\sigma/\mathrm{{d}}|\eta^{{\,j1}}|$  [{cs_units}]",
        "ht_full": fr"$\mathrm{{d}}\sigma/\mathrm{{d}}H_T$  [{cs_units}/GeV]",
        "ht_jets": fr"$\mathrm{{d}}\sigma/\mathrm{{d}}H_T^{{\,jets}}$  [{cs_units}/GeV]",

    }

    # --- HEPData cached once per observable -------------------------------- #
    HEP: Dict[str, Tuple[pd.Series, pd.Series, List[pd.Series]]] = {
        "ptj1":    read_hepdata("HEPdata/HEPData_WpJ_ptj1.csv"),
        "abs_yj1": read_hepdata("HEPdata/HEPData_WpJ_abs_yj1.csv"),
        "abs_etaj1": read_hepdata("HEPdata/HEPData_WpJ_abs_etaj1.csv"),
        "ht_full": read_hepdata("HEPdata/HEPData_WpJ_HT.csv"),
        "ht_jets": read_hepdata("HEPdata/HEPData_WpJ_HTjets.csv"),
        "ptw":     read_hepdata("HEPdata/HEPData_WpJ_ptw.csv"),
    }

    nnlojet_dat_folder = "combined/Final/"
    grid_folder = "combine_grid/"

    # --- (1) individual (order, observable) figures ------------------------ #
    for order in ORDERS:
        for obs in OBSERVABLES:
            # choose input file name depending on order type
            grid_file = (
                Path(f"{grid_folder}{order}.{obs}.tab.gz.txt.out")
                if order in CONTRIB_ORDERS else
                Path(f"{grid_folder}NNLO.{obs}.tab.gz.txt.out")
            )

            plot_single_order(
                unitsfactor_grid_nnlojet,
                central_scale_factor,
                order,
                obs,
                grid_specs,
                grid_out=grid_file,
                nnlojet_dat=Path(f"{nnlojet_dat_folder}{order}.{obs}_var.dat"),
                hep_tuple=HEP[obs],
                out_name=Path(f"{PLOT_DIR}/figure_{order}_{obs}.png"),
            )
    print("\nAll per-order plots done.\n")

    # --- (2) combined canvases -------------------------------------------- #
    if set(BASE_ORDERS).issubset(set(ORDERS)):
        for obs in OBSERVABLES:
            plot_combined(
                unitsfactor_grid_nnlojet,
                central_scale_factor,
                BASE_ORDERS,
                obs,
                grid_specs,
                grid_out=Path(f"{grid_folder}NNLO.{obs}.tab.gz.txt.out"),
                nnlojet_dat_folder = nnlojet_dat_folder,
                hep_tuple=HEP[obs],
                out_name =Path(f"{PLOT_DIR}/figure_{obs}_combined.png"),
            )
        print("All combined overview canvases done.")
    else:
        print("Skipped combined canvasses (some contribution missing).")