#!/usr/bin/env python3
"""
Grid-closure plots for W + jet at √s = 8 TeV (ATLAS).
"""
# --------------------------------------------------------------------------- #
#  Imports & matplotlib style                                                 #
# --------------------------------------------------------------------------- #

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


'''
# ─────────────────────────────────────────────────────────────────────────────
# Global “cosmetic” settings for all figures
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    # base font sizes
    "font.size":        12,    # controls default text size
    "axes.titlesize":   14,    # subplot titles
    "axes.labelsize":   14,    # x- and y-axis labels
    "xtick.labelsize":  14,    # tick labels
    "ytick.labelsize":  14,
    "legend.fontsize":  12,    # legend text

    # line widths
    "axes.linewidth":     1.2,  # axis spine thickness
    "grid.linewidth":     0.6,  # grid-line thickness

    # tick properties
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  6,    # length of major ticks
    "ytick.major.size":  6,
    "xtick.major.width": 1.0,  # thickness of tick lines
    "ytick.major.width": 1.0,

    # legend frame
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "black",
    "legend.fontsize":   14,    # controls text size in all legends
    "legend.title_fontsize": 15, # if you ever use legend titles
    "legend.handlelength": 2.5,  # length of the little line/marker swatch
    "legend.handletextpad": 0.8, # space between swatch and text

    # grid style
    "grid.linestyle":    "--",
    "grid.alpha":        0.3,

})
# ─────────────────────────────────────────────────────────────────────────────
'''

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
    """Return bin centres, cross-sections and ±errors from a HEPData CSV."""
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
    """Parse grid *.out → dataframe of all bins × scale points."""
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


def read_nnlojet_dat(nnlojet_dat: Path, unitsfactor_grid_nnlojet: float = 1.) -> pd.DataFrame:
    var_df = pd.read_csv(
        nnlojet_dat, sep=r"\s+", comment="#", header=None,
        names=["BinMin", "BinCenter", "BinMax",
               "cs", "cs_err",
               "tot02", "tot02_err", "tot03", "tot03_err"],
    )
    cols_to_scale = ["cs", "cs_err", "tot02", "tot02_err", "tot03", "tot03_err"]
    var_df[cols_to_scale] = var_df[cols_to_scale] / unitsfactor_grid_nnlojet
    return var_df

# --------------------------------------------------------------------------- #
#  Maths & helpers                                                            #
# --------------------------------------------------------------------------- #

def ratio_and_err(a: pd.Series, a_err: pd.Series,
                  b: pd.Series, b_err: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ratio  = a / b
    rel_sq = (a_err / a) ** 2 + (b_err / b) ** 2
    return ratio, abs(ratio) * np.sqrt(rel_sq)


def envelope(df: pd.DataFrame, order: str, central_scale_factor: float) -> pd.DataFrame:
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
    #print(order, merged)
    return merged.reset_index()


def _rounded(df: pd.DataFrame, col: str = "BinCenter") -> pd.DataFrame:
    out = df.copy()
    out["__bc__"] = out[col].round(ROUND_DECIMALS)
    return out


def attach_stat_err(env_df: pd.DataFrame, var_df: pd.DataFrame) -> pd.DataFrame:
    env_r = _rounded(env_df)
    var_r = _rounded(var_df)
    stat_map = var_r.set_index("__bc__")["cs_err"]
    out = env_r.copy()
    out["stat_err"] = out["__bc__"].map(stat_map).fillna(0.0)
    return out.drop(columns="__bc__")


def _ratio_scale_band_edges(num: pd.Series, num_err_dn: pd.Series, num_err_up: pd.Series,
                      den: pd.Series, den_err_dn: pd.Series, den_err_up: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return *lo*, *hi* series for the ratio (num±err)/(den±err)."""
    lo = (num - num_err_dn) / (den + den_err_up)
    hi = (num + num_err_up) / (den - den_err_dn)
    #print(num/den, lo, hi)
    return lo, hi

# ------------------------------------------------------------------------ #
#  k-factor panel                                                          #
# ------------------------------------------------------------------------ #
def plot_scale_ratio_panel(ax,
                           env_num: pd.DataFrame,
                           env_den: pd.DataFrame,
                           *,
                           color: str,
                           label: str) -> None:
    """
    Draw <env_num>/<env_den> with
      • scale band per bin
      • statistical uncertainty bar
    """
    if env_num.empty or env_den.empty:
        ax.set_visible(False)
        return

    # ---------- merge envelopes ---------------------------------------- #
    num_r = _rounded(env_num)
    den_r = _rounded(env_den)
    keep  = ["__bc__", "BinCenter", "BinMin", "BinMax",
             "sigma", "err_dn", "err_up", "stat_err"]

    num_m = num_r[keep].rename(columns={
        "sigma": "num_sig", "err_dn": "num_dn",
        "err_up": "num_up", "stat_err": "num_stat"})
    den_m = den_r[keep].rename(columns={
        "sigma": "den_sig", "err_dn": "den_dn",
        "err_up": "den_up", "stat_err": "den_stat"})

    m = num_m.merge(den_m, on=["__bc__", "BinCenter", "BinMin", "BinMax"],
                    how="inner")

    # ---------- central ratio & errors --------------------------------- #
    ratio, ratio_stat = ratio_and_err(
        m["num_sig"], m["num_stat"], m["den_sig"], m["den_stat"])

    lo, hi = _ratio_scale_band_edges(
        m["num_sig"], m["num_dn"], m["num_up"],
        m["den_sig"], m["den_dn"], m["den_up"])

    # ---------- draw ---------------------------------------------------- #
    first = True
    for i, row in m.iterrows():
        ax.fill_between([row.BinMin, row.BinMax],
                        [lo[i]] * 2, [hi[i]] * 2,
                        color=color, alpha=0.30, step="pre",
                        label= "scale unc." if first else None)
        first = False

    ax.scatter(m["BinCenter"], ratio, s=20, color=color)
    ax.errorbar(m["BinCenter"], ratio, yerr=ratio_stat,
                fmt="none", ecolor=color, alpha=.6,
                label="stat. unc.")                      

    # ---------- cosmetics ---------------------------------------------- #
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(label)
    ax.grid(True, ls="--", alpha=.35)
    ax.legend(frameon=False, fontsize=8)

# --------------------------------------------------------------------------- #
#  Column‑drawing helper (one order)                                          #
# --------------------------------------------------------------------------- #

def _make_one_column(
    ax_main, ax_r1, ax_r2, *,
    order: str,
    observable: str,
    env_df: pd.DataFrame,
    var_df: pd.DataFrame,
    hep_tuple: Tuple[pd.Series, pd.Series, List[pd.Series]],
    show_legend: bool,
    ratio2_mode: str = "hepdata",        # "hepdata" | "zoom" | "none"
    show_ratio2_xlabel: bool = True,
    plot_nnlojet_main: bool = True,      #  whether to plot NNLOJET in main
    plot_ratio1: bool = True,            #  whether to plot Grid/NNLOJET ratio
) -> None:
    """
    Draw main distribution + two ratio panels.

    • ax_r1   - Grid/NNLOJET
    • ax_r2   - mode-dependent
    """

    # ---------------- MAIN PANEL --------------------------- #
    has_hep = not hep_tuple[0].empty
    first_scale, first_stat = True, True
    for _, row in env_df.iterrows():
        ax_main.fill_between([row.BinMin, row.BinMax],
                             [row.sigma - row.err_dn] * 2,
                             [row.sigma + row.err_up] * 2,
                             color="tab:blue", alpha=0.25, step="pre",
                             label="scale unc." if first_scale else None)
        first_scale = False
        ax_main.fill_between([row.BinMin, row.BinMax],
                             [row.sigma - row.stat_err] * 2,
                             [row.sigma + row.stat_err] * 2,
                             color="tab:orange", alpha=0.40, step="pre",
                             label="stat. unc." if first_stat else None)
        first_stat = False

    ax_main.scatter(env_df["BinCenter"], env_df["sigma"],
                    s=20, color="tab:blue", zorder=3, label="Grid central")
    ax_main.errorbar(env_df["BinCenter"], env_df["sigma"],
                     xerr=env_df["width"] * .5, yerr=None,
                     ecolor="tab:blue", fmt="none", alpha=.7)

    if plot_nnlojet_main:
        ax_main.errorbar(var_df["BinCenter"], var_df["cs"], yerr=var_df["cs_err"],
                         fmt="s", color="red", ms=4, label="NNLOJET prediction")

    if has_hep and order not in CONTRIB_ORDERS:
        hep_x, hep_y, hep_err = hep_tuple
        ax_main.errorbar(hep_x, hep_y, yerr=hep_err,
                         fmt="^", color="k", ms=5, label="HEP data")

    ax_main.set_ylabel(Y_LABEL[observable])
    ax_main.grid(True, ls="--", alpha=.35)
    if show_legend:
        ax_main.legend(frameon=False, fontsize=8)

    env_r = _rounded(env_df)
    var_r = _rounded(var_df)        

    # ---------------- RATIO 1 : Grid / NNLOJET ------------------------- #
    if not plot_ratio1:
        ax_r1.set_visible(False)
    else:
        m1 = env_r.merge(var_r[["__bc__", "cs", "cs_err"]], on="__bc__", how="inner")
        r1_y, r1_err = ratio_and_err(m1["sigma"], m1["stat_err"],
                                     m1["cs"],    m1["cs_err"])

        ax_r1.errorbar(m1["BinCenter"], r1_y, yerr=r1_err,
                       fmt="s", color="red", ms=4, label="stat. unc.")
        ax_r1.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_r1.set_ylabel("Grid/\nNNLOJET")
        ax_r1.grid(True, ls="--", alpha=.35)
        ax_r1.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax_r1.yaxis.get_offset_text().set_visible(False)
        plt.setp(ax_r1.get_xticklabels(), visible=False)

    # ---------------- RATIO 2 : mode‑dependent ------------------------- #
    if ratio2_mode == "none":
        ax_r2.set_visible(False)

    elif ratio2_mode == "zoom":
        # ... zoom code ...
        ax_r2.errorbar(m1["BinCenter"], r1_y, yerr=r1_err,
                       fmt="s", color="red", ms=4, label="stat. unc.")
        ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_r2.set_ylabel("Grid/NNLOJET (‰)")
        ax_r2.set_ylim(1 - ZOOM_RATIO_LIMIT, 1 + ZOOM_RATIO_LIMIT)
        if show_ratio2_xlabel:
            ax_r2.set_xlabel(X_LABEL[observable])
        ax_r2.grid(True, ls="--", alpha=.35)
        ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax_r2.yaxis.get_offset_text().set_visible(False)
        if not show_ratio2_xlabel:
            ax_r2.tick_params(axis='x', which='both', labelbottom=False)

    else:  # "hepdata"
        if has_hep and order not in CONTRIB_ORDERS:
            hep_x, hep_y, hep_err = hep_tuple
            hep_df = pd.DataFrame({
                "BinCenter": hep_x,
                "hep_sig":   hep_y,
                "hep_err":   0.5 * (hep_err[0] + hep_err[1]),
            })
            hep_r = _rounded(hep_df)
            m2 = env_r.merge(hep_r[["__bc__", "hep_sig", "hep_err"]],
                             on="__bc__", how="inner")
            r2_y, r2_err = ratio_and_err(
                m2["sigma"], m2["stat_err"], m2["hep_sig"], m2["hep_err"])

            ax_r2.errorbar(m2["BinCenter"], r2_y, yerr=r2_err,
                           fmt="^", color="k", ms=5, label="stat. unc.")
            ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
            ax_r2.set_ylabel("Grid/HEPData")
            if show_ratio2_xlabel:
                ax_r2.set_xlabel(X_LABEL[observable])
            ax_r2.grid(True, ls="--", alpha=.35)
            ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax_r2.yaxis.get_offset_text().set_visible(False)
            if not show_ratio2_xlabel:
                ax_r2.tick_params(axis='x', which='both', labelbottom=False)
        else:
            ax_r2.set_visible(False)
            ax_r1.set_xlabel(X_LABEL[observable])
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
    """Produce the three-panel **closure-test** figure for a single order."""
    df_grid = parse_grid_out(grid_out, order)
    env_df  = envelope(df_grid, order, central_scale_factor)
    var_df  = read_nnlojet_dat(nnlojet_dat, unitsfactor_grid_nnlojet)
    env_df  = attach_stat_err(env_df, var_df)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05, figure=fig)

    ax_main = fig.add_subplot(gs[0])
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
        show_legend=True,
        ratio2_mode="zoom",          # zoom behaviour for closure tests
        # use default plot_nnlojet_main=True, plot_ratio1=True
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
    plot_closure: bool,
    width: int = 8,
    height: int = 8,
) -> None:
    """Make one wide canvas (4 rows x 3 columns) with per-column scale panels,
       but without Grid/NNLOJET ratio panels and without NNLOJET in main."""

    fig = plt.figure(figsize=(width * 3, height + 2), constrained_layout=True)
    
    if plot_closure:
        gs  = gridspec.GridSpec(
            4, 3,
            width_ratios=[1, 1, 1],
            height_ratios=[3, 1, 1, 1],   # extra bottom row
            wspace=0.03,
            hspace=0.02,
            figure=fig,
        )
    else:
        gs  = gridspec.GridSpec(
            4, 3,
            width_ratios=[1, 1, 1],
            height_ratios=[3, 0.1, 1, 1],   # extra bottom row
            wspace=0.03,
            hspace=0.02,
            figure=fig,
        )

    # ­collect y‑range info (optional – unchanged)
    main_min = r1_min = r2_min = np.inf
    main_max = r1_max = r2_max = -np.inf
    axes_main, axes_r1, axes_r2 = [], [], []
    env_store: Dict[str, pd.DataFrame] = {}
    axes_kfactor = []

    for col, order in enumerate(orders):

        nnlojet_dat = Path(f"{nnlojet_dat_folder}{order}.{observable}_var.dat")
        df_grid = parse_grid_out(grid_out, order)
        env_df  = envelope(df_grid, order, central_scale_factor)
        var_df  = read_nnlojet_dat(nnlojet_dat, unitsfactor_grid_nnlojet)
        env_df  = attach_stat_err(env_df, var_df)
        env_store[order] = env_df.copy()

        # -- three rows ------------------------------------------------- #
        ax_main = fig.add_subplot(gs[0, col])
        ax_main.set_yscale("log")
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_r1   = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_r2   = fig.add_subplot(gs[2, col], sharex=ax_main)

        axes_main.append(ax_main)
        axes_r1.append(ax_r1)
        axes_r2.append(ax_r2)

        # here we disable the NNLOJET in main, and disable ratio1
        _make_one_column(
            ax_main, ax_r1, ax_r2,
            order=order,
            observable=observable,
            env_df=env_df,
            var_df=var_df,
            hep_tuple=hep_tuple,
            show_legend=(order == "NNLO"),
            ratio2_mode="hepdata",
            show_ratio2_xlabel=(order == "LO"),
            plot_nnlojet_main=False,
            plot_ratio1=plot_closure,
        )

        ax_main.set_title(order, fontsize=12, pad=8, fontweight="bold")
        if col > 0:
            for a in (ax_main, ax_r1, ax_r2):
                a.set_ylabel("")
                a.tick_params(labelleft=False)

        # --- bottom‑row axis for this column --------------------------- #
        ax_kfactor = fig.add_subplot(gs[3, col], sharex=ax_main)
        axes_kfactor.append(ax_kfactor)

        if order == "NLO":
            plot_scale_ratio_panel(
                ax_kfactor,
                env_num=env_store["NLO"],
                env_den=env_store["LO"],
                color="tab:purple",
                label="kNLO",
            )
            ax_kfactor.set_xlabel(X_LABEL[observable])

        elif order == "NNLO":
            plot_scale_ratio_panel(
                ax_kfactor,
                env_num=env_store["NNLO"],
                env_den=env_store["NLO"],
                color="tab:red",
                label="kNNLO",
            )
            ax_kfactor.set_xlabel(X_LABEL[observable])

        else:
            ax_kfactor.set_visible(False)

        # track y‑ranges (unchanged)
        main_lo, main_hi = ax_main.get_ylim()
        r1_lo,   r1_hi   = ax_r1.get_ylim()
        r2_lo,   r2_hi   = ax_r2.get_ylim()

        main_min, main_max = min(main_min, main_lo), max(main_max, main_hi)
        r1_min,   r1_max   = min(r1_min,   r1_lo),   max(r1_max,   r1_hi)
        r2_min,   r2_max   = min(r2_min,   r2_lo),   max(r2_max,   r2_hi)

    # Enforce identical y‑ranges on shared rows (unchanged)
    def _apply_limits(ax_list, lo, hi):
        for a in ax_list:
            if a.get_yscale() == "log":
                lo_pos = max(lo, 1e-6)
                a.set_ylim(lo_pos, hi * 1.1)
            else:
                pad = 0.05 * (hi - lo)
                a.set_ylim(lo - pad, hi + pad)

    _apply_limits(axes_main, main_min, main_max)
    _apply_limits(axes_r1,   r1_min,   r1_max)
    _apply_limits(axes_r2,   r2_min,   r2_max)

    fig.suptitle(f"{grid_specs}", fontsize=14, x=0.1, y=0.995)
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_name} written")


def plot_relative_uncertainties(
    unitsfactor_grid_nnlojet: float,
    central_scale_factor: float,
    orders: List[str],
    observables: List[str],
    grid_specs: str,
    grid_folder: str,
    nnlojet_dat_folder: str,
    hep_tuple_map: Dict[str, Tuple[pd.Series, pd.Series, List[pd.Series]]],
    out_dir: Path,
) -> None:
    """
    For each order×observable, compute relative uncertainties:
      scale_rel = err_{scale}/sigma
      stat_rel  = stat_err/sigma

    And plot both as a band/error‑bar around 1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for order in orders:
        for obs in observables:
            # — load grid & envelope
            grid_file = (
                Path(f"{grid_folder}{order}.{obs}.tab.gz.txt.out")
                if order in CONTRIB_ORDERS else
                Path(f"{grid_folder}NNLO.{obs}.tab.gz.txt.out")
            )
            df_grid = parse_grid_out(grid_file, order)
            env_df  = envelope(df_grid, order, central_scale_factor)

            # — attach stat error from NNLOJET
            var_df  = read_nnlojet_dat(Path(f"{nnlojet_dat_folder}{order}.{obs}_var.dat"),
                                       unitsfactor_grid_nnlojet)
            env_df  = attach_stat_err(env_df, var_df)

            # — compute relative uncertainties
            env_df["scale_rel_up"]   = abs(env_df["err_up"]   / env_df["sigma"])
            env_df["scale_rel_down"] = abs(env_df["err_dn"]   / env_df["sigma"])
            env_df["stat_rel"]       = abs(env_df["stat_err"] / env_df["sigma"])

            # — plot
            fig, ax = plt.subplots(figsize=(6,4))

            # now fill each bin individually:
            first = True
            for _, row in env_df.iterrows():
                ax.fill_between(
                    [row.BinMin, row.BinMax],
                    [1.0 - row.scale_rel_down] * 2,
                    [1.0 + row.scale_rel_up] * 2,
                    step="pre",
                    color="tab:blue",
                    alpha=0.3,
                    label="scale unc. (rel.)" if first else None
                )
                first = False

            # stat errorbars around 1
            ax.errorbar(
                env_df["BinCenter"],
                [1.0]*len(env_df),
                yerr=env_df["stat_rel"],
                fmt="o", label="stat unc. (rel.)"
            )

            ax.axhline(1.0, color="k", ls="--", lw=0.8)
            ax.set_xlabel(X_LABEL[obs])
            ax.set_ylabel("Ratio to central (±unc.)")
            ax.set_title(f"{order} relative uncertainties  –  {obs}. {grid_specs}")
            ax.legend()
            ax.grid(True, ls="--", alpha=0.3)

            out_path = out_dir / f"rel_unc_{order}_{obs}.png"
            #plt.show()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  ✓ {out_path} written")



# --------------------------------------------------------------------------- #
#  Main driver                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PLOT_DIR = Path("Analysis_plots_2")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    OBSERVABLES = ["ptj1", "ptw", "abs_yj1", "ht_full"]
    ORDERS      = ["LO", "NLO", "NNLO", "R", "V", "RRa", "RRb", "RV", "VV"]
    grid_specs    = "25"
    central_scale_factor = 1.0
    grid_specs = f"{grid_specs} x-nodes, μ$_R$= μ$_F$ = {central_scale_factor} m$_W$"
    cs_units = "fb"
    unitsfactor_grid_nnlojet = 1.
    ZOOM_RATIO_LIMIT = 0.002        # ±0.2 % for the zoomed Grid/NNLOJET panel


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

    # --- (1) individual closure‑test figures ----------------------------- #
    for order in ORDERS:
        for obs in OBSERVABLES:
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

    # --- (2) combined overview canvases ---------------------------------- #
    if set(BASE_ORDERS).issubset(set(ORDERS)):
        for obs in OBSERVABLES:
            plot_combined(
                unitsfactor_grid_nnlojet,
                central_scale_factor,
                BASE_ORDERS,
                obs,
                grid_specs,
                grid_out=Path(f"{grid_folder}NNLO.{obs}.tab.gz.txt.out"),
                nnlojet_dat_folder=nnlojet_dat_folder,
                hep_tuple=HEP[obs],
                out_name=Path(f"{PLOT_DIR}/figure_{obs}_combined.png"),
                plot_closure = False
            )
        print("All combined overview canvases done.")
    else:
        print("Skipped combined canvasses (some contribution missing).")

        # --- (3) relative‐uncertainty plots --------------------------------- #
    from pathlib import Path
    UNC_DIR = Path("Analysis_plots_2/uncertainties")
    plot_relative_uncertainties(
        unitsfactor_grid_nnlojet,
        central_scale_factor,
        ORDERS,
        OBSERVABLES,
        grid_specs,
        grid_folder,
        nnlojet_dat_folder,
        HEP,
        UNC_DIR,
    )
    print("All relative-uncertainty plots done.")

