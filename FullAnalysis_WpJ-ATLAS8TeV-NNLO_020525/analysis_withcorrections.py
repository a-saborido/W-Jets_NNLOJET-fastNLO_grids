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
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec



# ─────────────────────────────────────────────────────────────────────────────
# Global “cosmetic” settings for all figures
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    # base font sizes
    "font.size":        12,    # controls default text size
    "axes.titlesize":   14,    # subplot titles
    "axes.labelsize":   20,    # x- and y-axis labels
    "xtick.labelsize":  18,    # tick labels
    "ytick.labelsize":  18,
    "legend.fontsize":  12,    # legend text

    # line widths
    "axes.linewidth":     1.2,  # axis spine thickness
    "grid.linewidth":     0.6,  # grid-line thickness

    # tick properties
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  6,    # length of major ticks
    "ytick.major.size":  6,
    "xtick.major.width": 1.1,  # thickness of tick lines
    "ytick.major.width": 1.1,

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
    """Parse grid *.out → dataframe of all bins x scale points."""
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


# ------------------------------------------------------------------------ #
#  k-factor panel                                                          #
# ------------------------------------------------------------------------ #

'''
def _ratio_scale_band_edges(num: pd.Series, num_err_dn: pd.Series, num_err_up: pd.Series,
                      den: pd.Series, den_err_dn: pd.Series, den_err_up: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return *lo*, *hi* series for the ratio (num±err)/(den±err)."""
    lo = (num - num_err_dn) / (den + den_err_up)
    hi = (num + num_err_up) / (den - den_err_dn)
    #print(num/den, lo, hi)
    return lo, hi
'''

'''
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
                        label= "Scale unc." if first else None)
        first = False

    ax.scatter(m["BinCenter"], ratio, s=20, color=color)
    ax.errorbar(m["BinCenter"], ratio, yerr=ratio_stat,
                fmt="none", ecolor=color, alpha=.6,
                label="Stat. unc.")                      

    # ---------- cosmetics ---------------------------------------------- #
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(label)
    ax.grid(True, ls="--", alpha=.35)
    ax.legend(frameon=False, fontsize=8)


def kfactor_envelope(
    df_num: pd.DataFrame,
    df_den: pd.DataFrame,
    num_col: str,
    den_col: str,
    central_scale: float
) -> pd.DataFrame:
    """
    Build per-bin envelope for the k-factor using **matched** scale points.
    Returns columns:
        BinCenter, BinMin, BinMax, central, lo, hi
    """
    # keep only what we need and rename for clarity
    num = df_num[["BinCenter", "BinMin", "BinMax", "xmur", "xmuf", num_col]]\
             .rename(columns={num_col: "num"})
    den = df_den[["BinCenter",                "xmur", "xmuf", den_col]]\
             .rename(columns={den_col: "den"})

    merged = num.merge(den, on=["BinCenter", "xmur", "xmuf"], how="inner")
    merged["ratio"] = merged["num"] / merged["den"]

    band = merged.groupby("BinCenter")["ratio"].agg(lo="min", hi="max")

    # central value (same central scale for num & den)
    cen_num = num[(num.xmur == central_scale) & (num.xmuf == central_scale)]\
                 .set_index("BinCenter")["num"]
    cen_den = den[(den.xmur == central_scale) & (den.xmuf == central_scale)]\
                 .set_index("BinCenter")["den"]
    central = (cen_num / cen_den).rename("central")

    out = pd.concat([central, band], axis=1).reset_index()
    # add the bin edges for plotting
    out = out.merge(num[["BinCenter", "BinMin", "BinMax"]].drop_duplicates(),
                    on="BinCenter")
    return out


def plot_kfactor_panel(ax, kenv: pd.DataFrame, *, color: str, label: str) -> None:
    """Draw the k-factor band built with `kfactor_envelope`."""
    first = True
    for _, row in kenv.iterrows():
        #print(row.lo, row.hi)
        ax.fill_between([row.BinMin, row.BinMax],
                        [row.lo] * 2, [row.hi] * 2,
                        step="pre", color=color, alpha=0.30,
                        label="Scale unc." if first else None)
        first = False

    ax.scatter(kenv["BinCenter"], kenv["central"], s=20, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(label)
    ax.grid(True, ls="--", alpha=.35)
    ax.legend(frameon=False, fontsize=14)
'''


def kfactor_ratio(
    env_num: pd.DataFrame,
    env_den: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-bin bands for

      •  num / den   (numerator envelope divided by *central* denominator)
      •  den / den   (=1, but with denominator’s scale band)

    Parameters
    ----------
    env_num, env_den
        Envelopes returned by `envelope()` for any two perturbative orders
        that share binning.

    Returns
    -------
    ratio_numden, ratio_denden : pd.DataFrame
        Each has columns  BinCenter, BinMin, BinMax, central, lo, hi
    """
    denom_c = env_den.set_index("BinCenter")["sigma"]  # central denominator

    # num / den
    numden = env_num.copy()
    numden["central"] = env_num["sigma"]                  / denom_c.values
    numden["lo"]      = (env_num["sigma"]-env_num["err_dn"]) / denom_c.values
    numden["hi"]      = (env_num["sigma"]+env_num["err_up"]) / denom_c.values

    # den / den  (centre = 1)
    denden = env_den.copy()
    denden["central"] = 1.0
    denden["lo"]      = (env_den["sigma"]-env_den["err_dn"]) / env_den["sigma"]
    denden["hi"]      = (env_den["sigma"]+env_den["err_up"]) / env_den["sigma"]

    keep = ["BinCenter", "BinMin", "BinMax", "central", "lo", "hi"]
    return numden[keep], denden[keep]


def plot_kfactor_bands(
    ax,
    bands: Sequence[pd.DataFrame],          # 1…N dataframes (lo/hi/central)
    colors: Sequence[str],
    labels: Sequence[str],
    *,
    ylabel: str = r"$k$"
) -> None:
    """
    Draw an arbitrary collection of scale-variation bands (and their
    central markers) on *ax*.

    All *bands[i]* must have the columns
        BinCenter, BinMin, BinMax, central, lo, hi
    as produced e.g. by `kfactor_ratio`.
    """
    first = [True] * len(bands)

    for df, c, lab, f in zip(bands, colors, labels, first):
        for _, row in df.iterrows():
            ax.fill_between(
                [row.BinMin, row.BinMax],
                [row.lo] * 2, [row.hi] * 2,
                step="pre", alpha=0.30,
                color=c,
                label=lab if f else None
            )
            f = False                            # only first patch gets label
        ax.scatter(df["BinCenter"], df["central"], s=20, color=c)

    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True, ls="--", alpha=.35)
    ax.legend(frameon=False, fontsize=12)

# --------------------------------------------------------------------------- #
#  NPC / EwC helpers                                                          #
# --------------------------------------------------------------------------- #
def read_correction_file(path: Path) -> pd.DataFrame:
    """
    Read a three-column file   BinMin  BinMax  Factor
    and return a dataframe with an extra BinCenter column.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"correction file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["BinMin", "BinMax", "Factor"])
    df["BinCenter"] = 0.5 * (df["BinMin"] + df["BinMax"])
    return df


def _apply_factor_to_env(env_df: pd.DataFrame, factor_map: pd.Series) -> None:
    """
    In–place multiplication of sigma and all error columns of *env_df*
    by the factor that corresponds to each BinCenter.
    """
    f = env_df["BinCenter"].round(ROUND_DECIMALS).map(factor_map)
    for col in ["sigma", "err_dn", "err_up", "stat_err"]:
        if col in env_df:
            env_df[col] *= f


def _apply_factor_to_var(var_df: pd.DataFrame, factor_map: pd.Series) -> None:
    """
    In–place multiplication of NNLOJET columns that carry cross sections /
    errors by the same factor.
    """
    f = var_df["BinCenter"].round(ROUND_DECIMALS).map(factor_map)
    for col in ["cs", "cs_err", "tot02", "tot02_err", "tot03", "tot03_err"]:
        if col in var_df:
            var_df[col] *= f


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
                             label="Scale unc." if first_scale else None)
        first_scale = False
        ax_main.fill_between([row.BinMin, row.BinMax],
                             [row.sigma - row.stat_err] * 2,
                             [row.sigma + row.stat_err] * 2,
                             color="tab:orange", alpha=0.40, step="pre",
                             label="Stat. unc." if first_stat else None)
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
        ax_main.legend(frameon=False, fontsize=14)

    env_r = _rounded(env_df)
    var_r = _rounded(var_df)        

    # ---------------- RATIO 1 : Grid / NNLOJET ------------------------- #
    if not plot_ratio1:
        ax_r1.set_visible(False)
    else:
        m1 = env_r.merge(var_r[["__bc__", "cs", "cs_err"]], on="__bc__", how="inner")
        r1_y, r1_err = ratio_and_err(m1["sigma"], m1["stat_err"],
                                     m1["cs"],    m1["cs_err"])

        #ax_r1.errorbar(m1["BinCenter"], r1_y, yerr=r1_err,
        #               fmt="s", color="red", ms=4, label="Stat. unc.")
        ax_r1.scatter(m1["BinCenter"], r1_y, s=20, color="red")
        ax_r1.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_r1.set_ylabel("Grid/NNLOJET")
        ax_r1.grid(True, ls="--", alpha=.35)
        ax_r1.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax_r1.yaxis.get_offset_text().set_visible(False)
        ax_r1.tick_params(axis='x', which='both', labelbottom=False)


    # ---------------- RATIO 2 : mode‑dependent ------------------------- #
    if ratio2_mode == "none":
        ax_r2.set_visible(False)
        ax_r1.set_xlabel(X_LABEL[observable])       
        ax_r1.tick_params(axis='x', which='both', labelbottom=True)

    elif ratio2_mode == "zoom":
        plt.setp(ax_r1.get_xticklabels(), visible=False)
        #ax_r2.errorbar(m1["BinCenter"], r1_y, yerr=r1_err,
        #               fmt="s", color="red", ms=4, label="Stat. unc.")
        ax_r2.scatter(m1["BinCenter"], r1_y, s=20, color="red")
        ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_r2.set_ylabel("Grid/NNLOJET (‰)")
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
                           fmt="^", color="k", ms=5, label="Stat. unc.")
            ax_r2.axhline(1.0, color="k", ls="--", lw=0.8)
            ax_r2.set_ylabel("Grid/Data")
            if show_ratio2_xlabel:                                                      #####
                ax_r2.set_xlabel(X_LABEL[observable])
            ax_r2.grid(True, ls="--", alpha=.35)
            ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax_r2.yaxis.get_offset_text().set_visible(False)
            if not show_ratio2_xlabel:                                                  #####
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
    
    # --- NPC × EwC --------------------------------------------------------- #
    if order in {"LO", "NLO", "NNLO"}:
        npc_df = NPC_FACTORS[observable]          # prepared in main()
        ewc_df = EWC_FACTORS[observable]

        # build a rounded BinCenter → factor map
        npc_map = npc_df.set_index(
            npc_df["BinCenter"].round(ROUND_DECIMALS)
        )["Factor"]
        ewc_map = ewc_df.set_index(
            ewc_df["BinCenter"].round(ROUND_DECIMALS)
        )["Factor"]
        total_factor_map = npc_map * ewc_map       # Bin-by-bin product

        _apply_factor_to_var(var_df,  total_factor_map)
        _apply_factor_to_env(env_df,  total_factor_map)

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
        ratio2_mode="none",          # zoom behaviour for closure tests. Set as "zoom" to add an extra panel zoomed around 1
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
            wspace=0.001,
            hspace=0.02,
            figure=fig,
        )
    else:
        gs  = gridspec.GridSpec(
            4, 3,
            width_ratios=[1, 1, 1],
            height_ratios=[3, 0.1, 1, 1],   # extra bottom row
            wspace=0.001,
            hspace=0.02,
            figure=fig,
        )

    # ­collect y‑range info (optional – unchanged)
    main_min = r1_min = r2_min = np.inf
    main_max = r1_max = r2_max = -np.inf
    axes_main, axes_r1, axes_r2 = [], [], []
    env_store: Dict[str, pd.DataFrame] = {}
    df_store:  Dict[str, pd.DataFrame] = {}
    axes_kfactor = []

    for col, order in enumerate(orders):

        nnlojet_dat = Path(f"{nnlojet_dat_folder}{order}.{observable}_var.dat")
        df_grid = parse_grid_out(grid_out, order)
        env_df  = envelope(df_grid, order, central_scale_factor)
        var_df  = read_nnlojet_dat(nnlojet_dat, unitsfactor_grid_nnlojet)
        
        # --- NPC × EwC --------------------------------------------------------- #
        if order in {"LO", "NLO", "NNLO"}:
            npc_df = NPC_FACTORS[observable]          # prepared in main()
            ewc_df = EWC_FACTORS[observable]

            # build a rounded BinCenter → factor map
            npc_map = npc_df.set_index(
                npc_df["BinCenter"].round(ROUND_DECIMALS)
            )["Factor"]
            ewc_map = ewc_df.set_index(
                ewc_df["BinCenter"].round(ROUND_DECIMALS)
            )["Factor"]
            total_factor_map = npc_map * ewc_map       # Bin-by-bin product

            _apply_factor_to_var(var_df,  total_factor_map)
            _apply_factor_to_env(env_df,  total_factor_map)

        env_df  = attach_stat_err(env_df, var_df)
        env_store[order] = env_df.copy()
        df_store[order]  = df_grid        # keep full scale table

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
            plot_nnlojet_main=plot_closure,
            plot_ratio1=plot_closure,
        )

        ax_main.set_title(order, fontsize=18, pad=8, fontweight="bold")
        if col > 0:                                                                     #####
            for a in (ax_main, ax_r1, ax_r2):
                a.set_ylabel("")
                a.tick_params(labelleft=False)

        # --- bottom‑row axis for this column --------------------------- #
        ax_kfactor = fig.add_subplot(gs[3, col], sharex=ax_main)
        axes_kfactor.append(ax_kfactor)
        
        if order == "NLO":
            # bands:  NLO/LO  +  LO/LO
            nlo_lo, lo_lo = kfactor_ratio(env_store["NLO"], env_store["LO"])
            plot_kfactor_bands(
                ax_kfactor,
                bands=[nlo_lo, lo_lo],
                colors=["tab:purple", "tab:grey"],
                labels=["NLO / LO", "LO / LO"],
                ylabel=r"Ratio to LO"
            )
            ax_kfactor.set_xlabel(X_LABEL[observable])

        elif order == "NNLO":
            # bands:  NNLO/NLO  +  NLO/NLO  +  LO/NLO
            nnlo_nlo, nlo_nlo = kfactor_ratio(env_store["NNLO"], env_store["NLO"])
            lo_nlo, _         = kfactor_ratio(env_store["LO"],  env_store["NLO"])
            plot_kfactor_bands(
                ax_kfactor,
                bands=[nnlo_nlo, nlo_nlo, lo_nlo],
                colors=["tab:red", "tab:purple", "tab:grey"],
                labels=["NNLO / NLO", "NLO / NLO", "LO / NLO"],
                ylabel=r"Ratio to NLO"
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

    fig.suptitle(f"{grid_specs}", fontsize=15, x=0.1, y=0.05)
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
    """For each *order × observable* pair, compute and plot relative uncertainties.

    scale_rel = err_{scale} / σ (envelope from scale variations)
    stat_rel  = stat_err / σ  (NNLOJET statistical error)

    The plot shows scale uncertainty as a filled band and statistical uncertainty
    as error bars, each centred on ratio 1.
    """
    # ---------------------------------------------------------------------
    # Prepare output folder
    # ---------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Iterate over perturbative orders and observables
    # ---------------------------------------------------------------------
    for order in orders:
        for obs in observables:
            # — Load grid & envelope
            grid_file = (
                Path(f"{grid_folder}{order}.{obs}.tab.gz.txt.out")
                if order in CONTRIB_ORDERS
                else Path(f"{grid_folder}NNLO.{obs}.tab.gz.txt.out")
            )
            df_grid = parse_grid_out(grid_file, order)
            env_df = envelope(df_grid, order, central_scale_factor)

            # — Attach statistical error from NNLOJET
            var_df = read_nnlojet_dat(
                Path(f"{nnlojet_dat_folder}{order}.{obs}_var.dat"),
                unitsfactor_grid_nnlojet,
            )

            # --- NPC × EwC --------------------------------------------------------- #
            if order in {"LO", "NLO", "NNLO"}:
                npc_df = NPC_FACTORS[obs]          # prepared in main()
                ewc_df = EWC_FACTORS[obs]

                # build a rounded BinCenter → factor map
                npc_map = npc_df.set_index(
                    npc_df["BinCenter"].round(ROUND_DECIMALS)
                )["Factor"]
                ewc_map = ewc_df.set_index(
                    ewc_df["BinCenter"].round(ROUND_DECIMALS)
                )["Factor"]
                total_factor_map = npc_map * ewc_map       # Bin-by-bin product

                _apply_factor_to_var(var_df,  total_factor_map)
                _apply_factor_to_env(env_df,  total_factor_map)

            env_df = attach_stat_err(env_df, var_df)

            # — Compute relative uncertainties
            env_df["scale_rel_up"] = abs(env_df["err_up"] / env_df["sigma"])
            env_df["scale_rel_down"] = abs(env_df["err_dn"] / env_df["sigma"])
            env_df["stat_rel"] = abs(env_df["stat_err"] / env_df["sigma"])

            # -----------------------------------------------------------------
            # Plot (font sizes hard‑coded to Matplotlib default equivalents)
            # -----------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))

            # Scale‑variation band (step‑style)
            first = True
            for _, row in env_df.iterrows():
                ax.fill_between(
                    [row.BinMin, row.BinMax],
                    [1.0 - row.scale_rel_down] * 2,
                    [1.0 + row.scale_rel_up] * 2,
                    step="pre",
                    color="tab:blue",
                    alpha=0.3,
                    label="Scale unc. (rel.)" if first else None,
                )
                first = False

            # Statistical error bars around 1
            ax.errorbar(
                env_df["BinCenter"],
                [1.0] * len(env_df),
                yerr=env_df["stat_rel"],
                fmt="o",
                label="stat unc. (rel.)",
            )

            # Axis styling with explicit default‑size fonts
            ax.axhline(1.0, color="k", ls="--", lw=0.8)
            ax.set_xlabel(X_LABEL[obs], fontsize=10)                 # default axes.labelsize ≈ 10 pt
            ax.set_ylabel("Ratio to central (±unc.)", fontsize=10)
            ax.set_title(
                f"{order} relative uncertainties  –  {obs}. {grid_specs}",
                fontsize=12,                                      # default axes.titlesize ≈ 12 pt
            )
            ax.legend(fontsize=10)                                  # default legend.fontsize ≈ 10 pt
            ax.tick_params(axis="both", which="major", labelsize=8)  # default tick label ≈ 8 pt
            ax.grid(True, ls="--", alpha=0.3)

            # -----------------------------------------------------------------
            # Save image
            # -----------------------------------------------------------------
            out_path = out_dir / f"rel_unc_{order}_{obs}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  ✓ {out_path} written")





# --------------------------------------------------------------------------- #
#  Main driver                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    PLOT_DIR = Path("Analysis_withcorrections")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    OBSERVABLES = ["ptj1", "ptw", "abs_yj1", "ht_full"]
    ORDERS      = ["LO", "NLO", "NNLO", "R", "V", "RRa", "RRb", "RV", "VV"]
    grid_specs    = "25"
    central_scale_factor = 1.0
    grid_specs = f"{grid_specs} x-nodes, μ$_R$= μ$_F$ = {central_scale_factor} m$_W$"
    cs_units = "fb"
    Wsign = "+"
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
        "ptj1":    fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}p_T^{{\,j1}}$  [{cs_units}/GeV]",
        "ptw":     fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}p_T^{{\,W}}$  [{cs_units}/GeV]",
        "abs_yj1": fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}|y^{{\,j1}}|$  [{cs_units}]",
        "abs_etaj1": fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}|\eta^{{\,j1}}|$  [{cs_units}]",
        "ht_full": fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}H_T$  [{cs_units}/GeV]",
        "ht_jets": fr"$\mathrm{{d}}\sigma^{{W^{Wsign}}}/\mathrm{{d}}H_T^{{\,jets}}$  [{cs_units}/GeV]",
    }

    HEP: Dict[str, Tuple[pd.Series, pd.Series, List[pd.Series]]] = {
        "ptj1":    read_hepdata("HEPdata/HEPData_WpJ_ptj1.csv"),
        "abs_yj1": read_hepdata("HEPdata/HEPData_WpJ_abs_yj1.csv"),
        "abs_etaj1": read_hepdata("HEPdata/HEPData_WpJ_abs_etaj1.csv"),
        "ht_full": read_hepdata("HEPdata/HEPData_WpJ_HT.csv"),
        "ht_jets": read_hepdata("HEPdata/HEPData_WpJ_HTjets.csv"),
        "ptw":     read_hepdata("HEPdata/HEPData_WpJ_ptw.csv"),
    }

    # ---------------------------------------------------------------------------
    #  Correction-file templates  (one per observable)
    # ---------------------------------------------------------------------------
    NPC_FILE_TEMPLATE = "corrections/Wplus_NPC_{obs}.dat"
    EWC_FILE_TEMPLATE = "corrections/Wplus_EWC_{obs}.dat"

    # read once and cache
    NPC_FACTORS: Dict[str, pd.DataFrame] = {
        obs: read_correction_file(Path(NPC_FILE_TEMPLATE.format(obs=obs)))
        for obs in OBSERVABLES
    }
    EWC_FACTORS: Dict[str, pd.DataFrame] = {
        obs: read_correction_file(Path(EWC_FILE_TEMPLATE.format(obs=obs)))
        for obs in OBSERVABLES
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
                plot_closure = True
            )
        print("All combined overview canvases done.")
    else:
        print("Skipped combined canvasses (some contribution missing).")

        # --- (3) relative‐uncertainty plots --------------------------------- #
    from pathlib import Path
    UNC_DIR = PLOT_DIR/"uncertainties"
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

