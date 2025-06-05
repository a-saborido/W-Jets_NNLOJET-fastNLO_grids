#!/usr/bin/env python3
"""
Parse fnlo-tk-cppread output files that may hold several scale-factor blocks
and plot cross section vs scale for every observable/bin at LO, NLO, NNLO.

Usage:
    python make_scale_variation_plots.py  /path/to/out/files  [--out plots]
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import json

import matplotlib.pyplot as plt

############################
# 1. CLI
############################
parser = argparse.ArgumentParser(
    description="Plot scale‑factor dependence of LO/NLO/NNLO cross sections."
)
parser.add_argument("indir", type=Path,
                    help="Directory containing *_mw.txt.out files")
parser.add_argument("--out", type=Path, default=Path("plots"),
                    help="Output directory for the PNGs (default: ./plots)")
args = parser.parse_args()

############################
# 2. containers
############################
orders = ("LO", "NLO", "NNLO")
# data[observable][bin][order] -> list of (scale, value)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# edges[observable][bin] -> (low_edge, high_edge)
edges = defaultdict(dict)

############################
# 3. regex helpers
############################
# ── filename pieces ────────────────────────────────────────────────
fn_re = re.compile(
    r"""
    (?P<prefix>.+?)\.(?P<observable>ptj1|ht_full|abs_yj1|ptw)\.tab\.gz
    _[\d.]+mw\.txt\.out$
    """, re.VERBOSE,
)

# ── inside‑file patterns ───────────────────────────────────────────
scale_re = re.compile(
    r"scale factors xmur, xmuf chosen here are:\s*([0-9.]+)\s*,\s*[0-9.]+"
)
row_re = re.compile(r"^\s*(\d+)\s")   # data rows start with a bin number

def parse_file(fp: Path):
    """
    Yield (observable, scale, rows) for every block in the file.
    rows is a list of (bin, lo, nlo, nnlo)
    """
    m = fn_re.search(fp.name)
    if not m:
        raise ValueError(f"Cannot decode filename: {fp.name}")
    observable = m.group("observable")

    rows_current_block = []
    current_scale = None

    with fp.open() as fh:
        for line in fh:
            # (a) start of a new block: look for the scale‑factor line
            ms = scale_re.search(line)
            if ms:
                # flush any finished block
                if current_scale is not None and rows_current_block:
                    yield observable, current_scale, rows_current_block
                    rows_current_block = []
                current_scale = float(ms.group(1))
                continue

            # (b) data rows
            if row_re.match(line):
                cols = line.split(maxsplit=10)          # grab all 11 columns
                bin_idx   = int(cols[0])
                low_edge  = float(cols[3])              # lower edge of the observable bin
                high_edge = float(cols[4])              # upper edge

                lo, nlo, nnlo = map(float, cols[6:9])

                rows_current_block.append((bin_idx, lo, nlo, nnlo))

                # store the edges once (they're identical in every scale block)
                if bin_idx not in edges[observable]:
                    edges[observable][bin_idx] = (low_edge, high_edge)

    # final block
    if current_scale is not None and rows_current_block:
        yield observable, current_scale, rows_current_block


############################
# 4. scan directory
############################
files = sorted(args.indir.glob("*mw.txt.out"))
if not files:
    raise SystemExit(f"No *_mw.txt.out files found in {args.indir}")

for fp in files:
    for observable, scale, rows in parse_file(fp):
        for bin_idx, lo, nlo, nnlo in rows:
            for order, val in zip(orders, (lo, nlo, nnlo)):
                data[observable][bin_idx][order].append((scale, val))

############################
# 5. plotting
############################
args.out.mkdir(parents=True, exist_ok=True)

for obs, bins in data.items():
    for bin_idx, odict in bins.items():
        plt.figure()
        for order in orders:
            if order not in odict:
                continue
            # sort → monotonic scale axis
            sx = sorted(odict[order])
            #x, y = zip(*sx)
            x, y = zip(*[(s, v) for s, v in sx if 0.05 <= s <= 20.0])
            plt.plot(x, y, linewidth=1, label=order)
        #plt.xlabel(r"Scale factor ($\mu_{R}=\mu_{F}$) in units of $m_W$")
        plt.xlabel(r"Scale factor ($\mu_{R}$) in units of $m_W$")
        plt.ylabel("Differential cross section [fb]")
        plt.xscale("log")
        low, high = edges[obs][bin_idx]
        plt.title(f"{obs}: {low:g} – {high:g}")
        plt.legend()
        plt.xlim(min(x), max(x))
        plt.tight_layout()
        outpng = args.out / f"{obs}_bin{bin_idx:02d}.png"
        plt.savefig(outpng, dpi=150)
        plt.close()
        print(f"✓ {outpng}")
        print(min(x),max(x))

print(f"Done. All plots are in {args.out.resolve()}")

json_out = args.out / "data_scale_dependence.json"
with json_out.open("w") as f:
    json.dump(data, f, indent=2)

print(f"✓ Full data saved to {json_out}")