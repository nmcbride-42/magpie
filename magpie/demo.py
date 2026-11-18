"""Offline demo: the real Tier-1 sweep, common-mode veto and Tier-4 period
resolution, run on a synthetic sector. No network, no LLM, about a minute
on a laptop.

    python -m magpie.demo                 # temp dir, printed funnel
    python -m magpie.demo --out demo-run --plot

What it builds: N synthetic light curves in the SPOC FITS layout (TIME,
PDCSAP_FLUX, QUALITY), plus a MAST-style bulk script that points at them
with file:// URLs, so `magpie.sweep` runs exactly as it does on a real
sector. The planted population:

  quiet       white noise + slow starspot-like modulation   -> nothing
  common-mode the SAME 2.5% dip at the SAME cadences on several
              unrelated stars (a spacecraft event, not the sky)
                                                            -> common-mode veto
  dipper      one asymmetric Boyajian-style dip per star    -> survives,
                                                               unresolved
  EB          a detached eclipsing binary                   -> survives, period
                                                               resolved by fold

The catalog veto (ExoFOP / SIMBAD) needs the network, so the demo marks
every Tier-1-flagged star as catalog-clean and says so. The LLM judges are
skipped: survivors are what they would see.
"""

import argparse
import os
import sqlite3
import tempfile

import numpy as np

from magpie import common_mode, tier4
from magpie.sweep import sweep
from magpie.tier1 import run_tier1
from magpie.tier2 import SCHEMA as TIER2_SCHEMA

SECTOR = 999                # not a real TESS sector
CADENCE_D = 2.0 / 1440.0    # 2-minute cadence
DAYS = 10.0
GLITCH = (3000, 3040)       # cadence window of the shared spacecraft event
EB_PERIOD = 1.7
TIER2_THRESHOLD = 10.0      # same default as magpie.tier2.run_tier2


def _base(rng, n):
    t = 3000.0 + np.arange(n) * CADENCE_D          # BTJD-like
    rot = 1.0 + 0.002 * np.sin(2 * np.pi * t / rng.uniform(3, 8))
    return t, rot * (1.0 + rng.normal(0, 5e-4, n)) * 1.0e4


def make_population(n_quiet=24, n_glitch=8, n_dipper=2, n_eb=2, seed=42):
    """Return [(tic, kind, time, flux)] for the synthetic sector."""
    rng = np.random.default_rng(seed)
    n = int(DAYS / CADENCE_D)
    out, tic = [], 100000001
    for kind, count in (("quiet", n_quiet), ("common-mode", n_glitch),
                        ("dipper", n_dipper), ("EB", n_eb)):
        for _ in range(count):
            t, f = _base(rng, n)
            if kind == "common-mode":
                f[GLITCH[0]:GLITCH[1]] *= 0.975
            elif kind == "dipper":
                c = int(rng.uniform(0.2, 0.8) * n)
                ingress = np.linspace(0, 1, 240)      # slow ingress
                egress = np.linspace(1, 0, 60)        # fast egress
                prof = np.concatenate([ingress, egress]) * 0.02
                f[c - 240:c + 60] *= 1 - prof
            elif kind == "EB":
                ph = ((t - t[0] - 0.4) / EB_PERIOD) % 1.0
                ph[ph > 0.5] -= 1.0
                f[np.abs(ph) < 0.02] *= 0.96           # primary, ~1.6 h
                f[np.abs(ph - 0.5) < 0.02] *= 0.985    # secondary
            out.append((tic, kind, t, f))
            tic += 1
    return out


def write_fits(path, t, f):
    from astropy.io import fits
    cols = fits.ColDefs([
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="E", array=f.astype(np.float32)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(t.size, int)),
    ])
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]
                 ).writeto(path, overwrite=True)


def write_sector(pop, src_dir):
    """FITS files + a MAST-style bulk script with file:// URLs."""
    os.makedirs(src_dir, exist_ok=True)
    script = os.path.join(src_dir, f"sector{SECTOR}_lc.sh")
    with open(script, "w") as fh:
        fh.write("#!/bin/sh\n")
        for tic, _, t, f in pop:
            name = f"demo-s{SECTOR:04d}-{tic:016d}-0000-s_lc.fits"
            path = os.path.abspath(os.path.join(src_dir, name))
            write_fits(path, t, f)
            fh.write(f"curl -C - -L -o {name} file://{path}\n")
    return script


def run_demo(out_dir, plot=False, quiet_sweep=True):
    pop = make_population()
    kinds = {tic: kind for tic, kind, _, _ in pop}
    curves = {tic: (t, f) for tic, _, t, f in pop}
    db_path = os.path.join(out_dir, "demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    script = write_sector(pop, os.path.join(out_dir, "src"))

    # Tier 0/1 — the real streaming sweep (curl -> FITS -> filters -> sqlite)
    sweep(script, SECTOR, db_path=db_path)

    db = sqlite3.connect(db_path)
    flagged = [tic for (tic,) in db.execute(
        "SELECT tic FROM tier1 WHERE sector=? AND status='ok' "
        "AND deep_dip + odd_brightening > ?", (SECTOR, TIER2_THRESHOLD))]

    # Tier 2 stand-in — offline, so every flagged star is "catalog-clean"
    db.executescript(TIER2_SCHEMA)
    db.executemany(
        "INSERT OR REPLACE INTO tier2 (sector,tic,verdict,reasons) "
        "VALUES (?,?,'survive','offline demo: catalog veto skipped')",
        [(SECTOR, tic) for tic in flagged])
    db.commit()
    db.close()

    # Tier 2.5 — the real common-mode veto
    common_mode.run(SECTOR, db_path=db_path)

    db = sqlite3.connect(db_path)
    survivors = [tic for (tic,) in db.execute(
        "SELECT tic FROM tier2 WHERE sector=? AND verdict='survive'",
        (SECTOR,))]
    db.close()

    # Tier 4 period resolution on the survivors, with the same rule as
    # tier4.dossier (minus the multi-sector download): resolved if the
    # dip-time fold succeeds OR BLS is overwhelming; reported period is BLS's.
    resolved = {}
    for tic in survivors:
        t, f = curves[tic]
        r = run_tier1(t, f)
        dips = [(t[s] + t[min(e, len(t) - 1)]) / 2
                for s, e, kind in r.events if "dip" in kind]
        bls = tier4.bls_search([(SECTOR, t, f)])
        fold = tier4.resolve_period(dips, bls=bls)
        is_res = bool(fold.get("resolved")) or (bls.get("power_ratio") or 0) > 50
        period = bls.get("best_period_d") or fold.get("period_d")
        resolved[tic] = {"resolved": is_res, "period_d": period,
                         "fold": fold, "bls": bls}

    rows = []
    for tic, kind, _, _ in pop:
        if tic not in flagged:
            fate = "tier1: below threshold"
        elif tic not in survivors:
            fate = "tier2.5: common-mode veto"
        else:
            res = resolved[tic]
            fold = res["fold"]
            if res["resolved"]:
                fate = f"tier4: periodic, BLS P={res['period_d']:.2f} d"
                if fold.get("period_d") and abs(fold["period_d"]
                                                - res["period_d"]) > 0.01:
                    fate += f" (dip fold resolves at {fold['period_d']:.2f} d)"
            else:
                fate = f"tier4: unresolved ({fold.get('reason')}) -> LLM judges"
        rows.append((kind, tic, fate))

    print("\n=== demo funnel (synthetic sector, planted truth on the left) ===")
    summary = {}
    for kind, tic, fate in rows:
        summary.setdefault((kind, fate), []).append(tic)
    for (kind, fate), tics in summary.items():
        print(f"  {kind:<12} x{len(tics):<3} {fate}")
    print(f"\n  {len(pop)} curves swept, {len(flagged)} flagged by Tier 1, "
          f"{len(survivors)} survive the common-mode veto")

    if plot:
        png = plot_examples(pop, os.path.join(out_dir, "demo.png"))
        print(f"  plot: {png}")
    return {"rows": rows, "flagged": flagged, "survivors": survivors,
            "kinds": kinds, "resolved": resolved}


def plot_examples(pop, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = {}
    for tic, kind, t, f in pop:
        picks.setdefault(kind, (tic, t, f))
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 1.8 * len(picks)),
                             sharex=True)
    labels = {"quiet": "quiet star: below Tier-1 threshold",
              "common-mode": "shared cadence-window dip: common-mode veto",
              "dipper": "asymmetric single dip: survives, unresolved",
              "EB": "eclipsing binary: survives, period resolved"}
    for ax, (kind, (tic, t, f)) in zip(axes, picks.items()):
        ax.plot(t - t[0], f / np.median(f), ".", ms=0.8, color="#333")
        ax.set_title(labels[kind], fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("days")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="output dir (default: a new temp dir)")
    ap.add_argument("--plot", action="store_true",
                    help="also write demo.png with one curve per planted class")
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="magpie-demo-")
    os.makedirs(out, exist_ok=True)
    print(f"demo output: {out}")
    run_demo(out, plot=a.plot)


if __name__ == "__main__":
    main()
