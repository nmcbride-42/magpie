"""On-demand light-curve plot for one TIC/sector.

Renders raw + detrended PNG with the Tier-1 event windows shaded, plus a
JSON sidecar of gap context. Cached in $MAGPIE_DATA_DIR/plots/. Used by the
Tier-3.5 vision judge, and handy by hand:

    python -m magpie.plot --tic 25078924 --sector 95
prints the absolute PNG path on success.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magpie.config import CACHE_DIR, PLOT_DIR
from magpie.tier1 import detrend, run_tier1


def gap_context(t, events):
    """Gaps (dt > 0.05 d ≈ 72 min) + each event's hours to the nearest gap
    edge or sector boundary. The Tier-3 text judge was blind to exactly
    this — all three of its first 'interesting' verdicts were gap-adjacent
    artifacts it had no way to see."""
    dt = np.diff(t)
    gap_idx = np.where(dt > 0.05)[0]
    edges = [t[0], t[-1]]
    for i in gap_idx:
        edges += [t[i], t[i + 1]]
    ev_ctx = []
    for s_i, e_i, kind in events:
        t0, t1 = t[s_i], t[min(e_i, len(t) - 1)]
        d = min(min(abs(t0 - g), abs(t1 - g)) for g in edges) * 24.0
        ev_ctx.append({"t_start": round(float(t0), 4),
                       "t_end": round(float(t1), 4),
                       "kind": kind,
                       "hours_to_nearest_gap": round(float(d), 2)})
    return ev_ctx, [round(float(g), 4) for g in edges]


def render(tic: int, sector: int) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.abspath(os.path.join(PLOT_DIR, f"s{sector}-tic{tic}.png"))
    sidecar = out.replace(".png", ".json")
    if os.path.exists(out) and os.path.exists(sidecar):
        return out

    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC",
                              sector=sector, exptime=120)
    if len(sr) == 0:
        raise SystemExit(f"no SPOC LC for TIC {tic} sector {sector}")
    lc = sr[0].download(download_dir=CACHE_DIR)
    lc = lc.remove_nans()
    t = lc.time.value
    f = np.asarray(lc.flux.value, dtype=float)

    resid, noise = detrend(t, f)
    r = run_tier1(t, f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                   gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("#14141a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#14141a")
        ax.tick_params(colors="#9a9aa5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#33333d")
    ax1.plot(t, f / np.median(f), ".", ms=1.5, color="#b7a6e8")
    ax1.set_ylabel("norm flux", color="#9a9aa5", fontsize=9)
    ax1.set_title(f"TIC {tic} — sector {sector}", color="#d8d8df", fontsize=11)
    ax2.plot(t, resid, ".", ms=1.5, color="#8fb8a0")
    ax2.axhline(0, lw=0.5, color="#555")
    ax2.set_ylabel("detrended (ppm)", color="#9a9aa5", fontsize=9)
    ax2.set_xlabel("BTJD", color="#9a9aa5", fontsize=9)
    for s_i, e_i, kind in r.events:
        for ax in (ax1, ax2):
            ax.axvspan(t[s_i], t[min(e_i, len(t) - 1)],
                       color="#e8a6a6", alpha=0.25, lw=0)
    fig.savefig(out, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    ev_ctx, gap_edges = gap_context(t, r.events)
    with open(sidecar, "w") as fh:
        json.dump({"tic": tic, "sector": sector, "events": ev_ctx,
                   "gap_edges_btjd": gap_edges,
                   "scores": {k: round(v, 2) for k, v in r.scores.items()}},
                  fh, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--sector", type=int, required=True)
    a = ap.parse_args()
    print(render(a.tic, a.sector))
