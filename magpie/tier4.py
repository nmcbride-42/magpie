"""Tier 4 — full-history dossier for a finalist.

For one TIC: download every available 2-min SPOC sector, run the Tier-1
filters on each, collect all events in absolute time (BTJD), then hunt for
a period two ways:
  1. BLS (BoxLeastSquares) over the stitched multi-sector curve — finds
     strictly periodic transit-like signals (EBs, planets) even when any
     single sector looks chaotic.
  2. Event-interval scan — pairwise dip-time differences; a common divisor
     means the per-sector "aperiodicity" was sampling, not physics.

Output: $MAGPIE_DATA_DIR/dossiers/tic{tic}.json with per-sector scores and
events. The dossier is what either kills the candidate (period found,
mundane class assigned) or graduates it to write-up.
"""

import json
import os
import shutil

import numpy as np

from magpie.config import CACHE_DIR as CACHE, DB_PATH, DOSSIER_DIR
from magpie.tier1 import run_tier1


def purge_download_cache():
    """Delete the lightkurve FITS download cache. tier4's fetch_all() pulls
    every sector of a star's history into CACHE/mastDownload; once the dossier
    JSON is written those FITS are dead weight. Unlike the sweep (which unlinks
    each FITS inline), lightkurve never cleans up after itself — left alone the
    cache grows ~unboundedly across finalists and fills the disk. Called per
    dossier so the cache never holds more than one star's history at a time."""
    md = os.path.join(CACHE, "mastDownload")
    if os.path.isdir(md):
        shutil.rmtree(md, ignore_errors=True)


def fetch_all(tic):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC",
                              exptime=120)
    curves = []
    for i in range(len(sr)):
        try:
            lc = sr[i].download(download_dir=CACHE).remove_nans()
            curves.append((int(lc.sector), lc.time.value,
                           np.asarray(lc.flux.value, dtype=float)))
        except Exception as e:
            print(f"  sector dl failed: {type(e).__name__}")
    return curves


def bls_search(curves, min_p=0.5, max_p=120.0):
    """BLS on the stitched normalized curve."""
    from astropy.timeseries import BoxLeastSquares
    ts = np.concatenate([t for _, t, _ in curves])
    fs = np.concatenate([f / np.median(f) for _, _, f in curves])
    order = np.argsort(ts)
    ts, fs = ts[order], fs[order]
    bls = BoxLeastSquares(ts, fs)
    periods = np.linspace(min_p, max_p, 20000)
    res = bls.power(periods, duration=[0.05, 0.1, 0.2, 0.4])
    i = int(np.argmax(res.power))
    return {"best_period_d": round(float(res.period[i]), 5),
            "depth": round(float(res.depth[i]), 5),
            "power": round(float(res.power[i]), 2),
            "power_median": round(float(np.median(res.power)), 2),
            "power_ratio": round(float(res.power[i] / np.median(res.power)), 1),
            "transit_time": round(float(res.transit_time[i]), 4)}


def interval_scan(dip_times, max_p=120.0, tol=0.02):
    """Try candidate periods: every pairwise interval / small integers.
    Score = fraction of dip times that phase up within tol*period."""
    if len(dip_times) < 3:
        return None
    t = np.asarray(sorted(dip_times))
    cands = set()
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            d = t[j] - t[i]
            for k in range(1, 13):
                p = d / k
                if 0.4 < p < max_p:
                    cands.add(round(p, 4))
    best = None
    for p in cands:
        ph = ((t - t[0]) / p) % 1.0
        ph = np.minimum(ph, 1 - ph)
        frac = float(np.mean(ph < tol))
        if best is None or frac > best["phased_fraction"] or (
                frac == best["phased_fraction"] and p > best["period_d"]):
            best = {"period_d": p, "phased_fraction": round(frac, 3)}
    return best


def _n_phase_clusters(t, p, tol_h):
    """How many tight phase clusters the dip times fold into at period p
    (wrap-aware), plus the widest cluster span in hours."""
    tol = tol_h / 24.0 / p
    ph = np.sort(((t - t[0]) / p) % 1.0)
    # wrap-aware: the largest gap is the true cluster break, so rotate it
    # to the end before counting.
    gaps = np.diff(np.concatenate([ph, [ph[0] + 1.0]]))
    n_clusters = int(np.sum(gaps > tol))
    # widest within-cluster span, in hours
    widest_ph = 0.0
    edges = list(np.where(gaps > tol)[0])
    start = 0
    for e in edges:
        widest_ph = max(widest_ph, ph[e] - ph[start])
        start = e + 1
    return max(n_clusters, 1), widest_ph * p * 24.0


def resolve_period(dip_times, bls=None, max_p=120.0, tol_h=3.0):
    """Decide if the dip times are explained by a periodic eclipser —
    INCLUDING eccentric binaries whose secondary eclipse sits off-phase.

    Method: take BLS's best period (now searched to 120d) as the seed,
    test it and its 2x multiple (BLS routinely locks the half-period of an
    eccentric EB), and accept the SHORTEST candidate whose fold puts every
    dip into <=2 tight phase clusters. Shortest-wins kills the n*P aliases
    that split into >2 clusters; seeding from BLS (a power statistic, not a
    clustering one) kills the P/n harmonics that a pure fold-search would
    wrongly prefer.

    Fixes the 30-day blind spot: BLS searched only 0.5-30d, so the 52d
    eccentric EB (TIC 220481009) read as unresolved/interesting.
    """
    if len(dip_times) < 3:
        return {"resolved": False, "reason": "too few events"}
    t = np.asarray(sorted(dip_times))
    cands = []
    if bls and bls.get("best_period_d"):
        bp = bls["best_period_d"]
        cands += [bp, 2 * bp, bp / 2]
    if not cands:
        cands = list(np.arange(0.5, max_p, 0.01))
    # shortest first — fundamental beats multiples; BLS seed beats harmonics
    for p in sorted(c for c in cands if 0.4 < c <= max_p):
        n_clusters, widest_h = _n_phase_clusters(t, p, tol_h)
        if n_clusters <= 2 and widest_h < tol_h * 3:
            return {"resolved": True, "period_d": round(float(p), 4),
                    "n_eclipse_phases": n_clusters,
                    "mechanism": "eccentric eclipsing binary" if n_clusters == 2
                                 else "eclipsing binary / transit"}
    return {"resolved": False,
            "reason": "no period folds all dips into <=2 phases"}


def dossier(tic):
    os.makedirs(DOSSIER_DIR, exist_ok=True)
    print(f"TIC {tic}: fetching history...")
    curves = fetch_all(tic)
    print(f"  {len(curves)} sectors downloaded")
    sectors_out, dip_times = [], []
    for sector, t, f in sorted(curves):
        r = run_tier1(t, f)
        evs = []
        for s_i, e_i, kind in r.events:
            t0, t1 = float(t[s_i]), float(t[min(e_i, len(t) - 1)])
            evs.append({"t_start": round(t0, 4), "t_end": round(t1, 4),
                        "kind": kind})
            if "dip" in kind:
                dip_times.append((t0 + t1) / 2)
        sectors_out.append({"sector": sector, "n_cadences": len(t),
                            "scores": {k: round(v, 2) for k, v in r.scores.items()},
                            "events": evs})
        flag = " <-- events" if evs else ""
        print(f"  s{sector}: dip={r.scores['deep_dip']:.1f} "
              f"bright={r.scores['odd_brightening']:.1f} "
              f"n_ev={len(evs)}{flag}")
    bls = bls_search(curves) if curves else None
    period = resolve_period(dip_times, bls=bls)
    # Either signal resolves it: a strong flux-BLS detection (clean EBs), OR
    # dip times that fold into <=2 eclipse phases (eccentric EBs whose unequal
    # eclipses give BLS weak power). Neither alone covers both classes.
    bls_strong = bool(bls) and (bls.get("power_ratio") or 0) > 50
    resolved = bool(period.get("resolved")) or bls_strong

    # Reported period: prefer the flux-based BLS estimate (it correctly found
    # the 52d eccentric orbit of TIC 220481009 where the dip-time fold picked
    # the 26d half-alias). Fall back to the fold period only when BLS is junk.
    period_d = (bls.get("best_period_d") if bls and bls.get("best_period_d")
                else period.get("period_d"))
    # Depth measured from the folded curve at that period (BLS box-depth
    # understates V-shaped eclipses). Needs the stitched flux + an epoch.
    depth_measured = None
    if period_d and curves:
        ts = np.concatenate([t for _, t, _ in curves])
        fs = np.concatenate([f / np.median(f) for _, _, f in curves])
        lo = np.percentile(fs, 1.0)
        ep = float(np.median(ts[fs <= lo]))
        ph = (((ts - ep) / period_d) % 1.0)
        ph[ph > 0.5] -= 1.0
        near = np.abs(ph) < 0.03
        if near.sum() >= 5:
            depth_measured = round(float(1.0 - np.percentile(fs[near], 1.0)), 4)

    out = {
        "tic": tic,
        "n_sectors": len(curves),
        "n_dip_events_total": len(dip_times),
        "dip_times_btjd": [round(x, 4) for x in sorted(dip_times)],
        "bls": bls,
        "interval_scan": interval_scan(dip_times),
        "period_resolution": period,
        "period_d": round(float(period_d), 5) if period_d else None,
        "depth_measured": depth_measured,
        "period_caveat": "period from flux-BLS; verify against the phased "
                         "curve — may be 2x or 1/2 for some eclipse geometries",
        "resolved": resolved,
        "resolved_by": ("fold" if period.get("resolved")
                        else "bls" if bls_strong else None),
        "sectors": sectors_out,
    }
    path = os.path.join(DOSSIER_DIR, f"tic{tic}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  dossier: {path}")
    print(f"  BLS: {out['bls']}")
    print(f"  interval scan: {out['interval_scan']}")
    return out


def sector_interesting_tics(sector):
    """TICs the vision judge flagged interesting for a sector and that don't
    yet have a dossier."""
    import sqlite3
    db = sqlite3.connect(DB_PATH, timeout=60)
    rows = db.execute(
        "SELECT tic FROM tier35 WHERE sector=? AND verdict='interesting'",
        (sector,)).fetchall()
    db.close()
    return [t for (t,) in rows]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tics", type=int, nargs="*")
    ap.add_argument("--sector", type=int,
                    help="run on all of a sector's interesting candidates")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a dossier already exists")
    args = ap.parse_args()
    tics = list(args.tics)
    if args.sector:
        tics += sector_interesting_tics(args.sector)
    if not tics:
        print("nothing to do (no TICs and no --sector interesting candidates)")
        return
    for tic in tics:
        path = os.path.join(DOSSIER_DIR, f"tic{tic}.json")
        if os.path.exists(path) and not args.force:
            print(f"TIC {tic}: dossier exists, skipping (use --force)")
            continue
        try:
            dossier(tic)
        finally:
            purge_download_cache()


if __name__ == "__main__":
    main()
