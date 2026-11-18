"""Tier 6 — VSX submission packet generator.

For a submittable object, produce everything VSX requires that the funnel
didn't already have:
  - epoch (time of minimum) folded from the period
  - phase-folded light curve plot (all sectors stitched)
  - magnitude range (catalog mag + eclipse depth -> max/min)
  - finder chart (DSS2 Red cutout, WCS-oriented, target marked)
  - fields.txt: the filled VSX field list, ready to paste

Output: $MAGPIE_DATA_DIR/packets/tic{tic}/ with phase.png, finder.png,
blending.png and fields.txt. The packet only PREPARES a submission; a human
reviews every field and files it. A filed object is a candidate pending VSX
moderation, not a confirmed discovery.

Honest about provenance: TESS is a red bandpass, not Johnson V. We report
the calibrated catalog magnitude we have (Tmag, or V/G if present) and
LABEL the band — we never claim a V-band range we didn't measure.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magpie.config import (CACHE_DIR as CACHE, DB_PATH, DISCOVERER,
                           DOSSIER_DIR, PACKET_DIR)

DARK = "#14141a"
BTJD_OFFSET = 2457000.0  # TESS time = BJD_TDB - 2457000


def measured_depth(t, f, period, epoch):
    """Fractional eclipse depth from the FOLDED curve, not BLS's box fit.
    BLS systematically understates V-shaped eclipses (3.9% vs a true ~10%
    for TIC 257691369). Depth = 1 - (1st-percentile flux near phase 0)."""
    ph = (((t - epoch) / period) % 1.0)
    ph[ph > 0.5] -= 1.0
    near = np.abs(ph) < 0.03  # within the eclipse window
    if near.sum() < 5:
        return None
    return float(1.0 - np.percentile(f[near], 1.0))


def _style(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors="#9a9aa5", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#33333d")


def tic_row(tic):
    from astroquery.mast import Catalogs
    c = Catalogs.query_object(f"TIC {tic}", catalog="TIC", radius=0.001)[0]
    return c


def stitched_curve(tic):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC",
                              exptime=120)
    ts, fs = [], []
    for i in range(len(sr)):
        try:
            lc = sr[i].download(download_dir=CACHE).remove_nans()
            f = np.asarray(lc.flux.value, dtype=float)
            ts.append(np.asarray(lc.time.value, dtype=float))
            fs.append(f / np.median(f))
        except Exception:
            continue
    return np.concatenate(ts), np.concatenate(fs)


def fold_epoch(t, f, period):
    """Time of primary minimum. Seed on the single deepest point, then refine
    to the centroid of the deep points within a narrow window around it in
    phase. The old version took the deepest 1% of ALL points — with a
    secondary eclipse (or 37 sectors of junk dips) inside that percentile the
    median time was contaminated: TIC 220481009's epoch came out 2.3 h late
    (BLS transit_time had it right). Caught via the round-1 resubmission zoom."""
    t0 = float(t[np.argmin(f)])
    ph = ((t - t0) / period + 0.5) % 1.0 - 0.5  # primary near phase 0
    near = np.abs(ph) < 0.02
    lo = np.percentile(f[near], 3.0)
    deep = near & (f <= lo)
    if deep.sum() >= 5:
        t0 += float(np.median(ph[deep])) * period
    # bring epoch to the first cycle at/after the data start
    n = np.floor((t0 - t.min()) / period)
    return t0 - n * period


def gaia_v(g, bp_rp):
    """Johnson V from Gaia DR3 G and BP-RP — the polynomial behind the VSX
    FAQ transformation spreadsheet (Riello et al. 2021; valid -0.5<BP-RP<5.0,
    sigma ~0.03). Moderation's preferred source for the Maximum field.
    Verified against the moderator's own number: G=12.3288, BP-RP=0.8472
    -> V=12.49 (TIC 220481009, VSX moderation feedback, 2026-08)."""
    if g is None or bp_rp is None:
        return None
    x = float(bp_rp)
    if not (-0.5 < x < 5.0):
        return None
    g_minus_v = -0.02704 + 0.01424 * x - 0.2156 * x ** 2 + 0.01426 * x ** 3
    return float(g) - g_minus_v


def star_facts_row(tic):
    """(g_mag, bp_rp) from the enrichment table (real Gaia DR3 query), or None."""
    import sqlite3
    db = sqlite3.connect(DB_PATH, timeout=60)
    try:
        return db.execute("SELECT g_mag, bp_rp FROM star_facts WHERE tic=?",
                          (int(tic),)).fetchone()
    except sqlite3.OperationalError:
        return None  # star_facts not populated yet (enrich never ran)
    finally:
        db.close()


def max_mag(tic, row):
    """Mean magnitude for the VSX Maximum field, 2 decimals (never 3 — bounced).
    VSX-preferred: Gaia DR3 G transformed to V. Falls back to TIC V/G/Tmag."""
    sf = star_facts_row(tic)
    if sf and sf[0] is not None and sf[1] is not None:
        v = gaia_v(sf[0], sf[1])
        if v is not None:
            return "V", round(v, 2), "Gaia DR3 G + (BP-RP) -> V (VSX FAQ relation)"
    for band, key in (("V", "Vmag"), ("G", "GAIAmag"), ("TESS", "Tmag")):
        m = row.get(key)
        if m is not None and np.isfinite(m):
            return band, round(float(m), 2), f"TIC catalog {band}"
    return None, None, None


def eclipse_geometry(t, f, period, epoch):
    """Measure primary duration and the secondary eclipse from the folded,
    binned curve. Duration = contact-to-contact width of the dip at phase 0,
    as % of period (moderation: it is measurable, never 'Uncertain').
    Secondary: deepest binned dip away from primary, if >5 sigma.
    Returns dict(duration_pct, min2_amp, min2_phase) — entries None if absent."""
    from scipy.stats import binned_statistic
    ph = (((t - epoch) / period) % 1.0)
    ph[ph > 0.5] -= 1.0
    nb = 2000
    med, edges, _ = binned_statistic(ph, f, statistic="median",
                                     bins=nb, range=(-0.5, 0.5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(med)
    base = np.nanmedian(med[ok])
    sigma = 1.4826 * np.nanmedian(np.abs(med[ok] - base))
    thr = base - max(3.0 * sigma, 1e-4)
    below = ok & (med < thr)
    # primary: contiguous below-threshold run containing phase 0 (wraps)
    i0 = np.argmin(np.abs(centers))
    dur_pct = None
    if below[i0] or below[(i0 - 1) % nb] or below[(i0 + 1) % nb]:
        lo = i0
        while below[(lo - 1) % nb]:
            lo = (lo - 1) % nb
            if lo == i0:
                break
        hi = i0
        while below[(hi + 1) % nb]:
            hi = (hi + 1) % nb
            if hi == i0:
                break
        width = ((hi - lo) % nb + 1) / nb
        dur_pct = width * 100.0
        prim = np.zeros(nb, bool)
        j = lo
        while True:
            prim[j] = True
            if j == hi:
                break
            j = (j + 1) % nb
    else:
        prim = np.abs(centers) < 0.02
    # secondary: deepest remaining bin, needs >5 sigma to be believed
    rest = ok & ~prim
    min2_amp = min2_phase = None
    if rest.any():
        j = np.nanargmin(np.where(rest, med, np.nan))
        if base - med[j] > 5.0 * sigma:
            min2_amp = -2.5 * np.log10(max(1e-6, med[j] / base))
            min2_phase = float(centers[j] % 1.0)
    return {"duration_pct": dur_pct, "min2_amp": min2_amp,
            "min2_phase": min2_phase}


def bjd_tdb_to_hjd_utc(bjd_tdb, ra, dec):
    """Convert a BJD_TDB epoch to HJD_UTC, the system the VSX epoch field uses.

    The two differ by far more than the barycentric/heliocentric light-time
    gap (a few seconds): the TDB-UTC offset is ~69 s (32.184 s + leap
    seconds), ~0.0008 d, which shows up in the 4th decimal. Method: undo the
    barycentric light-time to get the geocentric TDB time, add the
    heliocentric light-time, express the result in UTC. Uses astropy's
    built-in ephemeris (no network)."""
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    geocenter = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
    t_bary = Time(bjd_tdb, format="jd", scale="tdb", location=geocenter)
    t_geo = t_bary
    for _ in range(3):  # light-time depends weakly on t; converges at once
        t_geo = t_bary - t_geo.light_travel_time(coord, kind="barycentric")
    t_helio = t_geo + t_geo.light_travel_time(coord, kind="heliocentric")
    return float(t_helio.utc.jd)


def refine_period(t, f, period):
    """Fine BLS around the hunt-tier period before anything folds. A BLS period
    from a coarse grid is only good to ~1e-4 relative; coherent folding over a
    stitched baseline T needs dP < 0.02 P^2/T. A period error of ~3e-4 d on
    a ~2 d signal is enough to smear dozens of stitched sectors into a band,
    and percentile statistics on a smeared fold read the band's lower edge as
    a much deeper eclipse. Refinement is necessary but not sufficient: a
    refined fold can still be incoherent (the signal may not be a phase-stable
    eclipse at all), which is what fold_coherence checks next. Returns the
    refined period (or the input unchanged when the grid can't improve on it)."""
    from astropy.timeseries import BoxLeastSquares
    order = np.argsort(t)
    ts, fs = t[order], f[order]
    T = ts.max() - ts.min()
    if T <= 2 * period:
        return period
    # 30-min binning: eclipses stay resolved, BLS cost drops ~15x
    edges = np.arange(ts.min(), ts.max() + 0.0209, 0.0208333)
    idx = np.digitize(ts, edges) - 1
    cnt = np.bincount(idx, minlength=edges.size - 1)
    sm = np.bincount(idx, weights=fs, minlength=edges.size - 1)
    ok = cnt >= 3
    tb = (0.5 * (edges[:-1] + edges[1:]))[ok]
    fb = sm[ok] / cnt[ok]
    span = 1e-3 * period
    step = max(min(0.004 * period * period / T, span / 10.0), 1e-6)
    periods = np.arange(period - span, period + span, step)
    durations = np.clip(np.array([0.005, 0.01, 0.02]) * period, 0.02, 0.5 * period)
    res = BoxLeastSquares(tb, fb).power(periods, durations, method="fast")
    best = float(res.period[np.argmax(res.power)])
    if abs(best - period) > step:
        print(f"  period refined: {period:.6f} -> {best:.6f} d "
              f"(shift {abs(best-period)*86400:.0f}s)")
    return best


def fold_coherence(t, f, period, epoch):
    """Gate: is the fold coherent enough to derive numbers from? Compares the
    point-percentile depth (what measured_depth reports) to the binned-median
    depth at primary. A real folded eclipse gives ratio ~1; a smeared fold
    gives percentile >> binned. In the case that motivated this gate, a fold
    read ~20% deep by percentile but ~2.4% by binned median (ratio ~0.12) and
    that artifact went into a live VSX submission; the object later proved to
    be a previously published dipper whose dips are not phase-stable, i.e. not
    an eclipsing binary at any period. Statistics on a fold inherit the fold's
    validity silently; this makes the check explicit. Returns
    (ok, diagnostics_dict)."""
    from scipy.stats import binned_statistic
    ph = (((t - epoch) / period) % 1.0)
    ph[ph > 0.5] -= 1.0
    med, edges, _ = binned_statistic(ph, f, statistic="median",
                                     bins=2000, range=(-0.5, 0.5))
    c = 0.5 * (edges[:-1] + edges[1:])
    okb = np.isfinite(med)
    base = float(np.nanmedian(med[okb]))
    sigma = float(1.4826 * np.nanmedian(np.abs(med[okb] - base)))
    near = okb & (np.abs(c) < 0.02)
    binned_depth = base - float(np.nanmin(med[near])) if near.any() else 0.0
    pnear = np.abs(ph) < 0.03
    pctl_depth = (1.0 - float(np.percentile(f[pnear], 1.0))) if pnear.sum() >= 5 else 0.0
    ratio = binned_depth / max(pctl_depth, 1e-9)
    snr = binned_depth / max(sigma, 1e-9)
    diag = {"binned_depth": binned_depth, "pctl_depth": pctl_depth,
            "smear_ratio": ratio, "snr": snr}
    return (snr >= 5.0 and ratio >= 0.5), diag


def phase_plot(tic, t, f, period, epoch, out, tmag=None):
    """VSX-style phase plot, per VSX moderation feedback (2026-08):
    magnitudes not fluxes (they must support the table), brighter at TOP,
    TWO cycles (phase -0.5..1.5, secondary displayed mid-frame per the FAQ's
    EB convention), phase 0 = the table epoch, plain white background."""
    ph = (((t - epoch) / period) % 1.0)
    anchor = float(tmag) if tmag is not None and np.isfinite(tmag) else 0.0
    mag = anchor - 2.5 * np.log10(np.clip(f, 1e-3, None))
    p = np.concatenate([ph - 1.0, ph, ph + 1.0])
    m = np.tile(mag, 3)
    keep = (p >= -0.5) & (p <= 1.5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(p[keep], m[keep], ".", ms=1.2, color="black", alpha=0.35)
    ax.set_xlim(-0.5, 1.5)
    ax.invert_yaxis()  # magnitudes: brighter at top
    ax.set_xlabel("Phase")
    ax.set_ylabel("TESS magnitude")
    ax.set_title(f"TIC {tic} — TESS, P = {period:.6g} d, "
                 f"E = {epoch + BTJD_OFFSET:.4f}")
    ax.grid(alpha=0.2, lw=0.5)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def neighbor_budget(ra, dec, tmag, radius_arcsec=120.0):
    """Flux-budget blending argument over TESS's ~120" aperture.
    Returns (verdict_str, brightest_neighbor_dict, max_single_dip_frac).
    A single neighbor can dim the blended aperture by at most its own flux
    fraction; if that ceiling is below the observed depth, no neighbor can
    fake the eclipse."""
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c0 = SkyCoord(ra * u.deg, dec * u.deg)
    nb = Catalogs.query_region(f"{ra} {dec}", radius=radius_arcsec / 3600.0,
                               catalog="TIC")
    f_tot, brightest = 1.0, None
    for r in nb:
        try:
            tm = float(r["Tmag"])
            sep = c0.separation(SkyCoord(float(r["ra"]) * u.deg,
                                         float(r["dec"]) * u.deg)).arcsec
        except Exception:
            continue
        if sep < 0.5 or not np.isfinite(tm):
            continue  # the target itself / junk
        fr = 10 ** (-0.4 * (tm - tmag))  # neighbor flux / target flux
        f_tot += fr
        if brightest is None or fr > brightest["frac"]:
            brightest = {"tic": str(r["ID"]), "sep": round(sep, 1),
                         "tmag": round(tm, 2), "dT": round(tm - tmag, 2),
                         "frac": fr}
    if brightest is None:
        return "no catalogued neighbor within aperture", None, 0.0
    max_dip = brightest["frac"] / f_tot  # deepest dip a single neighbor allows
    return (f"brightest contaminant TIC {brightest['tic']} at "
            f"{brightest['sep']}\" is {brightest['dT']:+.2f} mag "
            f"({brightest['frac']*100:.0f}% of target flux); a single neighbor "
            f"can remove at most {max_dip*100:.0f}% of aperture flux"), \
           brightest, max_dip


def difference_imaging(tic, ra, dec, dossier, outdir, max_sectors=3):
    """TESS difference imaging: (out-of-eclipse - in-eclipse) image locates
    the eclipsing source. Writes blending.png. Returns (verdict, median_off_px).
    On-target => the dip is the target's, not a blended neighbor's."""
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    tgt = SkyCoord(ra * u.deg, dec * u.deg)
    # rank sectors by deep_dip score, take the deepest few
    secs = sorted(dossier.get("sectors", []),
                  key=lambda s: -(s.get("scores", {}).get("deep_dip", 0)))
    secs = [s for s in secs if s.get("events")][:max_sectors]
    panels, offs = [], []
    for s in secs:
        sector = s["sector"]
        ev = s["events"][0]
        t0 = 0.5 * (ev["t_start"] + ev["t_end"])
        try:
            sr = lk.search_targetpixelfile(f"TIC {tic}", sector=sector,
                                           author="SPOC")
            if len(sr) == 0:
                continue
            tpf = sr.download(quality_bitmask="hard", download_dir=CACHE)
        except Exception:
            continue
        t, flux = tpf.time.value, tpf.flux.value
        good = np.all(np.isfinite(flux), axis=(1, 2))
        inecl = good & (np.abs(t - t0) < 0.07)
        outecl = good & (np.abs(t - t0) > 0.30) & (np.abs(t - t0) < 2.0)
        if inecl.sum() < 3 or outecl.sum() < 5:
            continue
        diff = np.nanmean(flux[outecl], axis=0) - np.nanmean(flux[inecl], axis=0)
        diff[diff < 0] = 0
        ny, nx = diff.shape
        ys, xs = np.mgrid[0:ny, 0:nx]
        tot = diff.sum()
        if tot <= 0:
            continue
        cx, cy = (xs * diff).sum() / tot, (ys * diff).sum() / tot
        try:
            px, py = tpf.wcs.world_to_pixel(tgt)
            px, py = float(px), float(py)
        except Exception:
            px, py = nx / 2 - 0.5, ny / 2 - 0.5
        offs.append(np.hypot(cx - px, cy - py))
        panels.append((sector, diff, cx, cy, px, py))
    if not panels:
        return None, None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    fig.patch.set_facecolor(DARK)
    if len(panels) == 1:
        axes = [axes]
    for ax, (sector, diff, cx, cy, px, py) in zip(axes, panels):
        ax.imshow(diff, origin="lower", cmap="magma")
        ax.scatter([px], [py], s=160, facecolors="none", edgecolors="#5ad1ff",
                   linewidths=1.5, label="target (WCS)")
        ax.scatter([cx], [cy], marker="x", s=90, color="#5aff8a",
                   label="lost-light centroid")
        ax.set_title(f"s{sector}: Δimg", color="#d8d8df", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=6, loc="upper left", framealpha=0.3)
    fig.suptitle(f"TIC {tic} — eclipse difference imaging (on-target test)",
                 color="#d8d8df", fontsize=10)
    fig.savefig(os.path.join(outdir, "blending.png"), dpi=120,
                bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    med = float(np.median(offs))
    verdict = ("ON-TARGET (not a blend)" if med < 1.0
               else "OFFSET <2.5px — inspect neighbor" if med < 2.5
               else "LIKELY BLEND on a neighbor")
    return verdict, med


def finder_chart(tic, ra, dec, out, fov_arcmin=8.0):
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u
    pos = SkyCoord(ra, dec, unit="deg")
    imgs = SkyView.get_images(position=pos, survey=["DSS2 Red"],
                              radius=fov_arcmin * u.arcmin, pixels=400)
    hdu = imgs[0][0]
    wcs = WCS(hdu.header)
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection=wcs)
    data = hdu.data
    vmin, vmax = np.percentile(data, [5, 99.5])
    ax.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    # target marker (gap so the star isn't covered)
    px, py = wcs.world_to_pixel(pos)
    ax.scatter([px], [py], s=400, facecolors="none", edgecolors="#ff5a5a",
               linewidths=1.5)
    ax.set_title(f"TIC {tic}  ({fov_arcmin:.0f}' FOV, DSS2 Red)", fontsize=10)
    ax.coords[0].set_axislabel("RA"); ax.coords[1].set_axislabel("Dec")
    ax.coords.grid(color="#5a5a66", alpha=0.4, ls=":")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# TESS mission paper — the data-source reference for every Magpie find.
# Exact reference name+bibcode the VSX FAQ prescribes for TESS data — the
# 2015 JATIS bibcode with a free-form name was bounced by moderation (2026-08).
TESS_REF_NAME = "Ricker, G. R.; et al., 2014, Transiting Exoplanet Survey Satellite (TESS)"
TESS_REF_BIBCODE = "2014SPIE.9143E..20R"

# How max_mag's band code maps to the VSX passband control.
VSX_PASSBAND = {"V": "V (Johnson)", "G": "G (Gaia)", "TESS": "TESS (custom passband)"}


def to_sexagesimal(ra, dec):
    """Decimal degrees -> VSX sexagesimal 'HH MM SS.ss +DD MM SS.s' (J2000).
    Sexagesimal is the safer paste format: the decimal position field rejects a
    Unicode minus (U+2212), which is trivially easy to copy by accident."""
    rh = ra / 15.0
    h = int(rh); m = int((rh - h) * 60); s = ((rh - h) * 60 - m) * 60
    sign = "-" if dec < 0 else "+"; a = abs(dec)
    dd = int(a); mm = int((a - dd) * 60); ss = ((a - dd) * 60 - mm) * 60
    return f"{h:02d} {m:02d} {s:05.2f} {sign}{dd:02d} {mm:02d} {ss:04.1f}"


# eclipse duration now comes from eclipse_geometry() — measured on the folded
# curve, not from per-sector detector event windows (which pad the true width).


def sector_range_str(d):
    """Compress sector list to runs: 87,88,89,94,95,96 -> '87-89, 94-96'."""
    secs = sorted({s.get("sector") for s in d.get("sectors", []) if s.get("sector")})
    if not secs:
        return f"{d.get('n_sectors', '?')} sectors"
    runs, start, prev = [], secs[0], secs[0]
    for x in secs[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev)); start = prev = x
    runs.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def build_remarks(min2_amp=None, min2_phase=None):
    """VSX Remarks — ONLY what no form field can hold. Moderation (2026-08)
    bounced the old methodology paragraph in full: no discovery
    narrative, no blending defense, no justification — 'The information is
    already in the table. A phase plot is enough.' Admissible per the manual
    (VI.b): magnitude at secondary minimum, and for an eccentric eclipser the
    secondary's placement."""
    if min2_amp is None:
        return ""
    r = f"Min II amplitude {min2_amp:.2f} TESS."
    if min2_phase is not None and abs(min2_phase - 0.5) > 0.02:
        r += f" Min II at phase {min2_phase:.2f}."
    return r


def make_packet(tic, light=False):
    """light=True rebuilds phase.png + fields.txt only (still needs the LC),
    reusing existing finder/blending artifacts — the resubmission path when
    only the plot and form values changed."""
    p = os.path.join(DOSSIER_DIR, f"tic{tic}.json")
    if not os.path.exists(p):
        raise SystemExit(f"no dossier for TIC {tic} — run Tier-4 first")
    d = json.loads(open(p).read())
    pr = d.get("period_resolution") or {}
    bls = d.get("bls") or {}
    period = d.get("period_d") or pr.get("period_d") or bls.get("best_period_d")
    depth = d.get("depth_measured") or bls.get("depth")
    mech = pr.get("mechanism", "eclipsing binary")
    if not period:
        raise SystemExit(f"TIC {tic} has no period — not submittable")

    outdir = os.path.join(PACKET_DIR, f"tic{tic}")
    os.makedirs(outdir, exist_ok=True)

    row = tic_row(tic)
    ra, dec = float(row["ra"]), float(row["dec"])
    t, f = stitched_curve(tic)
    period = refine_period(t, f, period)
    epoch = fold_epoch(t, f, period)
    tmag = float(row["Tmag"]) if np.isfinite(row.get("Tmag", np.nan)) else None

    # gate before deriving ANYTHING from the fold — an incoherent fold returns
    # confident artifacts (depth, duration, epoch all garbage), and one of
    # those went into a live VSX submission. Refuse loudly instead.
    coh_ok, coh = fold_coherence(t, f, period, epoch)
    if not coh_ok:
        phase_plot(tic, t, f, period, epoch,
                   os.path.join(outdir, "fold-diagnostic.png"), tmag=tmag)
        msg = (f"REFUSED: fold not coherent at P={period:.6f} d — "
               f"binned depth {coh['binned_depth']*100:.2f}% vs percentile "
               f"{coh['pctl_depth']*100:.2f}% (smear ratio "
               f"{coh['smear_ratio']:.2f}, need >=0.5; SNR {coh['snr']:.1f}, "
               f"need >=5). Not an EA at this ephemeris, or period still "
               f"wrong. See fold-diagnostic.png; per-era folds discriminate "
               f"wrong-period from astrophysical drift (CPV/dipper).")
        with open(os.path.join(outdir, "REFUSED.txt"), "w") as fh:
            fh.write(msg + "\n")
        # clear stale submission artifacts so nobody pastes from a leftover
        for stale in ("fields.txt", "phase.png"):
            sp = os.path.join(outdir, stale)
            if os.path.exists(sp):
                os.remove(sp)
        print(f"  {msg}")
        return None

    # depth from the folded curve (BLS box depth understates V-eclipses)
    fold_depth = measured_depth(t, f, period, epoch)
    depth = fold_depth if fold_depth else depth
    epoch_bjd = epoch + BTJD_OFFSET
    epoch_hjd = bjd_tdb_to_hjd_utc(epoch_bjd, ra, dec)
    band, mmax, mmax_prov = max_mag(tic, row)
    geom = eclipse_geometry(t, f, period, epoch)

    for stale in ("REFUSED.txt", "fold-diagnostic.png"):
        sp = os.path.join(outdir, stale)
        if os.path.exists(sp):
            os.remove(sp)
    phase_plot(tic, t, f, period, epoch, os.path.join(outdir, "phase.png"),
               tmag=tmag)
    print(f"  phase plot written (mags, 2 cycles)")
    if geom["duration_pct"] is not None:
        print(f"  eclipse duration: {geom['duration_pct']:.1f}% of period")
    if geom["min2_amp"] is not None:
        print(f"  Min II: {geom['min2_amp']:.2f} mag at phase "
              f"{geom['min2_phase']:.2f}")

    if light:
        finder_ok = os.path.exists(os.path.join(outdir, "finder.png"))
        blending_ok = os.path.exists(os.path.join(outdir, "blending.png"))
        budget_str, brightest, max_dip = (None, None, None)
        # reuse the verdict the full run MEASURED; never assume one
        diff_verdict, diff_off = None, None
        bj = os.path.join(outdir, "blending.json")
        if blending_ok and os.path.exists(bj):
            try:
                saved = json.loads(open(bj).read())
                diff_verdict = saved.get("verdict")
                diff_off = saved.get("median_offset_px")
            except (OSError, ValueError):
                pass
        if blending_ok and not diff_verdict:
            print("  blending.png has no saved verdict (blending.json) — "
                  "rerun the full packet")
            blending_ok = False
    else:
        try:
            finder_chart(tic, ra, dec, os.path.join(outdir, "finder.png"))
            print(f"  finder chart written")
            finder_ok = True
        except Exception as e:
            print(f"  finder chart FAILED ({type(e).__name__}: {e})")
            finder_ok = False

        # --- blending assessment (VSX V.e/V.f: TESS ~120" aperture) ---
        budget_str, brightest, max_dip = (None, None, None)
        if tmag is not None:
            try:
                budget_str, brightest, max_dip = neighbor_budget(ra, dec, tmag)
                print(f"  neighbor budget: {budget_str}")
            except Exception as e:
                print(f"  neighbor budget FAILED ({type(e).__name__}: {e})")
        diff_verdict, diff_off = (None, None)
        bj = os.path.join(outdir, "blending.json")
        if os.path.exists(bj):
            os.remove(bj)
        try:
            diff_verdict, diff_off = difference_imaging(tic, ra, dec, d, outdir)
            if diff_verdict:
                print(f"  difference imaging: {diff_verdict} "
                      f"(median offset {diff_off:.2f}px)")
                with open(bj, "w") as fh:
                    json.dump({"verdict": diff_verdict,
                               "median_offset_px": diff_off}, fh)
                blending_ok = True
            else:
                print("  difference imaging: no usable sectors")
                blending_ok = False
        except Exception as e:
            print(f"  difference imaging FAILED ({type(e).__name__}: {e})")
            blending_ok = False
    # only an ON-TARGET verdict may be presented as on-target evidence
    on_target = bool(blending_ok and diff_verdict
                     and diff_verdict.startswith("ON-TARGET"))
    # budget verdict: can any single neighbor fake the observed depth?
    budget_clears = (max_dip is not None and depth is not None
                     and max_dip < depth)

    def _num(v):
        try:
            if v is None or np.ma.is_masked(v):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None
    def _gaia_id(v):  # 19-digit Gaia source id — never route through float
        try:
            if v is None or np.ma.is_masked(v):
                return None
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None
    gaia = _gaia_id(row.get("GAIA"))
    plx = _num(row.get("plx"))

    # cross-identifications (moderation: submissions without them bounce).
    # Faint-star set per the VSX FAQ: 2MASS (with the J prefix), UCAC4, Gaia.
    def _sval(key):
        v = row.get(key)
        if v is None or np.ma.is_masked(v):
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("nan", "--", "none") else None
    cross_ids = []
    if _sval("TWOMASS"):
        cross_ids.append(f"2MASS J{_sval('TWOMASS')}")
    if _sval("UCAC"):
        cross_ids.append(f"UCAC4 {_sval('UCAC')}")
    if gaia is not None:
        cross_ids.append(f"Gaia DR3 {gaia}")

    # derived submission values
    sexag = to_sexagesimal(ra, dec)
    amp_mag = (-2.5 * np.log10(max(1e-6, 1.0 - depth))) if depth else None
    band_label = VSX_PASSBAND.get(band, band or "?")
    dur_pct = geom["duration_pct"]
    ecc = "eccentric" in mech
    remarks = build_remarks(geom["min2_amp"], geom["min2_phase"])
    sec_str = sector_range_str(d)

    # fields.txt mirrors the VSX New Star Wizard page-by-page, paste-ready.
    fields = [
        "=== VSX New Star Wizard — paste-ready fields (verify every line) ===",
        f"TIC {tic}   |   file at vsx.aavso.org  (Submit > New Star Wizard)",
        "",
    ]
    if blending_ok and not on_target:
        fields += [
            f"*** BLEND SUSPECTED — DO NOT FILE. Difference imaging: {diff_verdict}. ***",
            "*** The eclipse may belong to a neighbor; resolve before any submission. ***",
            "",
        ]
    fields += [
        "[1] Novice discovery : do NOT check. That flag is for discoveries from",
        "                       your OWN observations only (manual/video, 2018+2021);",
        "                       we use TESS survey data. If it was checked in the",
        "                       original submission, UNcheck it while editing.",
        "",
        "[2] Position (J2000) — PASTE THE SEXAGESIMAL LINE",
        f"    Sexagesimal      : {sexag}",
        f"    Decimal (deg)    : {ra:.6f} {dec:+.6f}",
        ("    Provenance       : Gaia DR3 {0}{1}".format(
            gaia, f" (plx {plx:.2f} mas)" if plx is not None else "")
         if gaia is not None else "    Provenance       : TIC position"),
        "",
        f"[3] Catalogue name   : TIC {tic}   (primary)",
        "    Other names — ONE PER LINE in the textarea (wizard requirement);",
        "    cross-IDs are REQUIRED (omitting them bounced round 1):",
    ] + ([f"        {cid}" for cid in cross_ids] if cross_ids
         else ["        *** none resolved — look up in VizieR before filing ***"]) + [
        "",
        "[4] Variable type    : EA   (detached eclipsing binary"
        + ("; eccentric)" if ecc else ")"),
        "    Spectral type    : leave blank (no spectrum; TIC Teff is a single-star "
        "photometric fit)",
        "",
        "[5] Magnitude range  — mixed-band per VSX IV.c"
        if (band and band != "TESS") else "[5] Magnitude range",
    ]
    if band and amp_mag is not None:
        fields += [
            f"    Maximum mag      : {mmax:.2f}   passband: {band_label}",
            f"                       ({mmax_prov}; 2 decimals max — 3 bounced)",
            f"    Minimum          : {amp_mag:.2f}   -> CHECK \"Minimum is amplitude\""
            "; custom passband: TESS",
        ]
    else:
        fields.append("    *** NEEDS CALIBRATED MAG — none in TIC ***")
    fields += [
        "",
        f"[6] Period (d)       : {period:.6g}",
        "                       (significant figures only — 6 decimals bounced "
        "for a 52 d period)",
        f"    Epoch (HJD)      : {epoch_hjd:.4f}      <- paste THIS bare number only",
        f"                       (HJD_UTC, converted from BJD_TDB {epoch_bjd:.4f};",
        "                        the ~69 s TDB-UTC offset matters at 4 decimals)",
        f"                       (do NOT paste \"= BTJD {epoch:.4f} + 2457000\" — VSX "
        "evaluates no arithmetic)",
        (f"    Rise/ecl duration: {dur_pct:.1f} %   measured from the folded curve "
         "— do NOT check Uncertain"
         if dur_pct is not None else "    Rise/ecl duration: leave blank"),
        "    Outburst year    : leave blank",
        "",
        "[7] Supporting files (>=1 REQUIRED for unpublished originals; 1 MB max each)",
        "    EVERY file needs a TITLE and a brief DESCRIPTION (both wizard fields).",
        "    File 1 phase.png",
        "        title       : Phase plot, TESS magnitudes, 2 cycles",
        f"        description : Phase plot with TESS SPOC 2-min data, sectors "
        f"{sec_str}",
    ]
    if on_target:
        fields += [
            "    File 2 blending.png",
            "        title       : Difference imaging - on-target confirmation",
            "        description : TESS difference image (out-of-eclipse minus "
            "in-eclipse), eclipse source coincides with the target",
        ]
    elif blending_ok:
        fields.append(f"    File 2 *** blend suspected ({diff_verdict}) — "
                      "do not file ***")
    else:
        fields.append("    File 2 *** blending.png not generated — rerun packet ***")
    fields.append("    (finder.png optional — add via draft-edit if the field is crowded)"
                  if finder_ok else "    (finder.png missing — regenerate before filing)")
    fields += [
        "",
        f"[8] Discoverer       : {DISCOVERER}   (your name, NOT the observer code)",
        "",
        "[9] References — TESS is the primary (exact FAQ name + bibcode):",
        f"    name             : {TESS_REF_NAME}",
        f"    bibcode          : {TESS_REF_BIBCODE}   (click \"Check bibcode\")",
        "    PLUS a second reference for the Gaia-derived V maximum: save as",
        "    DRAFT, reopen via My Submissions, and add (FAQ-prescribed):",
        "    name             : Gaia collaboration; et al., 2022, Gaia Data "
        "Release 3 (Gaia DR3) Part 1 Main source",
        "    bibcode          : 2022yCat.1355....0G",
        "",
        "[10] Remarks — ONLY the line below (if any). No methodology, no",
        "     justification, no discovery narrative: all of that bounced round 1.",
        "     (Min II phase is admissible for an eccentric eclipser — manual VI.b.)",
        (remarks if remarks else "     (leave the field empty)"),
        "",
        "[11] Revision comment (the form field, NOT Remarks) — state what changed",
        "     and its sources, e.g.:",
        "     \"V max from Gaia DR3 photometry (G, BP-RP transformed to V per the",
        "     VSX FAQ relation); epoch corrected to time of primary minimum;",
        "     eclipse duration measured from the folded curve; cross-IDs and",
        "     TESS reference format corrected.\"",
        "",
        "--- Blending evidence (our notes — NOT for the form; keep for moderator "
        "questions) ---",
    ]
    if budget_str:
        fields.append(f"    Flux budget       : {budget_str}.")
        if budget_clears:
            fields.append(f"                        depth {depth*100:.1f}% EXCEEDS that "
                          "ceiling -> no single neighbor can fake it.")
    if blending_ok:
        off_s = f"(centroid {diff_off:.2f}px from target; " if diff_off is not None \
            else "("
        fields.append(f"    Difference imaging: {diff_verdict} "
                      f"{off_s}see blending.png).")
    fields += [
        "",
        "--- Filing gotchas (learned the hard way) ---",
        "  - VSX login is SEPARATE from aavso.org — use your USERNAME, not email.",
        "  - On a VPN? VSX blacklists VPN exit IPs. Turn it off to log in.",
        "  - Type minus signs by hand (ASCII '-'); a pasted Unicode minus is rejected.",
        "  - Paste bare numbers only — VSX evaluates no arithmetic or annotations.",
        "  - Round 1 bounce list (VSX moderation): flux plots, 1-cycle plots,",
        "    faint-up axes, 3-decimal mags, 6-decimal periods, 'Uncertain' on a",
        "    measurable duration, missing cross-IDs, wrong TESS reference, untitled",
        "    files, methodology in Remarks. All fixed structurally in this sheet.",
        "  - Rate limits (policy page): 1 submission/day, 5 reviewed/week per user.",
        "    Space out the ready-candidate filings.",
        "  - 60-DAY CLOCK: a submission awaiting required modifications is DELETED",
        "    after two months. Resubmit bounced stars promptly.",
    ]
    with open(os.path.join(outdir, "fields.txt"), "w") as fh:
        fh.write("\n".join(fields) + "\n")
    print(f"  fields.txt written → {outdir}")
    return outdir


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tics", type=int, nargs="*")
    ap.add_argument("--sector", type=int,
                    help="generate packets for all submittable objects in a sector")
    ap.add_argument("--light", action="store_true",
                    help="rebuild phase.png + fields.txt only (reuse finder/blending)")
    args = ap.parse_args()
    tics = list(args.tics)
    if args.sector:
        import sqlite3
        db = sqlite3.connect(DB_PATH, timeout=60)
        tics += [t for (t,) in db.execute(
            "SELECT tic FROM catalog_check WHERE sector=? AND submittable=1",
            (args.sector,)).fetchall()]
        db.close()
    if not tics:
        print("nothing to do")
        return
    for tic in tics:
        print(f"TIC {tic}: building packet...")
        make_packet(tic, light=args.light)


if __name__ == "__main__":
    main()
