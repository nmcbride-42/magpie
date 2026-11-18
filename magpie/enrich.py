"""Star facts — enrich a candidate with its astrophysical context.

For a TIC, pull Gaia DR3 (distance, temperature, radius, colour, proper
motion, the RUWE binary-wobble tell) and derive the human-facing bits
(light-years, spectral type, constellation, a colour swatch). Cached in the
`star_facts` table so readers never have to hit Gaia again, and used by the
packet stage for the Gaia-derived V magnitude.

  python -m magpie.enrich --tic <TIC>
  python -m magpie.enrich --all-submittable     # every submittable + filed find
"""
import argparse
import math
import sqlite3
import warnings

from magpie.config import DB_PATH as DB

warnings.filterwarnings("ignore")

SCHEMA = """
CREATE TABLE IF NOT EXISTS star_facts (
    tic INTEGER PRIMARY KEY,
    ra REAL, dec REAL,
    dist_ly REAL, dist_pc REAL, dist_err_pct REAL,
    g_mag REAL, bp_rp REAL,
    teff REAL, radius_rsun REAL, lum_lsun REAL,
    ruwe REAL, pm_total REAL, rv REAL,
    spectral_type TEXT, color_hex TEXT, constellation TEXT,
    enriched_at TEXT DEFAULT (datetime('now'))
);
"""


def spectral_type(teff):
    if not teff:
        return None
    for lo, t in [(30000, "O"), (10000, "B"), (7500, "A"), (6000, "F"),
                  (5200, "G"), (3700, "K"), (0, "M")]:
        if teff >= lo:
            return t
    return None


def color_hex(teff):
    """Rough star colour from temperature — for a UI swatch, not science."""
    if not teff:
        return "#cccccc"
    stops = [(2500, "#ffb56c"), (3700, "#ffd2a1"), (5200, "#fff4e8"),
             (6000, "#fffaf3"), (7500, "#ffffff"), (10000, "#cad7ff"),
             (30000, "#9bb0ff")]
    teff = max(2500, min(30000, teff))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= teff <= t1:
            f = (teff - t0) / (t1 - t0)
            rgb = [round(int(c0[j:j+2], 16) + f * (int(c1[j:j+2], 16) - int(c0[j:j+2], 16)))
                   for j in (1, 3, 5)]
            return "#%02x%02x%02x" % tuple(rgb)
    return stops[-1][1]


def constellation(ra, dec):
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        return SkyCoord(ra * u.deg, dec * u.deg).get_constellation()
    except Exception:
        return None


def coords_for(db, tic):
    r = db.execute("SELECT ra, dec FROM catalog_check WHERE tic=? AND ra IS NOT NULL "
                   "LIMIT 1", (tic,)).fetchone()
    if r:
        return float(r[0]), float(r[1])
    from astroquery.mast import Catalogs
    cat = Catalogs.query_object(f"TIC {tic}", catalog="TIC", radius=0.001)
    return float(cat[0]["ra"]), float(cat[0]["dec"])


def gaia_row(ra, dec):
    from astroquery.gaia import Gaia
    # radius_gspphot / lum_flame live in astrophysical_parameters in DR3, not
    # gaia_source — join on source_id to pull them alongside the core columns.
    q = (f"SELECT TOP 1 s.parallax, s.parallax_error, s.phot_g_mean_mag, "
         f"s.bp_rp, s.teff_gspphot, ap.radius_gspphot, ap.lum_flame, s.ruwe, "
         f"s.pmra, s.pmdec, s.radial_velocity "
         f"FROM gaiadr3.gaia_source s "
         f"LEFT JOIN gaiadr3.astrophysical_parameters ap "
         f"ON ap.source_id = s.source_id "
         f"WHERE 1=CONTAINS(POINT('ICRS',s.ra,s.dec),"
         f"CIRCLE('ICRS',{ra},{dec},0.0014)) ORDER BY s.phot_g_mean_mag ASC")
    t = Gaia.launch_job(q).get_results()
    return t[0] if len(t) else None


def fnum(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def enrich(tic):
    db = sqlite3.connect(DB, timeout=60)
    db.executescript(SCHEMA)
    db.commit()
    ra, dec = coords_for(db, tic)
    g = gaia_row(ra, dec)
    if g is None:
        print(f"TIC {tic}: no Gaia DR3 match — skipping")
        db.close()
        return
    plx = fnum(g["parallax"]); perr = fnum(g["parallax_error"])
    teff = fnum(g["teff_gspphot"])
    dist_pc = 1000.0 / plx if plx and plx > 0 else None
    dist_ly = dist_pc * 3.26156 if dist_pc else None
    pmra = fnum(g["pmra"]); pmdec = fnum(g["pmdec"])
    pm_total = math.hypot(pmra, pmdec) if pmra is not None and pmdec is not None else None
    row = (tic, ra, dec, dist_ly, dist_pc,
           (perr / plx * 100 if plx and plx > 0 and perr else None),
           fnum(g["phot_g_mean_mag"]), fnum(g["bp_rp"]),
           teff, fnum(g["radius_gspphot"]), fnum(g["lum_flame"]),
           fnum(g["ruwe"]), pm_total, fnum(g["radial_velocity"]),
           spectral_type(teff), color_hex(teff), constellation(ra, dec))
    db.execute(
        "INSERT OR REPLACE INTO star_facts "
        "(tic,ra,dec,dist_ly,dist_pc,dist_err_pct,g_mag,bp_rp,teff,radius_rsun,"
        "lum_lsun,ruwe,pm_total,rv,spectral_type,color_hex,constellation) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    db.commit()
    db.close()
    d = f"{dist_ly:,.0f} ly" if dist_ly else "distance unknown"
    print(f"TIC {tic}: {d}, {spectral_type(teff) or '?'}-type "
          f"{f'{teff:.0f}K' if teff else ''}, in {constellation(ra, dec)}")


def enrich_if_missing(tic):
    """Enrich only if not already cached — keeps auto-enrich cheap (no repeat
    Gaia call for a TIC we already have). Best-effort: never raises."""
    try:
        db = sqlite3.connect(DB, timeout=60)
        db.executescript(SCHEMA)
        db.commit()
        have = db.execute("SELECT 1 FROM star_facts WHERE tic=?", (tic,)).fetchone()
        db.close()
        if have:
            return
        enrich(tic)
    except Exception as e:
        print(f"  auto-enrich TIC {tic} skipped ({type(e).__name__}: {e})")


def all_submittable():
    db = sqlite3.connect(DB, timeout=60)
    tics = {t for (t,) in db.execute(
        "SELECT DISTINCT tic FROM catalog_check WHERE submittable=1").fetchall()}
    try:
        tics |= {t for (t,) in db.execute("SELECT tic FROM submitted").fetchall()}
    except sqlite3.OperationalError:
        pass
    db.close()
    return sorted(tics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int)
    ap.add_argument("--all-submittable", action="store_true")
    args = ap.parse_args()
    tics = [args.tic] if args.tic else (all_submittable() if args.all_submittable else [])
    if not tics:
        ap.error("need --tic N or --all-submittable")
    for tic in tics:
        try:
            enrich(tic)
        except Exception as e:
            print(f"TIC {tic}: enrich failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
