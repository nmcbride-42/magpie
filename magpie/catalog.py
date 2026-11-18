"""Tier 5 — catalog check for submittable discoveries.

For each Tier-4-resolved eclipsing binary in a sector, ask the two
authoritative variable-star catalogs whether it's already known:
  - VSX (AAVSO) — the submission target itself; absence == submittable
  - SIMBAD — broader "is this object typed as anything" cross-check

An object is SUBMITTABLE if Tier-4 resolved it to a period AND neither
catalog knows it as a variable. Writes coordinates, period, depth, and a
ready-to-paste VSX summary into the catalog_check table.
"""

import json
import os
import sqlite3
import urllib.parse
import urllib.request

from magpie.config import DB_PATH, DOSSIER_DIR, USER_AGENT

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_check (
    sector INTEGER NOT NULL,
    tic INTEGER NOT NULL,
    ra REAL, dec REAL,
    period_d REAL, depth REAL, mechanism TEXT,
    vsx_hits INTEGER, simbad_id TEXT, simbad_otype TEXT,
    tess_eb INTEGER, gaia_var INTEGER,
    submittable INTEGER NOT NULL DEFAULT 0,
    vsx_summary TEXT,
    checked_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (sector, tic)
);
CREATE TABLE IF NOT EXISTS submitted (
    tic INTEGER PRIMARY KEY,
    designation TEXT,            -- our internal/working name, if any
    submitted_at TEXT DEFAULT (datetime('now')),
    note TEXT,                   -- e.g. 'VSX, pending moderation'
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected
    vsx_name TEXT,               -- set by vsx_poll once VSX lists the object
    updated_at TEXT
);
"""

# Idempotent column adds for pre-existing DBs (the catalog_check table shipped
# before the professional-catalog gates existed; submission status tracking
# came later still).
MIGRATIONS = [
    "ALTER TABLE catalog_check ADD COLUMN tess_eb INTEGER",
    "ALTER TABLE catalog_check ADD COLUMN gaia_var INTEGER",
    "ALTER TABLE submitted ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'",
    "ALTER TABLE submitted ADD COLUMN vsx_name TEXT",
    "ALTER TABLE submitted ADD COLUMN updated_at TEXT",
]


def ensure_schema(db):
    """Create tables and apply idempotent migrations on an open connection."""
    db.executescript(SCHEMA)
    for m in MIGRATIONS:
        try:
            db.execute(m)
        except sqlite3.OperationalError:
            pass  # column already exists
    db.commit()

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
# SIMBAD otypes that mean "already known as a variable" — a hit here also
# blocks submittability, not just a VSX hit.
KNOWN_VAR_OTYPES = {
    "EB*", "EclBin", "Al*", "bL*", "WU*", "RR*", "RRLyr", "Ce*", "Cepheid",
    "dS*", "Pu*", "PulsV*", "Ro*", "RotV*", "BY*", "RS*", "Er*", "Fl*",
    "TT*", "Mi*", "LP*", "sr*", "SB*", "CV*", "V*", "Variable",
}


def submitted_tics():
    """TICs we've already filed (any sector). A multi-sector EB re-detected in
    a later sector must NOT re-flag as a fresh find — and a candidate pending
    VSX moderation is invisible to the VSX cone search (vsx_hits=0), so without
    this it would look 'uncatalogued' forever (TIC 220481009 was the case that
    surfaced this)."""
    db = sqlite3.connect(DB_PATH, timeout=60)
    ensure_schema(db)
    out = {t for (t,) in db.execute("SELECT tic FROM submitted").fetchall()}
    db.close()
    return out


def tic_coords(tic):
    from astroquery.mast import Catalogs
    cat = Catalogs.query_object(f"TIC {tic}", catalog="TIC", radius=0.001)
    return float(cat[0]["ra"]), float(cat[0]["dec"])


def vsx_count(ra, dec, radius_deg=0.0084):  # 30 arcsec
    q = urllib.parse.urlencode({"view": "api.list", "ra": ra, "dec": dec,
                                "radius": radius_deg, "format": "json"})
    req = urllib.request.Request(f"https://www.aavso.org/vsx/index.php?{q}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    if isinstance(data, list):
        return 0
    obj = data.get("VSXObjects")
    if not obj or isinstance(obj, list):
        return 0
    h = obj.get("VSXObject", [])
    return len([h] if isinstance(h, dict) else h)


def tess_eb_count(tic):
    """Rows in the professional TESS Eclipsing Binary catalog (Prsa+ 2022,
    VizieR J/ApJS/258/16) keyed by TIC. >0 means the pros already have it."""
    from astroquery.vizier import Vizier
    r = Vizier(row_limit=5).query_constraints(
        catalog="J/ApJS/258/16/tess-ebs", TIC=str(tic))
    return len(r[0]) if r else 0


def gaia_var_count(ra, dec, radius_deg=0.0014):  # ~5 arcsec
    """Gaia DR3 entries classified as eclipsing-binary variables near the
    position. >0 means Gaia already flagged it variable."""
    from astroquery.gaia import Gaia
    q = (f"SELECT TOP 5 source_id FROM gaiadr3.vari_eclipsing_binary v "
         f"JOIN gaiadr3.gaia_source g USING (source_id) "
         f"WHERE 1=CONTAINS(POINT('ICRS',g.ra,g.dec),"
         f"CIRCLE('ICRS',{ra},{dec},{radius_deg}))")
    return len(Gaia.launch_job(q).get_results())


def simbad_lookup(tic):
    adql = ("SELECT basic.main_id, basic.otype FROM ident "
            "JOIN basic ON ident.oidref = basic.oid "
            f"WHERE ident.id = 'TIC {tic}'")
    q = urllib.parse.urlencode({"request": "doQuery", "lang": "adql",
                                "format": "json", "query": adql})
    req = urllib.request.Request(f"{SIMBAD_TAP}?{q}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    rows = out.get("data") or []
    return (rows[0][0], rows[0][1]) if rows else (None, None)


def load_finalists(sector):
    """Every Tier-3.5 'interesting' candidate, resolved OR not, as
    (tic, period, depth, mechanism, resolved).

    Resolved EBs carry their period/depth/mechanism. UNRESOLVED finalists
    carry None — they still get the catalog cross-check (it's positional and
    needs no period), so the pipeline records whether each interesting-but-
    unresolved object is already catalogued. Without this, the unresolved
    bucket was a blind spot: we couldn't tell a genuinely-new anomaly from a
    long-known YSO/dipper without checking by hand (TIC 412308868, sector 100).
    submittable stays gated on a real period downstream, so unresolved
    finalists are checked and labelled but never marked submittable."""
    db = sqlite3.connect(DB_PATH, timeout=60)
    tics = [t for (t,) in db.execute(
        "SELECT tic FROM tier35 WHERE sector=? AND verdict='interesting'",
        (sector,)).fetchall()]
    db.close()
    out = []
    for tic in tics:
        period = depth = None
        mech = "unresolved"
        resolved = False
        p = os.path.join(DOSSIER_DIR, f"tic{tic}.json")
        if os.path.exists(p):
            d = json.loads(open(p).read())
            pr = d.get("period_resolution") or {}
            bls = d.get("bls") or {}
            # schema-tolerant: top-level flag (new dossiers) OR derive it from
            # fold / strong-BLS (older dossiers written before the flag existed)
            resolved = bool(d.get("resolved") or pr.get("resolved")
                            or (bool(bls) and (bls.get("power_ratio") or 0) > 50))
            if resolved:
                period = d.get("period_d") or pr.get("period_d") \
                    or bls.get("best_period_d")
                depth = d.get("depth_measured") or bls.get("depth")
                mech = pr.get("mechanism", "eclipsing binary")
        out.append((tic, period, depth, mech, resolved))
    return out


def vsx_blurb(tic, ra, dec, period, depth, mechanism):
    dpct = f"{depth*100:.1f}%" if depth else "?"
    return (f"TIC {tic}  RA={ra:.5f} Dec={dec:+.5f}  "
            f"type=EA (eclipsing)  period={period:.4f} d  "
            f"depth~{dpct}  source=TESS SPOC 2-min  "
            f"note={mechanism}, multi-sector eclipse timing")


def run(sector):
    import time
    os.makedirs(DOSSIER_DIR, exist_ok=True)
    # Phase 1 — schema/migrate, then close (don't hold a writer during the
    # slow catalog lookups, which can take minutes and would block sweeps).
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    ensure_schema(db)
    db.close()

    finalists = load_finalists(sector)
    already_filed = submitted_tics()
    n_res = sum(1 for f in finalists if f[4])
    print(f"catalog: {len(finalists)} interesting finalist(s) to check in "
          f"sector {sector} ({n_res} resolved EB, {len(finalists)-n_res} unresolved)")

    # Phase 2 — slow external catalog lookups, NO db connection held.
    rows, n_sub, n_new_unres = [], 0, 0
    for tic, period, depth, mech, is_resolved in finalists:
        try:
            ra, dec = tic_coords(tic)
            vh = vsx_count(ra, dec)
            sid, sot = simbad_lookup(tic)
            teb = tess_eb_count(tic)
            gv = gaia_var_count(ra, dec)
        except Exception as e:
            print(f"  TIC {tic}: check failed ({type(e).__name__}) — skip")
            continue
        filed = tic in already_filed
        known = (vh > 0 or sot in KNOWN_VAR_OTYPES or teb > 0 or gv > 0 or filed)
        # only a resolved EB with a period can be submitted; unresolved
        # finalists are recorded with their known/uncatalogued status so the
        # bucket is triageable, but never marked submittable.
        # Depth sanity: an eclipse can't block >100% of the light. A depth
        # >=1.0 is a detrending blowout / measurement artifact, not a real
        # eclipse — never submittable (sector 98 surfaced 156% and 719% "depths").
        physical_depth = depth is None or (0 < depth < 1.0)
        submittable = (not known) and period is not None and physical_depth
        blurb = vsx_blurb(tic, ra, dec, period, depth, mech) if submittable else None
        rows.append((sector, tic, ra, dec, period, depth, mech, vh, sid, sot,
                     teb, gv, int(submittable), blurb))
        n_sub += submittable
        why = []
        if filed: why.append("already-filed-by-us")
        if vh: why.append(f"vsx={vh}")
        if sot in KNOWN_VAR_OTYPES: why.append(f"simbad={sot}")
        if teb: why.append("TESS-EB-catalog")
        if gv: why.append("Gaia-var")
        if submittable:
            disp = "SUBMITTABLE"
        elif filed:
            disp = "already filed by us (re-detection in this sector)"
        elif is_resolved:
            disp = f"resolved but known ({', '.join(why) or sot})"
        elif known:
            disp = f"unresolved, already catalogued ({', '.join(why) or sot})"
        else:
            disp = f"unresolved + UNCATALOGUED ({sot or 'no SIMBAD'}) — needs more data"
            n_new_unres += 1
        print(f"  TIC {tic}: {disp}")

    # Phase 3 — one short retried write burst against any active sweep writers.
    for attempt in range(10):
        try:
            db = sqlite3.connect(DB_PATH, timeout=60)
            db.execute("BEGIN IMMEDIATE")
            db.executemany(
                "INSERT OR REPLACE INTO catalog_check "
                "(sector,tic,ra,dec,period_d,depth,mechanism,vsx_hits,simbad_id,"
                "simbad_otype,tess_eb,gaia_var,submittable,vsx_summary) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            db.commit()
            db.close()
            break
        except sqlite3.OperationalError:
            time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit("catalog: could not acquire write lock after 10 tries")
    print(f"catalog done: {len(finalists)} checked · {n_sub} submittable · "
          f"{n_new_unres} unresolved+uncatalogued (worth re-observation)")

    # Auto-enrich every new submittable with its stellar facts (distance,
    # temperature, etc.) so the star_facts table is populated with no manual
    # step. Best-effort and cached, so it only spends a Gaia call on
    # a TIC we don't already have. No DB writer held during the slow lookups.
    sub_tics = [r[1] for r in rows if r[12]]
    if sub_tics:
        try:
            from magpie.enrich import enrich_if_missing
            print(f"catalog: enriching {len(sub_tics)} submittable find(s)…")
            for t in sub_tics:
                enrich_if_missing(t)
        except Exception as e:
            print(f"catalog: auto-enrich skipped ({type(e).__name__}: {e})")


def mark_submitted(tic, note=""):
    """Record a TIC as filed so it stops re-flagging as a fresh find."""
    db = sqlite3.connect(DB_PATH, timeout=60)
    ensure_schema(db)
    db.execute("INSERT OR REPLACE INTO submitted (tic, note) VALUES (?,?)",
               (tic, note))
    db.commit()
    db.close()
    print(f"marked TIC {tic} submitted ({note})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int)
    ap.add_argument("--mark-submitted", type=int, metavar="TIC",
                    help="record a TIC as already filed (stops re-detection flags)")
    ap.add_argument("--note", default="", help="note for --mark-submitted")
    args = ap.parse_args()
    if args.mark_submitted:
        mark_submitted(args.mark_submitted, args.note)
    elif args.sector:
        run(args.sector)
    else:
        ap.error("need --sector or --mark-submitted")
