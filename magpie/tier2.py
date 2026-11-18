"""Tier 2 — known-object veto. Kill every Tier-1 survivor that's already
explained: TOIs/planet candidates (ExoFOP), catalogued variables and other
typed objects (SIMBAD). VSX coordinate match is a TODO (needs RA/Dec via
TIC catalog lookup); SIMBAD covers most VSX objects in practice.

A veto is a *recorded* veto: tic, source, why. Survivors carry their
clean-check provenance into Tier 3 so judges see what was already ruled out.
"""

import csv
import json
import os
import urllib.parse
import urllib.request

from magpie.config import DATA_DIR, DB_PATH, USER_AGENT

TOI_CSV = os.path.join(DATA_DIR, "exofop_toi.csv")
TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

# SIMBAD otypes that explain variability mundanely. Anything typed at all is
# suspicious of being known, but these are hard vetoes for dip/brightening
# morphology specifically.
VETO_OTYPES = {
    "EB*", "EclBin", "Al*", "bL*", "WU*", "RR*", "RRLyr", "Ce*", "Cepheid",
    "dS*", "deltaCep", "Pu*", "PulsV*", "Ro*", "RotV*", "BY*", "RS*",
    "Er*", "Eruptive", "Fl*", "Flare", "TT*", "TTau*", "Mi*", "Mira",
    "LP*", "LPV*", "sr*", "SB*", "CV*", "No*", "Sy*",
}


def refresh_toi_list(force=False) -> set:
    """Download/load the ExoFOP TOI list; return set of TIC ids."""
    if force or not os.path.exists(TOI_CSV):
        req = urllib.request.Request(TOI_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read().decode("utf-8", "replace")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TOI_CSV, "w") as fh:
            fh.write(data)
    tics = set()
    with open(TOI_CSV) as fh:
        for row in csv.DictReader(fh):
            tic = row.get("TIC ID") or row.get("TIC")
            if tic and tic.strip().isdigit():
                tics.add(int(tic.strip()))
    return tics


def simbad_lookup(tic: int) -> dict | None:
    """Resolve 'TIC <id>' in SIMBAD; return {main_id, otype} or None."""
    adql = (
        "SELECT basic.main_id, basic.otype FROM ident "
        "JOIN basic ON ident.oidref = basic.oid "
        f"WHERE ident.id = 'TIC {tic}'"
    )
    q = urllib.parse.urlencode({
        "request": "doQuery", "lang": "adql", "format": "json", "query": adql})
    req = urllib.request.Request(f"{SIMBAD_TAP}?{q}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    rows = out.get("data") or []
    if not rows:
        return None
    return {"main_id": rows[0][0], "otype": rows[0][1]}


def veto(tic: int, toi_set: set) -> dict:
    """Full Tier-2 check for one TIC. Returns
    {verdict: 'veto'|'survive', reasons: [...], simbad: {...}|None}."""
    reasons = []
    if tic in toi_set:
        reasons.append("ExoFOP TOI (known planet candidate)")
    sim = None
    try:
        sim = simbad_lookup(tic)
    except Exception as e:
        reasons.append(f"SIMBAD lookup failed ({type(e).__name__}) — NOT a veto")
    if sim and sim["otype"] in VETO_OTYPES:
        reasons.append(f"SIMBAD {sim['otype']} ({sim['main_id']})")
    hard = [r for r in reasons if not r.endswith("NOT a veto")]
    return {"verdict": "veto" if hard else "survive",
            "reasons": reasons, "simbad": sim}


SCHEMA = """
CREATE TABLE IF NOT EXISTS tier2 (
    sector INTEGER NOT NULL,
    tic INTEGER NOT NULL,
    verdict TEXT NOT NULL,           -- 'veto' | 'survive'
    reasons TEXT,
    simbad_otype TEXT,
    checked_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (sector, tic)
);
"""


def run_tier2(sector: int, threshold: float = 10.0):
    """Veto-check every Tier-1 row whose anomaly score clears threshold.

    Threshold applies to deep_dip + odd_brightening (secular is advisory-only
    until calibrated — TESS systematics ramps dominate it).
    """
    import sqlite3
    import time
    db_path = DB_PATH

    # Phase 1 — read the work list, close immediately.
    db = sqlite3.connect(db_path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(SCHEMA)
    db.commit()
    rows = db.execute(
        """SELECT t1.tic FROM tier1 t1
           LEFT JOIN tier2 t2 ON t2.sector=t1.sector AND t2.tic=t1.tic
           WHERE t1.sector=? AND t1.status='ok' AND t2.tic IS NULL
             AND (t1.deep_dip + t1.odd_brightening) > ?""",
        (sector, threshold)).fetchall()
    db.close()

    # Phase 2 — slow network lookups, NO db connection held.
    toi_set = refresh_toi_list()
    print(f"tier2: {len(rows)} flagged targets to check (threshold {threshold})")
    results = []
    for (tic,) in rows:
        v = veto(tic, toi_set)
        results.append((sector, tic, v["verdict"], "; ".join(v["reasons"]),
                        (v["simbad"] or {}).get("otype")))
    n_survive = sum(1 for r in results if r[2] == "survive")

    # Phase 3 — one short write burst, retried against busy writers.
    for attempt in range(10):
        try:
            db = sqlite3.connect(db_path, timeout=60)
            db.execute("BEGIN IMMEDIATE")
            db.executemany(
                "INSERT OR REPLACE INTO tier2 "
                "(sector,tic,verdict,reasons,simbad_otype) VALUES (?,?,?,?,?)",
                results)
            db.commit()
            db.close()
            break
        except sqlite3.OperationalError:
            time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit("tier2: could not acquire write lock after 10 tries")
    print(f"tier2 done: {len(results)} checked, {n_survive} survive")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=10.0)
    args = ap.parse_args()
    run_tier2(args.sector, args.threshold)
