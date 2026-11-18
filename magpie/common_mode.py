"""Tier 2.5 — common-mode veto. Pure Python, no LLM.

If events from many *independent* targets land in the same cadence window,
the cause is the spacecraft (momentum dump, scattered light, detrend edge),
not the sky. Discovered by the Tier-3 judge on the first smoke batch
(4/5 candidates shared the ~10630-10860 window in sector 95); promoted to a
mechanical filter so it never costs LLM tokens again.

Caveat: event indices are per-target positions after the NaN/quality cut,
not absolute cadence numbers. Alignment across targets is approximate but
empirically tight (same sector grid, similar quality masks). v2 should
store BTJD times in tier1 events instead.
"""

import re
import sqlite3
from collections import Counter

from magpie.config import DB_PATH
from magpie.dbutil import retry_executemany, retry_commit

# matches "(8999, 9154," and "(np.int64(8999), np.int64(9154),"
_WIN_RE = re.compile(r"\((?:np\.int64\()?(\d+)\)?,\s*(?:np\.int64\()?(\d+)\)?,")

BIN = 50            # cadence bins (~100 min at 2-min cadence)
P_CUT = 1e-4        # Poisson tail: bin is common-mode if P(X>=k | bg) < this
MIN_TICS_FLOOR = 3  # never call a bin common-mode on fewer than 3 TICs
EDGE_FRAC = 0.01    # events in the first/last 1% of a curve are edge artifacts
MARKER = "common-mode/edge cadence window"


def event_bins(events_repr, n_cadences):
    """Yield (bin, is_edge) for each event window of one target."""
    for s, e in _WIN_RE.findall(events_repr):
        mid = (int(s) + int(e)) // 2
        edge = mid < n_cadences * EDGE_FRAC or mid > n_cadences * (1 - EDGE_FRAC)
        yield mid // BIN, edge


def build_badmap(db, sector):
    """Bins with a Poisson-significant excess of independent eventing targets.

    Background: marks spread uniformly over bins. A bin is common-mode only
    if its count is a P_CUT-tail outlier above that rate (and >= floor).
    Fixed-count thresholds don't survive density: at 13k targets random
    coincidence poisons most bins (learned at 2.7k targets, 73 false bins).
    """
    from scipy.stats import poisson
    per_bin = Counter()
    rows = db.execute(
        "SELECT tic, events, n_cadences FROM tier1 "
        "WHERE sector=? AND status='ok' AND n_events > 0", (sector,)).fetchall()
    for tic, ev, n in rows:
        for b in {b for b, _ in event_bins(ev, n)}:  # distinct bins per tic
            per_bin[b] += 1
    if not per_bin:
        return set()
    n_bins = max(per_bin) + 1
    lam = sum(per_bin.values()) / n_bins
    return {b for b, c in per_bin.items()
            if c >= MIN_TICS_FLOOR and poisson.sf(c - 1, lam) < P_CUT}


def run(sector, db_path=None):
    """Recomputable: every catalog-clean tier2 row (survive OR previously
    common-mode-demoted) is re-verdicted from the current badmap, so the
    veto self-corrects as sweep density changes."""
    db = sqlite3.connect(db_path or DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    bad = build_badmap(db, sector)
    print(f"common-mode bins (Poisson p<{P_CUT}): {sorted(bad)}")
    rows = db.execute(
        "SELECT t2.tic, t2.reasons, t1.events, t1.n_cadences FROM tier2 t2 "
        "JOIN tier1 t1 ON t1.sector=t2.sector AND t1.tic=t2.tic "
        "WHERE t2.sector=? AND (t2.verdict='survive' OR t2.reasons LIKE ?)",
        (sector, f"%{MARKER}%")).fetchall()
    n_veto = n_restored = 0
    updates = []
    for tic, reasons, ev, n in rows:
        base = (reasons or "").replace(MARKER, "").strip("; ")
        marks = list(event_bins(ev, n))
        if marks and all(b in bad or edge for b, edge in marks):
            updates.append(("veto", (base + "; " if base else "") + MARKER, sector, tic))
            n_veto += 1
        else:
            if reasons and MARKER in reasons:
                n_restored += 1
            updates.append(("survive", base, sector, tic))
    retry_executemany(db,
        "UPDATE tier2 SET verdict=?, reasons=? WHERE sector=? AND tic=?", updates)
    retry_commit(db)
    print(f"common-mode: {n_veto} vetoed, {n_restored} restored, "
          f"{len(rows)} catalog-clean rows evaluated")
    db.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    run(ap.parse_args().sector)
