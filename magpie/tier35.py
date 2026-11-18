"""Tier 3.5 — vision re-judgment of the shortlist.

Why this exists: the Tier-3 text judge receives scores and event windows
but cannot see gaps or local noise — and its first three 'interesting'
verdicts were all gap-adjacent artifacts, falsified in seconds by a human
looking at the plot. So the re-judge panel gets eyes.

Two passes:
  1. Mechanical gap veto (free): every event within GAP_VETO_H hours of a
     gap/sector edge -> 'gap-artifact', no LLM call.
  2. Vision judge: batches of plots (PNG read by claude itself) + gap
     context JSON. Hostile default, same as Tier 3.

Shortlist = tier3 'interesting' OR mundane with confidence < CONF_CUT.
"""

import json
import os
import sqlite3
import subprocess

from magpie.config import DB_PATH, PLOT_DIR, VISION_JUDGE_CMD
from magpie.dbutil import retry_write, retry_commit
from magpie.plot import render
from magpie.tier3 import parse_verdicts

CONF_CUT = 0.85
GAP_VETO_H = 6.0
BATCH = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS tier35 (
    sector INTEGER NOT NULL,
    tic INTEGER NOT NULL,
    verdict TEXT NOT NULL,      -- 'gap-artifact' | 'mundane' | 'interesting'
    reasoning TEXT,
    confidence REAL,
    morphology TEXT,            -- EB | dipper-YSO | flare | pulsator |
                                -- single-transit | artifact | unknown
    judged_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (sector, tic)
);
"""

MIGRATIONS = [
    "ALTER TABLE tier35 ADD COLUMN morphology TEXT",
]

# What the vision judge is allowed to call a candidate. Naming the morphology
# (not just interesting/mundane) is what made a human read beat the pipeline on
# sector 100: 'interesting' alone can't distinguish a new eclipser worth submitting
# from a long-known YSO dipper. The catalog stage then says known-vs-new; this
# says what-kind. Together they make the unresolved bucket triageable.
MORPHOLOGIES = ["EB", "dipper-YSO", "flare", "pulsator",
                "single-transit", "artifact", "unknown"]

PROMPT = """You are a hostile referee for stellar light-curve anomalies, and
this time you can SEE. For each candidate below, READ the PNG plot file with
your Read tool, study both panels (top: raw normalized flux; bottom:
detrended residual in ppm; red shaded bands: the flagged event windows),
and use the gap context JSON (hours from each event to the nearest data
gap or sector edge — gaps are downlinks/momentum dumps; events near them
are usually detrending artifacts).

Default verdict: mundane. Mundane accounts to try hard: detrend overshoot
at gap edges, scattered light (noise blowups near sector start/end),
eclipsing binary (regular, symmetric, often alternating deep/shallow),
starspot rotation (smooth quasi-sinusoidal waves), flares (fast rise, slow
decay), background blends. 'Uncatalogued variable star' is mundane.
Answer 'interesting' ONLY if the visual morphology strains every mundane
account: aperiodic deep dips far from gaps, strong asymmetry on a clean
baseline, depth chaos without periodic structure, slow symmetric
brightening over days.

ALSO classify the morphology you see — your best single guess regardless of
verdict (this is what the object IS, separate from whether it's interesting):
  - "EB"             regular symmetric eclipses, often deep, period evident
  - "dipper-YSO"     deep but VARIABLE-depth dips that don't fold cleanly,
                     often on an active/spotted baseline (young star + disk)
  - "flare"          fast-rise/slow-decay brightenings, often on a spotted star
  - "pulsator"       smooth coherent oscillation (sinusoidal/sawtooth)
  - "single-transit" one isolated dip, no repeat in this sector
  - "artifact"       gap/scattered-light/detrend junk
  - "unknown"        morphology genuinely unclear
Distinguish EB from dipper-YSO carefully: clean equal-depth folding eclipses
= EB; messy varying-depth dips on an active baseline = dipper-YSO. (In TESS
sector 100, TIC 412308868 LOOKED like a deep EB but the varying depths were
a dipper.)

Candidates:
%s

Reply with ONLY a JSON array, one object per candidate:
{"tic": int, "verdict": "mundane"|"interesting",
 "morphology": "EB"|"dipper-YSO"|"flare"|"pulsator"|"single-transit"|"artifact"|"unknown",
 "reasoning": str,        // what you SAW that decides verdict + morphology
 "confidence": float}     // 0-1
"""


def shortlist(db, sector):
    return db.execute(
        """SELECT t3.tic FROM tier3 t3
           LEFT JOIN tier35 t35 ON t35.sector=t3.sector AND t35.tic=t3.tic
           WHERE t3.sector=? AND t35.tic IS NULL
             AND (t3.verdict='interesting' OR t3.confidence < ?)
           ORDER BY (t3.verdict='interesting') DESC, t3.confidence ASC""",
        (sector, CONF_CUT)).fetchall()


def load_sidecar(tic, sector):
    p = os.path.join(PLOT_DIR, f"s{sector}-tic{tic}.json")
    with open(p) as fh:
        return json.load(fh)


def judge_batch(items):
    """items: [{tic, png, context}]"""
    desc = json.dumps(
        [{"tic": it["tic"], "plot_png": it["png"],
          "gap_context": it["context"]} for it in items], indent=1)
    out = subprocess.run(
        list(VISION_JUDGE_CMD),
        input=PROMPT % desc,
        capture_output=True, text=True, timeout=900).stdout
    return parse_verdicts(out)


def run(sector, max_llm=None, db_path=None):
    db = sqlite3.connect(db_path or DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(SCHEMA)
    for m in MIGRATIONS:
        try:
            db.execute(m)
        except sqlite3.OperationalError:
            pass  # column already exists
    retry_commit(db)
    tics = [r[0] for r in shortlist(db, sector)]
    print(f"tier35: {len(tics)} shortlist candidates")

    # Pass 0: render plots + sidecars (downloads LC if needed)
    items, n_gap = [], 0
    for tic in tics:
        try:
            png = render(tic, sector)
            ctx = load_sidecar(tic, sector)
        except SystemExit as e:
            print(f"  TIC {tic}: no data ({e}) — skipping")
            continue
        except Exception as e:
            print(f"  TIC {tic}: render failed ({type(e).__name__}) — skipping")
            continue
        evs = ctx["events"]
        # Pass 1: mechanical gap veto
        if evs and all(e["hours_to_nearest_gap"] < GAP_VETO_H for e in evs):
            retry_write(db,
                "INSERT OR REPLACE INTO tier35 "
                "(sector,tic,verdict,reasoning,confidence,morphology) "
                "VALUES (?,?,?,?,?,?)",
                (sector, tic, "gap-artifact",
                 f"all {len(evs)} events within {GAP_VETO_H}h of a gap/edge",
                 0.9, "artifact"))
            n_gap += 1
        else:
            items.append({"tic": tic, "png": png, "context": evs})
    retry_commit(db)
    print(f"  gap veto killed {n_gap}; {len(items)} go to vision judge")
    if max_llm is not None:
        items = items[:max_llm]

    # Pass 2: vision judging
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        try:
            verdicts = judge_batch(batch)
        except Exception as e:
            print(f"  batch {i//BATCH}: FAILED {type(e).__name__}: {e}")
            continue
        for v in verdicts:
            # a single malformed verdict (LLM dropped a key) must not crash the
            # round — skip it; the shortlist query will resurface that TIC.
            if not isinstance(v, dict) or "tic" not in v or "verdict" not in v:
                print(f"  skipping malformed verdict: {v}")
                continue
            retry_write(db,
                "INSERT OR REPLACE INTO tier35 "
                "(sector,tic,verdict,reasoning,confidence,morphology) "
                "VALUES (?,?,?,?,?,?)",
                (sector, v["tic"], v["verdict"], v.get("reasoning", ""),
                 v.get("confidence", 0.0), v.get("morphology", "unknown")))
        retry_commit(db)
        n_int = sum(1 for v in verdicts
                    if isinstance(v, dict) and v.get("verdict") == "interesting")
        print(f"  batch {i//BATCH}: {len(verdicts)} judged, {n_int} interesting",
              flush=True)
    db.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--max-llm", type=int)
    args = ap.parse_args()
    run(args.sector, args.max_llm)
