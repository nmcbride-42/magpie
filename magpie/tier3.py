"""Tier 3 — adversarial LLM judge panel.

Input: Tier-2 survivors (already cleared of TOIs and catalogued variables).
The judge's default position is HOSTILE: "this is an instrumental artifact,
a background eclipsing binary, scattered light, or a data-quality glitch —
prove otherwise." A candidate survives only if the judge cannot construct a
plausible mundane account from the evidence given.

Cost discipline (docs/design.md): candidates are judged in batches of BATCH
(one judge call amortizes the fixed per-call prompt/agent overhead across the
batch), and a single run never judges more than MAX_PER_RUN candidates.
Tier 3 must stay the SMALL tier — if it sees thousands, Tier 1/2 thresholds
are wrong, and the fix is upstream, not a bigger budget here.

The judge is any command that reads the prompt on stdin and prints a JSON
array (MAGPIE_JUDGE_CMD, default: the Claude Code CLI in print mode).
"""

import json
import sqlite3
import subprocess

from magpie.config import DB_PATH, JUDGE_CMD, JUDGE_MODEL
from magpie.dbutil import retry_write, retry_commit

BATCH = 8
MAX_PER_RUN = 48

SCHEMA = """
CREATE TABLE IF NOT EXISTS tier3 (
    sector INTEGER NOT NULL,
    tic INTEGER NOT NULL,
    verdict TEXT NOT NULL,           -- 'mundane' | 'interesting'
    mundane_account TEXT,            -- the judge's best mundane explanation
    interest TEXT,                   -- why it survived, if it did
    confidence REAL,
    judged_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (sector, tic)
);
"""

PROMPT = """You are a hostile referee for stellar light-curve anomalies.
Your default verdict is "mundane". Real anomalies are vanishingly rare;
instrumental artifacts, background eclipsing binaries, scattered light at
sector edges, momentum dumps, and detrending failures are common. For each
candidate below, construct the most plausible mundane account you can. Only
if every mundane account is strained by the actual evidence may you answer
"interesting".

These candidates already passed: (a) mechanical filters tuned to aperiodic /
asymmetric events, (b) veto against ExoFOP TOIs and SIMBAD-typed variables.
"Uncatalogued variable star" is still a perfectly good mundane account.

Candidates (JSON):
%s

Reply with ONLY a JSON array, one object per candidate, schema:
{"tic": int, "verdict": "mundane"|"interesting",
 "mundane_account": str,  // your best mundane explanation
 "interest": str,         // empty if mundane
 "confidence": float}     // 0-1 in your verdict
"""


def fetch_survivors(db, sector, limit):
    """Tier-1 flagged, Tier-2 survived, not yet judged."""
    return db.execute(
        """SELECT t1.tic, t1.deep_dip, t1.secular, t1.odd_brightening,
                  t1.n_events, t1.events, t1.n_cadences
           FROM tier1 t1
           LEFT JOIN tier3 t3 ON t3.sector = t1.sector AND t3.tic = t1.tic
           JOIN tier2 t2 ON t2.sector = t1.sector AND t2.tic = t1.tic
           WHERE t1.sector = ? AND t1.status = 'ok'
             AND t2.verdict = 'survive' AND t3.tic IS NULL
           ORDER BY t1.deep_dip + t1.odd_brightening DESC
           LIMIT ?""", (sector, limit)).fetchall()


def parse_verdicts(out):
    """Extract the JSON array from a judge's stdout (tolerates prose around
    it). Raises ValueError when there is no array to parse."""
    start, end = out.find("["), out.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("judge output contained no JSON array")
    return json.loads(out[start:end + 1])


def judge_batch(rows, model=None):
    payload = [
        {"tic": r[0], "deep_dip_score": r[1], "secular_advisory": r[2],
         "odd_brightening_score": r[3], "n_events": r[4],
         "event_windows": r[5], "n_cadences": r[6]}
        for r in rows
    ]
    cmd = list(JUDGE_CMD)
    if model:
        cmd += ["--model", model]
    out = subprocess.run(
        cmd, input=PROMPT % json.dumps(payload, indent=1),
        capture_output=True, text=True, timeout=600).stdout
    return parse_verdicts(out)


def run(sector, model=JUDGE_MODEL, max_per_run=MAX_PER_RUN, db_path=None):
    db = sqlite3.connect(db_path or DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(SCHEMA)
    rows = fetch_survivors(db, sector, max_per_run)
    print(f"tier3: {len(rows)} survivors to judge (cap {max_per_run})")
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        try:
            verdicts = judge_batch(batch, model)
        except Exception as e:
            print(f"  batch {i//BATCH}: FAILED {type(e).__name__}: {e}")
            continue
        for v in verdicts:
            # one malformed verdict object (LLM dropped a key) must NOT crash
            # the whole round — skip it; the drain loop will re-judge that TIC.
            if not isinstance(v, dict) or "tic" not in v or "verdict" not in v:
                print(f"  skipping malformed verdict: {v}")
                continue
            retry_write(db,
                "INSERT OR REPLACE INTO tier3 "
                "(sector,tic,verdict,mundane_account,interest,confidence) "
                "VALUES (?,?,?,?,?,?)",
                (sector, v["tic"], v["verdict"], v.get("mundane_account", ""),
                 v.get("interest", ""), v.get("confidence", 0.0)))
        retry_commit(db)
        n_int = sum(1 for v in verdicts
                    if isinstance(v, dict) and v.get("verdict") == "interesting")
        print(f"  batch {i//BATCH}: {len(verdicts)} judged, {n_int} interesting")
    db.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--max", type=int, default=MAX_PER_RUN)
    args = ap.parse_args()
    run(args.sector, args.model, args.max)
