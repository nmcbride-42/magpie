"""Watchdog — find swept-but-incomplete sectors and resume them.

A *completed* hunt now leaves no gaps (it drains tier3 fully and runs the
funnel end-to-end through catalog). But a hunt KILLED mid-run — reboot, OOM,
a reaped `claude -p` judge subprocess — still strands a sector partway. This
scans for those: tier1 sweep complete, a downstream gap, and no hunt currently
working it. Each is resumed with `hunt --sector N --skip-sweep`.

Cheap by default: the scan is pure SQL (no LLM), so it can run often; it only
spends (launches a hunt) when there's a real gap and the per-sector cooldown
has elapsed. Honors hunt locks (never fights a live hunt), bounds work per
invocation, and gives up on a genuinely-stuck sector after MAX_ATTEMPTS with
an alert instead of hammering it forever.

  python -m magpie.watchdog            # scan + resume
  python -m magpie.watchdog --dry-run  # scan + report, do nothing
"""
import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time

from magpie.config import DATA_DIR as DATA, DB_PATH as DB, ROOT
from magpie.hunt import pid_alive

STATE = os.path.join(DATA, ".watchdog-state.json")
LOCK = os.path.join(DATA, ".watchdog.lock")
CONF_CUT = 0.85          # must match tier35's shortlist cut
COOLDOWN_S = 1800        # 30 min between resume attempts on the same sector
MAX_ATTEMPTS = 3         # after this many failed resumes, alert not retry
MAX_PER_RUN = 1          # resume at most this many sectors per invocation


def log(msg):
    print(f"[watchdog] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def expected_targets(sector):
    script = os.path.join(DATA, f"sector{sector}_lc.sh")
    try:
        with open(script, errors="replace") as fh:
            return sum(1 for line in fh if "curl" in line)
    except OSError:
        return 0


def hunt_locked(sector):
    """True if a hunt is actively working this sector (live pid in its lock)."""
    lk = os.path.join(DATA, f".hunt-s{sector}.lock")
    try:
        pid = open(lk).read().strip()
    except OSError:
        return False
    return bool(pid) and pid_alive(pid)


def gaps(db, sector):
    """The same funnel gaps report_completion checks, as a dict of counts.
    A downstream table that doesn't exist yet means the hunt died before
    that stage ever ran — itself a gap worth resuming."""
    try:
        return _gaps(db, sector)
    except sqlite3.OperationalError as e:
        if "no such table" not in str(e):
            raise
        return {"funnel_not_started": 1}


def _gaps(db, sector):
    unjudged = db.execute(
        """SELECT count(*) FROM tier2 t2
           LEFT JOIN tier3 t3 ON t3.sector=t2.sector AND t3.tic=t2.tic
           WHERE t2.sector=? AND t2.verdict='survive' AND t3.tic IS NULL""",
        (sector,)).fetchone()[0]
    sl_unjudged = db.execute(
        """SELECT count(*) FROM tier3 t3
           WHERE t3.sector=? AND (t3.verdict='interesting' OR t3.confidence < ?)
             AND NOT EXISTS (SELECT 1 FROM tier35 t35
                             WHERE t35.sector=t3.sector AND t35.tic=t3.tic)""",
        (sector, CONF_CUT)).fetchone()[0]
    finalists_unchecked = db.execute(
        """SELECT count(*) FROM tier35 t35
           WHERE t35.sector=? AND t35.verdict='interesting'
             AND NOT EXISTS (SELECT 1 FROM catalog_check cc
                             WHERE cc.sector=t35.sector AND cc.tic=t35.tic)""",
        (sector,)).fetchone()[0]
    g = {}
    if unjudged:
        g["survivors_unjudged"] = unjudged
    if sl_unjudged:
        g["shortlist_unjudged"] = sl_unjudged
    if finalists_unchecked:
        g["finalists_unchecked"] = finalists_unchecked
    return g


def incomplete_sectors(db):
    """(sector, gaps) for every sector whose sweep is complete but funnel isn't.
    Mid-sweep sectors (tier1 short of the bulk-script target) are skipped so we
    never resume a sector that's still being swept."""
    try:
        sectors = [r[0] for r in db.execute(
            "SELECT DISTINCT sector FROM tier1 ORDER BY sector").fetchall()]
    except sqlite3.OperationalError:
        return []  # nothing swept yet
    out = []
    for s in sectors:
        exp = expected_targets(s)
        swept = db.execute(
            "SELECT count(*) FROM tier1 WHERE sector=? AND status='ok'",
            (s,)).fetchone()[0]
        if exp and swept < exp:
            continue  # sweep not finished — leave it alone
        g = gaps(db, s)
        if g:
            out.append((s, g))
    return out


def resume(sector):
    """Run the end-to-end funnel on a swept sector (skip the sweep itself)."""
    subprocess.run([sys.executable, "-m", "magpie.hunt",
                    "--sector", str(sector), "--skip-sweep"],
                   cwd=ROOT, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be resumed, do nothing")
    args = ap.parse_args()

    # single-instance: a resume can run 30+ min; the next tick must not overlap
    os.makedirs(DATA, exist_ok=True)
    lockf = open(LOCK, "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another watchdog is running — exiting")
        return

    db = sqlite3.connect(DB, timeout=60)
    state = {}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except (OSError, json.JSONDecodeError):
            state = {}

    cands = incomplete_sectors(db)
    if not cands:
        log("all swept sectors complete — nothing to do")
        db.close()
        return
    log(f"incomplete: {', '.join(f's{s} {g}' for s, g in cands)}")

    now = time.time()
    acted = 0
    for s, g in cands:
        if acted >= MAX_PER_RUN:
            log(f"per-run cap reached — leaving s{s} for next tick")
            break
        if hunt_locked(s):
            log(f"s{s}: a hunt is already working it — skip")
            continue
        st = state.get(str(s), {"attempts": 0, "last": 0})
        if st["attempts"] >= MAX_ATTEMPTS:
            log(f"s{s}: STUCK after {st['attempts']} resumes ({g}) — ALERT, "
                f"not retrying (clear .watchdog-state.json to reset)")
            continue
        if now - st.get("last", 0) < COOLDOWN_S:
            mins = int((COOLDOWN_S - (now - st["last"])) / 60)
            log(f"s{s}: in cooldown (~{mins}m left) — skip")
            continue

        if args.dry_run:
            log(f"s{s}: WOULD resume ({g})")
            continue

        log(f"s{s}: resuming ({g}) — attempt {st['attempts'] + 1}")
        st["attempts"] += 1
        st["last"] = now
        state[str(s)] = st
        json.dump(state, open(STATE, "w"), indent=1)

        resume(s)

        remaining = gaps(db, s)
        if not remaining:
            state[str(s)] = {"attempts": 0, "last": now, "status": "complete"}
            log(f"s{s}: resume succeeded — funnel complete")
        else:
            log(f"s{s}: still incomplete after resume ({remaining})")
        json.dump(state, open(STATE, "w"), indent=1)
        acted += 1

    db.close()


if __name__ == "__main__":
    main()
