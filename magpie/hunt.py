"""The whole hunt for one sector, one command.

    python -m magpie.hunt --sector 97

Stages (each resumable; a re-run skips finished work):
  0. fetch the sector's MAST bulk script if missing
  1. sharded sweep (4 workers in-process via subprocess, restart-on-death)
  2. tier2 catalog veto
  3. common-mode veto
  4. tier3 text judge (capped batches)
  5. tier35 vision judge (gap veto first)
  6. tier4 dossiers for finalists (multi-sector history, BLS, period fold)
  7. catalog cross-check (VSX, SIMBAD, TESS-EB, Gaia DR3 variability)
Then prints an honest completion summary. A lock file prevents concurrent
hunts of the same sector. Long-running: run it under a process supervisor
(systemd unit, tmux, nohup) rather than an interactive shell.

    python -m magpie.hunt --sector 97 --fetch-only   # just download the
                                                     # sector's MAST bulk script
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request

from magpie.config import DATA_DIR as DATA, DB_PATH, ROOT

PY = sys.executable
NSHARDS = int(os.environ.get("MAGPIE_SHARDS", "4"))
SHARD_RESTARTS = 3  # a shard that dies (db lock, network) gets this many lives

BULK_URL = ("https://archive.stsci.edu/missions/tess/download_scripts/"
            "sector/tesscurl_sector_{n}_lc.sh")


def log(msg):
    print(f"[hunt] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def pid_alive(pid_str) -> bool:
    """True if pid_str names a live process (signal 0 probes, sends nothing)."""
    try:
        pid = int(str(pid_str).strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def ensure_script(sector: int) -> str:
    """Download MAST's per-sector bulk curl script (one line per 2-min SPOC
    light curve) unless it's already cached. Returns its local path."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"sector{sector}_lc.sh")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        log(f"fetching bulk script for sector {sector}")
        urllib.request.urlretrieve(BULK_URL.format(n=sector), path)
    return path


def run_stage(args, name):
    log(f"stage: {name}")
    r = subprocess.run([PY, "-m"] + args, cwd=ROOT)
    if r.returncode != 0:
        log(f"stage {name} exited {r.returncode} (continuing — stages are resumable)")


def sweep(sector: int, script: str):
    lives = {i: SHARD_RESTARTS for i in range(NSHARDS)}
    procs = {}

    def start(i):
        lf = open(os.path.join(DATA, f"sweep-s{sector}-shard{i}.log"), "a")
        procs[i] = subprocess.Popen(
            [PY, "-m", "magpie.sweep", script, "--sector", str(sector),
             "--shard", str(i), "--nshards", str(NSHARDS)],
            cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)

    for i in range(NSHARDS):
        start(i)
    log(f"sweep: {NSHARDS} shards launched")
    while procs:
        time.sleep(30)
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[i]
            if rc != 0 and lives[i] > 0:
                lives[i] -= 1
                log(f"shard {i} died rc={rc} — restarting ({lives[i]} lives left)")
                start(i)
            else:
                log(f"shard {i} finished rc={rc}")
    # mop-up pass catches anything shards dropped between death and restart
    run_stage(["magpie.sweep", script, "--sector", str(sector)], "sweep mop-up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--fetch-only", action="store_true",
                    help="download the sector's MAST bulk script and stop")
    args = ap.parse_args()
    s = args.sector

    if args.fetch_only:
        print(ensure_script(s))
        return

    os.makedirs(DATA, exist_ok=True)
    lock = os.path.join(DATA, f".hunt-s{s}.lock")
    if os.path.exists(lock):
        pid = open(lock).read().strip()
        if pid and pid_alive(pid):
            log(f"hunt for sector {s} already running (pid {pid}) — refusing")
            sys.exit(1)
    with open(lock, "w") as fh:
        fh.write(str(os.getpid()))
    try:
        script = ensure_script(s)
        if not args.skip_sweep:
            sweep(s, script)
        run_stage(["magpie.tier2", "--sector", str(s)], "tier2 catalog veto")
        run_stage(["magpie.common_mode", "--sector", str(s)], "common-mode veto")
        run_stage(["magpie.tier3", "--sector", str(s)], "tier3 text judge")
        # Drain the tier3 survivor queue FULLY. tier3 caps each run at
        # MAX_PER_RUN, so a sector with many survivors needs many rounds. The
        # old fixed 6-round cap silently abandoned the remainder and still
        # printed 'complete' (one sector was left with 130 unjudged).
        # Loop until zero, with a stall guard so a row that genuinely can't be
        # judged (persistent failure) can't spin the loop forever.
        prev_left = None
        for round_n in range(200):  # runaway guard, not a real ceiling
            db = sqlite3.connect(DB_PATH, timeout=60)
            left = db.execute(
                """SELECT count(*) FROM tier2 t2
                   LEFT JOIN tier3 t3 ON t3.sector=t2.sector AND t3.tic=t2.tic
                   WHERE t2.sector=? AND t2.verdict='survive'
                     AND t3.tic IS NULL""", (s,)).fetchone()[0]
            db.close()
            if left == 0:
                break
            if left == prev_left:
                log(f"tier3: {left} unjudged not shrinking — stopping (stuck rows)")
                break
            prev_left = left
            log(f"tier3: {left} unjudged remain — round {round_n + 1}")
            run_stage(["magpie.tier3", "--sector", str(s)], "tier3 (continued)")
        run_stage(["magpie.tier35", "--sector", str(s)], "tier35 vision judge")
        # End-to-end: resolve finalists and cross-check catalogs in the same
        # run, so a hunt produces a fully-processed, catalog-checked sector
        # instead of stopping at tier35 and leaving catalog at 0 forever.
        run_stage(["magpie.tier4", "--sector", str(s)], "tier4 dossiers")
        run_stage(["magpie.catalog", "--sector", str(s)], "catalog cross-check")
        report_completion(s)
    finally:
        os.unlink(lock)


def report_completion(s):
    """Honest end-of-hunt status: query the actual funnel and report any gap,
    instead of always printing 'complete'. A stage that crashed (run_stage
    swallows non-zero exits so one failure can't abort the hunt) or a capped
    queue shows up here as a named gap with the resume command."""
    db = sqlite3.connect(DB_PATH, timeout=60)
    unjudged = db.execute(
        """SELECT count(*) FROM tier2 t2
           LEFT JOIN tier3 t3 ON t3.sector=t2.sector AND t3.tic=t2.tic
           WHERE t2.sector=? AND t2.verdict='survive' AND t3.tic IS NULL""",
        (s,)).fetchone()[0]
    sl_unjudged = db.execute(
        """SELECT count(*) FROM tier3 t3
           WHERE t3.sector=? AND (t3.verdict='interesting' OR t3.confidence < 0.85)
             AND NOT EXISTS (SELECT 1 FROM tier35 t35
                             WHERE t35.sector=t3.sector AND t35.tic=t3.tic)""",
        (s,)).fetchone()[0]
    n_final = db.execute(
        "SELECT count(*) FROM tier35 WHERE sector=? AND verdict='interesting'",
        (s,)).fetchone()[0]
    n_checked = db.execute(
        "SELECT count(*) FROM catalog_check WHERE sector=?", (s,)).fetchone()[0]
    db.close()
    gaps = []
    if unjudged:
        gaps.append(f"{unjudged} survivors unjudged (tier3)")
    if sl_unjudged:
        gaps.append(f"{sl_unjudged} shortlist unjudged (tier35)")
    if n_final and not n_checked:
        gaps.append(f"{n_final} finalists not catalog-checked")
    if gaps:
        log(f"sector {s} INCOMPLETE: {'; '.join(gaps)} "
            f"— re-run: hunt --sector {s} --skip-sweep")
    else:
        log(f"sector {s} complete: funnel drained + catalog-checked")


if __name__ == "__main__":
    main()
