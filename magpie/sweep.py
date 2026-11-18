"""Tier 0/1 sweep runner — stream a TESS sector through the Tier-1 filters.

Streams: download FITS -> run_tier1 -> persist scores to sqlite -> delete FITS.
Disk stays flat regardless of sector size. Resumable: TICs already in the DB
for a sector are skipped, so an interrupted sweep continues where it left off
(the hunt-log discipline from SPEC.md).

Usage:
    python -m magpie.sweep data/sector95_lc.sh --sector 95 --limit 100
"""

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time as _time

import numpy as np

from magpie.config import DB_PATH
from magpie.dbutil import retry_commit, retry_write
from magpie.tier1 import run_tier1

SCHEMA = """
CREATE TABLE IF NOT EXISTS tier1 (
    sector INTEGER NOT NULL,
    tic INTEGER NOT NULL,
    deep_dip REAL, secular REAL, odd_brightening REAL,
    n_events INTEGER, events TEXT,
    n_cadences INTEGER, status TEXT NOT NULL DEFAULT 'ok',
    swept_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (sector, tic)
);
"""

# https:// for MAST; file:// lets the offline demo and tests reuse this path.
URL_RE = re.compile(r"-o\s+(\S+)\s+((?:https|file)://\S+)")
TIC_RE = re.compile(r"-(\d{16})-")


def parse_bulk_script(path):
    """Yield (tic, filename, url) from a MAST tesscurl bulk script."""
    with open(path) as fh:
        for line in fh:
            m = URL_RE.search(line)
            if not m:
                continue
            fname, url = m.groups()
            tm = TIC_RE.search(fname)
            if tm:
                yield int(tm.group(1)), fname, url


def load_flux(fits_path):
    """PDCSAP flux from a SPOC light-curve FITS, NaN-free."""
    from astropy.io import fits as pyfits

    with pyfits.open(fits_path, memmap=False) as hdul:
        d = hdul[1].data
        t = np.asarray(d["TIME"], dtype=float)
        f = np.asarray(d["PDCSAP_FLUX"], dtype=float)
        q = np.asarray(d["QUALITY"], dtype=int)
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    return t[ok], f[ok]


# Multiple sweep shards write one SQLite file; even under WAL only one may
# write at a time. A blocked write must NOT surface as a target error (the
# target's data is fine, only the write contended) nor crash the shard, so
# writes go through magpie.dbutil's jittered-backoff retry helpers.


def sweep(script_path, sector, limit=None, workdir=None, shard=0, nshards=1,
          db_path=None):
    db_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = sqlite3.connect(db_path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(SCHEMA)
    done = {r[0] for r in db.execute(
        "SELECT tic FROM tier1 WHERE sector=?", (sector,))}
    targets = [(t, f, u) for t, f, u in parse_bulk_script(script_path)
               if t not in done and t % nshards == shard]
    if limit:
        targets = targets[:limit]
    print(f"sector {sector}: {len(targets)} to sweep ({len(done)} already done)")

    own_workdir = workdir is None  # only remove what we created
    workdir = workdir or tempfile.mkdtemp(prefix="magpie-")
    t0, n_ok, n_err = _time.time(), 0, 0
    try:
      for i, (tic, fname, url) in enumerate(targets):
        local = os.path.join(workdir, fname)
        # Phase 1 — curl + analyse. Decide the row WITHOUT touching the DB, so a
        # write-lock can never be mistaken for a target failure. curl retries
        # ride out transient MAST hiccups (the bulk of the old err:OSError rows).
        try:
            subprocess.run(
                ["curl", "-sS", "-L", "--max-time", "120",
                 "--retry", "3", "--retry-delay", "2", "--retry-all-errors",
                 "-o", local, url],
                check=True, capture_output=True)
            t, f = load_flux(local)
            if len(t) < 500:
                raise ValueError(f"too few cadences: {len(t)}")
            r = run_tier1(t, f)
            sql = ("INSERT OR REPLACE INTO tier1 "
                   "(sector,tic,deep_dip,secular,odd_brightening,n_events,events,"
                   "n_cadences,status) VALUES (?,?,?,?,?,?,?,?,'ok')")
            params = (sector, tic, r.scores["deep_dip"], r.scores["secular"],
                      r.scores["odd_brightening"], len(r.events),
                      repr(r.events[:10]), len(t))
            n_ok += 1
        except Exception as e:  # a real target failure (download/parse) — record it
            sql = "INSERT OR REPLACE INTO tier1 (sector,tic,status) VALUES (?,?,?)"
            params = (sector, tic, f"err:{type(e).__name__}")
            n_err += 1
        finally:
            if os.path.exists(local):
                os.unlink(local)
        # Phase 2 — write, waiting out any cross-shard lock contention.
        retry_write(db, sql, params, jitter_key=tic)
        if (i + 1) % 25 == 0:
            retry_commit(db)
            rate = (i + 1) / (_time.time() - t0)
            print(f"  {i+1}/{len(targets)} ({rate:.1f}/s, {n_err} errors)",
                  flush=True)
      retry_commit(db)
      print(f"done: {n_ok} ok, {n_err} errors, "
            f"{(_time.time()-t0)/60:.1f} min")
    finally:
        db.close()
        if own_workdir:  # only remove the scratch dir this call created
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    sweep(args.script, args.sector, args.limit, shard=args.shard,
          nshards=args.nshards)


if __name__ == "__main__":
    main()
