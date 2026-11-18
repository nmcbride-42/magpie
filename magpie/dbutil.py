"""Shared SQLite write helpers that wait out cross-process lock contention.

Multiple pipeline stages (and multiple concurrent hunts or sweep shards) write
one magpie.db. Under WAL only one writer proceeds at a time; a blocked write
raises `database is locked` once the busy-timeout elapses. That once crashed
the tier35 vision judge when two hunts ran at once. These helpers retry with
jittered backoff so a busy DB is waited out instead of crashing a stage.
"""
import sqlite3
import time as _time


def retry_write(db, sql, params=(), tries=12, jitter_key=None):
    """Execute a write, waiting out `database is locked` instead of raising.
    Jitter de-syncs competing writers so they stop colliding on the same beat.
    It is keyed off `jitter_key` (e.g. the TIC, which differs per shard) or,
    failing that, the first integer param."""
    seed = jitter_key if isinstance(jitter_key, int) else next(
        (p for p in params if isinstance(p, int)), 0)
    for k in range(tries):
        try:
            db.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or k == tries - 1:
                raise
            _time.sleep(min(2.0, 0.05 * (2 ** k)) * (1 + (seed % 7) / 7.0))


def retry_executemany(db, sql, seq, tries=12):
    """executemany variant of retry_write — for a single batched write."""
    seq = list(seq)
    for k in range(tries):
        try:
            db.executemany(sql, seq)
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or k == tries - 1:
                raise
            _time.sleep(min(2.0, 0.05 * (2 ** k)))


def retry_commit(db, tries=12):
    """Commit, waiting out cross-writer lock contention (see retry_write)."""
    for k in range(tries):
        try:
            db.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or k == tries - 1:
                raise
            _time.sleep(min(2.0, 0.05 * (2 ** k)))
