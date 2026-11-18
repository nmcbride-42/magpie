"""Sweep (Tier 0/1) and common-mode veto (Tier 2.5), offline.

The sweep runs its real code path (curl -> FITS -> filters -> sqlite) against
file:// URLs, so these tests need curl on PATH but no network.
"""
import os
import shutil
import sqlite3

import numpy as np
import pytest

from magpie import common_mode
from magpie.demo import write_fits
from magpie.sweep import load_flux, parse_bulk_script, sweep
from magpie.tier2 import SCHEMA as TIER2_SCHEMA

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "tesscurl_sector95_sample.sh")
needs_curl = pytest.mark.skipif(shutil.which("curl") is None,
                                reason="curl not installed")


def test_parse_bulk_script_reads_mast_format():
    rows = list(parse_bulk_script(FIXTURE))
    assert [r[0] for r in rows] == [25078597, 25078674, 25078924,
                                    25079689, 25079777]
    tic, fname, url = rows[0]
    assert fname.endswith("_lc.fits") and url.startswith("https://mast.")


def test_load_flux_drops_nans_and_flagged_cadences(tmp_path):
    t = np.arange(10, dtype=float)
    f = np.ones(10)
    f[3] = np.nan
    p = tmp_path / "x.fits"
    write_fits(str(p), t, f)
    # write_fits sets QUALITY=0; flag one cadence by rewriting the column
    from astropy.io import fits
    with fits.open(p, mode="update") as h:
        h[1].data["QUALITY"][5] = 128
    tt, ff = load_flux(str(p))
    assert list(tt) == [0, 1, 2, 4, 6, 7, 8, 9]
    assert np.isfinite(ff).all()


def _sector(tmp_path, curves, extra_lines=()):
    script = tmp_path / "s.sh"
    lines = ["#!/bin/sh"]
    for tic, f in curves.items():
        name = f"x-s0001-{tic:016d}-0000-s_lc.fits"
        path = tmp_path / name
        write_fits(str(path), 1000 + np.arange(f.size) * 2 / 1440, f)
        lines.append(f"curl -C - -L -o {name} file://{path}")
    lines += list(extra_lines)
    script.write_text("\n".join(lines) + "\n")
    return str(script)


@needs_curl
def test_sweep_is_resumable_and_records_failures(tmp_path):
    rng = np.random.default_rng(1)
    curves = {tic: 1 + rng.normal(0, 5e-4, 2000) for tic in (11, 12, 13)}
    missing = tmp_path / "x-s0001-0000000000000099-0000-s_lc.fits"
    script = _sector(tmp_path, curves, [
        f"curl -C - -L -o {missing.name} file://{missing}"])
    db_path = str(tmp_path / "t.db")

    sweep(script, 1, db_path=db_path)
    db = sqlite3.connect(db_path)
    rows = dict(db.execute("SELECT tic, status FROM tier1").fetchall())
    assert rows == {11: "ok", 12: "ok", 13: "ok", 99: "err:CalledProcessError"}

    # second run: everything already recorded, nothing re-swept
    sweep(script, 1, db_path=db_path)
    assert db.execute("SELECT count(*) FROM tier1").fetchone()[0] == 4


def test_event_bins_parses_plain_and_numpy_reprs():
    plain = "[(1000, 1040, 'dip'), (5000, 5010, 'dip')]"
    numpy_repr = "[(np.int64(1000), np.int64(1040), 'dip')]"
    assert list(common_mode.event_bins(plain, 10000)) == [(20, False), (100, False)]
    assert list(common_mode.event_bins(numpy_repr, 10000)) == [(20, False)]
    # an event in the first 1% of the curve is an edge artifact
    assert list(common_mode.event_bins("[(10, 20, 'dip')]", 10000)) == [(0, True)]


def test_common_mode_vetoes_shared_window_and_keeps_independent(tmp_path):
    db_path = str(tmp_path / "cm.db")
    db = sqlite3.connect(db_path)
    db.execute("""CREATE TABLE tier1 (sector INTEGER, tic INTEGER,
                  deep_dip REAL, secular REAL, odd_brightening REAL,
                  n_events INTEGER, events TEXT, n_cadences INTEGER,
                  status TEXT, PRIMARY KEY (sector, tic))""")
    db.executescript(TIER2_SCHEMA)
    rows = []
    # 10 unrelated stars all eventing in the same cadence window
    for tic in range(1, 11):
        rows.append((tic, "[(3000, 3040, 'dip')]"))
    # 3 stars with their own, unshared events
    rows += [(101, "[(7000, 7100, 'dip')]"), (102, "[(9000, 9100, 'dip')]"),
             (103, "[(12000, 12100, 'dip')]")]
    # 40 quiet-ish stars with scattered single events, for a background rate
    for i, tic in enumerate(range(200, 240)):
        s = 1000 + 400 * i
        rows.append((tic, f"[({s}, {s + 20}, 'dip')]"))
    for tic, ev in rows:
        db.execute("INSERT INTO tier1 VALUES (5,?,20,0,0,1,?,18000,'ok')",
                   (tic, ev))
        db.execute("INSERT INTO tier2 (sector,tic,verdict,reasons) "
                   "VALUES (5,?,'survive','')", (tic,))
    db.commit()

    bad = common_mode.build_badmap(db, 5)
    assert bad == {3020 // common_mode.BIN}
    db.close()

    common_mode.run(5, db_path=db_path)
    db = sqlite3.connect(db_path)
    v = dict(db.execute("SELECT tic, verdict FROM tier2").fetchall())
    assert all(v[t] == "veto" for t in range(1, 11))
    assert v[101] == v[102] == v[103] == "survive"

    # recomputable: rerunning restores nothing and vetoes nothing new
    common_mode.run(5, db_path=db_path)
    v2 = dict(db.execute("SELECT tic, verdict FROM tier2").fetchall())
    assert v2 == v


@needs_curl
def test_offline_demo_recovers_planted_truth(tmp_path):
    from magpie.demo import run_demo
    res = run_demo(str(tmp_path))
    kinds, surv = res["kinds"], set(res["survivors"])
    assert {t for t in res["flagged"] if kinds[t] == "quiet"} == set()
    assert {t for t, k in kinds.items() if k == "common-mode"}.isdisjoint(surv)
    for tic in surv:
        r = res["resolved"][tic]
        if kinds[tic] == "EB":
            assert r["resolved"] and abs(r["period_d"] - 1.7) < 0.01
        else:
            assert kinds[tic] == "dipper" and not r["resolved"]
    assert sum(kinds[t] == "EB" for t in surv) == 2
    assert sum(kinds[t] == "dipper" for t in surv) == 2
