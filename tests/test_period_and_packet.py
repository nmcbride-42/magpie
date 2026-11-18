"""Tier-4 period resolution and the VSX packet numerics, all synthetic.

The packet gates come from real failures:
  1. gaia_v must reproduce the V magnitude VSX moderation computed for a
     submitted star (G=12.3288, BP-RP=0.8472 -> V=12.49).
  2. fold_epoch must land on the PRIMARY minimum despite a secondary eclipse
     (an earlier version put an epoch 2.3 h late).
  3. refine_period must recover a 0.0003 d period error over a multi-year
     baseline (an error that size smears a stitched fold and makes
     percentile depth statistics read far too deep).
  4. fold_coherence must pass a clean fold and refuse a smeared one.
  5. the VSX epoch must be HJD_UTC, not BJD_TDB pasted as-is (they differ by
     ~69 s, enough to change the 4th decimal of the epoch).
"""
import numpy as np
import pytest

from magpie.packet import (bjd_tdb_to_hjd_utc, build_remarks,
                           eclipse_geometry, fold_coherence, fold_epoch, gaia_v,
                           measured_depth, refine_period, sector_range_str,
                           to_sexagesimal)
from magpie.tier4 import bls_search, interval_scan, resolve_period

P_TRUE = 3.21173
T0 = 13.37
DEPTH = 0.05


@pytest.fixture(scope="module")
def eb():
    """8 TESS-like 27 d eras of a detached EB across ~5.5 years: box primary
    plus a displaced (eccentric) secondary. Many eras matter: the smear the
    coherence gate exists for needs each phase bin to mix one era's eclipse
    with other eras' baseline."""
    rng = np.random.default_rng(7)
    starts = np.linspace(0, 2000 - 27, 8)
    t = np.concatenate([np.arange(s, s + 27, 0.00139) for s in starts])
    f = 1.0 + rng.normal(0, 0.002, t.size)
    ph = ((t - T0) / P_TRUE) % 1.0
    ph[ph > 0.5] -= 1.0
    f[np.abs(ph) < 0.006] -= DEPTH
    f[np.abs(ph - 0.44) < 0.006] -= DEPTH * 0.4
    return t, f


# ---------------------------------------------------------------- tier 4

def test_bls_recovers_period_on_stitched_sectors():
    rng = np.random.default_rng(3)
    curves = []
    for k, s0 in enumerate((0.0, 60.0)):
        t = s0 + np.arange(0, 20, 2 / 1440)
        f = 1 + rng.normal(0, 1e-3, t.size)
        ph = ((t - 0.3) / 2.5) % 1.0
        f[(ph < 0.02) | (ph > 0.98)] -= 0.02
        curves.append((k + 1, t, f * 1e4))
    b = bls_search(curves, max_p=10.0)
    assert abs(b["best_period_d"] - 2.5) < 0.01
    assert b["power_ratio"] > 10


def test_interval_scan_finds_common_divisor():
    times = [10.0 + 4.2 * k for k in (0, 1, 3, 7, 8)]
    best = interval_scan(times)
    assert best["phased_fraction"] == 1.0
    assert abs(best["period_d"] / 4.2 - round(best["period_d"] / 4.2)) < 1e-3
    assert interval_scan([1.0, 2.0]) is None


def test_resolve_period_handles_eccentric_secondary():
    """Dip times from an eccentric EB (secondary off phase 0.5) fold into two
    tight phase clusters, so the star is resolved as periodic. Dip times
    alone cannot tell P from its divisors, and resolve_period returns the
    shortest alias that folds; that is why tier4.dossier reports the
    flux-BLS period instead. This test pins both facts."""
    p = 52.0
    prim = [100 + p * k for k in range(4)]
    sec = [100 + p * 0.37 + p * k for k in range(3)]     # off phase 0.5
    r = resolve_period(prim + sec, bls={"best_period_d": p / 2})
    assert r["resolved"]
    assert r["n_eclipse_phases"] == 2
    assert r["mechanism"] == "eccentric eclipsing binary"
    ratio = p / r["period_d"]
    assert ratio == pytest.approx(round(ratio)) and ratio >= 1


def test_resolve_period_refuses_aperiodic_dips():
    rng = np.random.default_rng(5)
    dips = sorted(rng.uniform(0, 300, 6))
    r = resolve_period(dips, bls={"best_period_d": 7.3})
    assert not r["resolved"]
    assert resolve_period([1.0, 2.0])["reason"] == "too few events"


# ---------------------------------------------------------------- packet

def test_gaia_v_matches_moderation_value():
    assert gaia_v(12.3288, 0.8472) == pytest.approx(12.49, abs=0.005)
    assert gaia_v(12.0, 6.0) is None           # outside the relation's range
    assert gaia_v(None, 0.5) is None


def test_fold_epoch_ignores_secondary(eb):
    t, f = eb
    e = fold_epoch(t, f, P_TRUE)
    off = abs(((e - T0) / P_TRUE + 0.5) % 1.0 - 0.5) * P_TRUE
    assert off < 0.01                           # < 14.4 minutes
    assert t.min() <= e < t.min() + P_TRUE      # first cycle in the data


def test_refine_period_recovers_small_error(eb):
    t, f = eb
    assert abs(refine_period(t, f, P_TRUE + 3e-4) - P_TRUE) < 5e-5


def test_fold_coherence_passes_clean_and_refuses_smear(eb):
    t, f = eb
    ok, diag = fold_coherence(t, f, P_TRUE, fold_epoch(t, f, P_TRUE))
    assert ok and diag["smear_ratio"] >= 0.5
    bad_p = P_TRUE + 3e-4
    ok_bad, _ = fold_coherence(t, f, bad_p, fold_epoch(t, f, bad_p))
    assert not ok_bad


def test_eclipse_geometry_and_depth(eb):
    t, f = eb
    e = fold_epoch(t, f, P_TRUE)
    g = eclipse_geometry(t, f, P_TRUE, e)
    amp_true = -2.5 * np.log10(1 - DEPTH * 0.4)
    assert g["min2_amp"] == pytest.approx(amp_true, abs=0.01)
    assert g["min2_phase"] == pytest.approx(0.44, abs=0.01)
    assert g["duration_pct"] is not None and 0.5 < g["duration_pct"] < 3
    d = measured_depth(t, f, P_TRUE, e)
    assert DEPTH < d < DEPTH + 0.01             # 1st percentile includes noise


def test_formatting_helpers():
    assert to_sexagesimal(0.0, 0.0) == "00 00 00.00 +00 00 00.0"
    assert to_sexagesimal(150.0, -30.5) == "10 00 00.00 -30 30 00.0"
    d = {"sectors": [{"sector": s} for s in (87, 88, 89, 94, 96)]}
    assert sector_range_str(d) == "87-89, 94, 96"
    assert build_remarks() == ""
    assert build_remarks(0.21, 0.5) == "Min II amplitude 0.21 TESS."
    assert "phase 0.37" in build_remarks(0.21, 0.37)


@pytest.mark.parametrize("ra,dec", [(0.0, 0.0), (90.0, 66.56), (270.0, -66.0)])
def test_bjd_tdb_to_hjd_utc_offset(ra, dec):
    bjd = 2460500.25
    diff_s = (bjd_tdb_to_hjd_utc(bjd, ra, dec) - bjd) * 86400.0
    # TDB-UTC is 32.184 s + 37 leap s = 69.184 s; BJD-HJD adds at most ~4 s
    assert -69.184 - 5.0 < diff_s < -69.184 + 5.0
