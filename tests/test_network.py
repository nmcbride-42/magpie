"""Validation on real archive data. Skipped unless MAGPIE_NETWORK_TESTS=1.

    MAGPIE_NETWORK_TESTS=1 pytest -m network

1. Boyajian recovery gate: a filter that can't re-find KIC 8462852's dips
   in Kepler data doesn't get to call anything in TESS anomalous.
2. Injection-recovery on a real TESS curve (Pi Mensae, 2-min SPOC): no false
   alarm on the quiet star (pi Men c's ~300 ppm symmetric transits must not
   dominate), and a planted asymmetric 1% dip is recovered and localized.

Downloads a few MB into $MAGPIE_DATA_DIR/cache (a temp dir under pytest).
"""
import numpy as np
import pytest

from magpie.config import CACHE_DIR
from magpie.tier1 import run_tier1

pytestmark = pytest.mark.network


@pytest.mark.parametrize("quarter", [8, 16])
def test_boyajian_dips_are_recovered(quarter):
    lk = pytest.importorskip("lightkurve")
    sr = lk.search_lightcurve("KIC 8462852", mission="Kepler", quarter=quarter)
    lc = sr[0].download(download_dir=CACHE_DIR).remove_nans()
    r = run_tier1(lc.time.value, np.asarray(lc.flux.value, dtype=float))
    assert r.scores["deep_dip"] > 50.0


def test_injection_recovery_on_real_tess_noise():
    lk = pytest.importorskip("lightkurve")
    sr = lk.search_lightcurve("Pi Mensae", mission="TESS", author="SPOC",
                              exptime=120)
    lc = sr[0].download(download_dir=CACHE_DIR).remove_nans()
    t, f = lc.time.value, np.asarray(lc.flux.value, dtype=float)

    clean = run_tier1(t, f)
    assert clean.scores["deep_dip"] < 5

    # sawtooth dip: slow ingress, fast egress (Boyajian morphology)
    half = 120
    prof = np.concatenate([np.linspace(0, 1, 2 * half),
                           np.linspace(1, 0, half // 3)]) * 0.01
    s = len(f) // 2 - 2 * half
    g = f.copy()
    g[s:s + len(prof)] *= 1 - prof
    inj = run_tier1(t, g)
    assert inj.scores["deep_dip"] > 10 * max(clean.scores["deep_dip"], 0.5)
    assert any(ev[0] < s + len(prof) and ev[1] > s for ev in inj.events)
