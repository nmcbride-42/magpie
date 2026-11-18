"""Tier-1 filters: injection-recovery on synthetic TESS-like noise.

The network suite (test_network.py) repeats the injection on a real TESS
curve and checks recovery of Boyajian's Star in Kepler data.
"""
import numpy as np

from magpie.tier1 import (_runs, deep_dip_finder, detrend, flare_oddity,
                          run_tier1, secular_trend)

CAD = 2.0 / 1440.0


def quiet(n=10000, noise=5e-4, seed=0):
    rng = np.random.default_rng(seed)
    t = 2000.0 + np.arange(n) * CAD
    f = (1 + 0.002 * np.sin(2 * np.pi * t / 5.0)) * (1 + rng.normal(0, noise, n))
    return t, f


def test_runs_finds_contiguous_segments():
    m = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert _runs(m) == [(0, 2), (4, 5), (6, 9)]
    assert _runs(np.zeros(4, bool)) == []


def test_detrend_removes_rotation_and_estimates_noise():
    t, f = quiet()
    resid, noise = detrend(t, f)
    assert abs(np.median(resid)) < 50          # ppm
    assert 400 < noise < 600                   # injected 500 ppm white noise


def test_quiet_star_has_no_false_alarm():
    t, f = quiet()
    r = run_tier1(t, f)
    assert r.scores["deep_dip"] == 0
    assert r.scores["odd_brightening"] == 0
    assert r.events == []


def test_asymmetric_dip_is_recovered_and_localized():
    t, f = quiet()
    # ~5 h event: slow ingress, fast egress, 1% deep (20 sigma)
    c = 5000
    prof = np.concatenate([np.linspace(0, 1, 120), np.linspace(1, 0, 40)]) * 0.01
    f = f.copy()
    f[c - 120:c + 40] *= 1 - prof
    r = run_tier1(t, f)
    assert r.scores["deep_dip"] > 10
    hits = [ev for ev in r.events if ev[0] < c + 40 and ev[1] > c - 120]
    assert hits and "dip" in hits[0][2]


def test_long_events_are_attenuated_by_the_detrend_window():
    """Documented limitation: the 12 h median detrend partly absorbs events
    that last a large fraction of the window, so they score lower."""
    t, f = quiet()
    scores = []
    for half in (60, 240):                      # ~2.7 h vs ~10.7 h events
        prof = np.concatenate([np.linspace(0, 1, 3 * half // 2),
                               np.linspace(1, 0, half // 2)]) * 0.01
        g = f.copy()
        g[5000 - len(prof):5000] *= 1 - prof
        scores.append(run_tier1(t, g).scores["deep_dip"])
    assert scores[0] > scores[1]


def test_symmetric_box_is_downweighted_vs_asymmetric():
    resid = np.zeros(400)
    sym = resid.copy()
    sym[100:140] = -np.concatenate([np.linspace(1, 20, 20),
                                    np.linspace(20, 1, 20)])  # symmetric V
    asym = resid.copy()
    asym[100:140] = -np.linspace(1, 20, 40)     # slow ingress, sharp egress
    s_sym, _ = deep_dip_finder(sym, noise=1.0)
    s_asym, _ = deep_dip_finder(asym, noise=1.0)
    assert s_sym > 0 and s_asym > 0
    # same peak depth; the asymmetry weight (0.3 .. 1.0) favours the sawtooth
    assert s_asym > 2 * s_sym


def test_slow_rise_brightening_flags_but_fast_flare_does_not():
    resid = np.zeros(300)
    flare = resid.copy()
    flare[50:70] = np.concatenate([[30.0], np.linspace(28, 9, 19)])   # fast rise
    slow = resid.copy()
    slow[150:170] = np.concatenate([np.linspace(9, 28, 19), [30.0]])  # slow rise
    s_flare, _ = flare_oddity(flare, noise=1.0)
    s_slow, ev = flare_oddity(slow, noise=1.0)
    assert s_flare == 0
    assert s_slow > 0 and "slow-rise" in ev[0][2]


def test_secular_trend_scales_with_slope():
    t, f = quiet(noise=1e-3)
    flat = secular_trend(t, f)
    ramp = secular_trend(t, f * (1 + 0.01 * (t - t[0]) / (t[-1] - t[0])))
    assert ramp > 5 * max(flat, 1.0)
