"""Tier 1 — mechanical anomaly filters. Pure numpy/scipy, no LLM.

Each filter returns a score >= 0; 0 means "nothing here." A light curve's
tier-1 record is the dict of filter scores plus the event windows that
triggered them. Thresholding/calibration happens in the runner, not here,
so filters stay pure functions of the flux array.

Design rule: these filters are deliberately blind to *periodic* signals.
Planets and eclipsing binaries are somebody else's catalog. We hunt the
aperiodic, the asymmetric, the secular, the one-off.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Tier1Result:
    scores: dict = field(default_factory=dict)
    events: list = field(default_factory=list)  # (start_idx, end_idx, kind)

    @property
    def total(self) -> float:
        return float(sum(self.scores.values()))


def detrend(time: np.ndarray, flux: np.ndarray, window_hours: float = 12.0):
    """Median-filter detrend. Returns (residual_in_ppm, noise_ppm).

    Window is wide relative to transit-like events (~hours) so dips survive
    detrending; stellar rotation (~days) is removed.
    """
    cadence_d = float(np.median(np.diff(time)))
    win = max(3, int(window_hours / 24.0 / cadence_d) | 1)  # odd
    trend = ndimage.median_filter(flux, size=win, mode="nearest")
    resid = (flux / trend - 1.0) * 1e6
    # robust noise: MAD -> sigma
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return resid, max(noise, 1e-3)


def _runs(mask: np.ndarray):
    """Contiguous True runs as (start, end) index pairs (end exclusive)."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def deep_dip_finder(resid: np.ndarray, noise: float, min_depth_sigma: float = 7.0,
                    min_points: int = 5) -> tuple[float, list]:
    """Aperiodic deep-dip detector (the Boyajian filter).

    Flags contiguous negative excursions deeper than min_depth_sigma whose
    shape is NOT well fit by a symmetric box/U (classic transit). Score is
    summed excess depth of qualifying dips.
    """
    mask = resid < -min_depth_sigma * noise
    score, events = 0.0, []
    for s, e in _runs(mask):
        if e - s < min_points:
            continue
        seg = resid[s:e]
        depth = -float(seg.min()) / noise
        # asymmetry: compare ingress/egress areas around the minimum
        m = int(np.argmin(seg))
        left, right = abs(seg[: m + 1].sum()), abs(seg[m:].sum())
        asym = abs(left - right) / max(left + right, 1e-9)  # 0 sym .. 1 one-sided
        # symmetric, flat-bottomed dips look planet/EB-like -> down-weight
        weight = 0.3 + 0.7 * asym
        score += depth * weight
        events.append((int(s), int(e), f"dip depth={depth:.1f}σ asym={asym:.2f}"))
    return score, events


def secular_trend(time: np.ndarray, flux: np.ndarray) -> float:
    """Monotonic dimming/brightening across the sector, in sigma of slope.

    Boyajian's star also showed secular dimming. Linear fit over the whole
    sector vs. scatter-derived slope error.
    """
    t = time - time.mean()
    f = flux / np.median(flux) - 1.0
    slope, intercept = np.polyfit(t, f, 1)
    pred = slope * t + intercept
    sig = float(np.std(f - pred)) / max(float(np.sqrt((t**2).sum())), 1e-9)
    return abs(float(slope)) / max(sig, 1e-12)


def flare_oddity(resid: np.ndarray, noise: float, min_sigma: float = 8.0) -> tuple[float, list]:
    """Positive one-off events with slow rise (flares decay fast; slow-rise
    brightenings are odd — microlensing, dust clearing, instrumentals)."""
    mask = resid > min_sigma * noise
    score, events = 0.0, []
    for s, e in _runs(mask):
        if e - s < 4:
            continue
        seg = resid[s:e]
        m = int(np.argmax(seg))
        rise, decay = m + 1, len(seg) - m
        if rise >= decay:  # flares rise fast/decay slow; inverse is weird
            amp = float(seg.max()) / noise
            score += amp
            events.append((int(s), int(e), f"slow-rise brightening {amp:.1f}σ"))
    return score, events


def run_tier1(time: np.ndarray, flux: np.ndarray) -> Tier1Result:
    """Run all filters on a single light curve (raw, NaN-free, any units)."""
    resid, noise = detrend(time, flux)
    res = Tier1Result()
    dip_score, dip_events = deep_dip_finder(resid, noise)
    flare_score, flare_events = flare_oddity(resid, noise)
    res.scores = {
        "deep_dip": dip_score,
        "secular": secular_trend(time, flux),
        "odd_brightening": flare_score,
    }
    res.events = dip_events + flare_events
    return res
