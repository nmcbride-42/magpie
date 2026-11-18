# Magpie design notes

*A long, patient sweep over open sky data, collecting the shiny weird things.*

This is the original design for Magpie, lightly edited, followed by what
changed once the pipeline met real data.

## The bet

Anomaly-hunting in open survey data has an unusual property: the verifier is
the data itself. A genuine outlier is self-evidencing, and no lab is needed.
The precedent is Boyajian's Star (KIC 8462852), found by citizen scientists
looking at Kepler light curves that the official pipeline had classified as
nothing. The gap between "what the pipeline flags" and "what is in the data"
is real and persistent, and TESS keeps shipping new sectors.

Any one of these would count as success:

1. A genuinely anomalous object, cross-checked against known catalogs
   (ExoFOP, SIMBAD, VSX) and written up with light curves and a falsifiable
   characterization.
2. A reusable anomaly-hunting harness that can be pointed at any time-series
   corpus. The infrastructure is a deliverable even if the sky is quiet.
3. A ranked list of candidates with judge-verified rationale: direction for
   follow-up even without a headline object.

## Architecture: smallest tool that works, escalating

A funnel. Each tier is roughly 10-100x cheaper than the next, and LLM
judgment runs only on survivors.

```
Tier 0  Bulk fetch          MAST bulk download, sector by sector, streamed
Tier 1  Mechanical filters  pure numpy/scipy, no LLM. Detrend, then flag
                            aperiodic deep dips, asymmetric dips, slow-rise
                            brightenings, secular trends
Tier 2  Known-object veto   cross-match survivors against ExoFOP TOIs and
                            SIMBAD object types; kill everything explained
Tier 3  LLM judge panel     hostile by default: argue FOR a mundane
                            explanation (instrumental artifact, eclipsing
                            binary, scattered light). Survives only if no
                            mundane account fits the evidence.
Tier 4  Deep dossier        per finalist: every available sector, period
                            search, pixel-level checks, falsifiable claims
```

## Budgets and stop conditions (set before any unattended run)

- **LLM budget:** judging is batched and capped per run. If Tier 3 sees
  thousands of candidates, the Tier 1/2 thresholds are wrong and the fix is
  upstream, not a bigger budget.
- **Disk budget:** light curves only (a few hundred KB each), never full-frame
  images. The sweep deletes each FITS file as soon as it is scored, so disk
  use stays flat regardless of sector size.
- **Wall clock:** every unattended run has an explicit end. The harness stops
  the run, not the model.
- **Honesty gate:** no candidate leaves the pipeline without a dossier listing
  the mundane explanations considered and why they fail. We do not announce a
  discovery. We report a candidate.

## Validation discipline

Tier 1 is checked two ways before any new flag is trusted. A single
planted dip in a real TESS curve must be recovered and localized (a smoke
check, not a recall measurement), and a known weird object must be
re-found: a filter that can't re-find Boyajian's Star in Kepler data doesn't
get to call anything in TESS anomalous. Both checks live in
`tests/test_network.py`. Recall and precision over real sectors have not been
measured.

## What changed once it met the sky

The plan above was written before the first sector was swept. The pipeline
that exists today differs in instructive ways:

- **Common-mode veto (Tier 2.5).** The first Tier-3 batch showed most
  "anomalies" sharing one cadence window across unrelated stars: a spacecraft
  event, not the sky. The LLM judge spotted it; the fix was to promote that
  observation to a free statistical filter (a Poisson-tail test on how many
  independent stars event in each cadence bin) so it never costs tokens again.
- **The judge needed eyes (Tier 3.5).** The text judge saw scores and event
  windows but not gaps, and its first "interesting" verdicts were all
  artifacts next to data gaps, obvious in seconds on a plot. Tier 3.5 adds a
  mechanical gap veto and a vision judge that reads the rendered light curve,
  and asks it to name the morphology (EB, dipper, flare, pulsator,
  single-transit, artifact), not just interesting/mundane.
- **Most anomalies are eclipsing binaries.** A sector's worth of "aperiodic"
  dips often folds cleanly once every TESS sector of the star is stitched
  together. Tier 4 became a period resolver (BLS over the stitched curve plus
  a dip-time fold that handles eccentric binaries), and the realistic output
  of the pipeline became *uncatalogued eclipsing binaries*: modest, but
  checkable and submittable.
- **Catalog check and submission packets (Tiers 5-6).** Resolved binaries are
  checked against VSX, SIMBAD, the TESS EB catalog and Gaia DR3 variability
  tables. The ones nobody has catalogued get a VSX submission packet. VSX
  moderation feedback then drove a round of fixes, each with a regression
  test: magnitude-scale phase plots, an epoch on the true primary minimum,
  period refinement before folding, and a fold-coherence gate that refuses
  to emit numbers from a smeared fold.
- **Operations.** Resumable stages, a write-retry layer for concurrent SQLite
  writers, a hunt that reports honest completion (named gaps, not
  "complete"), and a watchdog that resumes sectors whose hunt was killed.
