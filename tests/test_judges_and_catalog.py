"""LLM-judge plumbing, catalog finalist logic and the VSX poller, offline.

No LLM is called: judge_batch is replaced with fakes that return what real
judges have returned, including malformed output.
"""
import json
import sqlite3

import pytest

from magpie import catalog, tier3, tier35, vsx_poll
from magpie.enrich import color_hex, spectral_type


# ------------------------------------------------------------------ judges

def test_parse_verdicts_tolerates_prose_and_rejects_garbage():
    out = 'Sure! Here you go:\n[{"tic": 1, "verdict": "mundane"}]\nDone.'
    assert tier3.parse_verdicts(out) == [{"tic": 1, "verdict": "mundane"}]
    with pytest.raises(ValueError):
        tier3.parse_verdicts("I could not judge these.")


def _funnel_db(path, survivors):
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE tier1 (sector INTEGER, tic INTEGER,
                  deep_dip REAL, secular REAL, odd_brightening REAL,
                  n_events INTEGER, events TEXT, n_cadences INTEGER,
                  status TEXT, PRIMARY KEY (sector, tic))""")
    db.execute("""CREATE TABLE tier2 (sector INTEGER, tic INTEGER,
                  verdict TEXT, reasons TEXT, simbad_otype TEXT,
                  PRIMARY KEY (sector, tic))""")
    for tic, score in survivors:
        db.execute("INSERT INTO tier1 VALUES (7,?,?,0,0,1,'[]',18000,'ok')",
                   (tic, score))
        db.execute("INSERT INTO tier2 VALUES (7,?,'survive','',NULL)", (tic,))
    db.commit()
    db.close()


def test_tier3_skips_malformed_verdicts_and_leaves_them_queued(tmp_path,
                                                                monkeypatch):
    db_path = str(tmp_path / "f.db")
    _funnel_db(db_path, [(1, 50.0), (2, 40.0), (3, 30.0)])
    seen = []

    def fake_judge(rows, model=None):
        seen.append([r[0] for r in rows])
        return [{"tic": 1, "verdict": "mundane", "mundane_account": "EB",
                 "confidence": 0.9},
                {"tic": 2, "verdict": "interesting", "interest": "asym",
                 "confidence": 0.6},
                {"tic": 3, "mundane_account": "dropped its verdict key"},
                "not even a dict"]

    monkeypatch.setattr(tier3, "judge_batch", fake_judge)
    tier3.run(7, db_path=db_path)
    assert seen == [[1, 2, 3]]                  # highest score first, one batch
    db = sqlite3.connect(db_path)
    got = dict(db.execute("SELECT tic, verdict FROM tier3").fetchall())
    assert got == {1: "mundane", 2: "interesting"}
    # tic 3 is still unjudged, so the next round re-queues it
    left = tier3.fetch_survivors(db, 7, 10)
    assert [r[0] for r in left] == [3]


def test_tier3_respects_the_per_run_cap(tmp_path, monkeypatch):
    db_path = str(tmp_path / "cap.db")
    _funnel_db(db_path, [(t, float(t)) for t in range(1, 21)])
    batches = []
    monkeypatch.setattr(tier3, "judge_batch",
                        lambda rows, model=None: batches.append(len(rows)) or [])
    tier3.run(7, max_per_run=10, db_path=db_path)
    assert sum(batches) == 10 and max(batches) <= tier3.BATCH


def test_tier35_shortlist_is_interesting_or_low_confidence(tmp_path):
    db = sqlite3.connect(str(tmp_path / "s.db"))
    db.executescript(tier3.SCHEMA + tier35.SCHEMA)
    db.executemany("INSERT INTO tier3 (sector,tic,verdict,confidence) "
                   "VALUES (7,?,?,?)",
                   [(1, "mundane", 0.95), (2, "mundane", 0.5),
                    (3, "interesting", 0.7), (4, "interesting", 0.99)])
    db.execute("INSERT INTO tier35 (sector,tic,verdict) VALUES (7,4,'mundane')")
    assert [r[0] for r in tier35.shortlist(db, 7)] == [3, 2]


# ----------------------------------------------------------------- catalog

def test_load_finalists_is_schema_tolerant(tmp_path, monkeypatch):
    db_path = str(tmp_path / "c.db")
    ddir = tmp_path / "dossiers"
    ddir.mkdir()
    monkeypatch.setattr(catalog, "DB_PATH", db_path)
    monkeypatch.setattr(catalog, "DOSSIER_DIR", str(ddir))
    db = sqlite3.connect(db_path)
    db.executescript(tier35.SCHEMA)
    db.executemany("INSERT INTO tier35 (sector,tic,verdict) "
                   "VALUES (9,?,'interesting')", [(1,), (2,), (3,)])
    db.commit()
    db.close()
    # new-style dossier with a top-level resolved flag
    (ddir / "tic1.json").write_text(json.dumps({
        "resolved": True, "period_d": 4.2, "depth_measured": 0.08,
        "period_resolution": {"resolved": True, "mechanism": "eclipsing binary"}}))
    # old-style dossier: no flag, but an overwhelming BLS detection
    (ddir / "tic2.json").write_text(json.dumps({
        "bls": {"best_period_d": 1.1, "depth": 0.03, "power_ratio": 80}}))
    # tic 3: no dossier at all -> unresolved, still cross-checked
    got = {t: (p, d, m, r) for t, p, d, m, r in catalog.load_finalists(9)}
    assert got[1] == (4.2, 0.08, "eclipsing binary", True)
    assert got[2] == (1.1, 0.03, "eclipsing binary", True)
    assert got[3] == (None, None, "unresolved", False)


def test_mark_submitted_and_vsx_poll_state_change(tmp_path, monkeypatch):
    db_path = str(tmp_path / "v.db")
    monkeypatch.setattr(catalog, "DB_PATH", db_path)
    catalog.mark_submitted(111, "VSX, pending moderation")
    catalog.mark_submitted(222, "VSX, pending moderation")
    assert catalog.submitted_tics() == {111, 222}

    answers = {"TIC 111": [], "TIC 222": [{"Name": "TIC 222", "AUID": "000-AAA-000",
                                           "VariabilityType": "EA"}]}
    state = str(tmp_path / "poll.json")
    n = vsx_poll.poll(db_path=db_path, state_path=state,
                      query=lambda ident: answers[ident])
    assert n == 1
    db = sqlite3.connect(db_path)
    rows = dict(db.execute("SELECT tic, status FROM submitted").fetchall())
    assert rows == {111: "pending", 222: "accepted"}

    # a network failure (None) must never invent a state change
    n_after_error = vsx_poll.poll(db_path=db_path, state_path=state,
                       query=lambda ident: None)
    assert n_after_error == 0
    s = json.loads(open(state).read())
    assert s["111"]["checks"] == 2 and s["222"]["result"] == "terminal:accepted"


def test_vsx_blurb_and_star_facts_helpers():
    b = catalog.vsx_blurb(5, 10.0, -5.0, 2.345678, 0.123, "eclipsing binary")
    assert "period=2.3457 d" in b and "depth~12.3%" in b and "Dec=-5.00000" in b
    assert spectral_type(5800) == "G" and spectral_type(3000) == "M"
    assert spectral_type(None) is None
    assert color_hex(None) == "#cccccc"
    assert color_hex(100000) == "#9bb0ff"       # clamped to the hot end
