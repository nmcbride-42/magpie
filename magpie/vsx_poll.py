"""Poll VSX for adjudication state changes on submitted candidates.

A submission to the AAVSO International Variable Star Index (VSX) sits in a
moderation queue that can take weeks. An untracked "pending" status is how a
planning document can drift into claiming an acceptance that never happened,
so the acceptance channel is instrumented here instead of assumed.

For every pending TIC in the `submitted` table (see catalog.mark_submitted),
query the public VSX api.object endpoint. DB writes happen ONLY on a state
change (pending -> listed in VSX); every poll updates
$MAGPIE_DATA_DIR/.vsx-poll.json so "when did we last look" is answerable.

Stdlib only. Fail-open per TIC: a network error logs and skips — it never
invents a state change. Run daily from cron/systemd, or by hand:

    python -m magpie.vsx_poll
"""

import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from magpie.catalog import ensure_schema
from magpie.config import DATA_DIR, DB_PATH

STATE = os.path.join(DATA_DIR, ".vsx-poll.json")
API = "https://vsx.aavso.org/index.php?view=api.object&format=json&ident="
UA = "magpie-vsx-poll/1.0 (variable-star submission status check)"


def query_vsx(ident: str):
    """Return a list of VSXObject payloads for an identifier ([] if VSX has
    no record), or None on a network/parse error."""
    url = API + urllib.parse.quote(ident)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"  {ident}: poll error ({type(e).__name__}: {e}) — skipping")
        return None
    obj = data.get("VSXObject") if isinstance(data, dict) else None
    # The API returns {} or [] for no-match depending on path; treat any
    # falsy/empty payload as "not in VSX".
    if not obj:
        return []
    return obj if isinstance(obj, list) else [obj]


def poll(db_path=None, state_path=None, query=query_vsx):
    """One polling pass. Returns the number of state changes recorded."""
    db_path = db_path or DB_PATH
    state_path = state_path or STATE
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    state = {}
    if os.path.exists(state_path):
        try:
            with open(state_path) as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError):
            state = {}

    conn = sqlite3.connect(db_path, timeout=30)
    ensure_schema(conn)
    rows = conn.execute("SELECT tic, status FROM submitted").fetchall()

    changes = 0
    for tic, status in rows:
        rec = state.setdefault(str(tic), {"checks": 0})
        rec["checks"] += 1
        rec["last_checked"] = now

        if status != "pending":
            rec["result"] = f"terminal:{status}"
            continue

        objs = query(f"TIC {tic}")
        if objs is None:
            continue  # network error — no state claim
        if not objs:
            rec["result"] = "pending"
            print(f"  TIC {tic}: still pending (check #{rec['checks']})")
            continue

        # State change: VSX now lists the object.
        o = objs[0]
        name = o.get("Name", "?")
        auid = o.get("AUID", "")
        vtype = o.get("VariabilityType", "")
        rec["result"] = f"in_vsx:{name}"
        changes += 1
        note = (f" | {now.split()[0]}: listed in VSX — name={name}"
                + (f" AUID={auid}" if auid else "")
                + (f" type={vtype}" if vtype else ""))
        conn.execute("""
            UPDATE submitted
            SET status='accepted', vsx_name=?,
                note=COALESCE(note,'') || ?, updated_at=datetime('now')
            WHERE tic=?
        """, (name, note, tic))
        conn.commit()
        print(f"  TIC {tic}: STATE CHANGE — listed in VSX as "
              f"'{name}'{' (' + auid + ')' if auid else ''}")

    conn.close()
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as fh:
        json.dump(state, fh, indent=2)
    print(f"vsx_poll: {len(rows)} tracked, {changes} state change(s), {now}")
    return changes


def main() -> int:
    poll()
    return 0


if __name__ == "__main__":
    sys.exit(main())
