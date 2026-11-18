"""Runtime configuration, read from environment variables.

Every path and external command the pipeline uses is resolved here so the
stages carry no machine-specific assumptions. See `.env.example` and the
README for the full list.
"""

import os
import shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the sweep database, bulk scripts, plots, dossiers and packets live.
DATA_DIR = os.path.abspath(os.environ.get("MAGPIE_DATA_DIR",
                                          os.path.join(ROOT, "data")))
DB_PATH = os.path.abspath(os.environ.get("MAGPIE_DB",
                                         os.path.join(DATA_DIR, "magpie.db")))
CACHE_DIR = os.path.join(DATA_DIR, "cache")
PLOT_DIR = os.path.join(DATA_DIR, "plots")
DOSSIER_DIR = os.path.join(DATA_DIR, "dossiers")
PACKET_DIR = os.path.join(DATA_DIR, "packets")

# LLM judge commands. Each is run with the prompt on stdin and must print a
# JSON array on stdout. The defaults use the Claude Code CLI in headless
# (print) mode; the vision judge additionally needs a tool that can read
# the PNG plot files it is pointed at.
JUDGE_CMD = shlex.split(os.environ.get("MAGPIE_JUDGE_CMD", "claude -p"))
JUDGE_MODEL = os.environ.get("MAGPIE_JUDGE_MODEL") or None
VISION_JUDGE_CMD = shlex.split(os.environ.get(
    "MAGPIE_VISION_JUDGE_CMD", "claude -p --allowedTools Read"))

# Sent with catalog HTTP requests (ExoFOP, SIMBAD, VSX).
USER_AGENT = os.environ.get("MAGPIE_USER_AGENT", "magpie/0.1")

# Discoverer name written into VSX submission packets.
DISCOVERER = os.environ.get("MAGPIE_DISCOVERER", "<your name>")
