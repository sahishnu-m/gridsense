"""
run.py: the ONLY file you need to run GridSense.

    python run.py

Why this exists (two Windows annoyances it fixes automatically):
  1. On Python 3.14 for Windows, Streamlit's startup calls platform.system(),
     which can freeze for a long time on a slow WMI system query. We return the
     Windows version directly so startup is instant and never hangs.
  2. Streamlit asks for your email on first run and waits for input. We skip it.
"""

import platform

# --- Fix 1: skip the slow WMI query --------------------------------------- #
# Must be patched FIRST, before anything calls platform.system().
platform.win32_ver = lambda *a, **k: ("11", "10.0.26200", "SP0", "Multiprocessor Free")

import sys
from pathlib import Path

# --- Fix 2: skip Streamlit's first-run email prompt ------------------------ #
_cred = Path.home() / ".streamlit" / "credentials.toml"
if not _cred.exists():
    _cred.parent.mkdir(parents=True, exist_ok=True)
    _cred.write_text('[general]\nemail = ""\n', encoding="utf-8")

from streamlit.web.cli import main

if __name__ == "__main__":
    # Same as running:  streamlit run app.py   (extra flags are passed through)
    sys.argv = ["streamlit", "run", "app.py"] + sys.argv[1:]
    sys.exit(main())
