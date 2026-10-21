"""Where scripts write their outputs.

Defaults to results/ at the repo root; set ALGZOO_RESULTS_DIR to write
elsewhere (the script smoke tests use a temporary directory).
"""

import os
from pathlib import Path

RESULTS_DIR = Path(os.getenv("ALGZOO_RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
