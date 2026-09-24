#!/usr/bin/env python3
"""Run the shared escape probe against the separate direction-probe plan."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

import escape_probe as runner

runner.PLAN = HERE / "direction-probe" / "plan.json"
runner.PREFLIGHT = HERE / "direction-probe" / "preflight.json"

if __name__ == "__main__":
    runner.main()
