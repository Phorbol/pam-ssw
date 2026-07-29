from __future__ import annotations

from pathlib import Path
import runpy


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SOURCE_DRIVER = (
    RUN_ROOT.parent
    / "20260727-024648-safe-lbfgs-multiseed-gpu"
    / "run_multiseed.py"
)
FROZEN_CODE_COMMIT = "61f32ef70585adfd78b1484bd6f0cfcac8317352"


def main() -> int:
    namespace = runpy.run_path(str(SOURCE_DRIVER))
    source_main = namespace["main"]
    source_main.__globals__["OUTPUT_ROOT"] = OUTPUT_ROOT
    source_main.__globals__["FROZEN_CODE_COMMIT"] = FROZEN_CODE_COMMIT
    return source_main()


if __name__ == "__main__":
    raise SystemExit(main())
