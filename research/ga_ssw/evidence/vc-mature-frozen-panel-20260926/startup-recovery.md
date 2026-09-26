# Startup-only recovery

Parent: run-1499633/case0-safe-lbfgs-total-biased, process exit 124.
No task directory/task.json or source-validation/search ledger was produced. The
worker creates those after imports; observed stdout contains only torch/e3nn
import warnings. Thus no search evaluation is evidenced, and this is not an
observed optimizer failure. Exact import stall cause remains unknown.

Run that one task once in a new directory with the same frozen plan and numerical
source. Only external process timeout increases from 120 to 360 seconds to allow
startup; the search deadline stays 100 seconds, cap 600 requests, and all optimizer
parameters stay unchanged. Budget: 1 V100, 8 minutes, up to 602 E/F/stress requests.
Preserve the original exit in the final denominator and report recovered data
separately. No automatic additional retry if this attempt fails.
