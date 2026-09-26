"""Bounded diagnostic of the original EMT preflight; no algorithm changes."""
import faulthandler
import json
import time
from pathlib import Path
import numpy as np
from ase.calculators.emt import EMT
from research.ga_ssw.tests.test_vc_frozen_optimizer_panel import FrozenVCPanelTest
from research.ga_ssw.run_vc_frozen_optimizer_panel import run_stage

faulthandler.dump_traceback_later(45, exit=True)
test = FrozenVCPanelTest(); test.setUp()
root = Path(__file__).resolve().parent / 'preflight-diagnostic'
root.mkdir(exist_ok=False)
class TracingEMT(EMT):
    def calculate(self, atoms=None, *args, **kwargs):
        print(json.dumps({'event': 'before_emt', 'volume': atoms.get_volume(),
            'cell': atoms.cell.array.tolist()}), flush=True)
        result = super().calculate(atoms, *args, **kwargs)
        print(json.dumps({'event': 'after_emt'}), flush=True)
        return result
for method in ('safe-lbfgs-total', 'ase-lbfgs-linesearch', 'scipy-lbfgsb'):
    for phase, q, terms in (('biased', test.q_start, test.terms), ('unbiased', test.q_saved, [])):
        print(json.dumps({'event': 'stage_start', 'method': method, 'phase': phase}), flush=True)
        result = run_stage(method, phase, q, test.spec, test.chart, terms,
            TracingEMT, root / f'{method}-{phase}', time.monotonic()+40)
        print(json.dumps({'event': 'stage_end', 'status': result['status'],
            'requests': result['search_requests']}), flush=True)
faulthandler.cancel_dump_traceback_later()
