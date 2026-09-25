"""Zero-PES cost of the production snapshot copier on real history/payloads."""
import gc
import json
from pathlib import Path
import sys
import time
from dataclasses import replace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from pamssw.standalone.paper_reference import (
    SSWProgress, load_ssw_checkpoint, _checkpoint_copy,
)

SOURCE = Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/'
    'research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/ssw-seed61/checkpoint.pkl')
checkpoint = load_ssw_checkpoint(SOURCE)
progress = SSWProgress('outer_step', checkpoint.records[-1], None,
    checkpoint.current, checkpoint.current_energy, checkpoint.best,
    checkpoint.evaluation_requests, checkpoint.next_index)
rows = []
for n in (10, 50, 100, 200, 400):
    if n > len(checkpoint.records):
        continue
    prefix = replace(checkpoint, records=checkpoint.records[:n],
                     minima=checkpoint.minima[:n+1])
    row = {'history_records': n}
    for kind, payload in (('full', prefix), ('compact', progress)):
        samples = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            copied = _checkpoint_copy(payload)
            samples.append(time.perf_counter() - start)
            del copied
        row[kind + '_seconds'] = samples
    rows.append(row)
result = dict(source=str(SOURCE), rows=rows, PES_requests=0,
    scope='Copy-only comparison: same latest-step compact payload across history sizes. '
          'Prefixes are not valid restart files. Not a whole-search speedup.')
(HERE / 'copy-profile.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
