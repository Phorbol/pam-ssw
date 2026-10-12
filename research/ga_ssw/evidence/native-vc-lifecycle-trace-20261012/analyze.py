"""Offline correction/readout of native trace1754624; raw evidence stays immutable."""
import collections
import json
from pathlib import Path
import re
import sys

run, out = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
process = json.loads((run / 'process.json').read_text())
assert process['state'] == 'completed' and process['returncode'] == 0
assert not process['cleanup_survivors']
assert not (run / 'trace-error.json').exists()
corrected = []
per_step = collections.defaultdict(collections.Counter)
observed_types = []
for i, raw in enumerate(rows):
    row = dict(raw)
    if raw['label'] == 'climb_post_dispatch':
        assert raw['registers']['rbx'] == 0x53ed7a0
        after = rows[i + 1]
        assert after['label'] == 'update_mode0'
        assert after['registers']['rdi'] == raw['registers']['r13']
        # Instructions5f3750..5f3762 only read param/type-table and call update;
        # no write to nsswstep/lcellmove occurs before the next entry.
        for key in ('object', 'nsswstep', 'lcellmove_raw'):
            row[key] = after[key]
        row['object_source'] = 'inferred_from_adjacent_update_entry_same_descriptor'
        row['run_type_source'] = 'observed_para_plus_0x100'
        observed_types.append(row['run_type'])
    else:
        row['run_type'] = None
        row['run_type_source'] = 'not_observed_old_trace_read_wrong_object_offset'
    per_step[row['nsswstep']][row['label']] += 1
    row['call_count_within_step'] = per_step[row['nsswstep']][row['label']]
    corrected.append(row)
assert observed_types == [15, 15]
text = (run / 'lasp.out').read_text()
assert 'Variable Cell Structure Search:  Run_type=15' in text
assert 'SSW all done' in text
stages = collections.defaultdict(lambda: {'climb_iterations': []})
for line_number, line in enumerate(text.splitlines(), 1):
    for pattern, key in ((r'^\s*disp_step\s+(\d+)', 'displacement_line'),
                         (r'^\s*finish ng\s+(\d+)', 'finish_line')):
        match = re.search(pattern, line)
        if match:
            stages[int(match[1])][key] = line_number
    match = re.search(r'^\s*SSW_climb\s+(\d+)\s+(\d+)', line)
    if match:
        stages[int(match[1])]['climb_iterations'].append(int(match[2]))
generators = [row for row in corrected if row['label'] == 'gen_randommode']
assert len(generators) == len(stages) == 3
for stage, event in zip(sorted(stages), generators):
    assert event['nsswstep'] == 1 and event['lcellmove_raw'] == -1
    stages[stage]['generator_event'] = event['event']
    stages[stage]['generator_return_address'] = hex(event['return_address'])
    assert stages[stage]['climb_iterations'] == list(range(1, len(stages[stage]['climb_iterations']) + 1))
output = {'raw_run': str(run.resolve()), 'scope': 'one native cell outer step; no search or phase validation',
          'correction': 'run_type belongs to ssw_para; dispatch rbx is para, r13 is CSSW descriptor',
          'parameter_run_type_observations': observed_types,
          'counts': dict(collections.Counter(r['label'] for r in corrected)),
          'counts_by_nsswstep': {str(k): dict(v) for k, v in per_step.items()},
          'native_cell_gaussian_stages': dict(stages), 'process': process,
          'limits': 'random-generator invocation per observed stage; not proof of independent full direction redraw, paper2014 cycle identity, full stress qualification or efficacy'}
out.mkdir(exist_ok=False)
(out / 'corrected-trace.json').write_text(json.dumps(corrected, indent=2) + '\n')
(out / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
