"""Read-only tensor-handoff analysis; no calculator or native execution."""
import argparse
import json
from pathlib import Path


def flatten(matrix):
    return [x for row in matrix for x in row]


def difference(a, b):
    return max(abs(x-y) for x, y in zip(flatten(a), flatten(b)))


def analyze(root):
    provenance = json.loads((root/'provenance.json').read_text())
    events = [json.loads(x) for x in (root/'native-stress-trace.jsonl').read_text().splitlines()]
    callbacks = [json.loads(x) for x in (root/'callback-events.jsonl').read_text().splitlines()]
    assert len(callbacks) == provenance['external_requests'] == 3
    assert sum(x['actual_calculator_calls'] for x in callbacks) == provenance['actual_calculator_calls'] == 3
    assert all(x['status'] == 'calculated' for x in callbacks)
    assert provenance['process_status']['cleanup_survivors'] == []
    rows = []
    for row in events:
        if row['kind'] != 'class_struc_mp_stress2dedlatt_entry':
            continue
        cb = row['latest_callback']; sigma = cb['ase_stress_eV_A3_tensile']
        prior = row['nearest_otherpot_return']
        assert row['callback_request_id_matches_otherpot']
        assert row['effective_external_dedcell'] == prior['effective_external_dedcell'] == 0
        rows.append(dict(request=cb['request'],
            legacy_vs_ase_max_eV_A3=difference(prior['legacy_tensor']['fortran_column_major'], sigma),
            class_vs_ase_max_eV_A3=difference(row['class_stress']['fortran_column_major'], sigma),
            ase_max_stress_eV_A3=max(abs(x) for x in flatten(sigma)),
            native_cell_transpose_vs_callback_max_A=row['callback_vs_class_cell']['transposed_cell_max_abs'],
            cell_change_from_first_A=difference(cb['cell_A'], callbacks[0]['cell_A']),
            volume_A3=row['class_volume_A3'], external_pressure_stored=row['external_pressure_stored']))
    assert len(rows) == 3
    return dict(source_run=str(root.resolve()), source_head=provenance['git_head'],
        execution_status=provenance['status'], calculator_calls=3,
        parsed_flags=[x['global_flag'] for x in events if x['kind']=='external_dedcell_getter_return'],
        rows=rows,
        scope='Three changing-cell Cu4/EMT handoffs, explicit external.dedcell F and wire=-ASE stress. No search efficacy, full gradient or arbitrary-backend qualification.',
        limitations=['Class atomic positions not read; linkage uses callback ID and nearest otherpot return plus cell.',
                     'Symmetric physical stress cannot discriminate row versus column flattening.',
                     'Small class/legacy residual retained without precision tuning.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.run)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
