"""Reuse the existing saved-path true-quench harness; no search-policy changes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'early-quench'
SOURCE = ROOT / 'research/ga_ssw/evidence/climb-depth-ablation-20260925/run.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error('choose prepare or execute')
    spec = importlib.util.spec_from_file_location('saved_quench_harness', SOURCE)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.HERE = OUT
    if args.execute:
        harness.execute()
        return
    sys.path[:0] = [str(ROOT), str(ROOT / 'research/ga_ssw')]
    from analyze_c4h6_ls_reaction_coverage import atoms_from_dict
    import numpy as np
    rows, sources = [], []
    for arm in ('ssw_without_ls', 'native_ls'):
        for seed in (1101, 1102):
            p = HERE / 'runs' / f'{arm}-{seed}' / 'result.json'
            result = json.loads(p.read_text())
            record = result['records'][0]
            stages = record['climb']
            assert record['index'] == 0 and len(stages) == 12
            overhead = record['evaluation_requests'] - record['landing']['evaluation_requests'] - sum(s['requests'] for s in stages)
            assert overhead >= 0
            sources.append(dict(path=str(p), record_index=0))
            for depth in (2, 3, 6):
                point = dict(result['initial']['atoms'])
                point['positions'] = stages[depth]['center']
                a = atoms_from_dict(point)
                assert len(a) == 60 and (a.numbers == 6).all() and not a.pbc.any()
                assert np.isfinite(a.positions).all()
                rows.append(dict(case_id=f'C60-{arm}-{seed}-r0-k{depth}',
                    system='C60', arm=arm, seed=seed, record_index=0, depth=depth,
                    full_depth=12, start=result['initial']['atoms'], point=point,
                    full=record['landing'], prefix_requests=overhead+sum(s['requests'] for s in stages[:depth]),
                    nonstage_overhead=overhead, original_full_requests=record['evaluation_requests'],
                    steps=1000, fmax=.03, memory=500))
    assert len(rows) == 12
    OUT.mkdir(exist_ok=False)
    (OUT / 'wrapper.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'runner.py').write_bytes(SOURCE.read_bytes())
    harness.save(OUT / 'inputs.json', dict(rows=rows, sources=sources,
        core_tree=harness.git('rev-parse', 'HEAD:pamssw'), git_head=harness.git('rev-parse', 'HEAD'),
        model_sha256=harness.MODEL_SHA, search_cap_per_case=1000, fresh_cap_per_case=2, total_cap=12024))
    print('12 saved geometries prepared; zero PES requests')


if __name__ == '__main__':
    main()
