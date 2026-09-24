"""Real Cu13/EMT, independent-process active GA walk recovery qualification."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
RUNS = HERE / 'runs'
PHASES = ('quick', 'generation_short', 'fine', 'offspring_ssw')


def serializer():
    path = HERE.parent / 'periodic-rotation-priority-20260923/ledger.py'
    spec = importlib.util.spec_from_file_location('ga_replay_serializer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._jsonable


def strip_checkpoints(value):
    if isinstance(value, dict):
        return {k: strip_checkpoints(v) for k, v in value.items() if k != 'checkpoint'}
    if isinstance(value, list):
        return [strip_checkpoints(v) for v in value]
    return value


def run(name, request_allowance):
    import numpy as np
    from ase.io import read
    from ase.calculators.emt import EMT
    from pamssw.standalone import ASESurface, SSWConfig, PaperGAConfig
    from pamssw.standalone.ga_checkpoint import GACheckpoint
    from pamssw.standalone.legacy_descriptor import cluster_descriptor
    from pamssw.standalone.paper_ga import run_ga_ssw

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=False)
    source = HERE.parent / 'population-comparison-20260923'
    inherited = json.loads((source / 'cu13-seed3.json').read_text())
    initial = [read(source / f'inputs/cu13-{i}.extxyz') for i in range(3)]
    ssw = SSWConfig(**inherited['ssw'])
    options = dict(inherited['ga'])
    options.update(quick_steps=2, generation_steps=2, fine_steps=2,
                   offspring_steps=2, ga_candidates=4, generations=1, cycles=1)
    config = PaperGAConfig(**options)
    descriptor = inherited['descriptor']
    bonds = {tuple(row[:2]): row[2] for row in descriptor['bond_lengths']}
    references = tuple(cluster_descriptor(a.numbers, a.positions, bonds,
                        descriptor['neighbor_range']) for a in initial)
    resume = name.endswith('-resume')
    pause = name.endswith('-pause')
    phase = name.rsplit('-', 1)[0] if (resume or pause) else None
    checkpoint = GACheckpoint.load(RUNS / f'{phase}-pause' / 'ga.pkl') if resume else None
    rng = np.random.default_rng(999 if resume else inherited['seed'])
    class BoundedSurface(ASESurface):
        def evaluate(self, atoms):
            if self.requests >= request_allowance:
                raise RuntimeError('Series aggregate request allowance exhausted')
            return super().evaluate(atoms)
    surface = BoundedSurface(EMT())
    captured = []

    def callback(state):
        active = getattr(state, 'active_walk', None)
        if pause and active is not None and active.phase == phase:
            state.save(out / 'ga.pkl')
            captured.append(active.phase)
            return True
        return False

    result = run_ga_ssw(initial, surface, groups=None, references=references,
        descriptor_bonds=bonds, descriptor_weights=descriptor['weights'],
        neighbor_range=descriptor['neighbor_range'], proposal_bond_limits={},
        config=config, ssw_config=ssw, rng=rng, max_evaluations=20000,
        checkpoint=checkpoint, checkpoint_callback=callback,
        checkpoint_walk_steps=name != 'full-default')
    encode = serializer()
    canonical = dict(result=strip_checkpoints(encode(result)), rng=encode(rng.bit_generator.state))
    (out / 'canonical.json').write_text(json.dumps(canonical, sort_keys=True, allow_nan=False) + '\n')
    summary = dict(name=name, status=result.status, actual_requests=surface.requests,
        cumulative_requests=result.evaluation_requests, captured=captured,
        phases=[stage.phase for stage in result.stages], observations=len(result.observations))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if pause:
        assert captured == [phase] and result.checkpoint is not None, summary
    else:
        assert not result.budget_exhausted and result.status in ('completed', 'completed_with_failures'), summary
    print(json.dumps(summary))


def execute():
    if RUNS.exists():
        raise FileExistsError('No implicit replay/overwrite')
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('diff', 'HEAD', '--', 'pamssw'):
        raise RuntimeError('Commit the reviewed core before qualification')
    (HERE / f'{RUNS.name}-execution.json').write_text(json.dumps(dict(
        git_head=git('rev-parse', 'HEAD'), core_tree=git('rev-parse', 'HEAD:pamssw'),
        python=sys.version, command=[sys.executable, str(Path(__file__).resolve()), '--execute']), indent=2) + '\n')
    names = ['full-default', 'full-optin']
    names += [f'{phase}-{part}' for phase in PHASES for part in ('pause', 'resume')]
    for name in names:
        spent = sum(json.loads(p.read_text())['actual_requests']
                    for series in ('runs', 'runs-v2')
                    for p in (HERE / series).glob('*/summary.json'))
        allowance = 120000 - spent
        if allowance <= 0:
            raise RuntimeError('Aggregate search budget exhausted')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run', name,
                        '--series', RUNS.name, '--request-allowance', str(allowance)], check=True)
    reference = json.loads((RUNS / 'full-default/canonical.json').read_text())
    summaries = {name: json.loads((RUNS / name / 'summary.json').read_text()) for name in names}
    total = sum(row['actual_requests'] for row in summaries.values())
    previous = sum(json.loads(p.read_text())['actual_requests'] for p in (HERE / 'runs').glob('*/summary.json')) if RUNS.name != 'runs' else 0
    assert previous + total <= 120000, (previous, total)
    full_cost = summaries['full-default']['actual_requests']
    for name in ['full-optin'] + [f'{phase}-resume' for phase in PHASES]:
        actual = json.loads((RUNS / name / 'canonical.json').read_text())
        assert actual == reference, f'Full-result/RNG mismatch: {name}'
    for phase in PHASES:
        assert (summaries[f'{phase}-pause']['actual_requests'] +
                summaries[f'{phase}-resume']['actual_requests']) == full_cost, phase
    from ase import Atoms
    from ase.calculators.emt import EMT
    import numpy as np
    archive = reference['result']['archive']
    assert len(archive) <= 1000
    fresh = []
    for row in archive:
        atoms = Atoms(**row['atoms'])
        atoms.calc = EMT()
        energy = float(atoms.get_potential_energy())
        fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        fresh.append(dict(id=row['id'], energy=energy, fmax=fmax))
        assert abs(energy - row['energy']) <= 1e-10 and fmax <= .01 + 1e-12
    result = dict(status='passed', total_search_requests=total, fresh_requests=len(fresh),
                  comparisons=5, recovered_phases=list(PHASES), summaries=summaries, fresh=fresh)
    (HERE / f'{RUNS.name}-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('summaries', 'fresh')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--series', default='runs')
    parser.add_argument('--request-allowance', type=int, default=20000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.series not in ('runs', 'runs-v2'):
        parser.error('Only frozen original or focused repair run permitted')
    RUNS = HERE / args.series
    if args.run and args.execute:
        parser.error('Choose one mode')
    if args.run:
        run(args.run, args.request_allowance)
    elif args.execute:
        execute()
    else:
        print('Prepared only; no execution')
