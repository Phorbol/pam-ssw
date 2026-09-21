"""Small EMT check of GA boundary resume; this is not a search benchmark."""
import json
from pathlib import Path
import shutil
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from pamssw.standalone import ASESurface, PaperGAConfig, SSWConfig, run_ga_ssw
from pamssw.standalone.ga_checkpoint import GACheckpoint
from pamssw.standalone.legacy_descriptor import cluster_descriptor
from pamssw.standalone import paper_ga


class LedgerSurface(ASESurface):
    def __init__(self):
        super().__init__(EMT())
        self.ledger = []

    def evaluate(self, atoms):
        try:
            energy, forces = super().evaluate(atoms)
            self.ledger.append({'ok': True, 'positions': atoms.positions.tolist(),
                                'energy': float(energy), 'forces': np.asarray(forces).tolist()})
            return energy, forces
        except Exception as error:
            self.ledger.append({'ok': False, 'positions': atoms.positions.tolist(),
                                'error': f'{type(error).__name__}: {error}'})
            raise


def _inputs():
    source = Path('research/ga_ssw/evidence/cu13-safe-total/strict-validation/17-dimer.json')
    rows = json.loads(source.read_text())
    initial, groups = [], []
    for row in rows:
        if row.get('qualified') and row.get('fingerprint_group') not in groups:
            initial.append(Atoms('Cu13', positions=row['positions']))
            groups.append(row['fingerprint_group'])
        if len(initial) == 3:
            break
    if len(initial) != 3:
        raise RuntimeError('expected three qualified Cu13 source structures')
    bonds = {(29, 29): 2.259876}
    refs = tuple(cluster_descriptor(a.numbers, a.positions, bonds, 2.) for a in initial)
    refs = (refs * 3)[:3]
    return initial, bonds, refs, source, groups


def _kwargs(initial, bonds, refs, seed, surface):
    config = PaperGAConfig(
        quick_steps=1, generations=1, generation_steps=1, fine_steps=1,
        ga_candidates=8, regions=1, fine_regions=1, quench_fmax=.01,
        quench_steps=200, proposal_max_batches=2, proposal_max_cut_attempts=1000,
        proposal_max_pair_attempts=10000, partition_max_draws=10000,
        projection_tolerance=1e-4, energy_window=10., proposal_type=0,
        proposal_max_insertion_attempts=10000)
    ssw = SSWConfig(width=.2, rotation_bias=100., max_gaussians=14,
        temperature_K=300., fmax=.01, relax_steps=200, fd_step=1e-4,
        rotation_hvp=100, rotation_tol=.02, rotation_solver='dimer',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total')
    return dict(initial=initial, surface=surface, groups=None, references=refs,
        descriptor_bonds=bonds, descriptor_weights=(.3, .2, .2, .1, .1, .1),
        neighbor_range=2., proposal_bond_limits={}, config=config, ssw_config=ssw,
        rng=np.random.default_rng(seed), max_evaluations=20000)


def _fingerprint(result):
    return [(o.phase, o.generation, o.parent_ids, o.operator,
             o.eligible_for_archive, o.result.atoms.positions.tolist(),
             float(o.result.energy)) for o in result.observations]


def main():
    initial, bonds, refs, source, groups = _inputs()
    out = Path('research/ga_ssw/evidence/ga-checkpoint-emt-full-20260912-v2')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sources = out / 'sources'
    sources.mkdir()
    shutil.copytree('pamssw', sources / 'pamssw')
    (out / 'manifest.json').write_text(json.dumps(
        {'source': str(source), 'groups': groups, 'seed': 3, 'max_evaluations': 20000,
         'input_count': len(initial)}, indent=2) + '\n')
    rows = []
    for label, stop_phase in (('quick', 'quick_complete'),
                              ('generation', 'generation_complete')):
        split_surface = LedgerSurface()
        captured = []
        partial = run_ga_ssw(**_kwargs(initial, bonds, refs, 3, split_surface),
            checkpoint_callback=lambda state: captured.append(state) or state.phase == stop_phase)
        if partial.checkpoint is None:
            raise RuntimeError(f'expected {label} boundary checkpoint: '
                               f'status={partial.status}, phases={[s.phase for s in captured]}, '
                               f'requests={split_surface.requests}')
        path = out / f'{label}.chk'
        partial.checkpoint.save(path)
        resumed_surface = LedgerSurface()
        resumed = run_ga_ssw(**_kwargs(initial, bonds, refs, 3, resumed_surface),
            checkpoint=GACheckpoint.load(path))
        full_surface = LedgerSurface()
        full = run_ga_ssw(**_kwargs(initial, bonds, refs, 3, full_surface))
        fresh = []
        for row in resumed.archive:
            energy, forces = ASESurface(EMT()).evaluate(row['atoms'])
            error = abs(float(energy) - float(row['energy']))
            max_force = float(np.linalg.norm(forces, axis=1).max())
            fresh.append(dict(energy=float(energy), energy_error=error, max_force=max_force))
        (out / f'{label}-ledger.json').write_text(json.dumps(
            {'partial': split_surface.ledger, 'resumed': resumed_surface.ledger,
             'full': full_surface.ledger}, indent=2) + '\n')
        if _fingerprint(full) != _fingerprint(resumed):
            raise AssertionError(f'{label}: geometry/lineage mismatch')
        if full.evaluation_requests != resumed.evaluation_requests:
            raise AssertionError(f'{label}: request mismatch')
        if not full.archive:
            raise AssertionError(f'{label}: no qualified archive minimum')
        if any(row['energy_error'] > 1e-10 or row['max_force'] > 0.01 for row in fresh):
            raise AssertionError(f'{label}: fresh archive certificate failed')
        rows.append(dict(label=label, full_requests=full.evaluation_requests,
                         split_requests=partial.evaluation_requests,
                         resumed_requests=resumed.evaluation_requests,
                         split_ledger=len(split_surface.ledger),
                         resumed_ledger=len(resumed_surface.ledger),
                         full_ledger=len(full_surface.ledger),
                         observations=len(resumed.observations), archive=len(resumed.archive),
                         fresh_archive=fresh, status=resumed.status))
    (out / 'summary.json').write_text(json.dumps(
        dict(source=str(source), runs=rows, complete=True), indent=2) + '\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
