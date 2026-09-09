"""Bounded independent GA-SSW / GFN2-xTB water integration experiment.

Optional dependency: tblite==0.7.0. Set OMP_NUM_THREADS=1 and
OPENBLAS_NUM_THREADS=1 before Python starts; run under an external timeout.
No uploaded LASP/Java code executes. Numerical settings below are a declared
short workflow experiment, not tuned defaults or a paper-efficiency benchmark.
"""
from dataclasses import asdict, fields, is_dataclass
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from ase import Atoms
from ase.io import read, write
from tblite.ase import TBLite

from pamssw.standalone.surface import ASESurface
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.paper_ga import PaperGAConfig, run_ga_ssw
from pamssw.standalone.legacy_descriptor import cluster_descriptor


def serial(value):
    if isinstance(value, Atoms):
        return dict(numbers=value.numbers.tolist(), positions=value.positions.tolist(),
                    cell=value.cell.tolist(), pbc=value.pbc.tolist())
    if is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default='research/ga_ssw/evidence/independent-water-gfn2-ga')
    parser.add_argument('--quench-steps', type=int, default=200)
    parser.add_argument('--relax-steps', type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    calls = (output / 'evaluations.jsonl').open('w')

    class TracedSurface(ASESurface):
        def evaluate(self, atoms):
            e, f = super().evaluate(atoms)
            calls.write(json.dumps(dict(request=self.requests, elapsed=time.monotonic()-started,
                 energy=e, max_force=float(np.linalg.norm(f, axis=1).max()),
                 positions=atoms.positions.tolist()))+'\n')
            calls.flush()
            if self.requests % 100 == 0:
                print(f'physical requests={self.requests}, elapsed={time.monotonic()-started:.1f}s', flush=True)
            return e, f

    # tblite 0.7.0 ase.py:403-405 sets energy == free_energy, but omits
    # free_energy in implemented_properties. Explicitly request energy;
    # this is a verified backend convention, not a silent generic fallback.
    surface = TracedSurface(TBLite(method='GFN2-xTB', verbosity=0), force_consistent=False)
    raw = Path('research/ga_ssw/evidence/phase1/water-original-complete-04/final-arc')
    paths = [raw / f'{i}.arc' for i in (0, 4, 8)]
    initial = [read(path, format='dmol-arc') for path in paths]
    for atoms in initial:
        atoms.pbc = False
    groups = tuple(tuple(range(i, i+3)) for i in range(0, 45, 3))
    assert all(a.get_chemical_symbols() == ['O', 'H', 'H']*15 for a in initial)
    fixture = json.loads(Path('tests/fixtures/ga_ssw/water.json').read_text())
    bonds = {(int(a), int(b)): value for a, b, value in fixture['bond_lengths']}
    references = [cluster_descriptor(a.numbers, a.positions, bonds, fixture['neighbor_range']) for a in initial]
    ssw = SSWConfig(width=.2, rotation_bias=100., max_gaussians=2,
        temperature_K=300., fmax=.01, relax_steps=args.relax_steps, fd_step=.001,
        rotation_hvp=80, rotation_tol=.02, direction_sampling='paper')
    ga = PaperGAConfig(quick_steps=1, generations=1, generation_steps=1,
        fine_steps=1, ga_candidates=4, regions=1, fine_regions=1,
        quench_fmax=.01, quench_steps=args.quench_steps, proposal_max_batches=2,
        proposal_max_cut_attempts=100, proposal_max_pair_attempts=100,
        partition_max_draws=10000, projection_tolerance=.0001, energy_window=10.)
    # Explicit diagnostic close-contact filter; not a claimed universal cutoff.
    limits = {pair: .4 * length for pair, length in bonds.items()}
    config = dict(backend='tblite 0.7.0 GFN2-xTB', source_paths=list(map(str, paths)),
        force_consistent=False, energy_convention='tblite source sets energy == free_energy',
        seed=20260909, ssw=asdict(ssw), ga=asdict(ga), bond_lengths=serial(bonds),
        proposal_bond_limits=serial(limits), references=references,
        interpretation='independent paper-reference workflow, not LASP trajectory parity',
        budget='one CPU, externally limited to 300 seconds',
        settings='predeclared workflow diagnostics, no optimization-efficiency claim')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'run_script.py').write_text(Path(__file__).read_text())
    write(output/'initial.extxyz', initial)

    def terminate(signum, frame):
        raise KeyboardInterrupt('external bounded-run timeout')
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_ga_ssw(initial, surface, groups=groups, references=references,
            descriptor_bonds=bonds, descriptor_weights=[.3,.2,.2,.1,.1,.1],
            neighbor_range=fixture['neighbor_range'], proposal_bond_limits=limits,
            config=ga, ssw_config=ssw, rng=np.random.default_rng(20260909))
        (output/'result.json').write_text(json.dumps(serial(result), indent=2)+'\n')
        checks = []
        for row in result.archive:
            fresh = ASESurface(TBLite(method='GFN2-xTB', verbosity=0), force_consistent=False)
            e, f = fresh.evaluate(row['atoms'])
            checks.append(dict(id=row['id'], energy=e,
                max_force=float(np.linalg.norm(f, axis=1).max()),
                force_pass=bool(np.linalg.norm(f, axis=1).max() <= ga.quench_fmax)))
        summary = dict(status=result.status, archive_rows=len(result.archive),
            observations=len(result.observations), failures=len(result.failures),
            stages=[dict(phase=s.phase, status=s.status, requests=s.evaluation_requests) for s in result.stages],
            search_evaluation_requests=result.evaluation_requests,
            fresh_evaluation_requests=len(checks), checks=checks,
            wall_seconds=time.monotonic()-started,
            physical_stability_certified=False, different_basins_certified=False)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        (output/'interrupted.json').write_text(json.dumps(dict(
            error=repr(error), requests=surface.requests,
            wall_seconds=time.monotonic()-started), indent=2)+'\n')
        raise
    finally:
        calls.close()


if __name__ == '__main__':
    main()
