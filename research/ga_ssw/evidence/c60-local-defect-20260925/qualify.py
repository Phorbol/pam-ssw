"""Two sourced C60 geometries, true-PES qualification only; see plan.md."""
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/c60-defect-input-20260925/41524_2024_1410_MOESM3_ESM.zip')
MODEL = '/home/gengjianrui/.cache/mace/mace-mh-1.model'


def main():
    import ase
    import numpy as np
    import torch
    from ase.io import read, write
    from ase.optimize import LBFGS
    from mace.calculators import MACECalculator
    from research.ga_ssw.analyze_c60_random_development import graph_row
    import networkx as nx

    out = HERE / 'qualification'
    out.mkdir(exist_ok=False)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    class CountedMACE(MACECalculator):
        calls = 0

        def calculate(self, *args, **kwargs):
            if self.calls >= 404:
                raise RuntimeError('total E/F calculation cap reached')
            self.calls += 1
            return super().calculate(*args, **kwargs)

    calc = CountedMACE(model_paths=MODEL, head='omol', device='cuda',
                       default_dtype='float64', enable_cueq=False, enable_oeq=False)
    meta = dict(head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                model=MODEL, backend_head='omol', dtype='float64', device='cuda',
                ase=ase.__version__, torch=torch.__version__, source=str(SOURCE),
                fmax_eV_A=0.03, max_steps=200, memory=100, maxstep_A=0.2, alpha=70)
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    rows = []
    reference = None
    for index in (1, 2):
        member = f'c60/c60-iso-{index}_opt.xyz'
        case = out / f'isomer-{index}'
        case.mkdir()
        with zipfile.ZipFile(SOURCE) as archive:
            original = archive.read(member)
        (case / 'source.xyz').write_bytes(original)
        atoms = read(io.StringIO(original.decode()), format='xyz')
        atoms.pbc = False
        if len(atoms) != 60 or not np.all(atoms.numbers == 6) or not np.isfinite(atoms.positions).all():
            raise ValueError('invalid source geometry')
        if reference is None:
            d = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
            reference = nx.Graph()
            reference.add_nodes_from(range(60))
            reference.add_edges_from(zip(*np.where(np.triu((d > 0) & (d < 1.8), 1))))
        initial_graphs = [graph_row(atoms.numbers, atoms.positions, c, reference) for c in (1.64, 1.7, 1.8)]
        initial_positions = atoms.positions.copy()
        atoms.calc = calc
        calc.reset()
        start = calc.calls
        opt = LBFGS(atoms, memory=100, maxstep=0.2, alpha=70,
                    logfile=str(case / 'opt.log'), trajectory=str(case / 'opt.traj'))
        converged = bool(opt.run(fmax=0.03, steps=200))
        energy = float(atoms.get_potential_energy())
        search_calls = calc.calls - start
        if search_calls > 201:
            raise RuntimeError('per-geometry optimization exceeded planned calls')
        write(case / 'final.extxyz', atoms)
        calc.reset()
        fresh_start = calc.calls
        fresh_energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.linalg.norm(forces, axis=1).max())
        before = np.linalg.norm(initial_positions[:, None] - initial_positions[None, :], axis=2)
        after = np.linalg.norm(atoms.positions[:, None] - atoms.positions[None, :], axis=2)
        row = dict(member=member, converged=converged, steps=opt.nsteps,
                   optimization_calls=search_calls, fresh_calls=calc.calls-fresh_start,
                   energy_eV=energy, fresh_energy_eV=fresh_energy, fresh_fmax_eV_A=fmax,
                   force_qualified=bool(np.isfinite(fmax) and fmax <= 0.03),
                   labeled_edges_preserved={str(c): bool(np.array_equal(before < c, after < c))
                                            for c in (1.64, 1.7, 1.8)},
                   initial_graphs=initial_graphs,
                   final_graphs=[graph_row(atoms.numbers, atoms.positions, c, reference) for c in (1.64, 1.7, 1.8)])
        (case / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(dict(cases=rows, total_calculations=calc.calls), indent=2) + '\n')


if __name__ == '__main__':
    main()
