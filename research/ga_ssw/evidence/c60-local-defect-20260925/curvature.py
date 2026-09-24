"""Local finite-difference curvature at two qualified C60 inputs; no search."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    import numpy as np
    import torch
    from ase.io import read
    from scipy.linalg import null_space
    from mace.calculators import MACECalculator

    out = HERE / 'curvature'
    out.mkdir(exist_ok=False)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calc = MACECalculator(model_paths='/home/gengjianrui/.cache/mace/mace-mh-1.model',
                          head='omol', device='cuda', default_dtype='float64',
                          enable_cueq=False, enable_oeq=False)
    calls = 0
    rows = []
    for index in (1, 2):
        source = HERE / 'qualification' / f'isomer-{index}' / 'final.extxyz'
        atoms = read(source)
        atoms.calc = calc
        x = atoms.positions.copy()

        def force(positions):
            nonlocal calls
            if calls >= 730:
                raise RuntimeError('730 E/F cap exceeded')
            atoms.positions[:] = positions
            calc.reset()
            calls += 1
            return atoms.get_forces().ravel().copy()

        start = calls
        f0 = force(x)
        h = 0.01
        matrix = np.empty((180, 180))
        for j in range(180):
            displacement = np.zeros(180)
            displacement[j] = h
            matrix[:, j] = -(force(x + displacement.reshape(-1, 3)) -
                             force(x - displacement.reshape(-1, 3))) / (2*h)
        # Remove rigid translations and infinitesimal rotations. All masses
        # are equal; eigenvalues below are Cartesian stiffness, not frequency.
        centered = x - x.mean(axis=0)
        rigid = np.column_stack([np.tile(v, (60, 1)).ravel() for v in np.eye(3)] +
                                [np.cross(v, centered).ravel() for v in np.eye(3)])
        basis = null_space(rigid.T)
        assert basis.shape == (180, 174)
        projected = basis.T @ ((matrix + matrix.T)/2) @ basis
        values, vectors = np.linalg.eigh(projected)
        sensitivity = []
        for mode in (0, 1):
            direction = basis @ vectors[:, mode]
            s = 0.02
            curvature = -direction @ (force(x+s*direction.reshape(-1, 3)) -
                                       force(x-s*direction.reshape(-1, 3))) / (2*s)
            sensitivity.append(dict(mode=mode, displacement_norm_A=s,
                                    direct_curvature_eV_A2=float(curvature),
                                    matrix_eigenvalue_eV_A2=float(values[mode])))
        np.savez(out / f'isomer-{index}.npz', positions=x, raw_hessian=matrix,
                 internal_basis=basis, eigenvalues=values, eigenvectors=vectors)
        rows.append(dict(source=str(source), calls=calls-start,
                         fmax_eV_A=float(np.linalg.norm(f0.reshape(-1, 3), axis=1).max()),
                         displacement_A=h, eigenvalues_eV_A2=values.tolist(),
                         antisymmetric_spectral_norm=float(np.linalg.norm((matrix-matrix.T)/2, ord=2)),
                         lowest_modes_sensitivity=sensitivity))
        (out / 'results.json').write_text(json.dumps(dict(cases=rows, total_calls=calls), indent=2)+'\n')


if __name__ == '__main__':
    main()
