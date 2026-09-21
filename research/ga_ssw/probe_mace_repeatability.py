"""Bounded real-structure E/F repetition; no search or algorithm changes."""
import argparse
import json
import os
import platform
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads((args.input / 'plan.json').read_text())
    started = time.monotonic()
    calc = MACECalculator(model_paths=plan['model'], device='cuda',
                          default_dtype='float64', enable_cueq=False,
                          enable_oeq=False)
    output = dict(deterministic=args.deterministic, torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  host=platform.node(), cublas_workspace=os.getenv('CUBLAS_WORKSPACE_CONFIG'),
                  rows=[])
    for case in plan['cases']:
        atoms = read(args.input / 'inputs' / (case + '.traj'))
        atoms.calc = calc
        row = dict(case=case, evaluations=[])
        for repeat in range(3):
            calc.reset()  # Force a fresh model invocation at identical coordinates.
            try:
                e = float(atoms.get_potential_energy())
                f = atoms.get_forces()
                if not np.isfinite(e) or not np.isfinite(f).all():
                    raise ValueError('nonfinite energy/forces')
                row['evaluations'].append(dict(energy=e, forces=f.tolist()))
            except Exception as exc:
                row['error'] = repr(exc)
                break
        output['rows'].append(row)
    output['seconds'] = time.monotonic() - started
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
