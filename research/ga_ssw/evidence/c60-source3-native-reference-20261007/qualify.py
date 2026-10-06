"""Cold checks for preselected native minima; <=3 requests/run, no relaxation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from ase.io import read, write


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1<<20), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(prepared, readout, output, calculator_factory=None):
    manifest = json.loads((prepared/'manifest.json').read_text())
    analysis = json.loads((readout/'analysis.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plans = {entry['seed']:json.loads((prepared/entry['run_dir']/'plan.json').read_text())
        for entry in manifest['cases']}
    model = Path(next(iter(plans.values()))['model'])
    expected = next(iter(plans.values()))['model_sha256']
    if sha(model) != expected:
        raise ValueError('Model changed before cold checks')
    if calculator_factory is None:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mace.calculators import MACECalculator
        calculator_factory = lambda:MACECalculator(model_paths=str(model), head='omol',
            device='cuda', default_dtype='float64', enable_cueq=False, enable_oeq=False)
    calc = calculator_factory()
    actual = paid = 0
    original = calc.calculate
    def counted(*args, **kwargs):
        nonlocal actual
        actual += 1
        return original(*args, **kwargs)
    calc.calculate = counted
    records = []
    for row in analysis['rows']:
        selection = row.get('fresh_selection', {})
        if len(selection)>3 or set(selection)-{'initial','best','first_target'}:
            raise ValueError('Readout changed the frozen fresh-selection policy')
        if not row.get('comparison_eligible'):
            records.append(dict(seed=row['seed'],status='ineligible_search_no_cold_calls'))
            continue
        used = 0
        for label, event in selection.items():
            if used>=3 or paid>=6:
                raise RuntimeError('Cold-check budget exceeded')
            source = readout/event['path']
            atoms = read(source)
            if len(atoms)!=60 or list(atoms.numbers)!=[6]*60 or atoms.pbc.any() or atoms.constraints:
                raise ValueError('Fresh input must be an unconstrained isolated C60')
            atoms.calc = calc
            # Reserve before calculator access; failures consume this slot.
            used += 1
            paid += 1
            record = dict(seed=row['seed'],label=label,source=str(source),source_sha256=sha(source),
                event=event['event'],search_cost=event['search_cost'],paid_request=paid)
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(),dtype=float)
                fmax = float(np.linalg.norm(forces,axis=1).max())
                finite = np.isfinite(energy) and np.isfinite(forces).all()
                numerical = bool(finite and fmax<=.03 and abs(energy-event['energy_eV'])<=1e-4)
                centered = atoms.positions-atoms.positions.mean(axis=0)
                distances = np.linalg.norm(centered[:,None]-centered[None,:],axis=2)
                np.fill_diagonal(distances,np.inf)
                record.update(status='completed',energy_eV=energy,fmax_eV_A=fmax,
                    event_energy_difference_eV=energy-event['energy_eV'],numerical_qualified=numerical,
                    minimum_distance_A=float(distances.min()),
                    principal_rms_extents_A=(np.linalg.svd(centered,compute_uv=False)/np.sqrt(60)).tolist(),
                    physical_cage_review='required_for_positive_target')
                write(output/f"seed-{row['seed']}-{label}-cold.extxyz",atoms)
            except Exception as error:
                record.update(status='failed',error=repr(error),numerical_qualified=False)
            records.append(record)
            dump(output/'summary.json',dict(status='running',paid_requests=paid,
                actual_calculate_calls=actual,records=records,model=str(model),model_sha256=expected))
    dump(output/'summary.json',dict(status='completed',paid_requests=paid,actual_calculate_calls=actual,
        records=records,model=str(model),model_sha256=expected,readout=str(readout),
        elapsed_seconds=time.monotonic()-started,qualify_script=str(Path(__file__).resolve()),
        qualify_sha256=sha(Path(__file__).resolve()),python=sys.executable,
        scope='independent cold single points; no local refinement/physical-target promotion'))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--readout',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    run(args.prepared.resolve(),args.readout.resolve(),args.output.resolve())
