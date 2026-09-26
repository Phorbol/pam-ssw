"""One bounded frozen VC task per process, reusing the panel's numerical code."""
import argparse
import hashlib
import importlib.metadata
import platform
import json
import time
from pathlib import Path
import torch
from mace.calculators import MACECalculator
from research.ga_ssw.run_vc_frozen_optimizer_panel import (
    _read_case, _atom_dict, run_stage, fresh_endpoint,
    validate_saved_physical_objective, _failed_stage,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--phase', choices=('biased', 'unbiased'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.method not in plan['optimizers']:
        raise ValueError('method is not in the frozen panel')
    args.out.mkdir(parents=True, exist_ok=False)
    spec = {**plan['cases'][args.case_index],
        'maxiter_per_stage': plan['budget']['maxiter_per_stage'],
        'request_cap_per_stage': plan['budget']['request_cap_per_stage'],
        'max_step_A': plan['common_contract']['max_step_A'],
        'lbfgs_history_pairs': plan['lbfgs_history_pairs']}
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model = Path(plan['model'])
    if hashlib.sha256(model.read_bytes()).hexdigest() != plan['model_sha256']:
        raise ValueError('model differs from frozen plan')
    def factory():
        return MACECalculator(model_paths=str(model), device=plan['device'],
            default_dtype=plan['dtype'], head=plan['calculator']['head'],
            enable_cueq=False, enable_oeq=False)
    atoms, chart, q0, q_saved, terms, record, selected = _read_case(spec)
    frozen = dict(spec=spec, method=args.method, phase=args.phase,
        reference=_atom_dict(atoms), q_start=(q0 if args.phase=='biased' else q_saved).tolist(),
        gaussians=terms if args.phase=='biased' else [],
        source_note='formula-based reconstruction; no archived biased-gradient gold value')
    frozen['environment'] = {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'ase', 'mace-torch', 'torch')}}
    (args.out/'task.json').write_text(json.dumps(frozen, indent=2)+'\n')
    validation = validate_saved_physical_objective(q_saved, selected, spec, chart, factory)
    (args.out/'source-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    if validation['status'] != 'pass':
        (args.out/'result.json').write_text(json.dumps(dict(status='source_validation_failed', validation=validation), indent=2)+'\n')
        return 2
    q = q0 if args.phase=='biased' else q_saved
    active_terms = terms if args.phase=='biased' else []
    stage_dir = args.out/'stage'
    started = time.monotonic()
    try:
        result = run_stage(args.method, args.phase, q, spec, chart, active_terms,
                           factory, stage_dir, time.monotonic()+100)
    except Exception as exc:
        result = _failed_stage(args.method, args.phase, q, stage_dir, exc)
    result['source_validation'] = validation
    # Save before the separate fresh calculator check, so interruption is visible.
    result['fresh_endpoint'] = {'status': 'pending', 'requests': 0}
    path = args.out/'result.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if result.get('terminal') is not None:
        result['fresh_endpoint'] = fresh_endpoint(result['terminal_q'], spec, chart, active_terms, factory)
    else:
        result['fresh_endpoint'] = {'status': 'no_paid_terminal', 'requests': 0}
    result['stage_and_fresh_wall_seconds'] = time.monotonic()-started
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'case':spec['name'],'method':args.method,'phase':args.phase,
        'status':result['status'],'search_requests':result['search_requests'],
        'first_passage':result.get('first_common_qualified_request')}), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
