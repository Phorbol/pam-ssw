"""Bounded isolated minimal-history spectral restart probe; no PES calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from research.ga_ssw.probe_native_broyden_full import FullOracle, ELF_DEFAULT, ELF_SHA256, load_elf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if not args.execute:
        raise SystemExit('requires --execute for isolated numerical instructions')
    blob, segments = load_elf(ELF_DEFAULT)
    assert hashlib.sha256(blob).hexdigest() == ELF_SHA256
    oracle = FullOracle(segments)
    x = np.array([.3, -.1, .7])
    g0 = np.full(3, 1e-8)
    h = np.diag([.5, 1., 2.])
    rows = []
    for step in range(3):
        force = -h @ x
        before = x.copy()
        x, pointers = oracle.full_call(x, force, g0, step == 0)
        rows.append(dict(step=step, x_before=before.tolist(), force=force.tolist(),
                         x_after=x.tolist(), initial_flag_after=oracle.i(pointers[4]),
                         iteration=oracle.i(0x7942fa4),
                         initial_step_error=float(np.max(abs(x-(before+g0*force))))))
    result = dict(elf_sha256=ELF_SHA256, g0=g0.tolist(), hessian=h.tolist(),
                  boundary='Raw flags false, SciPy DGGEV substitution; numerical branch probe only',
                  steps=rows, spectral_checks=oracle.spectral_checks, dggev_calls=oracle.dggev_calls,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(steps=rows, spectral_checks=oracle.spectral_checks), indent=2))


if __name__ == '__main__':
    main()
