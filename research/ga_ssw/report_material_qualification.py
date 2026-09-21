"""Read-only derived report for running/terminal material qualification artifacts.

No calculator is imported or called. Does not change qualification decisions,
refine structures, classify phases, or discard missing/failed endpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def render(data):
    endpoints, tasks = data['endpoints'], data['tasks']
    lines = ['# Complex-material independent qualification', '',
             f"Recorded status: **{data['status']}**. Search denominator: "
             f"**{len(data['runs'])}/8**. Search E/F/stress requests: "
             f"**{sum(r.get('source_requests', 0) for r in data['runs'])}**. "
             f"Additional qualification requests: **{data['requests']}**. "
             f"Elapsed qualification seconds: {number(data.get('wall_seconds'))}.", '',
             'This is a snapshot of the saved artifact. Running/partial qualification '
             'is not a completed validation. Search and post-search costs are separate.', '',
             '## All source endpoints, including MC rejects', '',
             '| Run / endpoint | MC accepted | Fresh force | Fresh allowed stress | Original tolerance check | Task |',
             '|---|---|---:|---:|---|---|']
    for e in endpoints:
        stress = number(e.get('fresh_stress_residual')) if e['domain'] == 'variable' else 'fixed domain'
        lines.append(f"| {e['run']} / {e['index']} | {e['accepted']} | "
                     f"{number(e.get('fresh_fmax'))} | {stress} | "
                     f"{e.get('source_domain_certificate_pass', e['fresh_status'])} | {e.get('task_id', 'pending')} |")
    lines += ['', 'Forces use eV/Å and stress eV/Å³. Endpoint -1 is the prepared initial '
              'minimum. Exact duplicate geometry can share a fresh evaluation; all '
              'source aliases remain listed. Stress is not an acceptance condition '
              'in the fixed-cell domain.', '', '## Refinement and finite-cell curvature', '',
              '| Task / aliases | Status | Refined fmax | Stress residual | min eigenvalue, h1 / h2 | Matrix difference norm |',
              '|---|---|---:|---:|---|---:|']
    for t in tasks:
        aliases = ', '.join(f"{a['run']}:{a['index']}" for a in t['aliases'])
        cert = t.get('refined_certificate', {})
        spectra = t.get('spectra', [])
        minima = [number(min(s['eigenvalues'])) if s.get('status') == 'completed' else 'pending' for s in spectra]
        while len(minima) < 2:
            minima.append('pending')
        delta = None
        if len(spectra) == 2 and all(s.get('status') == 'completed' for s in spectra):
            matrices = [np.asarray(s['columns']).T for s in spectra]
            symmetric = [(m + m.T) / 2 for m in matrices]
            delta = float(np.linalg.norm(symmetric[0] - symmetric[1], ord=2))
        lines.append(f"| {t['id']} / {aliases} | {t['status']} | "
                     f"{number(cert.get('fmax'))} | "
                     f"{number(cert.get('stress_residual')) if t['domain'] == 'variable' else 'fixed domain'} | "
                     f"{' / '.join(minima)} | {number(delta)} |")
    lines += ['', 'Eigenvalues use each task’s explicitly scaled coordinate metric; '
              'different atomic/strain metrics are not directly comparable. Translations '
              'are removed. The two Hessian steps are 1e-4 and 5e-5 in those coordinates. '
              'The matrix-difference spectral norm diagnoses finite-difference sensitivity; '
              'it is not a rigorous bound on unknown derivative/model errors.', '',
              'A positive finite-cell spectrum applies to the strictly refined structure '
              'under the stated calculator and domain. It is not a full phonon dispersion, '
              'DFT validation, thermodynamic phase assignment, global-minimum certificate, '
              'or search-efficiency result. Raw-to-refined identity and geometry changes '
              'remain in result.json; no source landing is silently replaced.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    args.output.write_text(render(data))
    print(f"{data['status']}: {len(data['endpoints'])} endpoints, {data['requests']} qualification requests; 0 new PES calls")


if __name__ == '__main__':
    main()
