"""Zero-PES finite-deformation audit of existing block matrices."""
import json
from pathlib import Path
import numpy as np


def audit(root=Path('research/ga_ssw/fe7c3-block-baseline')):
    root = Path(root)
    rows = []
    for seed in (7, 101):
        data = json.loads((root / f'comparison/safe_total-seed{seed}/result.json').read_text())
        for record in data['records'][1:]:
            for cycle in record['cell_cycles']:
                before = np.asarray(cycle['cell_before'])
                after = np.asarray(cycle['cell_after'])
                increment = after - before
                deformation = np.linalg.solve(before, after)
                stretches = np.linalg.svd(deformation, compute_uv=False)
                displacement = np.linalg.norm(increment)
                assert np.isclose(displacement, cycle['distance'], atol=1e-12, rtol=1e-12)
                assert np.linalg.det(deformation) > 0
                bound = float(displacement / np.linalg.svd(before, compute_uv=False).min())
                actual = float(np.linalg.norm(deformation - np.eye(3), 2))
                assert actual <= bound + 1e-12
                rows.append(dict(seed=seed, outer_index=record['index'], cycle=cycle['index'],
                    relative_lattice_frobenius=float(displacement / np.linalg.norm(before)),
                    principal_stretches=stretches.tolist(),
                    max_absolute_principal_strain=float(np.max(np.abs(stretches - 1))),
                    volume_ratio=float(np.linalg.det(deformation)),
                    deformation_increment_spectral_norm=actual, increment_bound=bound))
    result = dict(zero_pes=True, convention='ASE row cell; F=inv(L_before)@L_after; '
                  'stretches=svd(F); geometric diagnostics only', rows=rows)
    (root / 'cell-deformation-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps({'cycles': len(audit()['rows']), 'zero_pes': True}))
