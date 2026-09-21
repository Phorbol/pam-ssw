"""Compare Python rigid subspace with archived original-instruction outputs.

The oracle uses a valid warm cache, not complete native initialization. Generic
nonlinear geometries only; rank-deficient native anomalies are not reproduced.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone.cluster_frame import ClusterFrame


@pytest.mark.parametrize('n',[4,7,13])
def test_nonlinear_projection_matches_native_warm_cache(n):
    root=Path(__file__).resolve().parents[2]
    report=json.loads((root/'research/ga_ssw/evidence/native-setconstraints-emulated/result.json').read_text())
    case=next(c for c in report['cases'] if c['n']==n)
    frame=ClusterFrame(Atoms(numbers=[29]*n,positions=case['positions']))
    matrix=np.column_stack([frame.project(e.reshape(n,3)).ravel() for e in np.eye(3*n)])
    np.testing.assert_allclose(matrix,case['native_matrix'],rtol=0,atol=1e-12)
