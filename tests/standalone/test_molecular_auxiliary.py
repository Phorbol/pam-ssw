from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.molecular_auxiliary import MolecularLJChart,optimize_molecular_lj


def water():
    path=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/staged-water/final-arc/0.arc'
    rows=[line.split() for line in path.read_text().splitlines() if 'CORE' in line]
    a=Atoms([r[0] for r in rows],positions=[[float(x) for x in r[1:4]] for r in rows])
    return a,[range(i,i+3) for i in range(0,len(a),3)]


# Explicit numerical-check sigmas; not claimed water force-field parameters.
SIGMA={(1,1):1.,(1,8):1.5,(8,8):3.}


def test_all_rigid_euler_components_at_finite_rotation_have_correct_gradient():
    a,groups=water();chart=MolecularLJChart(a,groups,SIGMA);rng=np.random.default_rng(3)
    q=chart.initial.copy();q[:,3:]=rng.normal(scale=.2,size=(len(groups),3));q=q.ravel()
    _,g=chart.evaluate(q);h=1e-6
    # Check every translation and angular component, including second partner.
    for i in range(len(q)):
        d=np.zeros_like(q);d[i]=h
        fd=(chart.evaluate(q+d)[0]-chart.evaluate(q-d)[0])/(2*h)
        np.testing.assert_allclose(fd,g[i],rtol=2e-5,atol=2e-6)


def test_bounded_real_water_auxiliary_optimization_preserves_all_internal_geometry():
    a,groups=water();before=a.positions.copy()
    result=optimize_molecular_lj(a,groups,SIGMA,max_evaluations=20)
    assert 1<=result.auxiliary_evaluations<=20
    assert not result.certified_physical_minimum
    for group in groups:
        np.testing.assert_allclose(result.atoms[list(group)].get_all_distances(),a[list(group)].get_all_distances(),atol=1e-12)
    np.testing.assert_array_equal(a.positions,before)
    chart=MolecularLJChart(result.atoms,groups,SIGMA)
    np.testing.assert_allclose(chart.evaluate(chart.initial)[0],result.energy_aux,rtol=1e-10)
