"""Periodic geometry contracts plus bounded Cu/EMT physical endpoint checks.

No test establishes basin discovery, global optimality or variable-cell support.
"""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from pamssw.standalone.periodic_geometry import FixedCellTranslationFrame, global_translation_free_direction
from pamssw.standalone.surface import ASESurface, quench
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.direction import paper_biased_direction
from pamssw.standalone.dimer import paper_dimer_direction


def copper():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.positions += np.random.default_rng(22).normal(scale=.035, size=(4,3))
    return atoms


def test_translation_projector_is_orthogonal_retains_rotation_and_never_wraps():
    atoms = copper(); frame = FixedCellTranslationFrame(atoms)
    rng = np.random.default_rng(9); a, b = rng.normal(size=(2,4,3))
    pa, pb = frame.project(a), frame.project(b)
    np.testing.assert_allclose(pa.sum(0), 0., atol=1e-15)
    np.testing.assert_allclose(frame.project(pa), pa, atol=1e-15)
    assert np.sum(pa*b) == pytest.approx(np.sum(a*pb), abs=1e-14)
    rotation = np.cross([0.,0.,1.], atoms.positions-atoms.positions.mean(0))
    np.testing.assert_allclose(frame.project(rotation), rotation, atol=1e-15)
    lifted = atoms.positions.copy(); lifted[0] += 4*atoms.cell[0]
    projected = frame.positions(lifted)
    # Removing a global drift preserves relative unwrapped displacement.
    np.testing.assert_allclose(projected[0]-projected[1], lifted[0]-lifted[1])
    assert np.max(projected[:,0]) > atoms.cell[0,0]
    np.testing.assert_array_equal(atoms.positions, frame.reference)


@pytest.mark.parametrize('change', ['cell','pbc','numbers','constraint'])
def test_frame_rejects_metadata_change_before_oracle(change):
    atoms=copper(); frame=FixedCellTranslationFrame(atoms)
    if change=='cell':atoms.cell[0,0]+=.1
    elif change=='pbc':atoms.pbc[2]=False
    elif change=='numbers':atoms.numbers[0]=47
    else:atoms.set_constraint(FixAtoms(indices=[0]))
    def unexpected(a):pytest.fail('changed metadata reached oracle')
    with pytest.raises(ValueError, match='unchanged'):
        frame.evaluate(atoms, unexpected)


def test_periodic_emt_energy_force_invariance_and_complete_pullback():
    atoms=copper(); surface=ASESurface(EMT()); frame=FixedCellTranslationFrame(atoms)
    direction=global_translation_free_direction(atoms,np.random.default_rng(3))
    np.testing.assert_allclose(direction.sum(0),0.,atol=1e-15)
    assert np.linalg.norm(direction)==pytest.approx(1.)
    e0,f0=surface.evaluate(atoms)
    translated=atoms.copy();translated.positions += [4.9,-2.3,7.1]
    et,ft=surface.evaluate(translated)
    assert et==pytest.approx(e0,abs=1e-12)
    np.testing.assert_allclose(ft,f0,atol=1e-12)
    imaged=atoms.copy();imaged.positions[0]+=atoms.cell[0]
    ei,fi=surface.evaluate(imaged)
    assert ei==pytest.approx(e0,abs=1e-12)
    np.testing.assert_allclose(fi,f0,atol=1e-12)
    bias=ProjectedGaussian(atoms.positions,direction,.2,.3)
    def complete(a):
        e,f=surface.evaluate(a); be,bf=bias.evaluate(a)
        return e+be,f+bf
    trial=atoms.copy();trial.positions+=.07*direction+[.1,.2,.3]
    _,force=frame.evaluate(trial,complete)
    tangent=np.random.default_rng(4).normal(size=(4,3));tangent/=np.linalg.norm(tangent)
    h=1e-5; plus=trial.copy();minus=trial.copy()
    plus.positions+=h*tangent;minus.positions-=h*tangent
    derivative=(frame.evaluate(plus,complete)[0]-frame.evaluate(minus,complete)[0])/(2*h)
    assert derivative==pytest.approx(-np.sum(force*tangent),abs=2e-7)
    # Physical periodic equivalence does not identify coordinates of path bias.
    assert bias.evaluate(imaged)[0] != pytest.approx(bias.evaluate(atoms)[0],abs=1e-5)


@pytest.mark.parametrize('solver',[paper_biased_direction,paper_dimer_direction])
def test_periodic_emt_direction_and_true_fixed_cell_quench(solver):
    atoms=copper(); surface=ASESurface(EMT()); initial_cell=atoms.cell.array.copy()
    result=quench(atoms,surface,fmax=.005,steps=100)
    assert result.converged
    np.testing.assert_array_equal(result.atoms.cell.array,initial_cell)
    frame=FixedCellTranslationFrame(result.atoms)
    anchor=global_translation_free_direction(result.atoms,np.random.default_rng(7))
    mode=solver(result.atoms,anchor,rotation_bias=3.,fd_step=1e-4,max_hvp=60,tol=.002,
        evaluate=lambda a:frame.evaluate(a,surface.evaluate))
    assert mode.converged
    np.testing.assert_allclose(mode.direction.sum(0),0.,atol=1e-9)
    np.testing.assert_array_equal(result.atoms.cell.array,initial_cell)
    fresh=ASESurface(EMT());e,f=fresh.evaluate(result.atoms)
    assert e==pytest.approx(result.energy,abs=1e-12)
    assert np.linalg.norm(f,axis=1).max()<=.005
    assert result.evaluation_requests>0 and mode.force_calls>0
