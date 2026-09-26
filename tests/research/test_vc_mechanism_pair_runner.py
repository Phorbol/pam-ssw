import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT

from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.run_vc_mechanism_pair import run_paired


def test_cell_cycles_are_only_added_stage_and_raw_pairing_streams_match():
    atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
    gate = cell_quench(
        atoms, ASEStressSurface(EMT()), strain_length=5.0, pressure=0.0,
        fmax=0.03, stress_tol=0.001, max_step=0.2, maxiter=400,
        lbfgs_memory=500,
    )
    assert gate.converged
    result = run_paired(
        gate.evaluation.atoms,
        EMT(),
        seed=17,
        steps=1,
        cell_cycles=1,
        request_cap=1200,
        seconds=60,
        fmax=0.03,
        atomic_gaussians=1,
        rotation_hvp=8,
        cell_rotation_requests=4,
    )

    on, off = result['cell_on'], result['cell_off']
    assert on['status'] == off['status'] == 'completed'
    assert len(on['steps']) == len(off['steps']) == 1
    assert len(on['steps'][0]['cell_cycles']) == 1
    assert off['steps'][0]['cell_cycles'] == []
    assert on['steps'][0]['atomic_rng_initial_state'] == off['steps'][0]['atomic_rng_initial_state']
    assert on['steps'][0]['mc_draw'] == off['steps'][0]['mc_draw']
    assert on['steps'][0]['true_quench'].converged
    assert off['steps'][0]['true_quench'].converged
    assert on['steps'][0]['atomic'].initial_direction is not None
    assert off['steps'][0]['atomic'].initial_direction is not None
    assert on['steps'][0]['true_quench'].certificate['certified']
    assert off['steps'][0]['true_quench'].certificate['certified']
    assert on['steps'][0]['full_escape_complete']
    assert off['steps'][0]['full_escape_complete']
