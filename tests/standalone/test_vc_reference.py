"""Real Cu EMT end-to-end joint-cell execution, not global-search efficacy."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw

class RecordingSurface(ASEStressSurface):
    def __init__(self):
        super().__init__(EMT());self.calls=[]
    def evaluate(self,atoms):
        self.calls.append((atoms.positions.copy(),atoms.cell.array.copy()))
        return super().evaluate(atoms)

def config(pressure=0.):
    return VCSSWConfig(strain_length=3.6,width=.2,rotation_bias=.5,
        pressure=pressure,max_gaussians=1,rotation_hvp=41,relax_steps=150)

@pytest.mark.parametrize('pressure',[0.,.005])
def test_real_cu_joint_escape_and_independent_physical_certificates(pressure):
    atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    atoms.positions[0]+=[.03,-.02,.01]
    original=atoms.copy();surface=RecordingSurface();c=config(pressure)
    result=run_vc_ssw(atoms,surface,steps=1,config=c,rng=np.random.default_rng(7))
    assert result.status=='completed' and len(result.minima)==2
    assert result.requests==surface.requests==len(surface.calls)
    assert result.records[0]['certificate']['certified']
    step=result.records[1]
    assert step['landing'] is not None and step['climb'][0]['status']=='converged'
    event=step['climb'][0]
    assert event['rotation_requests'] > 0
    assert event['height_requests'] == 1
    assert event['biased_quench_requests'] > 0
    assert event['true_check_requests'] == 1
    assert event['requests'] == sum(event[k] for k in ('rotation_requests', 'height_requests', 'biased_quench_requests', 'true_check_requests'))
    assert event['relaxation']['gradient_norm'] <= c.gradient_tol
    # Demonstrate cell coordinates in the actual escape direction and biased
    # minimum, not merely a posterior cell quench.
    assert np.linalg.norm(event['direction'][-6:])>1e-3
    assert np.linalg.norm(np.array(event['cell'])-result.initial.atoms.cell.array)>1e-3
    assert np.linalg.norm(event['direction'][:-6])>1e-5
    assert len(event['q'])==3*len(atoms)+6
    verify=ASEStressSurface(EMT())
    for minimum in result.minima:
        energy,forces,stress=verify.evaluate(minimum.atoms)
        assert np.isfinite(forces).all() and np.linalg.det(minimum.atoms.cell.array)>0
        assert np.linalg.norm(forces,axis=1).max()<=c.fmax
        assert np.abs(stress+pressure*np.eye(3)).max()<=c.stress_tol
        assert energy+pressure*minimum.atoms.get_volume()==pytest.approx(minimum.objective,abs=1e-12)
        np.testing.assert_allclose(forces,minimum.forces,atol=1e-12,rtol=0)
        np.testing.assert_allclose(stress,minimum.stress,atol=1e-12,rtol=0)
    assert result.best.objective==min(m.objective for m in result.minima)
    # Certification uses its own oracle and does not inflate the search counter.
    assert verify.requests==2 and result.requests==surface.requests
    np.testing.assert_array_equal(atoms.positions,original.positions)
    np.testing.assert_array_equal(atoms.cell.array,original.cell.array)

def test_failed_rotation_retains_certified_current_and_charges_endpoint_calls():
    atoms=bulk('Cu','fcc',a=3.65,cubic=True);surface=RecordingSurface()
    c=VCSSWConfig(strain_length=3.6,width=.2,rotation_bias=.5,max_gaussians=1,
                  rotation_hvp=1,rotation_tol=1e-10,relax_steps=150)
    result=run_vc_ssw(atoms,surface,steps=1,config=c,rng=np.random.default_rng(7))
    assert result.records[-1]['status']=='rotation_failed'
    assert not result.records[-1]['accepted'] and len(result.minima)==1
    assert result.records[-1]['requests']==2
    stage=result.records[-1]['climb'][0]
    assert stage['rotation_requests']==stage['requests']==2
    assert stage['height_requests']==stage['biased_quench_requests']==stage['true_check_requests']==0
    assert result.requests==len(surface.calls)==surface.requests
    np.testing.assert_array_equal(result.current.atoms.cell.array,result.initial.atoms.cell.array)
    np.testing.assert_array_equal(result.current.atoms.positions,result.initial.atoms.positions)

def test_initial_calculator_failure_is_a_counted_failed_result():
    from ase.calculators.calculator import Calculator, all_changes
    class FailedCalculator(Calculator):
        implemented_properties=['energy','forces','stress']
        def calculate(self,atoms=None,properties=None,system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            raise RuntimeError('intentional initial backend failure')
    surface=ASEStressSurface(FailedCalculator())
    r=run_vc_ssw(bulk('Cu','fcc',cubic=True),surface,steps=1,config=config(),rng=np.random.default_rng(7))
    assert r.status=='initial_quench_failed' and not r.minima
    assert r.initial is r.current is r.best is None
    assert r.requests==surface.requests==r.records[0]['requests']==1
    assert 'intentional initial backend failure' in r.records[0]['error']

@pytest.mark.parametrize('stop', ['maxiter', 'native_stop', 'evaluation_failed', 'request_limit'])
@pytest.mark.parametrize('policy', ['strict', 'numerical_stop'])
def test_bias_stop_release_keeps_physical_certificate(monkeypatch, stop, policy):
    from dataclasses import replace
    import pamssw.standalone.vc_reference as vc
    original = vc.safe_lbfgs
    def stopped(q, evaluate, **kwargs):
        # Only the biased optimizer is bound here; physical quench uses cell_relax.
        return replace(original(q, evaluate, **{**kwargs, 'maxiter': 1}), status=stop)
    monkeypatch.setattr(vc, 'safe_lbfgs', stopped)
    surface = RecordingSurface()
    result = run_vc_ssw(bulk('Cu', 'fcc', a=3.65, cubic=True), surface,
        steps=1, config=replace(config(), bias_release=policy), rng=np.random.default_rng(7))
    record = result.records[-1]
    assert record['climb'][0]['status'] == stop
    released = policy == 'numerical_stop' and stop in ('maxiter', 'native_stop')
    if released:
        assert record['status'] == 'biased_numerical_stop'
        assert record['certificate']['certified']
        assert record['landing_optimizer']['status'] == 'converged'
        check = ASEStressSurface(EMT())
        _, forces, stress = check.evaluate(record['landing'].atoms)
        assert np.linalg.norm(forces, axis=1).max() <= config().fmax
        assert np.abs(stress).max() <= config().stress_tol
    else:
        assert record['status'] == 'biased_quench_failed'
        assert record['landing'] is None
    assert result.requests == surface.requests

@pytest.mark.parametrize('invalid', [{'energy': None}, {'energy': float('nan')},
                                    {'gradient': None}, {'error': 'backend error'}])
def test_numerical_release_requires_valid_accepted_evaluation(monkeypatch, invalid):
    from dataclasses import replace
    import pamssw.standalone.vc_reference as vc
    original = vc.safe_lbfgs
    def stopped(q, evaluate, **kwargs):
        return replace(original(q, evaluate, **{**kwargs, 'maxiter': 1}),
                       status='maxiter', **invalid)
    monkeypatch.setattr(vc, 'safe_lbfgs', stopped)
    result = run_vc_ssw(bulk('Cu', 'fcc', a=3.65, cubic=True), RecordingSurface(),
        steps=1, config=replace(config(), bias_release='numerical_stop'),
        rng=np.random.default_rng(7))
    assert result.records[-1]['status'] == 'biased_quench_failed'
    assert result.records[-1]['landing'] is None
