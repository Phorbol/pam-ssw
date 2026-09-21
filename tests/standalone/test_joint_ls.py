import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening
from pamssw.standalone.vc_reference import VCSSWConfig
from pamssw.standalone.joint_ls import prepare_joint_ls_step
from pamssw.standalone.ls_cycle import LSCycleError

TABLE = dict(bond_energies={(29, 29): 1.0}, bond_lengths={(29, 29): 2.9})

def make():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    return a, FrozenPeriodicCellSoftening.from_atoms(a, **TABLE)

def cfg(p=0.):
    return VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=1., pressure=p,
                       fmax=.01, stress_tol=.01, gradient_tol=.01,
                       max_step=.2, relax_steps=100, lbfgs_memory=10)

def test_joint_preparation_relaxes_cell_gradient_and_keeps_accounting():
    a, soft = make(); surface = ASEStressSurface(EMT())
    result = prepare_joint_ls_step(a, surface, softening=soft, config=cfg())
    assert result.optimizer.converged
    assert result.requests == surface.requests
    assert result.post_soft_gradient.shape == (3 * len(a) + 6,)
    assert result.post_soft_fmax <= cfg().fmax
    assert result.post_soft_stress_max <= cfg().stress_tol
    assert result.true_energy_before == pytest.approx(result.true_enthalpy_before)
    assert result.true_energy_after == pytest.approx(result.true_enthalpy_after)
    assert result.ls_energy_before >= 0 and result.ls_energy_after >= 0
    assert np.linalg.norm(result.q[-6:]) > 1e-8

def test_nonzero_pressure_reports_enthalpy_separately():
    a, soft = make(); p=.002; surface=ASEStressSurface(EMT())
    result = prepare_joint_ls_step(a, surface, softening=soft, config=cfg(p))
    assert result.true_enthalpy_before == pytest.approx(result.true_energy_before + p*result.volume_before)
    assert result.true_enthalpy_after == pytest.approx(result.true_energy_after + p*result.volume_after)
    assert result.true_enthalpy_before != pytest.approx(result.true_energy_before)

def test_failed_joint_quench_preserves_last_state_and_paid_requests():
    a, soft = make()
    class Failing(ASEStressSurface):
        def evaluate(self, atoms):
            if self.requests >= 1: self.requests += 1; raise RuntimeError('injected joint failure')
            return super().evaluate(atoms)
    surface=Failing(EMT())
    with pytest.raises(LSCycleError) as exc:
        prepare_joint_ls_step(a, surface, softening=soft, config=cfg())
    err=exc.value
    assert err.stage == 'joint_soft_quench'
    assert err.result is not None
    assert hasattr(err.result, "optimizer")
    assert err.result.requests == surface.requests
    assert err.result.q.shape == (3*len(a)+6,)
    assert err.result.atoms is not None

def test_nonstationary_start_rejects_excess_physical_stress():
    a, soft = make()
    class HighStress(ASEStressSurface):
        def evaluate(self, atoms):
            energy, forces, stress = super().evaluate(atoms)
            return energy, forces, stress + np.eye(3)
    surface = HighStress(EMT())
    with pytest.raises(LSCycleError) as exc:
        prepare_joint_ls_step(a, surface, softening=soft, config=cfg())
    assert exc.value.stage == 'true_start'
    assert exc.value.result.true_start_stress_max > cfg().stress_tol
    assert exc.value.result.requests == surface.requests

def test_failed_final_physical_check_keeps_accepted_geometry_and_cost():
    a, soft = make()
    class FailFinal(ASEStressSurface):
        def evaluate(self, atoms):
            if self.requests >= 4:
                self.requests += 1
                raise RuntimeError('injected final check failure')
            return super().evaluate(atoms)
    surface = FailFinal(EMT())
    with pytest.raises(LSCycleError) as exc:
        prepare_joint_ls_step(a, surface, softening=soft, config=cfg())
    err = exc.value
    assert err.stage == 'true_finish'
    assert err.result.atoms is not None
    assert err.result.optimizer is not None
    assert err.result.requests == surface.requests
    assert err.result.softened_fmax is None


def test_final_fresh_joint_gradient_gate_is_not_replaced_by_stress_gate(monkeypatch):
    from pamssw.standalone import joint_ls
    original=joint_ls.safe_lbfgs; finished=[]
    def run(*args,**kwargs):
        result=original(*args,**kwargs);finished.append(True);return result
    monkeypatch.setattr(joint_ls,'safe_lbfgs',run)
    class ChangedFinalStress(ASEStressSurface):
        def evaluate(self,atoms):
            e,f,s=super().evaluate(atoms)
            if finished:s=s+.002*np.eye(3)
            return e,f,s
    a,soft=make();surface=ChangedFinalStress(EMT())
    with pytest.raises(LSCycleError) as exc:
        prepare_joint_ls_step(a,surface,softening=soft,config=cfg())
    snapshot=exc.value.result
    assert exc.value.stage=='true_finish'
    assert snapshot.softened_stress_max<cfg().stress_tol
    assert snapshot.joint_gradient_norm>cfg().gradient_tol
