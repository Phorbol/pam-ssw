"""Preserve the accepted soft point if its separate true-energy check fails."""
import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.paper_reference import LSSettings
from pamssw.standalone.constrained_reference import run_constrained_ssw,ConstrainedSSWConfig
from pamssw.standalone.generalized_numerics import GeneralizedRelaxResult


def test_failed_true_response_keeps_soft_optimizer_point(monkeypatch):
    from pamssw.standalone import constrained_ls
    class FailedResponse(ASESurface):
        reject_next=False
        def evaluate(self,a):
            if self.reject_next:
                self.requests+=1
                raise RuntimeError('injected failure in physical response evaluation')
            return super().evaluate(a)
    surface=FailedResponse(EMT())
    def soft_point(q,evaluate,**kwargs):
        q=q+.03;e,g=evaluate(q)
        surface.reject_next=True
        return GeneralizedRelaxResult(q,e,g,'converged',1,1,(),0,0,0)
    monkeypatch.setattr(constrained_ls,'safe_lbfgs',soft_point)
    atoms=bulk('Cu','fcc',a=3.6,cubic=True);atoms.set_constraint(FixAtoms(indices=[0]))
    r=run_constrained_ssw(atoms,surface,steps=2,
        config=ConstrainedSSWConfig(width=.1,rotation_bias=100.,max_gaussians=1),
        rng=np.random.default_rng(17),ls=LSSettings(bond_energies={(29,29):1.},bond_lengths={(29,29):2.9},target_per_atom=.001))
    event=r.records[-1];prepared=event['ls_preparation']
    assert r.status=='ls_prequench_failed' and r.current is r.initial
    assert prepared.optimizer.converged and prepared.energy_after is None
    assert prepared.energy_response is None and 'ls_update' not in event
    np.testing.assert_array_equal(event['last_work'].positions,prepared.atoms.positions)
    np.testing.assert_allclose(event['last_work'].positions[1:]-r.initial.atoms.positions[1:],.03)
    assert r.requests==surface.requests==sum(e['requests'] for e in r.records)
