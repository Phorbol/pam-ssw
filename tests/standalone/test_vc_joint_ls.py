import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.paper_reference import LSSettings
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig,run_vc_ssw


def test_vc_joint_ls_preparation_records_enthalpy_and_cost(monkeypatch):
    from pamssw.standalone.softening import LSResponseState
    original=LSResponseState.update; calls=[]
    def capture(self,*args,**kwargs):
        calls.append((kwargs['energy_before'],kwargs['energy_after']))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(LSResponseState,'update',capture)
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    cfg=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100,
        max_gaussians=1,relax_steps=200,rotation_hvp=40,pressure=.01)
    surface=ASEStressSurface(EMT())
    r=run_vc_ssw(a,surface,steps=1,config=cfg,rng=np.random.default_rng(17),
        ls=LSSettings({(29,29):1.},{(29,29):2.9},.001),ls_prequench='joint')
    ls=r.records[1]['ls']
    assert ls['prequench']=='joint_atoms_cell'
    assert calls==[(ls['enthalpy_before'],ls['enthalpy_after'])]
    assert abs(ls['volume_after']-ls['volume_before'])>1e-3
    assert ls['response_quantity']=='physical_enthalpy'
    assert ls['enthalpy_response']==pytest.approx((ls['enthalpy_after']-ls['enthalpy_before'])/len(a))
    assert ls['enthalpy_after']-ls['energy_after']==pytest.approx(cfg.pressure*ls['volume_after'])
    assert ls['soft_joint_gradient_norm']<=cfg.gradient_tol
    assert ls['soft_fmax']<=cfg.fmax and ls['soft_stress_max']<=cfg.stress_tol
    assert r.requests==surface.requests==sum(x['requests'] for x in r.records)


def test_joint_preparation_requires_ls_settings_before_pes():
    a=bulk('Cu','fcc',a=3.6,cubic=True);s=ASEStressSurface(EMT())
    with pytest.raises(ValueError,match='requires LS'):
        run_vc_ssw(a,s,steps=0,config=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100),
            rng=np.random.default_rng(1),ls_prequench='joint')
    assert s.requests==0


def test_joint_preparation_failure_keeps_failed_geometry_and_selected_minimum(monkeypatch):
    from pamssw.standalone import joint_ls
    from pamssw.standalone.ls_cycle import LSCycleError
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    failed=[]
    def failure(atoms,surface,*,softening,config):
        chart=SymmetricLogStrainChart(atoms,strain_length=config.strain_length)
        q=chart.pack(atoms);q[-6]=.02
        a=chart.unpack(q);surface.evaluate(a);failed.append(a.copy())
        raise LSCycleError('joint_soft_quench','injected last-state failure',
            result=joint_ls.JointLSFailureSnapshot(q,a,None,1))
    monkeypatch.setattr(joint_ls,'prepare_joint_ls_step',failure)
    a=bulk('Cu','fcc',a=3.6,cubic=True);s=ASEStressSurface(EMT())
    r=run_vc_ssw(a,s,steps=2,config=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100),
        rng=np.random.default_rng(17),ls=LSSettings({(29,29):1.},{(29,29):2.9},.001),ls_prequench='joint')
    assert r.status=='ls_prequench_failed' and len(r.minima)==1
    np.testing.assert_array_equal(r.records[1]['last_work'].cell.array,failed[0].cell.array)
    np.testing.assert_array_equal(r.current.atoms.cell.array,r.initial.atoms.cell.array)
    assert r.requests==s.requests==sum(x['requests'] for x in r.records)
