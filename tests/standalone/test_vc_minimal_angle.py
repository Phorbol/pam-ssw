import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy


def test_vc_minimal_angle_uses_joint_gradient_and_retains_inputs():
    atoms=bulk('Cu','fcc',a=3.6,cubic=True)
    cfg=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100,
        max_gaussians=1,relax_steps=200,rotation_hvp=40)
    surface=ASEStressSurface(EMT())
    result=run_vc_ssw(atoms,surface,steps=1,config=cfg,rng=np.random.default_rng(17),
        height_policy=MinimalAngleHeightPolicy())
    event=result.records[1]['climb'][0]
    preparation=event['height_preparation']
    assert preparation.status=='prepared'
    assert abs(preparation.angle_degrees-87)<1e-9
    assert abs(preparation.criterion_residual)<1e-10
    assert event['height_requests']==1
    assert len(event['height_input']['background_force'])==3*len(atoms)+6
    assert result.requests==surface.requests==sum(r['requests'] for r in result.records)


def test_vc_angle_background_counts_frozen_ls_and_old_gaussians_once():
    from pamssw.standalone.paper_reference import LSSettings
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    atoms=bulk('Cu','fcc',a=3.6,cubic=True)
    cfg=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100,
        max_gaussians=2,relax_steps=200,rotation_hvp=40)
    result=run_vc_ssw(atoms,ASEStressSurface(EMT()),steps=1,config=cfg,
        rng=np.random.default_rng(17),height_policy=MinimalAngleHeightPolicy(),
        ls=LSSettings({(29,29):1.},{(29,29):2.9},.001))
    record=result.records[1]
    chart=SymmetricLogStrainChart(record['chart_reference'],strain_length=cfg.strain_length)
    soft=record['frozen_softening']
    surface=ASEStressSurface(EMT())
    def total(a):
        e,f,s=surface.evaluate(a);b,bf,bs=soft.evaluate_stress(a)
        return e+b,f+bf,s+bs
    prepared=[stage for stage in record['climb'] if 'height_input' in stage]
    assert len(prepared)==2
    assert len(prepared[1]['height_input']['history'])==1
    for stage in prepared:
        data=stage['height_input'];point=data['point']
        force=-chart.evaluate(point,total).gradient
        for term in data['history']:
            force+=term.evaluate(point)[1]
        np.testing.assert_allclose(force,data['background_force'],atol=1e-11,rtol=0.)
        assert stage['height_requests']==1
    assert np.linalg.norm(prepared[0]['height_input']['direction'][-6:])>1e-6


def test_satisfied_angle_does_not_release_or_insert_zero_height():
    class Satisfied(MinimalAngleHeightPolicy):
        def prepare(self,history,**kwargs):
            n=kwargs['direction']; transverse=np.zeros_like(n); transverse[0]=1.
            transverse-=float(transverse@n)*n
            kwargs['background_force']=n+.1*transverse
            return super().prepare(history,**kwargs)
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    cfg=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100,max_gaussians=1)
    r=run_vc_ssw(a,ASEStressSurface(EMT()),steps=1,config=cfg,
        rng=np.random.default_rng(17),height_policy=Satisfied())
    event=r.records[1]
    assert event['status']=='nonpositive_height'
    assert event['landing'] is None and len(r.minima)==1
    assert event['frozen_gaussians']==[]
    assert event['climb'][0]['biased_quench_requests']==0
