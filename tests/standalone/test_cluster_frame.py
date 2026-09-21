"""Numerical contracts for an explicit isolated-cluster coordinate section."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.cluster_frame import ClusterFrame, ClusterFrameFilter
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.surface import ASESurface, SurfaceCalculator, quench


def cluster():
    return Atoms('Cu4', positions=[[0,0,0],[2.4,.1,0],[.2,2.5,.1],[.3,.4,2.3]])


def test_reject_inapplicable_geometry():
    for a in (Atoms('Cu2',positions=[[0,0,0],[2,0,0]]),Atoms('Cu3',positions=[[0,0,0],[1,0,0],[2,0,0]])):
        with pytest.raises(ValueError): ClusterFrame(a)
    a=cluster();a.pbc=True
    with pytest.raises(ValueError): ClusterFrame(a)


def test_section_removes_translation_rotation_and_is_idempotent():
    a=cluster();frame=ClusterFrame(a);x=a.positions-a.positions.mean(axis=0)
    for axis in np.eye(3):
        assert np.linalg.norm(frame.project(np.broadcast_to(axis,x.shape)))<1e-12
        assert np.linalg.norm(frame.project(np.cross(axis,x)))<1e-12
    d=frame.project(np.random.default_rng(5).normal(size=x.shape))
    np.testing.assert_allclose(frame.project(d),d,atol=1e-12)
    np.testing.assert_allclose(frame.positions(a.positions+d),a.positions+d,atol=1e-12)


def test_full_biased_objective_force_matches_section_derivative():
    a=cluster();frame=ClusterFrame(a);rng=np.random.default_rng(7)
    n=frame.project(rng.normal(size=a.positions.shape));n/=np.linalg.norm(n)
    a.positions+=.13*n
    a.calc=SurfaceCalculator(ASESurface(EMT()),terms=[ProjectedGaussian(frame.reference,n,.2,.5)])
    filt=ClusterFrameFilter(a,frame);f=filt.get_forces();x=filt.get_positions()
    # Includes components normal to the section: the setter and force must use
    # the SAME pullback, not just project physical forces after moving freely.
    for _ in range(4):
        v=rng.normal(size=x.shape);v/=np.linalg.norm(v);h=1e-5
        filt.set_positions(x+h*v);ep=filt.get_potential_energy()
        filt.set_positions(x-h*v);em=filt.get_potential_energy()
        assert -(ep-em)/(2*h)==pytest.approx(float(np.sum(f*v)),abs=2e-7)
    filt.set_positions(x)


def test_biased_quench_stays_in_section_and_reports_section_force():
    a=cluster();frame=ClusterFrame(a);n=frame.project(np.random.default_rng(8).normal(size=a.positions.shape));n/=np.linalg.norm(n)
    a.positions+=.2*n
    q=quench(a,ASESurface(EMT()),terms=[ProjectedGaussian(frame.reference,n,.2,.5)],fmax=.03,steps=120,frame=frame)
    np.testing.assert_allclose(frame.positions(q.atoms.positions),q.atoms.positions,atol=1e-12)
    assert q.surface=='modified_cluster_section'
    with pytest.raises(ValueError):quench(a,ASESurface(EMT()),fmax=.03,steps=1,frame=frame)


def test_antipodal_rigid_copy_is_outside_positive_chart():
    a=cluster();frame=ClusterFrame(a)
    # In reference principal axes this proper 180-degree rotation still solves
    # linear Eckart equations, but belongs to the wrong alignment branch.
    x=frame.relative;_,axes=np.linalg.eigh(x.T@x)
    rotation=axes@np.diag([1.,-1.,-1.])@axes.T
    with pytest.raises(ValueError,match='alignment branch'):
        frame.positions(x@rotation+a.positions.mean(axis=0))


def test_frame_domain_failure_stays_in_ssw_records(monkeypatch):
    from pamssw.standalone import SSWConfig, run_ssw
    from pamssw.standalone.cluster_frame import ClusterFrameDomainError
    def failed_trial(self, candidate):
        raise ClusterFrameDomainError('injected chart crossing')
    monkeypatch.setattr(ClusterFrame,'positions',failed_trial)
    cfg=SSWConfig(width=.2,rotation_bias=100.,max_gaussians=2,
        temperature_K=300.,fmax=.03,relax_steps=200,fd_step=1e-4,
        rotation_hvp=20,rotation_tol=.02,cluster_frame='eckart',direction_sampling='global')
    result=run_ssw(cluster(),ASESurface(EMT()),steps=1,config=cfg,rng=np.random.default_rng(1))
    assert result.records[0].status=='cluster_frame_failed'
    assert result.records[0].landing is None
    assert result.records[0].evaluation_requests>=0
    assert len(result.minima)==1
