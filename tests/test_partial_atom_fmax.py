import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
import importlib
block=importlib.import_module('pamssw.standalone.block_ssw')
from pamssw.standalone.block_ssw import BlockSSWConfig, run_block_ssw
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.vc_geometry import ASEStressSurface

def atomic():
    return SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=0.,fmax=.05,relax_steps=20,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')

def test_partial_override_dispatches_only_cell_interleave():
    a=bulk('Cu','fcc',a=3.65,cubic=True).repeat((2,1,1)); del a[0]
    c=BlockSSWConfig(atomic(),3.6,cell_cycles=1,atomic_period=2,partial_atom_steps=2,partial_atom_fmax=.001)
    calls=[]; original=block.safe_lbfgs
    def spy(*args,**kw): calls.append(kw['gtol']); return original(*args,**kw)
    block.safe_lbfgs=spy
    try:
        r=run_block_ssw(a,ASEStressSurface(EMT()),steps=2,config=c,rng=np.random.default_rng(7))
    finally: block.safe_lbfgs=original
    assert calls and all(x==.001 for x in calls)
    assert r.records[1]['cell_cycles'][0]['partial_atom_fmax']==.001
    assert r.records[0]['requests'] > 0

def test_partial_override_validation():
    with pytest.raises(ValueError): BlockSSWConfig(atomic(),3.6,partial_atom_fmax=0.)
    with pytest.raises(ValueError): BlockSSWConfig(atomic(),3.6,partial_atom_fmax=float('nan'))
