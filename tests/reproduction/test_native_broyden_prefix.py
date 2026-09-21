"""Recovered executable arithmetic, not scientific efficacy tests."""
import numpy as np
import pytest
from pamssw.standalone.native_broyden import native_block_sum_product, initial_step, secant_prefix

def test_native_product_is_degenerate_and_not_rotation_invariant():
    v = np.array([[1., -1., 0.]])
    q = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    assert np.linalg.det(q) == 1
    assert native_block_sum_product(v, v) == 0.
    assert native_block_sum_product(v @ q, v @ q) == 4.
    assert np.sum(v*v) == np.sum((v@q)**2)

def test_initial_step_and_secant_prefix():
    x=np.array([[0.,1.,2.]])
    f=np.array([[1.,2.,3.]])
    g=np.array([[.2,.3,.4]])
    nxt=initial_step(x,f,g)
    np.testing.assert_allclose(nxt,x+g*f)
    now=x+np.array([[.1,.2,.3]])
    df=np.array([[2.,1.,1.]])
    r=secant_prefix(now,f+df,x,f,g)
    assert r.normalizer == 4.
    np.testing.assert_allclose(r.force_difference,df/4)
    np.testing.assert_allclose(r.displacement,now-x)
    np.testing.assert_allclose(r.u,(g*df+now-x)/4)

def test_explicit_domain_divergence_for_null_semimetric():
    z=np.zeros((1,3))
    with pytest.raises(ValueError,match='zero block-sum'):
        secant_prefix(z,np.array([[1.,-1.,0.]]),z,z,np.ones((1,3)))

def test_no_partial_xyz_blocks():
    with pytest.raises(ValueError,match='three'):
        native_block_sum_product(np.ones(4),np.ones(4))

def test_frozen_elf_prefix_records():
    import json
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/native-broyden-prefix/result.json'
    report=json.loads(path.read_text())
    assert report['passed']==report['total']==18
    for case in report['cases']:
        a={k:np.asarray(v) for k,v in case['input'].items()}
        observed=case['emulated']
        np.testing.assert_allclose(initial_step(a['x'],a['f'],a['g0']),observed['initial'],rtol=1e-12,atol=1e-12)
        r=secant_prefix(a['x2'],a['f2'],a['x'],a['f'],a['g0'])
        for value,key in [(r.force_difference,'df'),(r.displacement,'dx'),(r.u,'u')]:
            np.testing.assert_allclose(value,observed[key],rtol=1e-12,atol=1e-12)
