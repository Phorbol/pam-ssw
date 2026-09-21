import math
from pathlib import Path
import pytest
from research.ga_ssw.convert_xxxii_amber import convert,sections,verify_topology
SRC=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')


def test_actual_xxxii_preserves_all_identity_and_bonded_data(tmp_path):
    m=convert(SRC/'lmp.data',SRC/'in.simple',tmp_path/'out')
    before=sections((SRC/'lmp.data').read_text());after=sections((tmp_path/'out/lmp.data').read_text())
    for name,rows in before.items():
        if name not in ('Pair Coeffs','Dihedral Coeffs'):assert after[name]==rows
    assert m['topology']['endpoint_pairs']==392 and m['topology']['zero_weight_duplicates']==40
    assert m['physical_requests']==0
    for original,new in zip(before['Dihedral Coeffs'],after['Dihedral Coeffs']):
        assert original[:4]==new[:4]
        assert float(new[4])==(5/6 if int(original[0])<=8 else 0.)
    text=(tmp_path/'out/in.simple').read_text()
    assert 'lj/charmm/coul/long 9 10' in text and 'special_bonds  lj 0 0 0 coul 0 0 0' in text
    assert 'units           real' in text and 'kspace_style    ewald 1e-6' in text


def test_normal_and_unswitched_endpoint_algebra_for_all_type_pairs(tmp_path):
    convert(SRC/'lmp.data',SRC/'in.simple',tmp_path/'out')
    old=sections((SRC/'lmp.data').read_text())['Pair Coeffs'];new=sections((tmp_path/'out/lmp.data').read_text())['Pair Coeffs']
    for i,left in enumerate(old):
        for j,right in enumerate(old):
            eps=math.sqrt(float(left[1])*float(right[1]));R=(float(left[2])+float(right[2]))/2
            sigma=(float(new[i][2])+float(new[j][2]))/2;e14=math.sqrt(float(new[i][3])*float(new[j][3]))
            for r in (1.7,3.,9.5,12.):
                t=(R/r)**6;u=(sigma/r)**6
                assert 4*eps*(u*u-u)==pytest.approx(eps*(t*t-2*t),rel=1e-12,abs=1e-12)
                assert (5/6)*4*e14*(u*u-u)==pytest.approx(.5*eps*(t*t-2*t),rel=1e-12,abs=1e-12)
                assert (5/6)*24*e14*(2*u*u-u)/r==pytest.approx(.5*12*eps*(t*t-t)/r,rel=1e-12,abs=1e-12)


def test_rejects_modified_source_and_nonunique_weighted_endpoint(tmp_path):
    changed=(SRC/'lmp.data').read_text().replace('0.8333333333','0.7',1);p=tmp_path/'changed';p.write_text(changed)
    with pytest.raises(ValueError,match='source SHA mismatch'):convert(p,SRC/'in.simple',tmp_path/'out')
    d=sections((SRC/'lmp.data').read_text());extra=next(r for r in d['Dihedrals'] if int(r[1])>8);extra[1]=str(int(extra[1])-8)
    with pytest.raises(ValueError,match='exactly one'):verify_topology(d)
