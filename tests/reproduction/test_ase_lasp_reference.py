"""ASE contract tests, separate from recorded real NN calculations."""
from pathlib import Path
import numpy as np
import pytest
import research.ga_ssw.ase_lasp_reference as module

ARC=Path(__file__).parents[2]/'research/ga_ssw/evidence/staged-water/final-arc/0.arc'

class CountingOracle:
    def __init__(self,*args):self.calls=0
    def evaluate(self,text,label):
        self.calls+=1
        return -1.,np.zeros((45,3)),{'wall_seconds':0.}


def test_ase_energy_force_cache_and_position_invalidation(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'Oracle',CountingOracle)
    atoms=module.read_water_atoms(ARC);calc=module.LaspWaterReference(tmp_path,tmp_path/'out',ARC);atoms.calc=calc
    assert atoms.get_potential_energy()==-1
    assert atoms.get_forces().shape==(45,3)
    assert calc.oracle.calls==1
    atoms.positions[0,0]+=.01;atoms.get_forces()
    assert calc.oracle.calls==2


def test_unsupported_cell_change_fails_before_oracle_call(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'Oracle',CountingOracle)
    atoms=module.read_water_atoms(ARC);calc=module.LaspWaterReference(tmp_path,tmp_path/'out',ARC);atoms.calc=calc
    atoms.set_cell(atoms.cell*1.01)
    with pytest.raises(ValueError,match='fixed input vacuum cell'):atoms.get_forces()
    assert calc.oracle.calls==0


def test_single_point_config_handles_whitespace_and_rejects_ambiguous_steps():
    from research.ga_ssw.validate_water_archive import single_point_config
    text='Run_type 5\nSSW.SSWsteps   100 # original search\nSSW.output F\nSSW.printevery F\n'
    result=single_point_config(text)
    assert 'SSW.SSWsteps   0 # original search' in result
    with pytest.raises(ValueError,match='ssw.sswsteps'):single_point_config(text+'ssw.sswsteps 3\n')
