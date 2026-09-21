"""No LAMMPS import/PES required: fixed-ID and coordinate transport contract."""
import ctypes
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from scipy.spatial.transform import Rotation
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator,ENERGY_TO_EV,REAL_NKTV2P
ROOT=Path(__file__).resolve().parents[2]
CONVERTED=ROOT/'research/ga_ssw/evidence/xxxii-stock-charmm-converted'


def setup():
    a=read(ROOT/'tests/standalone/fixtures/type2_xxxii.extxyz')
    c=XXXIILammpsCalculator(data_path=CONVERTED/'lmp.data',input_path=CONVERTED/'in.simple',model_manifest=CONVERTED/'manifest.json',reference_atoms=a)
    return a,c


class Engine:
    def __init__(self,c):
        self.types=(ctypes.c_int*172)(*c.atom_types);self.charges=(ctypes.c_double*172)(*c.charges)
        self.commands=[];self.positions=None;self.closed=False
    def command(self,s):self.commands.append(s)
    def get_natoms(self):return 172
    def gather_atoms(self,name,*args):
        if name=='type':return self.types
        if name=='q':return self.charges
        self.force=(ctypes.c_double*(172*3))(*(-self.positions).ravel());return self.force
    def scatter_atoms(self,name,typ,n,p):
        assert (name,typ,n)==('x',1,3)
        self.positions=np.ctypeslib.as_array(p,shape=(172*3,)).copy().reshape(172,3)
    def extract_compute(self,*args):
        assert args==('pam_virial',0,1)
        return [1.,2.,3.,.1,.2,.3]
    def get_thermo(self,name):assert name=='pe';return .5*np.sum(self.positions**2)
    def close(self):self.closed=True


def test_persistent_topology_unwrapped_coordinates_and_rotated_stress(monkeypatch):
    a,c=setup();e=Engine(c);monkeypatch.setattr(c,'_new_engine',lambda:e)
    # Deliberately use unwrapped coordinates several cells away.
    a.positions[0]+=2*a.cell.array[0]
    a.calc=c;energy=a.get_potential_energy();f=a.get_forces();stress=a.get_stress(voigt=False)
    np.testing.assert_allclose(f,-a.positions*ENERGY_TO_EV,atol=1e-13)
    assert energy==pytest.approx(.5*np.sum(a.positions**2)*ENERGY_TO_EV)
    u=Rotation.from_rotvec([.4,-.2,.3]).as_matrix();b=a.copy();b.positions=b.positions@u.T;b.cell=b.cell.array@u.T;b.calc=c
    np.testing.assert_allclose(b.get_forces(),f@u.T,atol=1e-13)
    np.testing.assert_allclose(b.get_stress(voigt=False),u@stress@u.T,atol=1e-16)
    assert c.requests==2
    assert len([x for x in e.commands if x.startswith('read_data')])==1
    assert len([x for x in e.commands if x=='set atom * image 0 0 0'])==2
    assert all(' remap ' not in x for x in e.commands if x.startswith('change_box'))
    assert not any(' type ' in x for x in e.commands)
    c.close();assert e.closed
    with pytest.raises(RuntimeError,match='closed'):c.calculate(a)


def test_rejects_changed_species_and_partial_pbc_before_engine():
    a,c=setup();a.pbc=[True,True,False]
    with pytest.raises(ValueError,match='fully periodic'):c.calculate(a)
    a.pbc=True;a.numbers[0]=1
    with pytest.raises(ValueError,match='ordered'):c.calculate(a)
    assert c.requests==0 and c._lmp is None


def test_failed_identity_preserves_attempt_cost_and_no_stale_results(monkeypatch):
    a,c=setup();e=Engine(c);monkeypatch.setattr(c,'_new_engine',lambda:e);a.calc=c;a.get_potential_energy()
    e.types[0]=2;a.positions[0,0]+=.01
    with pytest.raises(RuntimeError,match='types/charges'):a.get_potential_energy()
    assert c.requests==2 and c.results=={}
    c.close()
