"""Narrow experimental ASE adapter for the audited stock-CHARMM XXXII model."""
import ctypes,hashlib,json
from pathlib import Path
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator,all_changes
from ase.calculators.lammps.coordinatetransform import Prism
from ase.stress import full_3x3_to_voigt_6_stress
from research.ga_ssw.convert_xxxii_amber import DATA_SHA,INPUT_SHA,sections

ENERGY_TO_EV=units.kcal/units.mol
# LAMMPS real-unit nktv2p: (kcal/mol)/A^3 -> atmosphere.
REAL_NKTV2P=68568.415
TYPE_NUMBERS={1:8,2:6,3:8,4:1,5:6,6:1,7:7,8:1,9:6,10:1,11:17}

class XXXIILammpsCalculator(Calculator):
    """Persistent fixed-ID topology; changing species/count/PBC is unsupported.

    model_manifest must be the converter manifest; paths are explicit. LAMMPS
    imports only on first evaluation. Call close() or use the context manager.
    """
    implemented_properties=['energy','forces','stress']
    def __init__(self,*,data_path,input_path,model_manifest,reference_atoms):
        super().__init__()
        self.data_path=Path(data_path).resolve();self.input_path=Path(input_path).resolve()
        self.model_manifest=Path(model_manifest).resolve()
        manifest=json.loads(self.model_manifest.read_text())
        if manifest['source']['data_sha256']!=DATA_SHA or manifest['source']['input_sha256']!=INPUT_SHA:
            raise ValueError('audited XXXII source manifest required')
        for name,path in [('data',self.data_path),('input',self.input_path)]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['output_sha256'][name]:raise ValueError('converted file SHA mismatch')
        data=sections(self.data_path.read_text());rows=sorted(data['Atoms'],key=lambda r:int(r[0]))
        if [int(r[0]) for r in rows]!=list(range(1,173)):raise ValueError('172 contiguous original atom IDs required')
        self.numbers=np.array([TYPE_NUMBERS[int(r[2])] for r in rows])
        self.atom_types=np.array([int(r[2]) for r in rows],dtype=np.int32)
        self.charges=np.array([float(r[3]) for r in rows])
        self._check(reference_atoms)
        self._lmp=None;self.requests=0;self.closed=False
    def _check(self,atoms):
        if not np.array_equal(atoms.numbers,self.numbers):raise ValueError('fixed ordered XXXII species required')
        if not np.asarray(atoms.pbc).all():raise ValueError('fully periodic XXXII cell required')
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all() or np.linalg.det(atoms.cell.array)<=0:
            raise ValueError('finite positions and positive cell determinant required')
    def _new_engine(self):
        from lammps import lammps
        return lammps(cmdargs=['-log','none','-screen','none'])
    def _initialize(self):
        engine=self._new_engine()
        try:
            for raw in self.input_path.read_text().splitlines():
                command=raw.split('#')[0].strip()
                if not command:continue
                if command.startswith('read_data '):command=f'read_data "{self.data_path}"'
                engine.command(command)
            engine.command('compute pam_virial all pressure NULL virial')
            engine.command('thermo_style custom step pe c_pam_virial[1] c_pam_virial[2] c_pam_virial[3] c_pam_virial[4] c_pam_virial[5] c_pam_virial[6]')
            engine.command('thermo 1')
            self._verify_identity(engine)
        except BaseException:
            engine.close();raise
        self._lmp=engine
    def _verify_identity(self,engine):
        if int(engine.get_natoms())!=172:raise RuntimeError('LAMMPS atom count changed')
        types=np.ctypeslib.as_array(engine.gather_atoms('type',0,1),shape=(172,))
        charges=np.ctypeslib.as_array(engine.gather_atoms('q',1,1),shape=(172,))
        if not np.array_equal(types,self.atom_types) or not np.array_equal(charges,self.charges):
            raise RuntimeError('LAMMPS original per-ID force-field types/charges changed')
    def calculate(self,atoms=None,properties=('energy','forces','stress'),system_changes=all_changes):
        if self.closed:raise RuntimeError('calculator explicitly closed')
        self._check(atoms)
        super().calculate(atoms,properties,system_changes)
        self.results={}
        if self._lmp is None:self._initialize()
        lmp=self._lmp;prism=Prism(atoms.cell.array,pbc=True,reduce_cell=False)
        xhi,yhi,zhi,xy,xz,yz=prism.get_lammps_prism()
        self.requests+=1
        # No remap: scatter supplies the complete new positions exactly once.
        lmp.command('change_box all '+f'x final 0 {xhi:.17g} y final 0 {yhi:.17g} z final 0 {zhi:.17g} xy final {xy:.17g} xz final {xz:.17g} yz final {yz:.17g} units box')
        # Old image counters describe the previous frame, not this fresh scatter.
        # run 0 performs PBC remapping and reconstructs image flags/ghosts.
        lmp.command('set atom * image 0 0 0')
        pos=np.ascontiguousarray(prism.vector_to_lammps(atoms.positions,wrap=False),dtype=np.float64)
        lmp.scatter_atoms('x',1,3,pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        lmp.command('run 0 post no')
        self._verify_identity(lmp)
        force=np.ctypeslib.as_array(lmp.gather_atoms('f',1,3),shape=(172*3,)).copy().reshape(172,3)
        pv=lmp.extract_compute('pam_virial',0,1)
        xx,yy,zz,xyv,xzv,yzv=[float(pv[i]) for i in range(6)]
        pressure=np.array([[xx,xyv,xzv],[xyv,yy,yzv],[xzv,yzv,zz]])
        stress=prism.tensor2_to_ase(-pressure*ENERGY_TO_EV/REAL_NKTV2P)
        energy=float(lmp.get_thermo('pe'))*ENERGY_TO_EV
        force=prism.vector_to_ase(force)*ENERGY_TO_EV
        if not np.isfinite(energy) or not np.isfinite(force).all() or not np.isfinite(stress).all():raise RuntimeError('nonfinite XXXII energy/forces/stress')
        self.results=dict(energy=energy,forces=force,stress=full_3x3_to_voigt_6_stress(stress))
    def close(self):
        if self._lmp is not None:self._lmp.close();self._lmp=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
