"""Independent periodic RC forest geometry: affine centers, rigid interiors.

Exact finite coordinate derivatives and ASE E/F/stress enthalpy pullback. No
native Kabsch/lambda parity, periodic lifting inference, or SSW driver here.
"""
import numpy as np
from scipy.linalg import expm,expm_frechet
from .rc_forest import RigidForestChart
from .vc_geometry import SYMMETRIC_BASIS,VCEvaluation,_valid_atoms


class RigidForestCellChart:
    """Fixed lifted molecular forest with six symmetric log-strain coordinates.

    L=L0 exp(S), each root center=(c0+t) exp(S); local rotated/torsioned
    vectors stay Cartesian and rigid. Only anchor translation is fixed; all
    root rotations remain physical relative to the cell. Caller must supply
    consistent lifted molecular atom positions, not individually wrapped ones.
    q uses Angstrom translations, Lrot*p, Ltor*theta, Lcell*s6.
    """
    def __init__(self,atoms,trees,*,anchor=0,rotation_length,torsion_length,strain_length):
        _valid_atoms(atoms)
        for value in (rotation_length,torsion_length,strain_length):
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:
                raise ValueError('positive finite metric lengths required')
        local=atoms.copy();local.pbc=False
        forest=RigidForestChart(local,trees,anchor=anchor)
        self.reference=atoms.copy();self.reference.calc=None
        self.reference_cell=atoms.cell.array.copy()
        self.strain_length=float(strain_length);self.parts=[];self.kinds=[]
        for k,(ids,chart,oldcolumns,oldspan) in enumerate(forest.parts):
            columns=np.arange(3 if k==anchor else 0,chart.dimension)
            start=len(self.kinds)
            self.kinds.extend('translation' if j<3 else 'rotation' if j<6 else 'torsion' for j in columns)
            span=slice(start,len(self.kinds))
            center=chart.reference.positions[list(chart.bodies[0])].mean(axis=0)
            self.parts.append((ids,chart,columns,span,center))
        self.scales=np.array([1. if kind=='translation' else rotation_length if kind=='rotation' else torsion_length for kind in self.kinds])
        self.dimension=self.ndof=len(self.kinds)+6

    def geometry(self,q):
        """Return Atoms, dR/dq [N,3,D], dL/dq [3,3,D], without EFS calls."""
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all():raise ValueError('finite full chart coordinate required')
        strain=np.einsum('i,ijk->jk',q[-6:]/self.strain_length,SYMMETRIC_BASIS)
        deformation=expm(strain)
        dexps=[expm_frechet(strain,B/self.strain_length,compute_expm=False) for B in SYMMETRIC_BASIS]
        out=self.reference.copy();out.cell=self.reference_cell@deformation
        J=np.zeros((len(out),3,self.dimension));C=np.zeros((3,3,self.dimension))
        for ids,chart,columns,span,center in self.parts:
            local=np.zeros(chart.dimension);local[columns]=q[span]/self.scales[span]
            a,j=chart.evaluate(local);rootcenter=center+local[:3]
            out.positions[ids]=a.positions+rootcenter@(deformation-np.eye(3))
            jr=j[:,:,columns].copy()
            for k,col in enumerate(columns):
                if col<3:jr[:,:,k]+=(deformation-np.eye(3))[col]
            J[ids,:,span]=jr/self.scales[span]
            for k,dexp in enumerate(dexps):J[ids,:,-6+k]=rootcenter@dexp
        for k,dexp in enumerate(dexps):C[:,:,-6+k]=self.reference_cell@dexp
        _valid_atoms(out)
        if not np.isfinite(J).all() or not np.isfinite(C).all():raise ValueError('nonfinite chart derivative')
        return out,J,C

    def unpack(self,q):return self.geometry(q)[0]

    def evaluate(self,q,evaluate,*,pressure=0.):
        """Exact E+pV derivative including nonaffine force/virial correction.

        For A=L^-1 dL: dH=-F:(dR-R A)+V*(stress+pI):A.
        ASE stress must be tensile-positive symmetric eV/Angstrom^3.
        """
        if not np.isscalar(pressure) or not np.isfinite(pressure):raise ValueError('finite scalar pressure required')
        a,J,C=self.geometry(q);energy,forces,stress=evaluate(a)
        energy=float(energy);forces=np.asarray(forces,dtype=float);stress=np.asarray(stress,dtype=float)
        if not np.isfinite(energy) or forces.shape!=a.positions.shape or stress.shape!=(3,3) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError('invalid energy/forces/full stress')
        tolerance=64*np.finfo(float).eps*max(1.,np.linalg.norm(stress))
        if np.linalg.norm(stress-stress.T)>tolerance:raise ValueError('symmetric ASE stress required')
        volume=a.get_volume();gradient=np.empty(self.dimension)
        for k in range(self.dimension):
            A=np.linalg.solve(a.cell.array,C[:,:,k])
            nonaffine=J[:,:,k]-a.positions@A
            gradient[k]=-np.sum(forces*nonaffine)+volume*np.sum((stress+pressure*np.eye(3))*A)
        objective=energy+pressure*volume
        if not np.isfinite(objective) or not np.isfinite(gradient).all():raise ValueError('nonfinite objective or gradient')
        return VCEvaluation(objective,gradient,a,energy,forces.copy(),stress.copy(),volume)
