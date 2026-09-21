"""Corrected source LJBase.monoOpt: rigid-unit intermolecular auxiliary LJ.

No physical PES calls and no true-minimum certification. Preserve source
Euler geometry/energy; replace inconsistent paired angular derivatives by
individual analytic chain rules and use an independent bounded L-BFGS solver.
"""
from dataclasses import dataclass
import numpy as np
from .ga_operators import _partition,_rotation


class MolecularLJChart:
    def __init__(self,atoms,groups,pair_sigma):
        self.groups=_partition(atoms,groups)
        self.reference=atoms.copy();self.reference.calc=None
        self.local=[];self.initial=np.zeros((len(self.groups),6))
        self.labels=np.zeros(len(atoms),dtype=int)
        for k,group in enumerate(self.groups):
            x=atoms.positions[list(group)];center=x.mean(axis=0)
            self.initial[k,:3]=center;self.local.append(x-center);self.labels[list(group)]=k
        i,j=np.triu_indices(len(atoms),1);mask=self.labels[i]!=self.labels[j]
        self.i,self.j=i[mask],j[mask];sigma=[]
        for a,b in zip(atoms.numbers[self.i],atoms.numbers[self.j]):
            key=tuple(sorted((int(a),int(b))))
            if key not in pair_sigma:raise ValueError(f'missing explicit intermolecular sigma {key}')
            value=float(pair_sigma[key])
            if not np.isfinite(value) or value<=0:raise ValueError('positive finite pair sigmas required')
            sigma.append(value)
        self.sigma=np.array(sigma)

    def geometry(self,q):
        q=np.asarray(q,dtype=float)
        if q.size!=self.initial.size or not np.isfinite(q).all():raise ValueError('finite six coordinates per molecular group required')
        q=q.reshape(-1,6);a=self.reference.copy();jac=[]
        generators=(np.array([[0,0,0],[0,0,1],[0,-1,0]]),np.array([[0,0,-1],[0,0,0],[1,0,0]]),np.array([[0,1,0],[-1,0,0],[0,0,0]]))
        for k,group in enumerate(self.groups):
            r=[_rotation(axis,angle) for axis,angle in enumerate(q[k,3:])]
            full=r[0]@r[1]@r[2];derivatives=[]
            for axis in range(3):
                parts=list(r);parts[axis]=parts[axis]@generators[axis]
                derivatives.append(self.local[k]@(parts[0]@parts[1]@parts[2]))
            a.positions[list(group)]=self.local[k]@full+q[k,:3];jac.append(derivatives)
        return a,jac

    def evaluate(self,q):
        a,jac=self.geometry(q);d=a.positions[self.i]-a.positions[self.j]
        r2=np.einsum('ij,ij->i',d,d)
        if np.any(r2<=0):raise ValueError('intermolecular auxiliary LJ overlap')
        t=(self.sigma**2/r2)**3;energy=float(np.sum(4*(t*t-t)))
        pair=(24*(t-2*t*t)/r2)[:,None]*d;cart=np.zeros_like(a.positions)
        np.add.at(cart,self.i,pair);np.add.at(cart,self.j,-pair)
        g=np.zeros_like(self.initial)
        for k,group in enumerate(self.groups):
            grad=cart[list(group)];g[k,:3]=grad.sum(axis=0)
            g[k,3:]=[np.sum(grad*derivative) for derivative in jac[k]]
        return energy,g.ravel()


@dataclass(frozen=True)
class MolecularAuxiliaryResult:
    atoms: object
    energy_aux: float
    gradient_norm: float
    status: str
    auxiliary_evaluations: int
    pair_evaluations: int
    certified_physical_minimum: bool = False


def optimize_molecular_lj(atoms,groups,pair_sigma,*,max_evaluations=1000,gradient_tol=.1):
    """LJ_Monomer source energy on rigid translations/Euler rotations.

    Source defaults eps .1, maxSteps1000; independent solver stopping norm
    and line-search trajectory differ. Budget counts every auxiliary E/G call.
    Returned geometry/energy are always a paired evaluation, including failure.
    """
    from scipy.optimize import minimize
    if isinstance(max_evaluations,bool) or not isinstance(max_evaluations,(int,np.integer)) or max_evaluations<1:raise ValueError('positive auxiliary evaluation budget required')
    if not np.isfinite(gradient_tol) or gradient_tol<=0:raise ValueError('positive auxiliary gradient tolerance required')
    chart=MolecularLJChart(atoms,groups,pair_sigma);state=dict(calls=0,q=None,energy=None,g=None)
    class Budget(Exception):pass
    def evaluate(q):
        if state['calls']>=max_evaluations:raise Budget()
        # Count failed evaluations too; preserve previous successfully paired state.
        state['calls']+=1;energy,g=chart.evaluate(q);state.update(q=q.copy(),energy=energy,g=g.copy())
        return energy,g
    try:
        result=minimize(evaluate,chart.initial.ravel(),jac=True,method='L-BFGS-B',options=dict(maxcor=5,gtol=gradient_tol,ftol=0.,maxiter=max_evaluations,maxls=20))
        status='auxiliary_converged' if result.success else 'auxiliary_optimizer_stopped'
    except Budget:status='auxiliary_budget_exhausted'
    except (ValueError,RuntimeError,FloatingPointError) as error:
        if state['q'] is None:
            error.auxiliary_evaluations=state['calls']
            raise
        status='auxiliary_evaluation_failed: '+str(error)
    a,_=chart.geometry(state['q'])
    return MolecularAuxiliaryResult(a,float(state['energy']),float(np.linalg.norm(state['g'])),status,
        state['calls'],state['calls']*len(chart.i))
