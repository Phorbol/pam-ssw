"""Conservative, stage-frozen variant of the recovered native height policy.

Original set_initial_gaussw history writes are preserved. Old Gaussian forces
are counted once, unlike the inspected native addgaussian. No walker is changed.
"""
from dataclasses import dataclass,replace
import math
import numpy as np


@dataclass(frozen=True)
class WeightChange:
    index: int  # zero based
    before: float
    after: float


@dataclass(frozen=True)
class InitialWeightHistory:
    weights: tuple
    changed_history: tuple
    initial_new_weight: float


@dataclass(frozen=True)
class ConservativeNativeHeightPolicy:
    initial_weight: float  # eV, para.w_initial
    negative_weight: float  # eV, para.w_neg
    level: int  # para.w_level: only 1/2 apply historical overrides
    max_weight: float  # eV; post-update stop, not clipping
    growth_step: float  # multiplier applied to growth_scale after each update
    growth_scale: float  # initial multiplicative growth factor

    def __post_init__(self):
        for name in ('initial_weight','negative_weight','max_weight','growth_step','growth_scale'):
            if not np.isscalar(getattr(self,name)) or not math.isfinite(getattr(self,name)):raise ValueError('finite scalar height parameters required')
        if self.initial_weight<=0 or self.negative_weight<=0 or self.max_weight<0 or self.growth_step<1 or self.growth_scale<=1:raise ValueError('positive growing finite-domain height policy required')
        if isinstance(self.level,(bool,np.bool_)) or not isinstance(self.level,(int,np.integer)):raise ValueError('integer level required')

    def initialize_history(self,weights,*,curvature):
        """Append Wng then apply exact recovered historical first/second writes.

        Curvature<0 selects w_neg. Level1 rewrites W1=5.6; level2 rewrites
        W1=.5 and, when ng>1, W2=.5. These native constants use eV here.
        Other integer levels do not overwrite history. No caller defaults guessed.
        """
        if not np.isscalar(curvature) or not np.isfinite(curvature):raise ValueError('finite curvature required')
        old=tuple(float(w) for w in weights)
        if any(not math.isfinite(w) or w<=0 for w in old):raise ValueError('positive finite old weights required')
        values=list(old)+( [self.negative_weight if curvature<0 else self.initial_weight] )
        if self.level==1:values[0]=5.6
        elif self.level==2:
            values[0]=.5
            if len(values)>1:values[1]=.5
        changes=tuple(WeightChange(i,w,values[i]) for i,w in enumerate(old) if w!=values[i])
        return InitialWeightHistory(tuple(values),changes,values[-1])

    def prepare(self,history,*,center,direction,width,point,background_force,curvature,curvature_scope,max_updates):
        """Initialize/adjust once, return immutable terms for subsequent objective.

        background_force MUST equal physical/LS force plus the INPUT history's
        single-count Gaussian forces at point, in the same scaled coordinates.
        This function corrects that force for any level-based history writes.
        curvature must exclude the rank-one rotation-only bias; if the rotation
        surface includes LS, it still includes LS curvature. curvature_scope
        records the caller declaration; this helper cannot verify it. Native
        curv_real LS inclusion remains unclosed.
        It does not evaluate a PES or mutate an input. max_updates is an explicit
        numerical work ceiling: exhaustion raises, never silently accepts.
        """
        if not isinstance(curvature_scope,str) or not curvature_scope.strip():raise ValueError("explicit curvature scope required")
        history=tuple(history)
        if any(not isinstance(t,FrozenHeightGaussian) for t in history):raise TypeError('FrozenHeightGaussian history required')
        if isinstance(max_updates,(bool,np.bool_)) or not isinstance(max_updates,(int,np.integer)) or max_updates<1:raise ValueError('positive integer numerical update budget required')
        init=self.initialize_history([t.weight for t in history],curvature=curvature)
        latest=FrozenHeightGaussian(center,direction,width,init.initial_new_weight)
        point=np.asarray(point,dtype=float);background=np.array(background_force,dtype=float,copy=True)
        if point.shape!=latest.center.shape or background.shape!=point.shape or not np.isfinite(point).all() or not np.isfinite(background).all():raise ValueError('finite compatible point/background force required')
        updated=[]
        for i,t in enumerate(history):
            if t.center.shape!=point.shape:raise ValueError('history coordinate dimension mismatch')
            new=replace(t,weight=init.weights[i]);background+=new.evaluate(point)[1]-t.evaluate(point)[1];updated.append(new)
        projection=float((point-latest.center)@latest.direction)
        d1=math.exp(-.5*(projection/latest.width)**2);d2=projection/latest.width**2
        if not 0<d1<=1 or not d2>0:raise ValueError('positive nonunderflowed forward projection required')
        w=latest.weight;scale=self.growth_scale;contribution=d1*w*d2*latest.direction;force=background+contribution
        def angle(f):
            norm=float(np.linalg.norm(f))
            if not math.isfinite(norm) or norm==0:raise ValueError('finite nonzero resultant required')
            cosine=float(f@latest.direction)/norm
            if not -1<=cosine<=1:raise ValueError('invalid acos operand')
            return math.acos(cosine)*180./3.1415926535900001
        angle_value=angle(force);trace=[];reason='angle_satisfied'
        if angle_value>87.:
            while True:
                if len(trace)>=max_updates:raise RuntimeError('explicit numerical height-update budget exhausted')
                force-=contribution
                next_weight=min(w*scale,w+2.)
                if not math.isfinite(next_weight) or next_weight<=w:raise ValueError('height update cannot make representable progress')
                w=next_weight;scale*=self.growth_step;contribution=d1*w*d2*latest.direction;force+=contribution;angle_value=angle(force)
                trace.append((w,angle_value))
                if w>self.max_weight:reason='maxw_exceeded';break
                if angle_value<=87.:break
        updated.append(replace(latest,weight=w));terms=tuple(updated)
        energy=sum(t.evaluate(point)[0] for t in terms)
        force.setflags(write=False)
        return PreparedNativeHeight(terms,init.changed_history,init.initial_new_weight,w,angle_value,reason,tuple(trace),float(energy),force,float(curvature),curvature_scope)


@dataclass(frozen=True)
class FrozenHeightGaussian:
    """Flat scaled-coordinate Gaussian with exact E and force; immutable arrays."""
    center: np.ndarray
    direction: np.ndarray
    width: float
    weight: float

    def __post_init__(self):
        center=np.array(self.center,dtype=float,copy=True);n=np.array(self.direction,dtype=float,copy=True)
        if center.ndim!=1 or not center.size or n.shape!=center.shape or not np.isfinite(center).all() or not np.isfinite(n).all():raise ValueError('finite flat Gaussian coordinates required')
        if abs(np.linalg.norm(n)-1)>1e-12:raise ValueError('unit direction required')
        if not np.isscalar(self.width) or not np.isscalar(self.weight) or not np.isfinite(self.width) or not np.isfinite(self.weight) or self.width<=0 or self.weight<=0:raise ValueError('positive finite width/weight required')
        center.setflags(write=False);n.setflags(write=False);object.__setattr__(self,'center',center);object.__setattr__(self,'direction',n)

    def evaluate(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=self.center.shape or not np.isfinite(q).all():raise ValueError('compatible finite evaluation coordinates required')
        z=float((q-self.center)@self.direction);energy=self.weight*math.exp(-.5*(z/self.width)**2)
        return energy,energy*z/self.width**2*self.direction


@dataclass(frozen=True)
class PreparedNativeHeight:
    terms: tuple
    changed_history: tuple
    initial_new_weight: float
    final_weight: float
    angle_degrees: float
    stop_reason: str
    update_trace: tuple
    bias_energy: float
    total_force_at_preparation: np.ndarray
    curvature_input: float
    curvature_scope: str
