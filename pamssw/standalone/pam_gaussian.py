"""PAM curvature Gaussian core, without trust or global feedback.

Source: SurfaceWalker._scaled_step_scale/_bias_weight in pamssw/walker.py.
Widths are Angstrom, weights eV, curvatures eV/Angstrom**2. Bounds and targets
are inherited experimental choices, not universal physical constants. The
current full PAM default uses per_atom_rms; height_width explicitly selects
its curvature_adaptive alternative. No native binary is called.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class PAMCurvatureGaussian:
    target_uphill_energy: float = .6
    target_negative_curvature: float = .05
    min_width: float = .15
    max_width: float = 1.5
    min_weight: float = 0.
    max_weight: float = 10.
    curvature_floor: float = 1e-4
    mode: str = 'height_width'

    def __post_init__(self):
        if self.mode not in ('height_only','height_width'): raise ValueError('mode must be height_only or height_width')
        for n in ('target_uphill_energy','target_negative_curvature','min_width','max_width','max_weight','curvature_floor'):
            if not np.isfinite(getattr(self,n)) or getattr(self,n)<=0: raise ValueError(f'{n} must be positive and finite')
        if self.min_width>self.max_width or self.min_weight<0 or self.min_weight>self.max_weight: raise ValueError('invalid PAM Gaussian bounds')
        if not np.isfinite(self.min_weight) or self.min_weight < 0: raise ValueError('min_weight must be finite and nonnegative')

    def parameters(self): return {"target_uphill_energy":self.target_uphill_energy,"target_negative_curvature":self.target_negative_curvature,"min_width":self.min_width,"max_width":self.max_width,"min_weight":self.min_weight,"max_weight":self.max_weight,"curvature_floor":self.curvature_floor,"mode":self.mode}

    def choose(self, *, mode, anchor, center, terms, base_width, rotation_bias=0.):
        direction=np.asarray(mode.direction,float); a=np.asarray(anchor,float).reshape(-1); d=direction.reshape(-1)
        if a.shape != d.shape or not np.isfinite(a).all() or not np.isfinite(d).all() or not np.isfinite(mode.curvature) or not np.isfinite(rotation_bias): raise ValueError('finite compatible mode and anchor required')
        if (direction.shape != np.asarray(center).shape or not np.isfinite(center).all()
            or np.linalg.norm(a)==0 or np.linalg.norm(d)==0
            or not np.isfinite(base_width) or base_width<=0 or rotation_bias<0):
            raise ValueError('finite nonzero directions, center and positive width required')
        a=a/np.linalg.norm(a); d=d/np.linalg.norm(d)
        k_true=float(mode.curvature + rotation_bias * float(np.dot(d,a))**2)
        # Existing ProjectedGaussian terms contribute their analytic Hessian
        # along the current direction at the current center.
        k_inner=k_true
        for term in terms:
            delta=(np.asarray(center)-term.center).reshape(-1); n=term.direction.reshape(-1); p=float(np.dot(delta,n)); z=p/term.sigma
            k_inner += float(term.weight*math.exp(-.5*z*z)*(p*p/term.sigma**4-1./term.sigma**2)*float(np.dot(n,d))**2)
        if not np.isfinite(k_true) or not np.isfinite(k_inner):
            raise ValueError('finite physical and inner curvature required')
        raw_width=base_width if self.mode=='height_only' else math.sqrt(2.*self.target_uphill_energy/max(abs(k_true),self.curvature_floor))
        width=float(raw_width if self.mode=='height_only' else np.clip(raw_width,self.min_width,self.max_width)); raw_weight=width*width*max(k_inner+self.target_negative_curvature,0.)
        weight=float(np.clip(raw_weight,self.min_weight,self.max_weight))
        return {"width":width,"weight":weight,"k_true":k_true,"k_inner":k_inner,"raw_width":raw_width,"raw_weight":raw_weight,"width_clamped":width!=raw_width,"weight_clamped":weight!=raw_weight,"parameters":self.parameters()}
