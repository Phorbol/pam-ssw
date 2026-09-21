"""Remaining source-defined TYPE0 packing geometries; no physical PES.

Caller chooses family, radii, density/box/template and explicit attempt cap.
Source empirical rules are retained and recorded, never marketed as optimal.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from ase import Atoms
from .surface_ga import _radii,_swap_order


@dataclass(frozen=True)
class PackingResult:
    atoms: object
    status: str
    source_atom_indices: tuple
    details: dict
    physical_requests: int = 0


def source_cubic_boxes(n):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:raise ValueError('positive integer atom count required')
    if n<=2:return ((1,1,n),)
    boxes=[]
    for i in range(1,n-1):
        for j in range(i,n):
            box=tuple(sorted((i,j,int(np.ceil(n/(i*j))))))
            if box not in boxes and not (box[0]==1 and box[1]*box[2]>np.ceil(np.sqrt(n))**2):boxes.append(box)
    return tuple(boxes)


def source_cage_occupancy(n):
    """Recovered empirical polynomial, truncated toward zero to 3 decimals."""
    if n<1:raise ValueError('positive atom count required')
    if n>100:return .446
    b=np.log10(n);x=-.283*b**5+1.966*b**4-5.232*b**3+6.5352*b**2-3.659*b+1.0799+.05
    return float(np.trunc(x*1000)/1000)


def _sphere(r,rng):
    elevation=(rng.random()-.5)*2*np.pi;azimuth=rng.random()*2*np.pi
    return r*np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])


def pack_type0(numbers,family,rng,*,atomic_radii,max_attempts,occupancy=None,box=None,
               template=None,template_atomic_radii=None):
    """One exact family geometry with bounded failure and partial work retained.

    families: unlimited, simple_cubic, irregular_ball, irregular_ball_ori,
    irregular_cage, custom_template. Density-based families require explicit
    occupancy; source_cage_occupancy gives original empirical default if desired.
    A failed result contains a partial structure, never a complete fake candidate.
    """
    started=perf_counter()
    raw=np.asarray(numbers)
    if raw.ndim!=1 or not len(raw) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<=0):raise ValueError('explicit positive integer atomic-number sequence required')
    numbers=raw.astype(int)
    if isinstance(max_attempts,bool) or not isinstance(max_attempts,(int,np.integer)) or max_attempts<1:raise ValueError('positive per-insertion attempt cap required')
    supported=('unlimited','simple_cubic','irregular_ball','irregular_ball_ori','irregular_cage','custom_template')
    if family not in supported:raise ValueError('unknown source packing family')
    radii=_radii(numbers,atomic_radii);radius=float(np.mean(radii))
    order=_swap_order(len(numbers),rng);order=order[_swap_order(len(numbers),rng)]
    zs=numbers[order];rs=radii[order];details=dict(family=family,mean_radius_A=radius,attempts=[],density_updates=[],corrections=[])
    def finish(x,status):
        details['total_attempts']=sum(details['attempts'])
        details['elapsed_seconds']=perf_counter()-started
        if status!='completed':details['failed_atom_index']=len(x)
        return PackingResult(Atoms(numbers=zs[:len(x)],positions=np.array(x).reshape(-1,3)),status,tuple(int(i) for i in order[:len(x)]),details)
    if family=='custom_template':
        if template is None or len(template)!=len(numbers) or not np.isfinite(template.positions).all():raise ValueError('matching finite caller template required')
        if template_atomic_radii is None:raise ValueError('explicit template-species radius table required')
        old=float(np.mean(_radii(template.numbers,template_atomic_radii)));scale=radius/old
        details.update(scale=scale,template_numbers=tuple(int(z) for z in template.numbers),template_source='caller-provided Atoms; no synthesized template corpus')
        return finish(template.positions*scale,'completed')
    if family=='irregular_ball':
        x=[np.zeros(3)];shell=2*radius
        while len(x)<len(zs):
            count=min(len(zs)-len(x),max(6,int(np.floor(4*shell**2/(1.3*radius**2)+.5))))
            if count==1:
                y=np.array([0.]);details['corrections'].append('single-point Fibonacci shell uses equator instead of division by zero')
            else:y=1-2*np.arange(count)/(count-1)
            angle=np.pi*(3-np.sqrt(5))*np.arange(count);v=np.sqrt(np.maximum(0,1-y*y))
            points=shell*np.column_stack((np.cos(angle)*v,y,np.sin(angle)*v))
            a,b,c=np.array([rng.random(),rng.random(),rng.random()])*[2*np.pi,np.pi,2*np.pi]
            ca,sa,cb,sb,cc,sc=np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(c),np.sin(c)
            rotation=np.array([[ca*cb*cc-sa*sc,-ca*cb*sc-sa*cc,ca*sb],[sa*cb*cc+ca*sc,-sa*cb*sc+ca*cc,sa*sb],[-sb*cc,sb*sc,cb]])
            x.extend(points@rotation.T);shell+=2*radius
        x=np.array(x)
        for row in x:row+=(np.array([rng.random() for _ in range(3)])*2-1)*.1*(2*radius)
        details.update(shell_spacing_A=2*radius,packing_factor=1.3,jitter_fraction_of_diameter=.1)
        return finish(x,'completed')
    if family=='simple_cubic':
        if box is None or len(box)!=3 or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or i<1 for i in box) or np.prod(box)<len(numbers):raise ValueError('explicit integer box with sufficient sites required')
        bounds=np.array(box);sites=[np.zeros(3,dtype=int)];directions=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
        details.update(box=tuple(box),corrections=['symmetric negative-y boundary','integer lattice excludes floating point occupancy ambiguity'])
        while len(sites)<len(zs):
            occupied={tuple(i) for i in sites};openings=[]
            for site in sites:
                trial=site+directions
                openings.append(np.array([np.all(p>=0) and np.all(p<bounds) and tuple(p) not in occupied for p in trial]))
            counts=np.array([m.sum() for m in openings]);positive=counts[counts>0]
            if not len(positive):return finish(radius+2*radius*np.array(sites),'no_open_site')
            choices=np.flatnonzero(counts==positive.min());index=int(choices[int(rng.random()*len(choices))])
            for attempt in range(1,max_attempts+1):
                direction=int(rng.random()*6)
                if openings[index][direction]:sites.append(sites[index]+directions[direction]);details['attempts'].append(attempt);break
            else:
                details['attempts'].append(max_attempts)
                return finish(radius+2*radius*np.array(sites),'direction_budget_exhausted')
        return finish(radius+2*radius*np.array(sites),'completed')
    if family in ('irregular_ball_ori','irregular_cage'):
        if occupancy is None or not np.isfinite(occupancy) or occupancy<=0:raise ValueError('explicit positive empirical occupancy required')
        density=float(occupancy);details['initial_occupancy']=density
        def bounds():
            if family=='irregular_ball_ori':return 0.,radius*(len(zs)/density)**(1/3)
            width=2.2*radius;volume=len(zs)*(4*np.pi/3)*radius**3/density
            roots=np.roots([3*width,-3*width**2,width**3-3*volume/(4*np.pi)])
            real=[float(r.real) for r in roots if abs(r.imag)<1e-12 and r.real>0]
            if not real:raise ValueError('source shell volume has no positive real outer radius')
            outer=max(real);inner=outer-width
            if inner<0:inner=0.;outer=width
            return inner,outer
        inner,outer=bounds();x=[np.array([0,0,(inner+outer)/2]) if family=='irregular_cage' else np.zeros(3)]
    else:x=[np.zeros(3)];density=None
    while len(x)<len(zs):
        for attempt in range(1,max_attempts+1):
            if density is not None and attempt>1 and (attempt-1)%(1000*len(x))==0:
                density-=.01
                if density<=0:return finish(x,'nonpositive_source_occupancy')
                inner,outer=bounds();details['density_updates'].append((len(x),attempt,density))
            index=int(rng.random()*len(x))
            distance=rs[index]+rs[len(x)] if family=='unlimited' else 2*radius*(1+.1*(rng.random()-.5))
            point=x[index]+_sphere(distance,rng)
            distances=np.linalg.norm(np.array(x)-point,axis=1)
            if family=='unlimited':
                minimum=rs[:len(x)]+rs[len(x)]
                valid=np.all(distances>=minimum-32*np.finfo(float).eps*minimum)
            else:
                valid=np.all(distances>=1.75*radius) and np.linalg.norm(point)<=outer-radius
                if family=='irregular_cage':valid=valid and np.linalg.norm(point)>=inner+radius
            if valid:x.append(point);details['attempts'].append(attempt);break
        else:
            details['attempts'].append(max_attempts)
            return finish(x,'insertion_budget_exhausted')
    if family=='unlimited':details['corrections'].append('32 machine-epsilon tolerance at exactly tangent source boundary')
    else:details.update(final_occupancy=density,outer_radius_A=outer,inner_radius_A=inner)
    return finish(x,'completed')
