"""Source-backed TYPE4 supported-cluster operators, with fixed support.

Distinct from TYPE2 molecular crystals: only adsorbates are cut/reloaded;
cell and substrate positions are preserved. Explicit 2D PBC and topology.
Implements CrossLoaded and all four MutateLoaded geometric families.
Cubic rebuilding includes a separately counted synthetic LJ/wall optimizer.
No ASE calculator calls. See docs/research/type4-surface-ga.md for corrections.
"""
from dataclasses import dataclass
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from .atomic_ga import build_atomic_pool, _positive_int
from .ga_operators import SamplingExhausted, rotate_coordinates, _docking_directions, _monomer_radius


@dataclass(frozen=True)
class SurfaceCandidate:
    atoms: Atoms
    substrate_indices: tuple
    adsorbate_indices: tuple
    atom_parent_indices: tuple
    source_atom_indices: tuple
    operation: str
    details: dict


def surface_topology(atoms, substrate_indices, adsorbate_indices):
    """Validate explicit exhaustive partition; only fully frozen support mode.

    The caller supplies an intact unwrapped adsorbate; no bond inference.
    Existing FixAtoms must equal the substrate set, not silently overwritten.
    """
    groups=[]
    for group in (substrate_indices,adsorbate_indices):
        values=tuple(group)
        if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in values):
            raise ValueError('topology indices must be integers')
        groups.append(tuple(int(i) for i in values))
    support,ads=groups
    if not support or len(ads)<2 or sorted(support+ads)!=list(range(len(atoms))):
        raise ValueError('support and adsorbates must be nonempty disjoint exhaustive groups; >=2 adsorbates')
    if not np.array_equal(atoms.pbc,[True,True,False]):
        raise ValueError('surface operators require explicit PBC=(True,True,False)')
    if np.linalg.det(atoms.cell.array)<=0 or not np.isfinite(atoms.cell.array).all() or not np.isfinite(atoms.positions).all():
        raise ValueError('finite positions and right-handed nonsingular fixed cell required')
    if 'masses' in atoms.arrays:
        raise ValueError('custom isotope/site masses require an additional exchange contract')
    fixed=set()
    for c in atoms.constraints:
        if not isinstance(c,FixAtoms):raise ValueError('only matching FixAtoms supported')
        fixed.update(int(i) for i in c.get_indices())
    if atoms.constraints and fixed!=set(support):
        raise ValueError('FixAtoms must exactly match explicitly frozen support')
    return support,ads


def _frame(atoms):
    a,b=atoms.cell.array[:2];e=a/np.linalg.norm(a)
    normal=np.cross(a,b);normal/=np.linalg.norm(normal)
    return np.array([e,np.cross(normal,e),normal])


def _rotation_between(a,b):
    a=np.asarray(a,dtype=float);a/=np.linalg.norm(a)
    b=np.asarray(b,dtype=float);b/=np.linalg.norm(b)
    v=np.cross(a,b);c=np.dot(a,b)
    if c<-1+1e-13:
        axis=np.eye(3)[np.argmin(abs(a))];axis-=axis.dot(a)*a;axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k/(1+c)


def suit_orientation(positions, accuracy):
    """SuitLoad.diff/getFit, row positions in a local z-normal frame."""
    accuracy=_positive_int(accuracy,'accuracy')
    x=np.array(positions,dtype=float);x-=x.mean(axis=0)
    trials=[]
    for axis in ([0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
        for direction in _docking_directions(accuracy):
            trials.append(x@_rotation_between(axis,direction).T)
    scores=[float(np.sum(t[:,2])-len(t)*np.min(t[:,2])) for t in trials]
    index=int(np.argmin(scores))
    return trials[index],dict(orientation_index=index,orientation_count=len(trials),height_sum_A=scores[index])


def _reload_positions(atoms,support,cluster,*,min_distance,site_fractional,accuracy):
    if not np.isfinite(min_distance) or min_distance<=0:raise ValueError('positive docking distance required')
    site=np.asarray(site_fractional,dtype=float)
    if site.shape!=(2,) or not np.isfinite(site).all():raise ValueError('explicit finite 2D fractional site required')
    frame=_frame(atoms);normal=frame[2]
    x=np.asarray(cluster,dtype=float);x=(x-x.mean(axis=0))@frame.T
    info={}
    if accuracy is not None:x,info=suit_orientation(x,accuracy)
    x=x@frame
    site_xyz=site@atoms.cell.array[:2]
    x+=site_xyz
    s=atoms.positions[list(support)]
    # Nearest in-plane image suffices for first contact along a fixed normal:
    # other images have >= lateral distance and hence <= top contact height.
    d=s[:,None,:]-x[None,:,:]
    height=d@normal;lateral=d-height[:,:,None]*normal
    lateral,_=find_mic(lateral.reshape(-1,3),atoms.cell.array,pbc=[True,True,False])
    lateral=np.linalg.norm(lateral,axis=1).reshape(height.shape)
    step=.125;cutoff=min_distance+step;valid=lateral<cutoff
    if not valid.any():raise SamplingExhausted('normal docking ray has no substrate contact')
    contact=float(np.max(height[valid]+np.sqrt(cutoff**2-lateral[valid]**2)))
    native_start=float(np.max(s@normal)+2*_monomer_radius(x@frame.T))
    start=max(native_start,contact+step)
    count=max(1,int(np.floor((start-contact)/step))+1)
    if count>10000:raise SamplingExhausted('surface descent exceeds source 10000-step bound')
    displacement=start-count*step
    x+=displacement*normal
    info.update(site_fractional=tuple(site),normal=tuple(normal),descent_steps=count,
                min_docking_distance_A=min_distance,descent_step_A=step,
                start_raised_for_clearance=bool(start>native_start),
                contact_geometry='nearest lateral periodic image, no vacuum image')
    return x,info


def _candidate(parent,support,ads,cluster,positions,parents,sources,operation,details):
    a=Atoms(numbers=parent.numbers.copy(),positions=parent.positions.copy(),cell=parent.cell.copy(),pbc=parent.pbc.copy())
    a.numbers[list(ads)]=cluster.numbers;a.positions[list(ads)]=positions
    a.set_constraint(FixAtoms(indices=support))
    return SurfaceCandidate(a,support,ads,tuple(parents),tuple(sources),operation,details)


def reload_surface(atoms, substrate_indices, adsorbate_indices, rng, *,
                   site_fractional, min_distance=2., accuracy=None, rotate=True,parent_index=0):
    """LoadedHandle rotating reload or SuitLoad, with support fixed exactly.

    Native values: min_distance=2 A, accuracy=30 for crossover / 10 rebuild.
    site_fractional is explicit; no inferred active site or random site rule.
    """
    support,ads=surface_topology(atoms,substrate_indices,adsorbate_indices)
    cluster=Atoms(numbers=atoms.numbers[list(ads)],positions=atoms.positions[list(ads)])
    if rotate:
        frame=_frame(atoms);local=(cluster.positions-cluster.positions.mean(axis=0))@frame.T
        cluster.positions=rotate_coordinates(local,rng)@frame
    positions,details=_reload_positions(atoms,support,cluster.positions,min_distance=min_distance,site_fractional=site_fractional,accuracy=accuracy)
    return _candidate(atoms,support,ads,cluster,positions,[parent_index]*len(atoms),range(len(atoms)),
                      'surface_reload',details)


def disturb_surface(atoms, substrate_indices, adsorbate_indices, rng, *, parent_index=0):
    """Native floor(.2*Nads) width-2 A disturbances, then 10*Nads swaps."""
    support,ads=surface_topology(atoms,substrate_indices,adsorbate_indices)
    cluster=Atoms(numbers=atoms.numbers[list(ads)],positions=atoms.positions[list(ads)])
    frame=_frame(atoms);moved=[];sources=np.array(ads)
    for _ in range(int(.2*len(ads))):
        index=int(rng.random()*len(ads));cluster.positions[index]+=2*(np.array([rng.random() for _ in range(3)])-.5)@frame;moved.append(index)
    for _ in range(10*len(ads)):
        i,j=(int(rng.random()*len(ads)) for _ in range(2))
        cluster.numbers[i],cluster.numbers[j]=int(cluster.numbers[j]),int(cluster.numbers[i])
        sources[i],sources[j]=sources[j],sources[i]
    lineage=list(range(len(atoms)))
    for index,source in zip(ads,sources):lineage[index]=int(source)
    return _candidate(atoms,support,ads,cluster,cluster.positions,[parent_index]*len(atoms),lineage,
                      'surface_disturbance_exchange',dict(moved_adsorbate_indices=tuple(moved),width_A=2.,swaps=10*len(ads)))


def surface_collision_free(atoms, bond_limits):
    """Full two-dimensional ASE periodic pair filter, including self images."""
    if not np.array_equal(atoms.pbc,[True,True,False]):raise ValueError('surface PBC must be explicit')
    symbols=atoms.get_chemical_symbols();cutoffs={}
    for a in set(symbols):
        for b in set(symbols):
            key=tuple(sorted((a,b)))
            if key not in bond_limits:raise ValueError(f'missing explicit bond limit {key}')
            value=float(bond_limits[key])
            if not np.isfinite(value) or value<=0:raise ValueError('positive bond limits required')
            cutoffs[key]=value
    i,j,d=neighbor_list('ijd',atoms,max(cutoffs.values()),self_interaction=False)
    return all(distance>=cutoffs[tuple(sorted((symbols[a],symbols[b])))] for a,b,distance in zip(i,j,d))


def cross_surface(parents,energies,substrate_indices,adsorbate_indices,rng,*,n,
                  site_fractional,max_cut_attempts,max_pair_attempts,cuts_per_parent_slot=None):
    """CrossLoaded geometry with fixed common support and source-held sperm.

    Return unfiltered candidates; caller must use explicit collision limits.
    Native bad-composition budget fallthrough is corrected to failure.
    """
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<0:raise ValueError('n must be nonnegative integer')
    if not parents:raise ValueError('parents required')
    support,ads=surface_topology(parents[0],substrate_indices,adsorbate_indices)
    first=parents[0];clusters=[]
    for a in parents:
        surface_topology(a,support,ads)
        if not np.array_equal(a.cell.array,first.cell.array) or not np.array_equal(a.positions[list(support)],first.positions[list(support)]) or not np.array_equal(a.numbers[list(support)],first.numbers[list(support)]):
            raise ValueError('all parents must share exactly the same frozen substrate and cell')
        clusters.append(Atoms(numbers=a.numbers[list(ads)],positions=a.positions[list(ads)]@_frame(a).T))
    pool=build_atomic_pool(clusters,energies,rng,max_cut_attempts=max_cut_attempts,cuts_per_parent_slot=cuts_per_parent_slot)
    _positive_int(max_pair_attempts,'max_pair_attempts');result=[]
    for k in range(n):
        i=int(rng.random()*len(pool.sons));son=pool.sons[i]
        for attempt in range(1,max_pair_attempts+1):
            j=int(rng.random()*len(pool.daughters));daughter=pool.daughters[j]
            cluster=son.atoms+daughter.atoms
            if tuple(sorted(int(z) for z in cluster.numbers))==pool.composition:break
        else:raise SamplingExhausted('CrossLoaded composition matching exhausted')
        parent_index=int(rng.random()*len(parents));parent=parents[parent_index]
        cluster.positions=cluster.positions@_frame(parent)
        accuracy=30 if k<(n//3)*2 else None
        positions,details=_reload_positions(parent,support,cluster.positions,min_distance=2.,site_fractional=site_fractional,accuracy=accuracy)
        lineage=[parent_index]*len(parent);sources=list(range(len(parent)))
        p=(son.parent_index,)*len(son.atoms)+(daughter.parent_index,)*len(daughter.atoms)
        source=son.source_atom_indices+daughter.source_atom_indices
        for index,pi,si in zip(ads,p,source):lineage[index]=pi;sources[index]=ads[si]
        details.update(son_pool_index=i,daughter_pool_index=j,pair_attempts=attempt,cuts_per_parent_slot=pool.cuts_per_parent_slot,substrate_parent_index=parent_index,orientation_accuracy=accuracy)
        result.append(_candidate(parent,support,ads,cluster,positions,lineage,sources,'surface_crossover',details))
    return tuple(result)


def _radii(numbers, atomic_radii):
    values=np.array([atomic_radii[int(z)] for z in numbers],dtype=float)
    if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('explicit positive atomic radii in Angstrom required')
    return values


def _swap_order(n,rng):
    order=np.arange(n)
    for _ in range(10*n):
        i,j=(int(rng.random()*n) for _ in range(2));order[i],order[j]=order[j],order[i]
    return order


def triple_tangency_cluster(numbers,rng,*,atomic_radii,max_face_attempts):
    """TripleTangencyBallsPacking with bounded face sampling and stable normals.

    Uses composition-average radius, not species pair radii. Coordinates are
    newly generated, so returned indices express species provenance only.
    """
    from itertools import combinations
    numbers=np.asarray(numbers,dtype=int)
    if not len(numbers):raise ValueError('nonempty composition required')
    radius=float(np.mean(_radii(numbers,atomic_radii)))
    _positive_int(max_face_attempts,'max_face_attempts')
    # Constructor and createOneStructure each perform a 10N swap pass.
    order=_swap_order(len(numbers),rng);order=order[_swap_order(len(numbers),rng)]
    xyz=[np.zeros(3)];draws=0
    if len(numbers)>1:
        elevation=(rng.random()-.5)*2*np.pi;azimuth=rng.random()*2*np.pi
        xyz.append(2*radius*np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)]))
    if len(numbers)>2:
        normal=xyz[0]-xyz[1];axis=np.cross(normal,[1.,0,0])
        if np.linalg.norm(axis)==0:axis=np.cross(normal,[0.,1,0])
        other=np.cross(normal,axis);axis/=np.linalg.norm(axis);other/=np.linalg.norm(other)
        phi=rng.random()*2*np.pi
        xyz.append((xyz[0]+xyz[1])/2+radius*np.sqrt(3)*(np.cos(phi)*axis+np.sin(phi)*other))
    while len(xyz)<len(numbers):
        x=np.array(xyz);distance=np.linalg.norm(x[:,None]-x[None,:],axis=2);faces=[]
        for a,b,c in combinations(range(len(xyz)),3):
            if not all(1.9*radius<=distance[i,j]<=2.1*radius for i,j in ((a,b),(a,c),(b,c))):continue
            normal=np.cross(x[a]-x[b],x[a]-x[c]);norm=np.linalg.norm(normal)
            if norm==0:continue
            normal/=norm
            # Stable coordinate-free formula replaces native divisions by zero.
            # Keep native positive quadratic-root orientation on its regular domain.
            component=0 if normal[0]!=0 and normal[1]!=0 and normal[2]!=0 else (2 if normal[0]==0 or normal[1]==0 else 0)
            if normal[component]==0:component=int(np.flatnonzero(normal)[0])
            if normal[component]<0:normal=-normal
            center=(x[a]+x[b]+x[c])/3;offset=2*radius*np.sqrt(2/3)*normal
            points=(center+offset,center-offset)
            valid=[np.all(np.linalg.norm(x-p,axis=1)>=1.9*radius) for p in points]
            faces.append((points,valid))
        if not faces or not any(any(v) for _,v in faces):raise SamplingExhausted('no unoccupied tangent face')
        for _ in range(max_face_attempts):
            draws+=1;points,valid=faces[int(rng.random()*len(faces))]
            if valid[0]:xyz.append(points[0]);break
            if valid[1]:xyz.append(points[1]);break
        else:raise SamplingExhausted('tangent face draw budget exhausted')
    return Atoms(numbers=numbers[order],positions=xyz),tuple(int(i) for i in order),dict(average_radius_A=radius,face_draws=draws)


def cubic_auxiliary_energy_gradient(positions,numbers,box,atomic_radii):
    """Source LJForDiff energy, corrected upper-wall derivative; no physical PES.

    Pair sigma=r_i+r_j, epsilon=1 in source auxiliary units. Wall energy
    10/(1+exp(10*x)) + 10/(1+exp(10*(L-x))) per coordinate.
    """
    from scipy.special import expit
    x=np.asarray(positions,dtype=float);radius=_radii(numbers,atomic_radii)
    i,j=np.triu_indices(len(x),1);d=x[i]-x[j];r2=np.einsum('ij,ij->i',d,d)
    if np.any(r2<=0):raise ValueError('auxiliary LJ overlapping atoms')
    t=((radius[i]+radius[j])**2/r2)**3
    energy=float(np.sum(4*(t*t-t)));gradient=np.zeros_like(x)
    pair=(24*(t-2*t*t)/r2)[:,None]*d
    np.add.at(gradient,i,pair);np.add.at(gradient,j,-pair)
    left=expit(-10*x);right=expit(-10*(np.asarray(box)-x))
    energy+=float(10*np.sum(left+right))
    gradient+=100*(-left*(1-left)+right*(1-right))
    return energy,gradient


def cubic_cluster(numbers,rng,*,atomic_radii,space,max_insertion_attempts,
                  auxiliary_evaluations=100):
    """10 source cubic starts, independent bounded L-BFGS auxiliary refinement.

    This incurs and reports synthetic LJ/wall evaluations, never ASE PES.
    Default 100 is source reverse-communication call budget, not iterations.
    """
    from scipy.optimize import minimize
    numbers=np.asarray(numbers,dtype=int);radii=_radii(numbers,atomic_radii)
    space=np.asarray(space,dtype=float)
    if space.shape!=(3,) or not np.isfinite(space).all() or np.any(space<=0):raise ValueError('positive three-component cubic aspect ratio required')
    _positive_int(max_insertion_attempts,'max_insertion_attempts');_positive_int(auxiliary_evaluations,'auxiliary_evaluations')
    box=(np.sum(8*radii**3)/np.prod(space))**(1/3)*space
    min_r=float(1.5*np.mean(radii));starts=[];ledger=[]
    for index in range(10):
        order=_swap_order(len(numbers),rng);x=[np.array([.1,.1,.1])]
        for _ in range(1,len(numbers)):
            for attempt in range(max_insertion_attempts):
                point=np.array([rng.random() for _ in range(3)])*box
                if np.all(np.linalg.norm(np.array(x)-point,axis=1)>=min_r):x.append(point);break
            else:raise SamplingExhausted('cubic placement budget exhausted')
        starts.append((order,np.array(x)))
    results=[]
    class Budget(Exception):pass
    for index,(order,x) in enumerate(starts):
        state=dict(calls=0,last_x=None,last_energy=None,last_gradient=None)
        def evaluate(flat):
            if state['calls']>=auxiliary_evaluations:raise Budget()
            state['calls']+=1  # attempted auxiliary evaluations include failures
            energy,gradient=cubic_auxiliary_energy_gradient(flat.reshape(-1,3),numbers[order],box,atomic_radii)
            state.update(last_x=flat.copy(),last_energy=energy,last_gradient=gradient.copy())
            return energy,gradient.ravel()
        try:
            result=minimize(evaluate,x.ravel(),jac=True,method='L-BFGS-B',options=dict(maxcor=5,gtol=.01,ftol=0.,maxiter=auxiliary_evaluations,maxls=20))
            status='converged' if result.success else 'optimizer_stopped'
            # Return last evaluated geometry and its paired energy, never an unevaluated iterate.
        except Budget:status='auxiliary_budget_exhausted'
        except (ValueError,RuntimeError,FloatingPointError) as error:
            failed=dict(start=index,auxiliary_evaluations=state['calls'],status='auxiliary_evaluation_failed',energy_aux=state['last_energy'],error=str(error))
            error.auxiliary_runs=tuple(ledger+[failed])
            error.auxiliary_evaluations=sum(r['auxiliary_evaluations'] for r in error.auxiliary_runs)
            raise
        ledger.append(dict(start=index,auxiliary_evaluations=state['calls'],status=status,energy_aux=state['last_energy'],gradient_norm=float(np.linalg.norm(state['last_gradient']))))
        results.append((Atoms(numbers=numbers[order],positions=state['last_x'].reshape(-1,3)),order))
    best=min(range(10),key=lambda k:ledger[k]['energy_aux'])
    atoms,order=results[best]
    return atoms,tuple(int(i) for i in order),dict(space=tuple(space),box_A=tuple(box),selected_start=best,
        auxiliary_evaluations=sum(r['auxiliary_evaluations'] for r in ledger),auxiliary_runs=tuple(ledger),
        corrections=('consistent positive upper-wall derivative','independent scipy L-BFGS-B solver, not Java iteration parity'))


def rebuild_surface(atoms,substrate_indices,adsorbate_indices,rng,*,site_fractional,
                    atomic_radii,max_face_attempts,max_insertion_attempts,
                    auxiliary_evaluations=100,parent_index=0):
    """Native four families: tangent packing plus cubic (1,1,1),(2,2,1),(3,3,1)."""
    support,ads=surface_topology(atoms,substrate_indices,adsorbate_indices);numbers=atoms.numbers[list(ads)]
    families=[('tangent',triple_tangency_cluster(numbers,rng,atomic_radii=atomic_radii,max_face_attempts=max_face_attempts))]
    for space in ((1.,1.,1.),(2.,2.,1.),(3.,3.,1.)):
        try:
            family=cubic_cluster(numbers,rng,atomic_radii=atomic_radii,space=space,max_insertion_attempts=max_insertion_attempts,auxiliary_evaluations=auxiliary_evaluations)
        except (SamplingExhausted,ValueError,RuntimeError,FloatingPointError) as error:
            error.completed_family_details=tuple(item[2] for _,item in families)
            error.auxiliary_evaluations=getattr(error,'auxiliary_evaluations',0)+sum(item[2].get('auxiliary_evaluations',0) for _,item in families)
            raise
        families.append(('cubic',family))
    result=[]
    for family,(cluster,order,info) in families:
        cluster.positions=cluster.positions@_frame(atoms)
        try:
            positions,details=_reload_positions(atoms,support,cluster.positions,min_distance=2.,site_fractional=site_fractional,accuracy=10)
        except (SamplingExhausted,ValueError,RuntimeError,FloatingPointError) as error:
            error.completed_family_details=tuple(item[2] for _,item in families)
            error.auxiliary_evaluations=getattr(error,'auxiliary_evaluations',0)+sum(item[2].get('auxiliary_evaluations',0) for _,item in families)
            error.partial_candidates=tuple(result)
            raise
        sources=list(range(len(atoms)))
        for index,source in zip(ads,order):sources[index]=ads[source]
        details.update(info);details['family']=family;details['lineage_kind']='species origin; geometry reconstructed'
        result.append(_candidate(atoms,support,ads,cluster,positions,[parent_index]*len(atoms),sources,'surface_rebuild',details))
    return tuple(result)


def reconstruct_surface(atoms,substrate_indices,adsorbate_indices,rng,*,
                        atomic_radii,max_face_attempts,parent_index=0):
    """Native >3 A upper-adsorbate split; two outputs or no upper fragment.

    Upper atoms are rotated pi around normal, or rebuilt by tangent packing.
    Contact is to the lower adsorbate fragment, not a hidden new whole-cluster
    reload. No lower fragment is an explicit failure, not 10000 downhill steps.
    """
    support,ads=surface_topology(atoms,substrate_indices,adsorbate_indices)
    frame=_frame(atoms);normal=frame[2];top=float(np.max(atoms.positions[list(support)]@normal))
    upper=tuple(i for i in ads if atoms.positions[i]@normal-top>3.)
    lower=tuple(i for i in ads if i not in upper)
    if not upper:return ()
    if not lower:raise SamplingExhausted('surface reconstruction has no lower adsorbate contact fragment')
    original=Atoms(numbers=atoms.numbers[list(upper)],positions=atoms.positions[list(upper)])
    local=(original.positions-original.positions.mean(axis=0))@frame.T;local[:,:2]*=-1
    rotated=Atoms(numbers=original.numbers,positions=local@frame)
    generated,order,info=triple_tangency_cluster(original.numbers,rng,atomic_radii=atomic_radii,max_face_attempts=max_face_attempts)
    generated.positions=generated.positions@frame
    # Source centers upper part at support center; explicit independent center
    # is mean support projection, avoiding mutation of the frozen support itself.
    center=atoms.positions[list(support)].mean(axis=0);site=(center@np.linalg.inv(atoms.cell.array))[:2]
    result=[]
    for cluster,order,info in ((rotated,tuple(range(len(upper))),{}),(generated,order,info)):
        # Source's cutoff=2.5, step=.125 -> guaranteed minimum=2.375 A.
        positions,details=_reload_positions(atoms,lower,cluster.positions,min_distance=2.375,site_fractional=site,accuracy=None)
        a=atoms.copy();a.set_constraint();a.calc=None
        a.numbers[list(upper)]=cluster.numbers;a.positions[list(upper)]=positions
        sources=list(range(len(atoms)))
        for index,source in zip(upper,order):sources[index]=upper[source]
        details.update(info);details.update(upper_indices=upper,lower_indices=lower,height_split_A=3.,
            correction='preserve lower fragment and original index placement; bounded periodic first contact',
            lineage_kind='species origin for rebuilt upper fragment')
        result.append(_candidate(atoms,support,ads,a[list(ads)],a.positions[list(ads)],[parent_index]*len(atoms),sources,'surface_reconstruction',details))
    return tuple(result)


@dataclass(frozen=True)
class SurfaceProposal:
    candidates: tuple
    status: str
    batches: tuple
    auxiliary_evaluations: int


def propose_type4(parents,energies,substrate_indices,adsorbate_indices,rng,*,
                  min_ga,bond_limits,atomic_radii,site_fractional,max_batches,
                  max_cut_attempts,max_pair_attempts,max_face_attempts,
                  max_insertion_attempts,auxiliary_evaluations=100,cuts_per_parent_slot=None):
    """Complete TYPE4 native family counts, bounded attempts and explicit costs.

    Each batch: G//4 cross, G//4 reload, G//4 disturbance, G//8
    reconstruction calls (0 or 2 outputs each), G//4 rebuild calls (4 each).
    Keep whole passing batches; never truncate to G. Legacy native standardizing
    mutations/filter defects are corrected as documented. No physical E/F.
    """
    for value,name in ((min_ga,'min_ga'),(max_batches,'max_batches'),(max_cut_attempts,'max_cut_attempts'),
            (max_pair_attempts,'max_pair_attempts'),(max_face_attempts,'max_face_attempts'),(max_insertion_attempts,'max_insertion_attempts')):_positive_int(value,name)
    if min_ga < 4:
        raise ValueError('min_ga must be >=4 for a nonzero native operator allocation')
    parents=tuple(parents);energy=np.asarray(energies,dtype=float)
    if not parents or energy.shape!=(len(parents),) or not np.isfinite(energy).all():raise ValueError('parents and matching finite energies required')
    support,ads=surface_topology(parents[0],substrate_indices,adsorbate_indices)
    for a in parents:
        surface_topology(a,support,ads)
        if not np.array_equal(a.cell.array,parents[0].cell.array) or not np.array_equal(a.positions[list(support)],parents[0].positions[list(support)]) or not np.array_equal(a.numbers[list(support)],parents[0].numbers[list(support)]):raise ValueError('common fixed support and cell required')
        if not np.array_equal(np.sort(a.numbers[list(ads)]),np.sort(parents[0].numbers[list(ads)])):raise ValueError('identical adsorbate composition required')
    _radii(parents[0].numbers[list(ads)],atomic_radii)
    # Validate cutoff coverage without using the filter result as parent rejection.
    surface_collision_free(parents[0],bond_limits)
    order=np.argsort(energy,kind='stable');outputs=[];batches=[];aux_total=0
    for batch in range(max_batches):
        generated=[];failures=[];aux=0
        def attempt(operation,callback):
            nonlocal aux
            try:
                result=callback();items=list(result) if isinstance(result,(tuple,list)) else [result]
                generated.extend(items);aux+=sum(c.details.get('auxiliary_evaluations',0) for c in items)
            except (SamplingExhausted,ValueError,RuntimeError,FloatingPointError) as error:
                spent=int(getattr(error,'auxiliary_evaluations',0));aux+=spent
                generated.extend(getattr(error,'partial_candidates',()))
                failures.append(dict(operation=operation,error=str(error),auxiliary_evaluations=spent,
                    completed_family_details=getattr(error,'completed_family_details',())))
        attempt('crossover',lambda:cross_surface(parents,energy,support,ads,rng,n=min_ga//4,site_fractional=site_fractional,max_cut_attempts=max_cut_attempts,max_pair_attempts=max_pair_attempts,cuts_per_parent_slot=cuts_per_parent_slot))
        for _ in range(min_ga//4):
            index=int(order[int(rng.random()*len(order))])
            attempt('reload',lambda:reload_surface(parents[index],support,ads,rng,site_fractional=site_fractional,parent_index=index))
        for _ in range(min_ga//4):
            index=int(order[int(rng.random()*len(order))])
            attempt('disturbance',lambda:disturb_surface(parents[index],support,ads,rng,parent_index=index))
        for _ in range(min_ga//8):
            index=int(order[int(rng.random()*len(order))])
            attempt('reconstruction',lambda:reconstruct_surface(parents[index],support,ads,rng,atomic_radii=atomic_radii,max_face_attempts=max_face_attempts,parent_index=index))
        for _ in range(min_ga//4):
            index=int(order[0])
            attempt('rebuild',lambda:rebuild_surface(parents[index],support,ads,rng,site_fractional=site_fractional,atomic_radii=atomic_radii,max_face_attempts=max_face_attempts,max_insertion_attempts=max_insertion_attempts,auxiliary_evaluations=auxiliary_evaluations,parent_index=index))
        passed=[surface_collision_free(c.atoms,bond_limits) for c in generated]
        outputs.extend(c for c,ok in zip(generated,passed) if ok);aux_total+=aux
        batches.append(dict(batch=batch,generated=len(generated),passed=sum(passed),rejected=len(passed)-sum(passed),
            candidates=tuple(generated),passed_mask=tuple(passed),failures=tuple(failures),auxiliary_evaluations=aux))
        if len(outputs)>=min_ga:return SurfaceProposal(tuple(outputs),'target_reached',tuple(batches),aux_total)
        if not any(passed):return SurfaceProposal(tuple(outputs),'empty_batch',tuple(batches),aux_total)
    return SurfaceProposal(tuple(outputs),'budget_exhausted',tuple(batches),aux_total)
