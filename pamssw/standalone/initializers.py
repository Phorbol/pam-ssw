"""Source-backed seed expansion; generated structures are not certified minima.

GaInitialStructure TYPE2/3/4 all require supplied physical seeds and topology.
No descriptor bases, physical energies, molecular bonds or substrate are guessed.
"""
from dataclasses import dataclass
import numpy as np
from .ga_operators import (_validate_parents,_molecular_pool,_cross_from_pool,
                           mutate_type3,_passes_bond_limit,SamplingExhausted)
from .molecular_periodic_ga import reconstruct_type2,_molecular_lift
from .surface_ga import propose_type4,surface_topology


@dataclass(frozen=True)
class InitialPopulation:
    structures: tuple
    origins: tuple
    status: str
    ledger: dict
    auxiliary_evaluations: int = 0
    physical_requests: int = 0


def _count(n):
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<0:raise ValueError('nonnegative initializer count required')


def initialize_type2(seeds,groups,*,count,rng,image_shifts=None):
    """Original seeds + count ReComMC(parent0), source generate2 semantics.

    ReComMC deterministic serial sibling corrects native shared-state races.
    No implicit collision filter or fake minima certificate is applied.
    """
    _count(count);seeds=tuple(a.copy() for a in seeds)
    if not seeds:raise ValueError('at least one physical molecular-crystal seed required')
    shifts=[None]*len(seeds) if image_shifts is None else tuple(image_shifts)
    if len(shifts)!=len(seeds):raise ValueError('one image-shift entry per seed required')
    for a,shift in zip(seeds,shifts):
        _molecular_lift(a,groups,shift)
        if not np.array_equal(a.numbers,seeds[0].numbers):raise ValueError('same ordered composition required')
    structures=list(seeds);origins=[dict(kind='supplied_seed',seed_index=i,certified=False) for i in range(len(seeds))]
    try:
        candidates=reconstruct_type2(seeds[0],groups,count,rng,image_shifts=shifts[0])
        structures.extend(c.atoms for c in candidates)
        origins.extend(dict(kind='periodic_reconstruction',seed_index=0,candidate=c,certified=False) for c in candidates)
        status='completed';error=None
    except (SamplingExhausted,ValueError,RuntimeError) as failure:status='generation_failed';error=str(failure)
    return InitialPopulation(tuple(structures),tuple(origins),status,
        dict(proposal_type=2,source='GaInitialStructure.generate2',requested_reconstructions=count,
             generated=len(structures)-len(seeds),error=error,certification='not performed'))


def initialize_type3(seeds,energies,groups,change_types,*,count,rng,bond_limits,
                     max_cut_attempts,max_pair_attempts,lj_monomer_optimization=False,
                     mutable_quota_policy='complete_library',mutable_cuts_per_parent_slot=None,
                     lj_pair_sigma=None,lj_max_evaluations=1000,lj_gradient_tol=.1):
    """Source generate3: C CrossMC + (C,C,C) mutations, seeds, filter/shuffle/cap C.

    Explicit seed energies are metadata needed by source Compete. Optional
    native LJBase.monoOpt is implemented with corrected analytic rigid-unit
    gradients, explicit sigma parameters and separately counted auxiliary work.
    Nonperiodic ASE domain represents the source's 50 A bookkeeping vacuum.
    """
    _count(count);seeds=tuple(a.copy() for a in seeds)
    groups,energies=_validate_parents(seeds,energies,groups,change_types)
    # Generated fragments are flattened in molecule order; seeds and auxiliary
    # rigid groups must share that same atom ordering throughout the population.
    if [i for group in groups for i in group]!=list(range(len(seeds[0]))):
        raise ValueError('TYPE3 initializer requires contiguous monomer order')
    if lj_monomer_optimization:
        from .molecular_auxiliary import MolecularLJChart,optimize_molecular_lj
        if lj_pair_sigma is None:raise ValueError('explicit LJ pair sigmas required for auxiliary monomer optimization')
        MolecularLJChart(seeds[0],groups,lj_pair_sigma)  # preflight before generation

    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<1 for x in (max_cut_attempts,max_pair_attempts)):raise ValueError('positive initializer execution budgets required')
    for key,value in bond_limits.items():
        if len(key)!=2 or not np.isfinite(value) or value<0:raise ValueError('finite nonnegative pair cutoff required')
    raw=list(seeds);origins=[dict(kind='supplied_seed',seed_index=i,certified=False) for i in range(len(seeds))]
    generated=[];error=None
    try:
        if count:
            working=[a.copy() for a in seeds]
            sons,daughters=_molecular_pool(working,energies,groups,rng,max_cut_attempts)
            for _ in range(count):generated.append(_cross_from_pool(sons,daughters,len(groups),rng,max_pair_attempts))
            generated.extend(mutate_type3(working,energies,groups,change_types,(count,count,count),rng,
                max_selection_attempts=max_pair_attempts,mutable_quota_policy=mutable_quota_policy,
                mutable_cuts_per_parent_slot=mutable_cuts_per_parent_slot))
        status='completed'
    except (SamplingExhausted,ValueError,RuntimeError) as failure:status='generation_failed';error=str(failure)
    raw.extend(c.atoms for c in generated);origins.extend(dict(kind=c.operation,candidate=c,certified=False) for c in generated)
    passed=[i for i,a in enumerate(raw) if _passes_bond_limit(a,bond_limits)]
    # Explicit Fisher-Yates is the Collections.shuffle distribution, not the
    # independent 10N-swap distribution used by mutation's TemTools.upset.
    order=list(passed)
    for i in range(len(order)-1,0,-1):
        j=int(rng.random()*(i+1));order[i],order[j]=order[j],order[i]
    shuffled_order=tuple(order);auxiliary=[];auxiliary_failures=[]
    if lj_monomer_optimization:
        for i in order:
            try:
                result=optimize_molecular_lj(raw[i],groups,lj_pair_sigma,max_evaluations=lj_max_evaluations,gradient_tol=lj_gradient_tol)
                raw[i]=result.atoms;auxiliary.append(dict(raw_index=i,result=result))
            except (ValueError,RuntimeError,FloatingPointError) as failure:
                auxiliary_failures.append(dict(raw_index=i,error=str(failure),auxiliary_evaluations=getattr(failure,'auxiliary_evaluations',0)))
                status='auxiliary_failed'
        score={item['raw_index']:item['result'].energy_aux for item in auxiliary}
        order=[i for i in order if i in score];order.sort(key=lambda i:score[i])
    kept=order[:count]
    return InitialPopulation(tuple(raw[i] for i in kept),tuple(origins[i] for i in kept),status,
        dict(proposal_type=3,source='GaInitialStructure.generate3',requested=count,
             generated_candidates=tuple(generated),raw_count=len(raw),passed_indices=tuple(passed),
             rejected_indices=tuple(i for i in range(len(raw)) if i not in passed),
             shuffled_indices=shuffled_order,auxiliary_sorted_indices=tuple(order),selected_indices=tuple(kept),error=error,
             bookkeeping_cell='nonperiodic ASE; native extents+50 A is not a periodic physical crystal',
             auxiliary_runs=tuple(auxiliary),auxiliary_failures=tuple(auxiliary_failures),certification='not performed'),
        sum(item['result'].auxiliary_evaluations for item in auxiliary)+sum(item['auxiliary_evaluations'] for item in auxiliary_failures))


def initialize_type4(seeds,energies,substrate_indices,adsorbate_indices,*,count,rng,
                     bond_limits,atomic_radii,site_fractional,max_cut_attempts,
                     max_pair_attempts,max_face_attempts,max_insertion_attempts,
                     auxiliary_evaluations=100,cuts_per_parent_slot=None):
    """Source generate4 supplied seeds + one complete CrossLoaded/MutateLoaded batch.

    Deliberate physical-domain corrections are those of surface_ga. Unlike
    source initializer, generated candidates are filtered using the explicitly
    supplied full 2D periodic cutoff table. Seeds remain explicit raw inputs.
    """
    _count(count);seeds=tuple(a.copy() for a in seeds)
    if not seeds:raise ValueError('physical supported-cluster seeds required')
    for a in seeds:surface_topology(a,substrate_indices,adsorbate_indices)
    origins=[dict(kind='supplied_seed',seed_index=i,certified=False) for i in range(len(seeds))]
    if count==0:return InitialPopulation(seeds,tuple(origins),'completed',dict(proposal_type=4,requested=0,generated=0,certification='not performed'))
    proposal=propose_type4(seeds,energies,substrate_indices,adsorbate_indices,rng,
        min_ga=count,bond_limits=bond_limits,atomic_radii=atomic_radii,site_fractional=site_fractional,
        max_batches=1,max_cut_attempts=max_cut_attempts,max_pair_attempts=max_pair_attempts,
        max_face_attempts=max_face_attempts,max_insertion_attempts=max_insertion_attempts,
        auxiliary_evaluations=auxiliary_evaluations,cuts_per_parent_slot=cuts_per_parent_slot)
    origins.extend(dict(kind=c.operation,candidate=c,certified=False) for c in proposal.candidates)
    failed=any(b['failures'] for b in proposal.batches)
    return InitialPopulation(seeds+tuple(c.atoms for c in proposal.candidates),tuple(origins),
        'completed_with_failures' if failed else 'completed',
        dict(proposal_type=4,source='GaInitialStructure.generate4',requested=count,proposal=proposal,
             filtering_correction='explicit full 2D periodic pair filter for generated candidates',certification='not performed'),proposal.auxiliary_evaluations)


def initialize_type1(seeds,energies,*,rng,bond_limits,target_new=30,
                     mutation_request=30,max_batches=100):
    """GaInitialStructure.generate1 forced Doping expansion of physical cells.

    Keep the terminal whole passing batch (correct source for-condition omission).
    Child energy metadata is inherited solely for subsequent source sorting;
    it is not an evaluated energy of the generated geometry.
    """
    from ase import Atoms
    from .periodic_softening import _identity
    from .periodic_ga import periodic_collision_free
    _count(target_new)
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1 for n in (mutation_request,max_batches)):raise ValueError('positive initializer batch counts required')
    structures=[a.copy() for a in seeds];energy=list(np.asarray(energies,dtype=float))
    if not structures or len(energy)!=len(structures) or not np.isfinite(energy).all():raise ValueError('physical cell seeds and finite selection metadata required')
    for a in structures:
        _identity(a)
        if 'masses' in a.arrays:raise ValueError('site isotope masses unsupported for species exchange')
        if not np.array_equal(np.sort(a.numbers),np.sort(structures[0].numbers)):raise ValueError('same seed composition required')
        periodic_collision_free(a,bond_limits)
    original_count=len(structures);origins=[dict(kind='supplied_seed',seed_index=i,certified=False) for i in range(original_count)];ledger=[]
    if target_new==0:return InitialPopulation(tuple(structures),tuple(origins),'completed',dict(proposal_type=1,batches=(),generated=0))
    for batch in range(max_batches):
        ranked=np.argsort(energy,kind='stable');n=len(structures[0]);candidates=[]
        def candidate(operation,moves=0,width=0.,random_parent=True):
            index=int(ranked[int(rng.random()*len(ranked))] if random_parent else ranked[0]);parent=structures[index]
            a=Atoms(numbers=parent.numbers.copy(),positions=parent.positions.copy(),cell=parent.cell.copy(),pbc=True);sources=list(range(n));changed=[]
            if operation=='exchange':
                for _ in range(10*n):
                    i,j=(int(rng.random()*n) for _ in range(2))
                    a.numbers[i],a.numbers[j]=int(a.numbers[j]),int(a.numbers[i]);sources[i],sources[j]=sources[j],sources[i]
            else:
                for _ in range(moves):
                    i=int(rng.random()*n);a.positions[i]+=width*(np.array([rng.random() for _ in range(3)])-.5);changed.append(i)
            candidates.append((a,dict(kind=operation,parent_population_index=index,source_atom_indices=tuple(sources),
                selection_energy_metadata=float(energy[index]),energy_is_inherited_not_evaluated=True,
                moves=tuple(changed),width_A=width,certified=False)))
        for _ in range(mutation_request//2):candidate('exchange')
        for count,moves,width,random_parent in ((mutation_request//8,n//5,.3,False),(mutation_request//8,n//2,.3,False),(mutation_request//4,n//5,.5,True),(mutation_request//4,n//2,.5,True)):
            for _ in range(count):candidate('disturbance',moves,width,random_parent)
        mask=[periodic_collision_free(a,bond_limits) for a,_ in candidates]
        for (a,origin),passed in zip(candidates,mask):
            if passed:structures.append(a);origins.append(origin);energy.append(origin['selection_energy_metadata'])
        ledger.append(dict(batch=batch,candidates=tuple(candidates),passed_mask=tuple(mask),generated=len(candidates),passed=sum(mask),rejected=len(mask)-sum(mask)))
        if len(structures)-original_count>=target_new:break
    status='completed' if len(structures)-original_count>=target_new else 'generation_budget_exhausted'
    return InitialPopulation(tuple(structures),tuple(origins),status,dict(proposal_type=1,source='GaInitialStructure.generate1',
        target_new=target_new,mutation_request=mutation_request,max_batches=max_batches,batches=tuple(ledger),
        generated=len(structures)-original_count,correction='append complete crossing batch before count termination',certification='not performed'))


def initialize_type0_regular(numbers,*,rng,atomic_radii,ring_sizes=(),
                             ring_multiplicity=0,cage_count=0,tangent_count=0,
                             max_face_attempts=1000):
    """Explicit recovered RegularRing/RegularCage/TripleTangency initial subset.

    No default mixture or claim to implement all generate0 packing families.
    Returned nonperiodic structures are geometry-only, not physical minima.
    """
    from ase import Atoms
    from .surface_ga import _radii,_swap_order,triple_tangency_cluster
    from .ga_operators import _docking_directions
    numbers=np.asarray(numbers,dtype=int)
    if len(numbers)<2:raise ValueError('regular packing needs at least two supplied atoms')
    for count in (ring_multiplicity,cage_count,tangent_count):_count(count)
    sizes=tuple(dict.fromkeys(ring_sizes))
    if any(isinstance(k,bool) or not isinstance(k,(int,np.integer)) or k<3 for k in sizes):raise ValueError('ring sizes must be integers >=3')
    if ring_multiplicity and not sizes:raise ValueError('explicit ring sizes required')
    radius=float(np.mean(_radii(numbers,atomic_radii)));structures=[];origins=[]
    def append(a,details):
        structures.append(a);origins.append(dict(certified=False,**details))
    order=_swap_order(len(numbers),rng)
    for _ in range(ring_multiplicity):
        for k in sizes:
            order=order[_swap_order(len(numbers),rng)]
            layers=int(np.ceil(len(numbers)/k));r=radius/np.sin(np.pi/k)
            x=[]
            for layer in range(layers):
                angles=2*np.pi*np.arange(k)/k+(np.pi/k if layer%2 else 0.)
                x.extend(np.column_stack((r*np.cos(angles),r*np.sin(angles),np.full(k,layer*radius*np.sqrt(3)))))
            append(Atoms(numbers=numbers[order],positions=np.asarray(x)[:len(numbers)]),
                dict(kind='regular_ring',ring_size=k,layers=layers,mean_radius_A=radius,source_atom_indices=tuple(order)))
    order=_swap_order(len(numbers),rng)
    for _ in range(cage_count):
        order=order[_swap_order(len(numbers),rng)];x=_docking_directions(len(numbers))
        adjacent=float(np.mean(np.linalg.norm(np.diff(x,axis=0),axis=1)));x*=2*radius/adjacent
        append(Atoms(numbers=numbers[order],positions=x),dict(kind='regular_cage',mean_radius_A=radius,
            source_atom_indices=tuple(order),scaling='mean successive Fibonacci-point distance = 2*mean atomic radius'))
    for _ in range(tangent_count):
        a,order,details=triple_tangency_cluster(numbers,rng,atomic_radii=atomic_radii,max_face_attempts=max_face_attempts)
        append(a,dict(kind='triple_tangency',source_atom_indices=order,**details))
    return InitialPopulation(tuple(structures),tuple(origins),'completed',dict(proposal_type=0,
        implemented_subset=('RegularRing','RegularCage','TripleTangency'),ring_multiplicity=ring_multiplicity,
        cage_count=cage_count,tangent_count=tangent_count,ring_sizes=sizes,
        source='GaInitialStructure.generate0 packing primitives; no default family allocation',certification='not performed'))
