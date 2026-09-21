"""Three-stage TYPE4 GA on an explicit fixed-support, 2D-periodic manifold."""
from dataclasses import dataclass
import numpy as np
from ase.constraints import FixAtoms
from .surface_ga import surface_topology,propose_type4,surface_collision_free,_radii
from .constrained_reference import run_constrained_ssw,ConstrainedSSWConfig
from .population import partition,rank_regions


@dataclass(frozen=True)
class SurfaceGAConfig:
    quick_steps: int
    generations: int
    generation_steps: int
    fine_steps: int
    regions: int
    fine_regions: int
    min_ga: int
    max_batches: int
    max_cut_attempts: int
    max_pair_attempts: int
    max_face_attempts: int
    max_insertion_attempts: int
    partition_max_draws: int
    auxiliary_evaluations: int = 100
    cuts_per_parent_slot: int | None = None

    def __post_init__(self):
        for key,value in vars(self).items():
            if key=='cuts_per_parent_slot' and value is None:continue
            lower=0 if key in ('quick_steps','generations','generation_steps','fine_steps') else 1
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<lower:raise ValueError(f'invalid {key}')


def run_surface_ga(initial,surface,*,config,walker_config,rng,substrate_indices,
                   adsorbate_indices,routing,matcher,bond_limits,atomic_radii,
                   site_fractional,walker=None,proposal_factory=None):
    """Quick -> complete TYPE4 proposals/offspring quick -> ranked fine walks.

    routing(atoms) must return three finite caller-defined features; matcher(a,b)
    independently decides identity on the fixed-support 2D periodic domain.
    No 3D crystal descriptor or stress certificate is silently substituted.
    Every physical call belongs to one walker; auxiliary synthetic LJ calls
    belong to the proposal ledger. Rejected valid landings remain observations.
    A caller matcher/routing is a scientific input, not proof of basin identity.
    """
    if not isinstance(config,SurfaceGAConfig) or not isinstance(walker_config,ConstrainedSSWConfig):raise TypeError('SurfaceGAConfig and ConstrainedSSWConfig required')
    if config.generations and proposal_factory is None and config.min_ga < 4:
        raise ValueError('min_ga must be >=4 for native TYPE4 generations')
    initial=tuple(a.copy() for a in initial)
    if not initial:raise ValueError('initial structures required')
    support,ads=surface_topology(initial[0],substrate_indices,adsorbate_indices)
    reference=initial[0].copy()
    def invariant(a):
        return (np.array_equal(a.cell.array,reference.cell.array) and np.array_equal(a.pbc,reference.pbc)
                and np.array_equal(a.positions[list(support)],reference.positions[list(support)])
                and np.array_equal(a.numbers[list(support)],reference.numbers[list(support)])
                and np.array_equal(np.sort(a.numbers[list(ads)]),np.sort(reference.numbers[list(ads)])))
    if not callable(routing) or not callable(matcher):raise TypeError('explicit surface routing and identity matcher required')
    def projection(a):
        value=np.asarray(routing(a.copy()),dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('routing must return exactly three finite features')
        return tuple(value)
    for a in initial:
        surface_topology(a,support,ads)
        if not invariant(a):raise ValueError('initial structures need common fixed support/cell and composition')
        projection(a);a.set_constraint(FixAtoms(indices=support))
    if config.generations and proposal_factory is None:
        _radii(reference.numbers[list(ads)],atomic_radii)
        surface_collision_free(reference,bond_limits)
        site=np.asarray(site_fractional,dtype=float)
        if site.shape!=(2,) or not np.isfinite(site).all():raise ValueError("explicit finite 2D docking site required")
    walker=run_constrained_ssw if walker is None else walker
    begin=surface.requests
    report=dict(status='running',config=config,walker_config=walker_config,sources=initial,
        substrate_indices=support,adsorbate_indices=ads,routing='caller three features',identity='caller matcher',
        bond_limits=dict(bond_limits),atomic_radii=dict(atomic_radii),site_fractional=tuple(site_fractional),
        archive=[],observations=[],walks=[],proposals=[],failures=[],auxiliary_evaluations=0)
    archive=report['archive']
    def rows():return [dict(energy=x['energy'],sims=x['projection']) for x in archive]
    def walk(atoms,steps,phase,generation,parent_ids=(),operation=None,source_atom_indices=()):
        before=surface.requests;entry=dict(phase=phase,generation=generation,input=atoms.copy(),
            parent_ids=tuple(parent_ids),operation=operation,source_atom_indices=tuple(source_atom_indices))
        try:
            result=walker(atoms.copy(),surface,steps=steps,config=walker_config,rng=rng,fixed_indices=support)
            entry.update(status=result.status,result=result)
            if result.status!='completed':report['failures'].append(dict(phase=phase,walk_index=len(report['walks']),reason=result.status))
            for minimum_index,ev in enumerate(result.minima):
                cert=dict(ev.certificate)
                active=ev.active_fmax;full=ev.full_raw_fmax
                valid=bool(ev.converged and active is not None and full is not None and ev.energy is not None
                    and np.isfinite([ev.energy,active,full]).all() and active<=walker_config.fmax and invariant(ev.atoms))
                cert.update(certified=valid,active_fmax=active,full_raw_fmax=full,fixed_and_cell_exact=invariant(ev.atoms),scope='fixed_support_active_atoms')
                obs=dict(id=len(report['observations']),walk_index=len(report['walks']),minimum_index=minimum_index,
                    phase=phase,generation=generation,parent_ids=tuple(parent_ids),operation=operation,
                    source_atom_indices=tuple(source_atom_indices),atoms=ev.atoms.copy(),energy=ev.energy,
                    certificate=cert,archive_id=None)
                report['observations'].append(obs)
                if not valid:
                    report['failures'].append(dict(phase=phase,reason='invalid active-force/support certificate',observation_id=obs['id']));continue
                try:
                    obs['projection']=projection(ev.atoms)
                    match=next((i for i,old in enumerate(archive) if matcher(ev.atoms.copy(),old['atoms'].copy())),None)
                    if match is None:
                        match=len(archive);archive.append(dict(obs,archive_id=match))
                    elif ev.energy<archive[match]['energy']:archive[match]=dict(obs,archive_id=match)
                    obs['archive_id']=match
                except (ValueError,RuntimeError) as error:
                    report['failures'].append(dict(phase=phase,reason='routing/identity: '+str(error),observation_id=obs['id']))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            entry.update(status='evaluation_failed',error=str(error));report['failures'].append(dict(phase=phase,reason=str(error)))
        finally:
            entry['requests']=surface.requests-before;report['walks'].append(entry)
        return not getattr(surface,'exhausted',False)
    def regions(phase):
        try:return partition(rows(),config.regions,rng,max_draws=config.partition_max_draws)
        except (ValueError,RuntimeError) as error:
            report['failures'].append(dict(phase=phase,reason=str(error)));return []
    for a in initial:
        if not walk(a,config.quick_steps,'quick',0):break
    for generation in range(1,config.generations+1):
        if getattr(surface,'exhausted',False):break
        groups=regions('generation_partition');selected=[i for group in groups for i in group]
        entry=dict(generation=generation,archive_indices=tuple(selected),regions=groups,status='no_proposal');report['proposals'].append(entry)
        if not selected:
            entry['reason']='no archived parents';continue
        energies=np.array([archive[i]['energy'] for i in selected])
        if proposal_factory is None and (len(selected)<=2 or np.ptp(energies)==0):
            entry['reason']='source Compete requires >2 parents and nonzero energy span';continue
        try:
            parents=[archive[i]['atoms'] for i in selected]
            if proposal_factory is None:
                proposal=propose_type4(parents,energies,support,ads,rng,min_ga=config.min_ga,bond_limits=bond_limits,
                    atomic_radii=atomic_radii,site_fractional=site_fractional,max_batches=config.max_batches,
                    max_cut_attempts=config.max_cut_attempts,max_pair_attempts=config.max_pair_attempts,
                    max_face_attempts=config.max_face_attempts,max_insertion_attempts=config.max_insertion_attempts,
                    auxiliary_evaluations=config.auxiliary_evaluations,cuts_per_parent_slot=config.cuts_per_parent_slot)
            else:proposal=proposal_factory(parents,energies,support,ads,rng,config=config,bond_limits=bond_limits,atomic_radii=atomic_radii,site_fractional=site_fractional)
            entry.update(status=proposal.status,result=proposal)
            report['auxiliary_evaluations']+=proposal.auxiliary_evaluations
            for child in proposal.candidates:
                ids=tuple(selected[i] for i in child.atom_parent_indices)
                if not invariant(child.atoms):raise ValueError('proposal changed frozen support/cell/composition')
                if not walk(child.atoms,config.generation_steps,'offspring_quick',generation,ids,child.operation,child.source_atom_indices):break
        except (ValueError,RuntimeError) as error:
            entry.update(status='proposal_failed',reason=str(error));report['failures'].append(dict(phase='proposal',reason=str(error)))
    if not getattr(surface,'exhausted',False):
        groups=regions('fine_partition');ranking=rank_regions(rows(),groups);report['fine_ranking']=ranking
        # Freeze selected archive IDs before fine walks can append/replace entries.
        chosen=[min(region.indices,key=lambda i:archive[i]['energy']) for region in ranking[:config.fine_regions]]
        for index in chosen:
            if not walk(archive[index]['atoms'],config.fine_steps,'fine',config.generations,(index,)):break
    qualified=[obs for obs in report['observations'] if obs['certificate']['certified']]
    report['best']=min(qualified,key=lambda obs:obs['energy']) if qualified else None
    report['requests']=surface.requests-begin;report['requests_reconciled']=report['requests']==sum(w['requests'] for w in report['walks'])
    report['status']='censored' if getattr(surface,'exhausted',False) else ('completed_with_failures' if report['failures'] else 'completed')
    return report
