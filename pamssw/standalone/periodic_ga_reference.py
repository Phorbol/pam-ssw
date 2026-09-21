"""Independent three-stage TYPE1 GA with explicit periodic routing and identity.

Quick walks -> region proposals and offspring quick walks -> ranked fine walks.
Each seed/offspring is initialized exactly once by its chosen walker. No extra
fixed-cell quench, inner MC duplication, or nonperiodic descriptor is used.
"""
from dataclasses import dataclass
import numpy as np
from .periodic_ga import propose_type1,_periodic_parents
from .periodic_descriptor import periodic_descriptor,periodic_projection
from .population import partition,rank_regions
from .vc_reference import run_vc_ssw
from .surface import QuenchResult


@dataclass(frozen=True)
class PeriodicGAConfig:
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
    partition_max_draws: int
    slots_per_parent: int = 100
    cuts_per_slot: int = 10

    def __post_init__(self):
        for key,value in vars(self).items():
            minimum=0 if key in ('quick_steps','generations','generation_steps','fine_steps') else 1
            if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError(f'invalid {key}')


def pymatgen_identity(*,ltol,stol,angle_tol):
    """Explicit optional identity backend; routing never decides duplicates.

    scale=False, primitive_cell=True, attempt_supercell=True are intentional.
    Caller owns tolerance qualification; an approximate match is not phase proof.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    if any(not np.isfinite(x) or x<=0 for x in (ltol,stol,angle_tol)):raise ValueError('positive identity tolerances required')
    matcher=StructureMatcher(ltol=ltol,stol=stol,angle_tol=angle_tol,scale=False,
                            primitive_cell=True,attempt_supercell=True)
    return lambda a,b:bool(matcher.fit(AseAtomsAdaptor.get_structure(a),AseAtomsAdaptor.get_structure(b)))


def run_periodic_ga(initial,surface,*,config,walker_config,rng,descriptor_basis,
                    bond_lengths,neighbor_range,projection_weights,bond_limits,
                    matcher,walker=None,proposal_factory=None,fixed_cell=False):
    """Supplied fully periodic seeds, counted E/F/stress surface and matcher.

    walker(atoms,surface,steps=...,config=...,rng=...) returns current/best,
    minima (ALL valid observations, including rejected landings), records/status.
    Default is the complete joint VC walker; block is a compatible callback.
    Certificate thresholds/pressure come from walker_config; block's atomic.fmax
    is recognized. Matching must be independent of the descriptor/routing score.
    Three supplied descriptor reference Atoms are copied and frozen before PES.
    Initial/final walker costs, failed stages, all observations and lineage remain
    in the returned report. No expensive independent certificates are hidden here.

    With fixed_cell=True, use SSWConfig and the fixed-cell run_ssw default.
    Initial cells must be exactly equal and remain in their original frame.
    Only the TYPE1 canonical proposal frame is mapped back before evaluation.
    True QuenchResult energy/max_force certificates replace stress criteria;
    objective is energy and unrequested full forces/stress are recorded as None.
    """
    if not isinstance(fixed_cell, (bool, np.bool_)):
        raise TypeError('fixed_cell must be boolean')
    initial=tuple(a.copy() for a in initial)
    _periodic_parents(initial,np.zeros(len(initial)))
    common_cell = None
    common_pbc = None
    if fixed_cell:
        if len(initial[0]) < 2 or not np.asarray(initial[0].pbc, dtype=bool).all():
            raise ValueError('fixed-cell periodic GA requires fully periodic inputs with at least two atoms')
        common_cell = np.asarray(initial[0].cell.array, dtype=float).copy()
        common_pbc = np.asarray(initial[0].pbc, dtype=bool).copy()
        for seed in initial[1:]:
            if not np.array_equal(seed.cell.array, common_cell) or not np.array_equal(seed.pbc, common_pbc):
                raise ValueError('fixed-cell TYPE1 requires an exact common fixed cell and pbc')
    if not callable(matcher):raise TypeError('explicit periodic identity matcher required')
    if len(descriptor_basis)!=3:raise ValueError('exactly three caller reference structures required')
    basis=tuple(periodic_descriptor(a.copy(),bond_lengths,neighbor_range) for a in descriptor_basis)
    # Validate projection definition before any oracle use.
    periodic_projection(periodic_descriptor(initial[0],bond_lengths,neighbor_range),basis,projection_weights)
    if fixed_cell:
        from .paper_reference import SSWConfig, run_ssw
        if not isinstance(walker_config, SSWConfig):
            raise TypeError('fixed-cell periodic GA requires SSWConfig')
        if walker_config.cluster_frame != 'translation_only':
            raise ValueError("fixed-cell periodic GA requires cluster_frame='translation_only'")
        if walker_config.direction_sampling not in ('global', 'isotropic'):
            raise ValueError("fixed-cell periodic GA requires global or isotropic direction sampling")
        walker=run_ssw if walker is None else walker
        fmax=walker_config.fmax
        pressure=stress_tol=None
    else:
        walker=run_vc_ssw if walker is None else walker
        fmax=getattr(walker_config,'fmax',None)
        if fmax is None:fmax=walker_config.atomic.fmax
        pressure=walker_config.pressure;stress_tol=walker_config.stress_tol
    begin=surface.requests
    report=dict(status='running',archive=[],observations=[],walks=[],proposals=[],failures=[],
        descriptor_basis=basis,config=config,walker_config=walker_config,
        bond_lengths=dict(bond_lengths),neighbor_range=neighbor_range,projection_weights=tuple(projection_weights),bond_limits=dict(bond_limits),
        identity='caller matcher; separate from descriptor',pressure=pressure,
        fixed_cell=bool(fixed_cell),
        common_cell=(common_cell.copy() if fixed_cell else None),
        common_pbc=(common_pbc.copy() if fixed_cell else None),
        fixed_cell_contract=('exact common input cell; TYPE1 canonical crossover frames are restored'
                             if fixed_cell else None),
        routing='periodic image-aware projections, legacy partition and empirical E/variance score',
        sources=initial)
    archive=report['archive']
    def rows():return [dict(energy=x['objective'],sims=x['projection']) for x in archive]
    def fixed_child(atoms):
        if not fixed_cell:
            return atoms, False
        if (not np.isfinite(atoms.positions).all() or
                tuple(sorted(map(int,atoms.numbers))) != tuple(sorted(map(int,initial[0].numbers))) or
                atoms.constraints):
            raise ValueError('fixed-cell TYPE1 proposal violates composition, finite-coordinate, or constraint contract')
        if (np.array_equal(atoms.cell.array, common_cell) and
                np.array_equal(atoms.pbc, common_pbc)):
            return atoms, False
        from ase.cell import Cell
        canonical = Cell.fromcellpar(Cell(common_cell).cellpar()).array
        if (np.array_equal(atoms.cell.array, canonical) and
                np.array_equal(atoms.pbc, common_pbc)):
            restored=atoms.copy()
            restored.positions=atoms.get_scaled_positions(wrap=False) @ common_cell
            restored.set_cell(common_cell, scale_atoms=False)
            restored.pbc=common_pbc
            return restored, True
        raise ValueError('fixed-cell TYPE1 proposal does not preserve the common cell')
    def walk(atoms,steps,phase,generation,parent_ids=(),operation=None):
        atoms,frame_transform=fixed_child(atoms)
        before=surface.requests;item=dict(phase=phase,generation=generation,parent_ids=tuple(parent_ids),operation=operation,input=atoms.copy(),
            frame_transform=('canonical_cellpar_to_common_cell' if frame_transform else None))
        try:
            result=walker(atoms.copy(),surface,steps=steps,config=walker_config,rng=rng)
            item.update(result=result,status=result.status)
            for minimum_index,ev in enumerate(result.minima):
                if fixed_cell:
                    candidate=getattr(ev, 'atoms', atoms)
                    energy=float(getattr(ev, 'energy', np.nan))
                    force=float(getattr(ev, 'max_force', np.nan))
                    valid_geometry=(np.isfinite(candidate.positions).all() and
                            np.array_equal(candidate.pbc, common_pbc) and
                            np.array_equal(candidate.cell.array, common_cell) and
                            tuple(sorted(map(int,candidate.numbers))) == tuple(sorted(map(int,initial[0].numbers))))
                    certified=bool(isinstance(ev, QuenchResult) and ev.surface == 'true' and ev.converged and
                        np.isfinite(energy) and np.isfinite(force) and force >= 0 and
                        force <= fmax and valid_geometry)
                    obs=dict(id=len(report['observations']),walk_index=len(report['walks']),minimum_index=minimum_index,phase=phase,generation=generation,
                        parent_ids=tuple(parent_ids),operation=operation,atoms=candidate.copy(),
                        energy=energy,objective=energy,forces=None,stress=None,
                        certificate=dict(max_force=force,certified=certified,
                            converged=bool(getattr(ev,'converged',False)),surface=getattr(ev,'surface',None),
                            fixed_cell_exact=bool(valid_geometry),
                            certificate_scope='fixed_cell_energy_maxforce'),archive_id=None)
                else:
                    # Generic callback must return physical E/F/stress certificates.
                    force=float(np.linalg.norm(ev.forces,axis=1).max())
                    stress=float(abs(ev.stress+pressure*np.eye(3)).max())
                    certified=bool(np.isfinite([ev.energy,ev.objective,force,stress]).all() and force<=fmax and stress<=stress_tol)
                    obs=dict(id=len(report['observations']),walk_index=len(report['walks']),minimum_index=minimum_index,phase=phase,generation=generation,
                        parent_ids=tuple(parent_ids),operation=operation,atoms=ev.atoms.copy(),
                        energy=ev.energy,objective=ev.objective,forces=ev.forces.copy(),stress=ev.stress.copy(),
                        certificate=dict(fmax=force,stress_max=stress,certified=certified),archive_id=None)
                report['observations'].append(obs)
                if not certified:
                    report['failures'].append(dict(phase=phase,reason='walker minimum failed physical residual certificate',observation_id=obs['id']))
                    continue
                try:
                    d=periodic_descriptor(obs['atoms'],bond_lengths,neighbor_range)
                    obs['projection']=tuple(periodic_projection(d,basis,projection_weights))
                    match=next((i for i,old in enumerate(archive) if matcher(obs['atoms'].copy(),old['atoms'].copy())),None)
                    if match is None:
                        match=len(archive);entry=dict(obs);entry['archive_id']=match;archive.append(entry)
                    elif obs['objective']<archive[match]['objective']:
                        entry=dict(obs);entry['archive_id']=match;archive[match]=entry
                    obs['archive_id']=match
                except (ValueError,RuntimeError) as error:
                    report['failures'].append(dict(phase=phase,reason='routing/identity: '+str(error),observation_id=obs['id']))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            item.update(status='evaluation_failed',error=str(error))
            report['failures'].append(dict(phase=phase,generation=generation,reason=str(error)))
        finally:
            item['requests']=surface.requests-before;report['walks'].append(item)
        return not bool(getattr(surface,'exhausted',False))
    def regions(phase):
        try:return partition(rows(),config.regions,rng,max_draws=config.partition_max_draws)
        except (ValueError,RuntimeError) as error:
            report['failures'].append(dict(phase=phase,reason=str(error)));return []
    for seed in initial:
        if not walk(seed,config.quick_steps,'quick',0):break
    for generation in range(1,config.generations+1):
        if getattr(surface,'exhausted',False):break
        groups=regions('generation_partition')
        selected=[i for group in groups for i in group]
        record=dict(generation=generation,archive_indices=tuple(selected),regions=groups,status='no_proposal')
        report['proposals'].append(record)
        if proposal_factory is None and (len(groups)<2 or len(selected)<=2):
            record['reason']='TYPE1 requires >=2 regions and >2 parents';continue
        if not selected:
            record['reason']='no archived parents';continue
        energies=np.array([archive[i]['objective'] for i in selected])
        if proposal_factory is None and np.ptp(energies)==0:
            record['reason']='TYPE1 source selection requires nonzero energy span';continue
        offset=0;local=[]
        for group in groups:local.append(list(range(offset,offset+len(group))));offset+=len(group)
        try:
            parents_atoms=[archive[i]['atoms'] for i in selected]
            if proposal_factory is None:
                proposal=propose_type1(parents_atoms,energies,local,rng,
                    min_ga=config.min_ga,bond_limits=bond_limits,max_batches=config.max_batches,
                    max_cut_attempts=config.max_cut_attempts,max_pair_attempts=config.max_pair_attempts,
                    slots_per_parent=config.slots_per_parent,cuts_per_slot=config.cuts_per_slot)
            else:
                proposal=proposal_factory(parents_atoms,energies,local,rng,
                    config=config,bond_limits=bond_limits)
            record.update(status=proposal.status,result=proposal)
            for child in proposal.candidates:
                if hasattr(child,'atom_parent_indices'):
                    lineage=child.atom_parent_indices
                else:
                    lineage=child.group_parent_indices
                parents=tuple(selected[i] for i in lineage)
                if not walk(child.atoms,config.generation_steps,'offspring_quick',generation,parents,child.operation):break
        except (ValueError,RuntimeError) as error:
            record.update(status='no_proposal',reason=str(error))
    if not getattr(surface,'exhausted',False):
        groups=regions('fine_partition')
        ranking=rank_regions(rows(),groups);report['fine_ranking']=ranking
        for region in ranking[:config.fine_regions]:
            index=min(region.indices,key=lambda i:archive[i]['objective'])
            if not walk(archive[index]['atoms'],config.fine_steps,'fine',config.generations,(index,)):break
    certified=[o for o in report['observations'] if o['certificate']['certified']]
    report['best']=min(certified,key=lambda o:o['objective']) if certified else None
    report['requests']=surface.requests-begin
    report['requests_reconciled']=report['requests']==sum(w['requests'] for w in report['walks'])
    report['status']='censored' if getattr(surface,'exhausted',False) else 'completed'
    return report
