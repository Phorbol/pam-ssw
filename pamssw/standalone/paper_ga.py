"""Experimental three-stage GA-SSW controller on independent Python surfaces.

This follows the paper's quick exploration -> GA/short-walk expansion -> ranked
fine exploration architecture, not the uploaded Java schedule's hidden 3/4/5
multipliers, forced fine iteration, carry rule, or integer divisor nine.

The available descriptor, KMeans and TYPE3 operators are separately identified
legacy reconstructions, not claims of exact paper DCCD or all-system coverage.
Only supplied nonperiodic, unconstrained, fixed-internal-monomer Atoms are
supported. Force stationarity is not a chemical or Hessian-stability certificate.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from ase import Atoms

from .ga_operators import _partition, propose_type3
from .legacy_descriptor import (cluster_descriptor, descriptor_similarity,
                                energy_window, merge_archive, remove_duplicates)
from .population import partition, rank_regions
from .surface import QuenchResult, quench


def run_ssw(*args, **kwargs):
    """Load the independent paper-reference walker only when a walk is requested."""
    from .paper_reference import run_ssw as walker
    return walker(*args, **kwargs)


@dataclass(frozen=True)
class PaperGAConfig:
    quick_steps: int
    generations: int
    generation_steps: int
    fine_steps: int
    ga_candidates: int
    regions: int
    fine_regions: int
    quench_fmax: float
    quench_steps: int
    proposal_max_batches: int
    proposal_max_cut_attempts: int
    proposal_max_pair_attempts: int
    partition_max_draws: int
    projection_tolerance: float
    energy_window: float

    def __post_init__(self):
        nonnegative = ('quick_steps', 'generations', 'generation_steps', 'fine_steps', 'quench_steps')
        positive = ('ga_candidates', 'regions', 'fine_regions', 'proposal_max_batches',
                    'proposal_max_cut_attempts', 'proposal_max_pair_attempts', 'partition_max_draws')
        for key in nonnegative + positive:
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (1 if key in positive else 0):
                raise ValueError(f'{key} must be an integer >= {1 if key in positive else 0}')
        for key in ('quench_fmax', 'projection_tolerance', 'energy_window'):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0 or (key == 'quench_fmax' and value == 0):
                raise ValueError(f'{key} has an invalid numerical value')


@dataclass(frozen=True)
class MinimumObservation:
    id: int
    phase: str
    generation: int
    seed_id: int | None
    result: QuenchResult
    projection: tuple | None
    eligible_for_archive: bool
    parent_ids: tuple
    operator: str | None
    details: dict


@dataclass(frozen=True)
class GAFailure:
    phase: str
    generation: int
    seed_id: int | None
    reason: str
    evaluation_requests: int


@dataclass(frozen=True)
class GAStage:
    phase: str
    generation: int
    seed_id: int | None
    status: str
    evaluation_requests: int
    observations: int
    details: dict


@dataclass(frozen=True)
class PaperGAResult:
    archive: tuple
    observations: tuple[MinimumObservation, ...]
    failures: tuple[GAFailure, ...]
    stages: tuple[GAStage, ...]
    walks: tuple
    evaluation_requests: int
    status: str
    physical_validation_performed: bool

    @property
    def best(self):
        """Lowest-energy retained force-stationary geometry, or None."""
        if not self.archive:
            return None
        return min(self.archive, key=lambda row: row['energy'])['atoms'].copy()


def run_ga_ssw(initial: Sequence[Atoms], surface, *, groups, references,
               descriptor_bonds, descriptor_weights, neighbor_range,
               proposal_bond_limits, config: PaperGAConfig, ssw_config,
               rng: np.random.Generator, ls=None,
               structure_validator: Callable[[Atoms], bool] | None = None) -> PaperGAResult:
    """Execute all three stages with explicit budgets and complete landing records.

    Each supplied initial structure and every GA offspring is quenched on the
    true surface (no terms). Initial quick walks, per-generation short walks and
    final ranked-region walks use the supplied SSWConfig unchanged. Every returned
    landing, including the walk's initial quench and unconverged outcomes, is
    retained and projected; only true/converged/optionally validated observations
    enter the archive. The explicit global energy window is applied to archive
    representatives after merging; all excluded observations remain available.

    references contains >=3 frozen descriptors because the available KMeans uses
    the first three projection coordinates. descriptor_bonds is the explicit
    NNA bond-length table; proposal_bond_limits is a separate explicit cutoff
    table in Angstrom. No structures, reference minima or radii are invented.

    A failed stage records its spent surface.requests. Degenerate GA parents or
    exhausted sampling stop further GA generations; independent final fine walks
    can still use the existing archive. No fallback parents/candidates are made.
    """
    if not isinstance(config, PaperGAConfig):
        raise TypeError('config must be PaperGAConfig')
    if not initial:
        raise ValueError('initial Atoms must be explicitly supplied')
    groups = _partition(initial[0], groups)
    if [i for group in groups for i in group] != list(range(len(initial[0]))):
        raise ValueError("controller requires contiguous monomer order matching frozen references")
    for atoms in initial[1:]:
        _partition(atoms, groups)
        if not np.array_equal(atoms.numbers, initial[0].numbers):
            raise ValueError('initial structures must share ordered element and monomer topology')
    if len(references) < 3:
        raise ValueError('at least three frozen references required by the current partition implementation')
    # Verify descriptor configuration before spending any calculator requests.
    sample = cluster_descriptor(initial[0].numbers, initial[0].positions, descriptor_bonds, neighbor_range)
    for reference in references:
        descriptor_similarity(sample, reference, descriptor_weights)

    started = surface.requests
    archive, observations, failures, stages, walks = [], [], [], [], []
    validation_attempted = False

    def fail(phase, generation, seed_id, reason, cost):
        failures.append(GAFailure(phase, generation, seed_id, reason, cost))

    def ingest(results, phase, generation, seed_id, *, parent_ids=(), operator=None, details=None):
        nonlocal archive, validation_attempted
        incoming = []
        ids = []
        for result in results:
            ident = len(observations)
            ids.append(ident)
            projection = None
            reason = None
            try:
                descriptor = cluster_descriptor(result.atoms.numbers, result.atoms.positions,
                                                descriptor_bonds, neighbor_range)
                projection = tuple(descriptor_similarity(descriptor, ref, descriptor_weights) for ref in references)
                if not np.isfinite(projection).all():
                    raise ValueError('projection is not finite')
                if result.surface != 'true':
                    reason = 'modified-surface landing cannot enter true-surface archive'
                elif (not result.converged or not np.isfinite(result.max_force)
                      or result.max_force > config.quench_fmax):
                    reason = 'true-surface quench did not satisfy the archive force tolerance'
                elif structure_validator is not None:
                    validation_attempted = True
                    if not structure_validator(result.atoms.copy()):
                        reason = 'structure_validator rejected the landing'
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
            eligible = reason is None
            observation = MinimumObservation(ident, phase, generation, seed_id, result, projection,
                                              eligible, tuple(parent_ids), operator, dict(details or {}))
            observations.append(observation)
            if eligible:
                incoming.append(dict(id=ident, atoms=result.atoms.copy(), energy=result.energy,
                                     sims=list(projection), observation_id=ident))
            else:
                fail(phase, generation, seed_id, reason, result.evaluation_requests)
        incoming = remove_duplicates(incoming, config.projection_tolerance)
        archive = merge_archive(archive, incoming, config.projection_tolerance)
        archive = sorted(energy_window(archive, config.energy_window), key=lambda row: row['energy'])
        return ids

    def relax(atoms, phase, generation, *, parent_ids=(), operator=None, details=None):
        before = surface.requests
        try:
            result = quench(atoms, surface, fmax=config.quench_fmax, steps=config.quench_steps)
            ids = ingest((result,), phase, generation, None, parent_ids=parent_ids,
                         operator=operator, details=details)
            status = 'completed' if observations[ids[0]].eligible_for_archive else 'failed'
            stages.append(GAStage(phase, generation, None, status, surface.requests-before, 1, dict(details or {})))
            return observations[ids[0]]
        except Exception as error:
            cost = surface.requests-before
            fail(phase, generation, None, f'{type(error).__name__}: {error}', cost)
            stages.append(GAStage(phase, generation, None, 'failed', cost, 0, dict(details or {})))
            return None

    def walk(row, phase, generation, steps):
        before = surface.requests
        try:
            result = run_ssw(row['atoms'].copy(), surface, steps=steps,
                             config=ssw_config, rng=rng, ls=ls)
            walks.append(result)
            landings = list(result.minima)
            seen = {id(q) for q in landings}
            for record in result.records:
                landing = getattr(record, 'landing', None)
                if isinstance(landing, QuenchResult) and id(landing) not in seen:
                    landings.append(landing)
                    seen.add(id(landing))
            ids = ingest(landings, phase, generation, row['id'])
            bad = any(not observations[i].eligible_for_archive for i in ids)
            # Keep failed biased/true landings and LS failures, even when the
            # walker correctly excludes them from its force-stationary minima.
            for record in result.records:
                state = getattr(record, 'status', '')
                reason = getattr(record, 'error', None)
                if state.endswith('_failed') or state == 'nonpositive_height' or reason:
                    bad = True
                    fail(phase, generation, row['id'], f'{state}: {reason or "step did not produce a qualified landing"}', 0)
            state = getattr(result, 'status', 'completed')
            if state != 'completed':
                bad = True
                fail(phase, generation, row['id'], f'walk ended with {state}', 0)
            stages.append(GAStage(phase, generation, row['id'], 'completed_with_failures' if bad else 'completed',
                                  surface.requests-before, len(ids), {'steps': steps, 'records': result.records}))
        except Exception as error:
            cost = surface.requests-before
            landing = getattr(error, 'result', None)
            ids = (ingest((landing,), phase, generation, row['id'])
                   if isinstance(landing, QuenchResult) else [])
            fail(phase, generation, row['id'], f'{type(error).__name__}: {error}', cost)
            stages.append(GAStage(phase, generation, row['id'], 'failed', cost, len(ids), {'steps': steps}))

    def regions(phase, generation):
        try:
            return partition(archive, config.regions, rng, max_draws=config.partition_max_draws)
        except Exception as error:
            fail(phase, generation, None, f'{type(error).__name__}: {error}', 0)
            return []

    def finish():
        status = ('no_eligible_minima' if not archive else
                  'completed_with_failures' if failures else 'completed')
        return PaperGAResult(tuple(archive), tuple(observations), tuple(failures), tuple(stages),
                             tuple(walks), surface.requests-started, status, validation_attempted)

    # 1. Supplied initialization -> certified true-surface quench -> quick walks.
    seeds = []
    for index, atoms in enumerate(initial):
        observation = relax(atoms, 'initial_quench', -1, details={'initial_index': index})
        if observation is not None and observation.eligible_for_archive:
            seeds.append(dict(id=observation.id, atoms=observation.result.atoms.copy()))
    for row in seeds:
        walk(row, 'quick', -1, config.quick_steps)
    if not archive:
        return finish()

    # 2. GA generations, true-surface offspring quenching, then short walks.
    for generation in range(config.generations):
        selected_regions = regions('generation_partition', generation)
        indices = [i for region in selected_regions for i in region]
        if not indices:
            fail('proposal', generation, None, 'no selected parents', 0)
            break
        parent_rows = [archive[i] for i in indices]
        try:
            proposal = propose_type3([row['atoms'] for row in parent_rows],
                                     [row['energy'] for row in parent_rows], groups, (0,) * len(groups), rng,
                                     min_ga=config.ga_candidates, bond_limits=proposal_bond_limits,
                                     max_batches=config.proposal_max_batches,
                                     max_cut_attempts=config.proposal_max_cut_attempts,
                                     max_pair_attempts=config.proposal_max_pair_attempts)
        except Exception as error:
            fail('proposal', generation, None, f'{type(error).__name__}: {error}', 0)
            stages.append(GAStage('proposal', generation, None, 'failed', 0, 0, {}))
            break
        stages.append(GAStage('proposal', generation, None, proposal.status, 0, 0,
                              {'batches': proposal.batches, 'candidates': len(proposal.candidates),
                               'rejected_by_bond_limit': proposal.rejected_by_bond_limit}))
        incomplete = proposal.status != 'target_reached'
        if incomplete:
            fail('proposal', generation, None, f'{proposal.status}: {proposal.reason}', 0)
        for candidate in proposal.candidates:
            parent_ids = tuple(parent_rows[i]['id'] for i in candidate.group_parent_indices)
            relax(candidate.atoms, 'offspring_quench', generation, parent_ids=parent_ids,
                  operator=candidate.operation, details=candidate.details)
        selected_regions = regions('generation_partition', generation)
        # Snapshot seeds before adding observations changes archive indices.
        generation_seeds = [archive[min(region, key=lambda i: archive[i]['energy'])].copy()
                            for region in selected_regions]
        for row in generation_seeds:
            walk(row, 'generation_short', generation, config.generation_steps)
        if incomplete:
            break

    # 3. Explicitly bounded ranked-region fine walks, at the unchanged SSWConfig.
    selected_regions = regions('fine_partition', config.generations)
    try:
        ranked = rank_regions(archive, selected_regions)[:config.fine_regions]
    except Exception as error:
        fail('fine_partition', config.generations, None, f'{type(error).__name__}: {error}', 0)
        return finish()
    fine_seeds = [archive[min(region.indices, key=lambda i: archive[i]['energy'])].copy() for region in ranked]
    for row in fine_seeds:
        walk(row, 'fine', config.generations, config.fine_steps)
    return finish()
