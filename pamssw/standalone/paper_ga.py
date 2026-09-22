"""Experimental three-stage GA-SSW controller on independent Python surfaces.

This follows the paper's quick exploration -> GA/short-walk expansion -> ranked
fine exploration architecture, not the uploaded Java schedule's hidden 3/4/5
multipliers, forced fine iteration, carry rule, or integer divisor nine.

The available descriptor, KMeans and TYPE3 operators are separately identified
legacy reconstructions, not claims of exact paper DCCD or all-system coverage.
Supplied nonperiodic unconstrained TYPE0 atomic clusters and TYPE3 fixed-internal
monomer clusters are supported. TYPE0 uses corrected collision acceptance during
reinsertion, not the uploaded release's inverted condition/origin fallback. Force stationarity is not a chemical or Hessian-stability certificate.
"""
from dataclasses import dataclass
import copy
import pickle
from typing import Callable, Sequence

import numpy as np
from ase import Atoms

from .ga_operators import _partition, propose_type3
from .legacy_descriptor import (cluster_descriptor, descriptor_similarity,
                                _full_fingerprint_order,
                                energy_window, merge_archive, remove_duplicates)
from .population import partition, rank_regions
from .surface import QuenchResult, quench
from .paper_reference import SSWConfig
from ase.optimize import BFGS
from .ga_checkpoint import GACheckpoint


def propose_type0(*args, **kwargs):
    from .atomic_ga import propose_type0 as proposal
    return proposal(*args, **kwargs)


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
    proposal_type: int = 3  # Uploaded TYPE0 atoms or TYPE3 fixed monomers.
    proposal_max_insertion_attempts: int = 10000  # Operational cap, no origin fallback.
    cycles: int = 1
    offspring_steps: int = 0

    def __post_init__(self):
        if (isinstance(self.proposal_type, (bool, np.bool_))
                or not isinstance(self.proposal_type, (int, np.integer))
                or self.proposal_type not in (0, 3)):
            raise ValueError('proposal_type must be integer 0 or 3')
        nonnegative = ('quick_steps', 'generations', 'generation_steps', 'fine_steps', 'quench_steps', 'offspring_steps')
        positive = ('ga_candidates', 'regions', 'fine_regions', 'proposal_max_batches',
                    'proposal_max_cut_attempts', 'proposal_max_pair_attempts', 'partition_max_draws',
                    'proposal_max_insertion_attempts', 'cycles')
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


class BudgetExhausted(RuntimeError):
    """The next physical surface request would exceed the global run cap."""


@dataclass(frozen=True)
class GAStage:
    phase: str
    generation: int
    seed_id: int | None
    status: str
    evaluation_requests: int
    observations: int
    details: dict
    cycle: int = 0


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
    identity_mode: str = 'projection'
    budget_limit: int | None = None
    budget_exhausted: bool = False
    checkpoint: GACheckpoint | None = None

    @property
    def best(self):
        """Lowest-energy retained force-stationary geometry, or None."""
        if not self.archive:
            return None
        return min(self.archive, key=lambda row: row['energy'])['atoms'].copy()


def run_ga_ssw(initial: Sequence[Atoms], surface, *, groups, references,
               descriptor_bonds, descriptor_weights, neighbor_range,
               descriptor_row_order='legacy_counts',
               proposal_bond_limits, config: PaperGAConfig, ssw_config,
               rng: np.random.Generator, ls=None, change_types=None,
               structure_validator: Callable[[Atoms], bool] | None = None,
               structure_matcher: Callable[[Atoms, Atoms], bool] | None = None,
               max_evaluations: int | None = None, height_policy=None, gaussian_policy=None,
               height_update_budget: int = 1000,
               offspring_ssw_config: SSWConfig | None = None,
               mc=None, recovered_direction=None, recovered_rotation=None,
               checkpoint: GACheckpoint | None = None,
               checkpoint_callback: Callable[[GACheckpoint], bool | None] | None = None) -> PaperGAResult:
    """Execute all three stages with explicit budgets and complete landing records.

    Set config.proposal_type=0 and groups=None for atomic/alloy proposals;
    TYPE3 (default) requires monomer groups; optional change_types marks fixed (0)
    or internally mutable (1) units. Mixed-unit lineage is recorded per atom.
    Atomic parents may have different
    atom ordering but must share composition. TYPE0 retains per-atom lineage.

    Each supplied initial structure and every GA offspring is quenched on the
    true surface (no terms). Initial quick walks, per-generation short walks and
    final ranked-region walks use the supplied SSWConfig unchanged. Every returned
    landing, including the walk's initial quench and unconverged outcomes, is
    retained and projected; only true/converged/optionally validated observations
    enter the archive. The explicit global energy window is applied to archive
    representatives after merging; all excluded observations remain available.

    When ``config.offspring_steps`` is positive, each candidate is sent through
    its own short SSW run. ``offspring_ssw_config`` may replace the normal
    walker configuration for that run; only its lowest eligible landing is
    admitted to the shared archive, while every landing remains observable.

    references contains >=3 frozen descriptors because the available KMeans uses
    the first three projection coordinates. descriptor_bonds is the explicit
    NNA bond-length table; proposal_bond_limits is a separate explicit cutoff
    table in Angstrom. No structures, reference minima or radii are invented.

    ``descriptor_row_order="legacy_counts"`` preserves the reconstructed Java
    count-only row ordering and old checkpoint contracts. Opt-in
    ``"full_fingerprint"`` sorts candidates and copies of frozen references by
    all existing descriptor components. It corrects tied-row label dependence,
    not the descriptor radial formulas. Checkpoints cannot cross these modes.

    A failed stage records its spent surface.requests. Degenerate GA parents or
    exhausted sampling stop further GA generations; independent final fine walks
    can still use the existing archive. No fallback parents/candidates are made.

    ``mc``, ``recovered_direction`` and ``recovered_rotation`` are optional controls forwarded to every
    new SSW walk (quick, offspring, generation, and fine). Each walk starts its
    own SSW state; no direction or MC state is shared across GA walks. When a GA
    checkpoint is resumed, the same non-None options must be supplied so the
    explicit checkpoint contract matches; the default None contract remains
    compatible with older checkpoints.
    """
    if not isinstance(config, PaperGAConfig):
        raise TypeError('config must be PaperGAConfig')
    if descriptor_row_order not in ('legacy_counts', 'full_fingerprint'):
        raise ValueError("descriptor_row_order must be 'legacy_counts' or 'full_fingerprint'")
    if checkpoint is not None and not isinstance(checkpoint, GACheckpoint):
        raise TypeError('checkpoint must be GACheckpoint or None')
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError('checkpoint_callback must be callable or None')
    if checkpoint is not None:
        if checkpoint.version != GACheckpoint.VERSION:
            raise ValueError('unsupported GA checkpoint version')
        if checkpoint.phase not in ('quick_complete', 'generation_complete', 'cycle_complete'):
            raise ValueError('checkpoint is not a completed GA boundary')
        if checkpoint.cycle < 0 or checkpoint.generation < 0 or checkpoint.evaluation_requests < 0:
            raise ValueError('checkpoint cursor or request count is invalid')
    if structure_matcher is not None and not callable(structure_matcher):
        raise TypeError('structure_matcher must be callable or None')
    if offspring_ssw_config is not None and not isinstance(offspring_ssw_config, SSWConfig):
        raise TypeError('offspring_ssw_config must be SSWConfig or None')
    from .native_mc import NativeMCSettings
    if mc is not None and not isinstance(mc, NativeMCSettings):
        raise TypeError('mc must be NativeMCSettings or None')
    if mc is not None:
        for candidate in (ssw_config, offspring_ssw_config):
            if candidate is not None and candidate.temperature_K <= 0:
                raise ValueError('native MC requires positive temperature_K')
    if recovered_direction is not None:
        from .recovered_direction import RecoveredDirectionSettings
        if not isinstance(recovered_direction, RecoveredDirectionSettings):
            raise TypeError('recovered_direction must be RecoveredDirectionSettings or None')
        if any(atoms.pbc.any() or atoms.constraints or len(atoms) < 2 or
               np.asarray(atoms.positions).shape != (len(atoms), 3) or
               not np.isfinite(atoms.positions).all() for atoms in initial):
            raise ValueError('recovered direction requires a free nonperiodic cluster with finite positions')
        for candidate in (ssw_config, offspring_ssw_config):
            if candidate is not None and candidate.cluster_frame != 'direction_only':
                raise ValueError('recovered direction requires cluster_frame=direction_only')
            if candidate is not None and candidate.pre_rotation_hvp is not None:
                raise ValueError('recovered direction owns its PreRot stages; do not combine pre_rotation_hvp')
        if recovered_rotation is not None:
            raise ValueError('recovered_rotation and recovered_direction are mutually exclusive')
    if recovered_rotation is not None:
        from .recovered_rotation import RecoveredRotationSettings
        if not isinstance(recovered_rotation, RecoveredRotationSettings):
            raise TypeError('recovered_rotation must be RecoveredRotationSettings or None')
        for candidate in (ssw_config, offspring_ssw_config):
            if candidate is not None and candidate.pre_rotation_hvp is not None:
                raise ValueError('recovered_rotation and pre_rotation_hvp are mutually exclusive')
        if any(atoms.constraints for atoms in initial):
            raise NotImplementedError('recovered_rotation requires unconstrained atoms')
    from .paper_reference import _validate_height_policy_options
    _validate_height_policy_options(ssw_config, height_policy, height_update_budget)
    if offspring_ssw_config is not None:
        _validate_height_policy_options(offspring_ssw_config, height_policy, height_update_budget)
    if gaussian_policy is not None:
        from .pam_gaussian import PAMCurvatureGaussian
        if not isinstance(gaussian_policy, PAMCurvatureGaussian):
            raise TypeError('gaussian_policy must be PAMCurvatureGaussian')
        if height_policy is not None:
            raise ValueError('gaussian_policy and height_policy are mutually exclusive')
        for candidate in (ssw_config, offspring_ssw_config):
            if candidate is not None and candidate.cluster_frame == 'eckart':
                raise NotImplementedError('PAM Gaussian policy has no eckart section contract')
    if (max_evaluations is not None and
            (isinstance(max_evaluations, (bool, np.bool_)) or
             not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 0)):
        raise ValueError('max_evaluations must be a nonnegative integer or None')
    if not initial:
        raise ValueError('initial Atoms must be explicitly supplied')
    if config.proposal_type == 0:
        if change_types is not None:
            raise ValueError('TYPE0 does not use molecular change_types')
        if groups is not None:
            raise ValueError('TYPE0 takes groups=None; every atom is independently mutable')
        composition = sorted(initial[0].numbers)
        for atoms in initial:
            if len(atoms) < 2 or atoms.pbc.any() or atoms.constraints:
                raise ValueError('TYPE0 requires nonperiodic unconstrained clusters')
            if sorted(atoms.numbers) != composition:
                raise ValueError('TYPE0 initial structures must share composition')
        if config.generations > 0 and len(composition) <= (10 if len(set(composition)) == 1 else 5):
            raise ValueError('TYPE0 full mutation requires N>10 for pure or N>5 for alloy')
        if (config.generations > 0 and len(set(composition)) > 1
                and config.ga_candidates < 4):
            # TYPE0.java:45-59 and Mutate.java:84-110: every mixed-element
            # crossover/mutation quota is zero at G<4, irrespective of parents.
            raise ValueError('multi-element TYPE0 requires ga_candidates>=4; '
                             'smaller integer quotas generate no candidates')
    else:
        groups = _partition(initial[0], groups)
        change_types = (0,) * len(groups) if change_types is None else tuple(change_types)
        if len(change_types) != len(groups) or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or t not in (0, 1) for t in change_types):
            raise ValueError('one integer change_type 0 or 1 per molecular group required')
        if [i for group in groups for i in group] != list(range(len(initial[0]))):
            raise ValueError("controller requires contiguous monomer order matching frozen references")
        for atoms in initial[1:]:
            _partition(atoms, groups)
            if not np.array_equal(atoms.numbers, initial[0].numbers):
                raise ValueError('initial structures must share ordered element and monomer topology')
        if (config.proposal_type == 3 and config.generations > 0
                and config.ga_candidates < 4):
            # TYPE3's native integer quotas are zero below four candidates.
            raise ValueError('TYPE3 requires ga_candidates>=4; smaller integer quotas generate no candidates')
    if len(references) < 3:
        raise ValueError('at least three frozen references required by the current partition implementation')
    # Verify descriptor configuration before spending any calculator requests.
    sample = cluster_descriptor(initial[0].numbers, initial[0].positions, descriptor_bonds, neighbor_range)
    ordered_references = references
    if descriptor_row_order == 'full_fingerprint':
        sample = _full_fingerprint_order(sample)
        ordered_references = tuple(_full_fingerprint_order(reference) for reference in references)
    for reference in ordered_references:
        descriptor_similarity(sample, reference, descriptor_weights)

    started = surface.requests
    prior_requests = checkpoint.evaluation_requests if checkpoint is not None else 0
    budget_surface = None
    if max_evaluations is not None:
        class _BudgetSurface:
            def __init__(self, wrapped, start, limit):
                self._wrapped, self._start, self._limit = wrapped, start, int(limit)
                self.blocked = False

            @property
            def requests(self):
                return self._wrapped.requests

            def mark_exhausted_if_full(self):
                if prior_requests + self._wrapped.requests - self._start >= self._limit:
                    self.blocked = True

            def evaluate(self, atoms):
                if prior_requests + self._wrapped.requests - self._start >= self._limit:
                    self.blocked = True
                    raise BudgetExhausted(
                        f'max_evaluations={self._limit} exhausted before surface request')
                return self._wrapped.evaluate(atoms)

        budget_surface = _BudgetSurface(surface, started, max_evaluations)
        active_surface = budget_surface
    else:
        active_surface = surface
    archive, observations, failures, stages, walks = [], [], [], [], []
    validation_attempted = False
    resume_phase = None
    resume_cycle = 0
    resume_generation = 0
    def input_contract():
        def encoded(value):
            return pickle.dumps(value, protocol=4)
        contract = {
            'references': encoded(references), 'descriptor_bonds': encoded(descriptor_bonds),
            'descriptor_weights': encoded(descriptor_weights), 'neighbor_range': encoded(neighbor_range),
            'proposal_bond_limits': encoded(proposal_bond_limits), 'groups': encoded(groups),
            'change_types': encoded(change_types),
            'ls': encoded(ls), 'height_policy': encoded(height_policy),
            'gaussian_policy': encoded(gaussian_policy),
            'height_update_budget': encoded(height_update_budget),
            'initial_topology': encoded([(tuple(a.numbers), tuple(a.pbc), tuple(a.cell.array.ravel())) for a in initial]),
            'has_validator': structure_validator is not None,
            'has_matcher': structure_matcher is not None,
        }
        if mc is not None:
            contract['mc'] = encoded(mc)
        if recovered_rotation is not None:
            contract['recovered_rotation'] = encoded(recovered_rotation)
        if recovered_direction is not None:
            contract['recovered_direction'] = encoded(recovered_direction)
        if descriptor_row_order == 'full_fingerprint':
            contract['descriptor_row_order'] = descriptor_row_order
        return contract
    if checkpoint is not None:
        if checkpoint.phase == 'quick_complete' and (checkpoint.cycle != 0 or checkpoint.generation != 0):
            raise ValueError('quick checkpoint cursor is invalid')
        if checkpoint.phase == 'generation_complete' and not (0 <= checkpoint.cycle < config.cycles and
                                                              1 <= checkpoint.generation <= config.generations):
            raise ValueError('generation checkpoint cursor is invalid')
        if checkpoint.phase == 'cycle_complete' and not (1 <= checkpoint.cycle <= config.cycles and
                                                         checkpoint.generation == 0):
            raise ValueError('cycle checkpoint cursor is invalid')
        if checkpoint.evaluation_requests != sum(stage.evaluation_requests for stage in checkpoint.stages):
            raise ValueError('checkpoint request count disagrees with stage ledger')
        if checkpoint.config != config or checkpoint.ssw_config != ssw_config:
            raise ValueError('checkpoint configuration does not match supplied GA settings')
        if checkpoint.offspring_ssw_config != offspring_ssw_config:
            raise ValueError('checkpoint offspring configuration does not match supplied GA settings')
        if checkpoint.max_evaluations != max_evaluations:
            raise ValueError('checkpoint budget does not match max_evaluations')
        if checkpoint.contract != input_contract():
            raise ValueError('checkpoint scientific/input contract does not match supplied settings')
        archive = copy.deepcopy(checkpoint.archive)
        observations = copy.deepcopy(checkpoint.observations)
        failures = copy.deepcopy(checkpoint.failures)
        stages = copy.deepcopy(checkpoint.stages)
        walks = copy.deepcopy(checkpoint.walks)
        validation_attempted = checkpoint.validation_attempted
        rng.bit_generator.state = copy.deepcopy(checkpoint.rng_state)
        resume_phase = checkpoint.phase
        resume_cycle = checkpoint.cycle
        resume_generation = checkpoint.generation

    def fail(phase, generation, seed_id, reason, cost):
        failures.append(GAFailure(phase, generation, seed_id, reason, cost))

    def ingest(results, phase, generation, seed_id, *, parent_ids=(), operator=None,
               details=None, archive_best=False):
        nonlocal archive, validation_attempted
        incoming = []
        ids = []
        for result in results:
            ident = len(observations)
            ids.append(ident)
            projection = None
            reason = None
            identity_match = None
            try:
                descriptor = cluster_descriptor(result.atoms.numbers, result.atoms.positions,
                                                descriptor_bonds, neighbor_range)
                if descriptor_row_order == 'full_fingerprint':
                    descriptor = _full_fingerprint_order(descriptor)
                projection = tuple(descriptor_similarity(descriptor, ref, descriptor_weights)
                                   for ref in ordered_references)
                if not np.isfinite(projection).all():
                    raise ValueError('projection is not finite')
                if result.surface != 'true':
                    reason = 'modified-surface landing cannot enter true-surface archive'
                elif (not result.converged or not np.isfinite(result.energy)
                      or not np.isfinite(result.max_force)
                      or result.max_force > config.quench_fmax):
                    reason = 'true-surface quench did not satisfy finite-energy and archive force tolerances'
                elif structure_validator is not None:
                    validation_attempted = True
                    if not structure_validator(result.atoms.copy()):
                        reason = 'structure_validator rejected the landing'
                if reason is None and structure_matcher is not None and not archive_best:
                    # Match against the working archive exactly once while
                    # this observation is still rejectable. The archive is
                    # updated below, so later rows in this batch see it too.
                    for index, row in enumerate(archive):
                        try:
                            same = structure_matcher(result.atoms.copy(), row['atoms'].copy())
                        except Exception as error:
                            raise ValueError(f'identity matcher failed: {error}') from error
                        if same:
                            identity_match = index
                            break
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
            eligible = reason is None
            observation = MinimumObservation(ident, phase, generation, seed_id, result, projection,
                                              eligible, tuple(parent_ids), operator, dict(details or {}))
            observations.append(observation)
            if eligible:
                row = dict(id=ident, atoms=result.atoms.copy(), energy=result.energy,
                           sims=list(projection), observation_id=ident)
                if archive_best:
                    incoming.append(row)
                elif structure_matcher is None:
                    incoming.append(row)
                elif identity_match is None:
                    archive.append(row)
                elif result.energy < archive[identity_match]['energy']:
                    archive[identity_match] = row
            else:
                fail(phase, generation, seed_id, reason, result.evaluation_requests)
        if archive_best:
            incoming = sorted(incoming, key=lambda row: row['energy'])[:1]
            if structure_matcher is not None and incoming:
                row = incoming[0]
                identity_match = None
                try:
                    for index, old in enumerate(archive):
                        if structure_matcher(row['atoms'].copy(), old['atoms'].copy()):
                            identity_match = index
                            break
                except Exception as error:
                    fail(phase, generation, seed_id, f'ValueError: identity matcher failed: {error}', 0)
                    incoming = []
                if incoming:
                    if identity_match is None:
                        archive.append(row)
                    elif row['energy'] < archive[identity_match]['energy']:
                        archive[identity_match] = row
            elif incoming:
                incoming = remove_duplicates(incoming, config.projection_tolerance)
                archive = merge_archive(archive, incoming, config.projection_tolerance)
        elif structure_matcher is None:
            incoming = remove_duplicates(incoming, config.projection_tolerance)
            archive = merge_archive(archive, incoming, config.projection_tolerance)
        archive = sorted(energy_window(archive, config.energy_window), key=lambda row: row['energy'])
        return ids

    def relax(atoms, phase, generation, *, parent_ids=(), operator=None, details=None, cycle=0):
        before = surface.requests
        try:
            qopt = getattr(ssw_config, 'quench_optimizer', None)
            backend = qopt if qopt in ('safe-lbfgs-total', 'scipy-lbfgsb', 'ase-lbfgs-linesearch') else BFGS
            result = quench(atoms, active_surface, fmax=config.quench_fmax,
                            steps=config.quench_steps, optimizer=backend,
                            lbfgs_memory=getattr(ssw_config, 'lbfgs_memory', None))
            ids = ingest((result,), phase, generation, None, parent_ids=parent_ids,
                         operator=operator, details=details)
            status = 'completed' if observations[ids[0]].eligible_for_archive else 'failed'
            stages.append(GAStage(phase, generation, None, status, surface.requests-before, 1,
                                  dict(details or {}), cycle=cycle))
            return observations[ids[0]]
        except Exception as error:
            cost = surface.requests-before
            fail(phase, generation, None, f'{type(error).__name__}: {error}', cost)
            stages.append(GAStage(phase, generation, None, 'failed', cost, 0,
                                  dict(details or {}), cycle=cycle))
            return None

    def walk(row, phase, generation, steps, cycle=0, *, walk_config=None,
             archive_best=False, parent_ids=(), operator=None, details=None):
        before = surface.requests
        if budget_surface:
            budget_surface.mark_exhausted_if_full()
            if budget_surface.blocked:
                fail(phase, generation, row['id'],
                     f'BudgetExhausted: max_evaluations={max_evaluations} exhausted before walk', 0)
                stages.append(GAStage(phase, generation, row['id'], 'failed', 0, 0,
                                      {'steps': steps}, cycle=cycle))
                return
        try:
            walker_options = dict(
                config=ssw_config if walk_config is None else walk_config,
                rng=rng, ls=ls, height_policy=height_policy,
                gaussian_policy=gaussian_policy,
                height_update_budget=height_update_budget)
            if mc is not None:
                walker_options['mc'] = mc
            if recovered_direction is not None:
                walker_options['recovered_direction'] = recovered_direction
            if recovered_rotation is not None:
                walker_options['recovered_rotation'] = recovered_rotation
            result = run_ssw(row['atoms'].copy(), active_surface, steps=steps,
                             **walker_options)
            walks.append(result)
            landings = list(result.minima)
            seen = {id(q) for q in landings}
            for record in result.records:
                landing = getattr(record, 'landing', None)
                if isinstance(landing, QuenchResult) and id(landing) not in seen:
                    landings.append(landing)
                    seen.add(id(landing))
            ids = ingest(landings, phase, generation, row['id'], parent_ids=parent_ids,
                         operator=operator, details=details, archive_best=archive_best)
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
            stage_details = dict(details or {})
            stage_details.update({'steps': steps, 'records': result.records})
            if archive_best:
                eligible_ids = [ident for ident in ids if observations[ident].eligible_for_archive]
                if eligible_ids:
                    stage_details['selected_observation_id'] = min(
                        eligible_ids, key=lambda ident: observations[ident].result.energy)
                stage_details['parent_ids'] = tuple(parent_ids)
                stage_details['operator'] = operator
            stages.append(GAStage(phase, generation, row['id'], 'completed_with_failures' if bad else 'completed',
                                  surface.requests-before, len(ids), stage_details, cycle=cycle))
        except Exception as error:
            cost = surface.requests-before
            landing = getattr(error, 'result', None)
            ids = (ingest((landing,), phase, generation, row['id'],
                          parent_ids=parent_ids, operator=operator, details=details,
                          archive_best=archive_best)
                   if isinstance(landing, QuenchResult) else [])
            fail(phase, generation, row['id'], f'{type(error).__name__}: {error}', cost)
            stage_details = dict(details or {})
            stage_details.update({'steps': steps, 'parent_ids': tuple(parent_ids), 'operator': operator})
            stages.append(GAStage(phase, generation, row['id'], 'failed', cost, len(ids), stage_details, cycle=cycle))

    def regions(phase, generation):
        try:
            return partition(archive, config.regions, rng, max_draws=config.partition_max_draws)
        except Exception as error:
            fail(phase, generation, None, f'{type(error).__name__}: {error}', 0)
            return []

    def make_checkpoint(phase, cycle, generation):
        return GACheckpoint(
            GACheckpoint.VERSION, phase, cycle, generation,
            copy.deepcopy(archive), copy.deepcopy(observations),
            copy.deepcopy(failures), copy.deepcopy(stages), copy.deepcopy(walks),
            validation_attempted, copy.deepcopy(rng.bit_generator.state),
            config, ssw_config, offspring_ssw_config,
            prior_requests + surface.requests - started, max_evaluations,
            input_contract())

    def boundary(phase, cycle, generation):
        if checkpoint_callback is None:
            return None
        if budget_surface:
            budget_surface.mark_exhausted_if_full()
            if budget_surface.blocked:
                return None
        state = make_checkpoint(phase, cycle, generation)
        if checkpoint_callback(state):
            return state
        return None

    def finish(checkpoint_state=None, status_override=None):
        blocked = bool(budget_surface and budget_surface.blocked)
        status = status_override or ('budget_exhausted' if blocked else
                  'no_eligible_minima' if not archive else
                  'completed_with_failures' if failures else 'completed')
        return PaperGAResult(tuple(archive), tuple(observations), tuple(failures), tuple(stages),
                             tuple(walks), prior_requests + surface.requests-started, status, validation_attempted,
                             'caller_structure_matcher' if structure_matcher is not None else 'projection',
                             max_evaluations, blocked, checkpoint_state)

    # 1. Supplied initialization -> certified true-surface quench -> quick walks.
    if checkpoint is None:
        seeds = []
        for index, atoms in enumerate(initial):
            observation = relax(atoms, 'initial_quench', -1, details={'initial_index': index})
            if budget_surface and budget_surface.blocked:
                return finish()
            if observation is not None and observation.eligible_for_archive:
                seeds.append(dict(id=observation.id, atoms=observation.result.atoms.copy()))
        for row in seeds:
            walk(row, 'quick', -1, config.quick_steps)
            if budget_surface and budget_surface.blocked:
                return finish()
        saved = boundary('quick_complete', 0, 0)
        if saved is not None:
            return finish(saved, 'checkpoint_boundary')
    if not archive:
        return finish()

    def run_cycle(cycle, start_generation=0):
        # 2. GA generations, true-surface offspring quenching, then short walks.
        for generation in range(start_generation, config.generations):
            if budget_surface:
                budget_surface.mark_exhausted_if_full()
            if budget_surface and budget_surface.blocked:
                return finish()
            selected_regions = regions('generation_partition', generation)
            indices = [i for region in selected_regions for i in region]
            if not indices:
                fail('proposal', generation, None, 'no selected parents', 0)
                break
            parent_rows = [archive[i] for i in indices]
            try:
                if config.proposal_type == 0:
                    offsets = []
                    offset = 0
                    for region in selected_regions:
                        offsets.append(tuple(range(offset, offset + len(region))))
                        offset += len(region)
                    proposal = propose_type0([row['atoms'] for row in parent_rows],
                        [row['energy'] for row in parent_rows], rng,
                        min_ga=config.ga_candidates, bond_limits=proposal_bond_limits,
                        max_batches=config.proposal_max_batches,
                        max_cut_attempts=config.proposal_max_cut_attempts,
                        max_pair_attempts=config.proposal_max_pair_attempts,
                        parent_regions=offsets,
                        max_insertion_attempts=config.proposal_max_insertion_attempts)
                else:
                    proposal = propose_type3([row['atoms'] for row in parent_rows],
                                             [row['energy'] for row in parent_rows], groups, change_types, rng,
                                             min_ga=config.ga_candidates, bond_limits=proposal_bond_limits,
                                             max_batches=config.proposal_max_batches,
                                             max_cut_attempts=config.proposal_max_cut_attempts,
                                             max_pair_attempts=config.proposal_max_pair_attempts)
            except Exception as error:
                fail('proposal', generation, None, f'{type(error).__name__}: {error}', 0)
                stages.append(GAStage('proposal', generation, None, 'failed', 0, 0, {}, cycle=cycle))
                break
            stages.append(GAStage('proposal', generation, None, proposal.status, 0, 0,
                                  {'batches': proposal.batches, 'candidates': len(proposal.candidates),
                                   'rejected_by_bond_limit': proposal.rejected_by_bond_limit}, cycle=cycle))
            incomplete = proposal.status != 'target_reached'
            if incomplete:
                fail('proposal', generation, None, f'{proposal.status}: {proposal.reason}', 0)
            for candidate in proposal.candidates:
                lineage = candidate.details.get('atom_parent_indices', candidate.group_parent_indices)
                parent_ids = tuple(parent_rows[i]['id'] for i in lineage)
                if config.offspring_steps == 0:
                    relax(candidate.atoms, 'offspring_quench', generation, parent_ids=parent_ids,
                          operator=candidate.operation, details=candidate.details, cycle=cycle)
                else:
                    walk({'id': None, 'atoms': candidate.atoms.copy()}, 'offspring_ssw', generation,
                         config.offspring_steps, cycle, walk_config=offspring_ssw_config,
                         archive_best=True, parent_ids=parent_ids,
                         operator=candidate.operation, details=candidate.details)
                if budget_surface and budget_surface.blocked:
                    return finish()
            selected_regions = regions('generation_partition', generation)
            generation_seeds = [archive[min(region, key=lambda i: archive[i]['energy'])].copy()
                                for region in selected_regions]
            for row in generation_seeds:
                walk(row, 'generation_short', generation, config.generation_steps, cycle)
                if budget_surface and budget_surface.blocked:
                    return finish()
            if incomplete:
                break

            saved = boundary('generation_complete', cycle, generation + 1)
            if saved is not None:
                return finish(saved, 'checkpoint_boundary')

        selected_regions = regions('fine_partition', config.generations)
        try:
            ranked = rank_regions(archive, selected_regions)[:config.fine_regions]
        except Exception as error:
            fail('fine_partition', config.generations, None, f'{type(error).__name__}: {error}', 0)
            return finish()
        fine_seeds = [archive[min(region.indices, key=lambda i: archive[i]['energy'])].copy() for region in ranked]
        for row in fine_seeds:
            walk(row, 'fine', config.generations, config.fine_steps, cycle)
            if budget_surface and budget_surface.blocked:
                return finish()
        saved = boundary('cycle_complete', cycle + 1, 0)
        if saved is not None:
            return finish(saved, 'checkpoint_boundary')
        return None

    start_cycle = resume_cycle
    start_generation = 0
    if checkpoint is not None:
        if resume_phase == 'generation_complete':
            start_generation = resume_generation
        elif resume_phase == 'cycle_complete':
            start_cycle = resume_cycle
        elif resume_phase != 'quick_complete':
            raise ValueError('checkpoint is not a resumable completed boundary')
    for cycle in range(start_cycle, config.cycles):
        cycle_result = run_cycle(cycle, start_generation if cycle == start_cycle else 0)
        if cycle_result is not None:
            return cycle_result
    return finish()
