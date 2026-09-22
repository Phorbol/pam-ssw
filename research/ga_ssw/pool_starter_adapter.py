"""Research-only bridge from continuous standalone SSW to existing PAM scoring.

No new acquisition weights. Ordered geometry matching is approximate; this
adapter is limited to unconstrained inputs. Its explicit checkpoint payload is
a caller-owned research state contract, not a core checkpoint/session API.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from pamssw.acquisition import BanditSelector, ProposalOutcome, ProposalScorer
from pamssw.archive import ArchivePrototype, MinimaArchive, MinimaEntry
from pamssw.fingerprint import structural_descriptor
from pamssw.state import State


class PoolStarterAdapter:
    CHECKPOINT_VERSION = 1

    def __init__(self, *, mode, energy_tol, rmsd_tol):
        if mode not in ('uniform', 'pam'):
            raise ValueError('mode must be uniform or pam')
        self.mode = mode
        self.archive = MinimaArchive(energy_tol, rmsd_tol)
        self.selector = BanditSelector()
        self.scorer = ProposalScorer()
        self.mapping = []
        self.representatives = {}
        self.outcomes = []
        self.decisions = []
        self._source = None
        self._executed = 0
        self._finalized = False

    @staticmethod
    def _state_payload(state):
        return {
            'numbers': state.numbers.copy(), 'positions': state.positions.copy(),
            'cell': None if state.cell is None else state.cell.copy(),
            'pbc': tuple(state.pbc), 'fixed_mask': state.fixed_mask.copy(),
            'metadata': copy.deepcopy(state.metadata),
        }

    @staticmethod
    def _state_from_payload(payload):
        if not isinstance(payload, Mapping):
            raise ValueError('checkpoint state must be a mapping')
        required = ('numbers', 'positions', 'cell', 'pbc', 'fixed_mask', 'metadata')
        if any(key not in payload for key in required):
            raise ValueError('checkpoint state is incomplete')
        return State(numbers=np.asarray(payload['numbers'], dtype=int).copy(),
                     positions=np.asarray(payload['positions'], dtype=float).copy(),
                     cell=None if payload['cell'] is None else np.asarray(payload['cell'], dtype=float).copy(),
                     pbc=tuple(payload['pbc']),
                     fixed_mask=np.asarray(payload['fixed_mask'], dtype=bool).copy(),
                     metadata=copy.deepcopy(payload['metadata']))

    @staticmethod
    def _policy_payload(policy):
        return {key: getattr(policy, key) for key in (
            'archive_density_weight', 'novelty_weight', 'frontier_weight',
            'exploration_weight', 'baseline_probability', 'beta_energy')}

    def checkpoint_contract(self):
        """Return the pure configuration identity required for restoration."""
        policy = self.selector.policy
        return {
            'version': self.CHECKPOINT_VERSION,
            'identity': 'research.ga_ssw.pool_starter_adapter.PoolStarterAdapter',
            'mode': self.mode,
            'archive': {
                'energy_tol': self.archive.energy_tol,
                'rmsd_tol': self.archive.rmsd_tol,
                'max_prototypes': self.archive.max_prototypes,
                'cell_tol': self.archive.cell_tol,
            },
            'scorer': {
                'mode': self.scorer.mode.value,
                'near_energy_window': self.scorer.near_energy_window,
            },
            'selector': {'policy': self._policy_payload(policy)},
        }

    def export_state(self):
        """Export adapter state as explicit data, without strategy instances."""
        if self._finalized:
            raise ValueError('cannot export finalized pool adapter')
        entries = []
        for entry in self.archive.entries:
            entries.append({
                'entry_id': entry.entry_id, 'state': self._state_payload(entry.state),
                'energy': entry.energy, 'parent_id': entry.parent_id, 'visits': entry.visits,
                'descriptor': None if entry.descriptor is None else entry.descriptor.copy(),
                'node_trials': entry.node_trials, 'node_successes': entry.node_successes,
                'frontier_value': entry.frontier_value, 'duplicate_hits': entry.duplicate_hits,
                'node_duplicate_failures': entry.node_duplicate_failures,
                'frontier_score': entry.frontier_score, 'is_frontier': entry.is_frontier,
                'is_dead': entry.is_dead,
            })
        prototypes = [{
            'descriptor': prototype.descriptor.copy(),
            'representative_entry_id': prototype.representative_entry_id,
            'weight': prototype.weight,
        } for prototype in self.archive.prototypes]
        outcomes = [dict(vars(outcome)) for outcome in self.outcomes]
        return {
            'version': self.CHECKPOINT_VERSION,
            'contract': self.checkpoint_contract(),
            'archive': {
                'entries': entries, 'prototypes': prototypes,
                'energy_mismatch_hits': self.archive.energy_mismatch_hits,
                'max_energy_mismatch': self.archive.max_energy_mismatch,
            },
            'mapping': list(self.mapping),
            'representatives': [[int(entry_id), int(observation_index)]
                                for entry_id, observation_index in self.representatives.items()],
            'outcomes': outcomes,
            'decisions': copy.deepcopy(self.decisions),
            'source': self._source,
            'executed': self._executed,
            'finalized': self._finalized,
        }

    @staticmethod
    def _require_mapping(value, label):
        if not isinstance(value, Mapping):
            raise ValueError(f'checkpoint {label} must be a mapping')
        return value

    def restore_state(self, payload):
        """Validate and restore a prior adapter state atomically."""
        if self._finalized:
            raise ValueError('cannot restore finalized pool adapter')
        payload = self._require_mapping(payload, 'payload')
        required_payload = ('version', 'contract', 'archive', 'mapping',
                            'representatives', 'outcomes', 'decisions',
                            'source', 'executed', 'finalized')
        if any(key not in payload for key in required_payload):
            raise ValueError('checkpoint payload is incomplete')
        if payload['version'] != self.CHECKPOINT_VERSION:
            raise ValueError('checkpoint version does not match')
        if payload['finalized']:
            raise ValueError('cannot restore finalized pool adapter')
        if payload['contract'] != self.checkpoint_contract():
            raise ValueError('checkpoint contract does not match adapter')
        archive_payload = self._require_mapping(payload.get('archive'), 'archive')
        required_archive = ('entries', 'prototypes', 'energy_mismatch_hits',
                            'max_energy_mismatch')
        if any(key not in archive_payload for key in required_archive):
            raise ValueError('checkpoint archive is incomplete')
        archive = MinimaArchive(**self.checkpoint_contract()['archive'])
        entries = []
        for index, item in enumerate(archive_payload.get('entries', ())):
            item = self._require_mapping(item, 'entry')
            if item.get('entry_id') != index:
                raise ValueError('checkpoint entry ids are not contiguous')
            descriptor = item.get('descriptor')
            entries.append(MinimaEntry(
                entry_id=index, state=self._state_from_payload(item['state']),
                energy=float(item['energy']), parent_id=item.get('parent_id'),
                visits=int(item['visits']),
                descriptor=None if descriptor is None else np.asarray(descriptor, dtype=float).copy(),
                node_trials=int(item['node_trials']), node_successes=int(item['node_successes']),
                frontier_value=float(item['frontier_value']), duplicate_hits=int(item['duplicate_hits']),
                node_duplicate_failures=int(item['node_duplicate_failures']),
                frontier_score=float(item['frontier_score']), is_frontier=bool(item['is_frontier']),
                is_dead=bool(item['is_dead'])))
        prototypes = []
        for item in archive_payload.get('prototypes', ()):
            item = self._require_mapping(item, 'prototype')
            prototypes.append(ArchivePrototype(
                descriptor=np.asarray(item['descriptor'], dtype=float).copy(),
                representative_entry_id=int(item['representative_entry_id']),
                weight=int(item['weight'])))
        if any(prototype.representative_entry_id < 0 or
               prototype.representative_entry_id >= len(entries) for prototype in prototypes):
            raise ValueError('checkpoint prototype references an unknown entry')
        archive.entries = entries
        archive.prototypes = prototypes
        archive.energy_mismatch_hits = int(archive_payload.get('energy_mismatch_hits', 0))
        archive.max_energy_mismatch = float(archive_payload.get('max_energy_mismatch', 0.0))
        outcomes = [ProposalOutcome(**self._require_mapping(item, 'outcome'))
                    for item in payload['outcomes']]
        mapping = list(payload['mapping'])
        if len(outcomes) != len(mapping):
            raise ValueError('checkpoint outcomes and mapping lengths differ')
        if any(int(entry_id) < 0 or int(entry_id) >= len(entries) for entry_id in mapping):
            raise ValueError('checkpoint mapping references an unknown entry')
        representatives = {}
        for pair in payload['representatives']:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('checkpoint representative pairs are invalid')
            entry_id, observation_index = int(pair[0]), int(pair[1])
            if (entry_id not in mapping or observation_index < 0 or
                    observation_index >= len(mapping) or mapping[observation_index] != entry_id):
                raise ValueError('checkpoint representative references an invalid observation')
            if entry_id in representatives:
                raise ValueError('checkpoint representative ids are duplicated')
            representatives[entry_id] = observation_index
        decisions = copy.deepcopy(list(payload['decisions']))
        source = payload['source']
        if source is not None and (int(source) < 0 or int(source) >= len(entries)):
            raise ValueError('checkpoint source references an unknown entry')
        executed = int(payload['executed'])
        if executed < 0:
            raise ValueError('checkpoint executed count is invalid')
        if decisions:
            if any(not isinstance(decision, Mapping) or 'step' not in decision
                   for decision in decisions):
                raise ValueError('checkpoint decisions are missing step indices')
            if executed != max(int(decision['step']) for decision in decisions) + 1:
                raise ValueError('checkpoint executed count disagrees with decision steps')
        elif executed != 0:
            raise ValueError('checkpoint executed count has no decision steps')
        # All parsing and validation above is complete before changing self.
        self.archive = archive
        self.mapping = mapping
        self.representatives = representatives
        self.outcomes = outcomes
        self.decisions = decisions
        self._source = source
        self._executed = executed
        self._finalized = False
        return self

    def _insert(self, atoms, energy, parent):
        if atoms.constraints:
            raise ValueError('research pool adapter does not support constrained inputs')
        state = State(atoms.numbers.copy(), atoms.positions.copy(),
                      atoms.cell.array.copy(), tuple(atoms.pbc))
        old_best = min((e.energy for e in self.archive.entries), default=energy)
        is_new = self.archive.find_match(state, energy) is None
        coverage = self.archive.coverage_gain(structural_descriptor(state))
        entry = self.archive.add(state, energy, parent)
        index = len(self.mapping)
        self.mapping.append(entry.entry_id)
        self.representatives.setdefault(entry.entry_id, index)
        self.outcomes.append(ProposalOutcome(energy, old_best, is_new, not is_new, coverage))
        return entry.entry_id

    def _credit(self, source, attempts, landing_index=None):
        entry = self.archive.entries[source]
        entry.node_trials += attempts
        entry.visits += attempts
        if landing_index is not None:
            outcome = self.outcomes[landing_index]
            self.archive.record_success(entry, self.scorer.score(outcome),
                                        duplicate_failures=int(outcome.is_duplicate))
        else:
            self.archive.refresh_frontier_status()

    def __call__(self, snapshot, rng):
        if self._finalized or snapshot.step + 1 <= self._executed:
            raise ValueError('adapter requires one increasing callback sequence per run')
        for observation in snapshot.observations[len(self.mapping):]:
            if observation.index != len(self.mapping):
                raise ValueError('noncontiguous observation indices')
            self._insert(observation.atoms, observation.energy, self._source)
            if self._source is None:
                self._source = self.mapping[0]
        self._credit(self._source, snapshot.step + 1 - self._executed,
                     snapshot.last_landing_index)
        self._executed = snapshot.step + 1
        if self.mode == 'uniform':
            selected = self.archive.entries[int(rng.integers(len(self.archive.entries)))]
        else:
            selected = self.selector.select(self.archive, rng)
        current_entry = self.mapping[snapshot.current_index]
        chosen = None if selected.entry_id == current_entry else self.representatives[selected.entry_id]
        actual_index = snapshot.current_index if chosen is None else chosen
        self.decisions.append(dict(step=snapshot.step, source_entry=self._source,
                                   selected_entry=selected.entry_id, chosen_index=chosen,
                                   actual_index=actual_index, cumulative_requests=snapshot.cost))
        self._source = self.mapping[actual_index]
        return chosen

    def finalize(self, result):
        """Reconcile all executed attempts, including terminal failures; no PES calls."""
        if self._finalized:
            raise ValueError('adapter already finalized')
        total = result.initial.evaluation_requests + sum(r.evaluation_requests for r in result.records)
        if total != result.evaluation_requests:
            raise ValueError('initial and outer request costs do not close')
        for minimum in result.minima[len(self.mapping):]:
            self._insert(minimum.atoms, minimum.energy, self._source)
            if self._source is None:
                self._source = self.mapping[0]
        # Reconstruct exact executed trial credit. The last selection is only a
        # proposed next start and must never create a phantom completed trial.
        for entry in self.archive.entries:
            entry.node_trials = entry.node_successes = entry.node_duplicate_failures = 0
            entry.visits = self.mapping.count(entry.entry_id)
        current = 0
        landing_index = 0
        attempts = []
        decisions_by_step = {decision['step']: decision for decision in self.decisions}
        for step, record in enumerate(result.records):
            source = self.mapping[current]
            qualified = record.landing is not None and record.landing.converged
            observed = None
            if qualified:
                landing_index += 1
                observed = landing_index
            self._credit(source, 1, observed)
            if qualified and record.accepted:
                current = landing_index
            selection = getattr(record, 'starter_selection', None)
            # ``chosen_index`` is the selector request.  The standalone SSW
            # record is authoritative about whether that request committed;
            # older records predate ``restarted`` and retain the historical
            # successful-selection behavior.
            if (selection is not None and selection['chosen_index'] is not None
                    and selection.get('restarted', True)):
                current = selection['chosen_index']
            if step in decisions_by_step:
                # The callback can only propose; finalized reporting uses the
                # committed core state, including a failed LS initialization.
                decisions_by_step[step]['actual_index'] = current
            attempts.append(dict(source_entry=source, status=record.status,
                                 qualified=qualified, landing_index=observed,
                                 evaluation_requests=record.evaluation_requests))
        if landing_index + 1 != len(self.mapping):
            raise ValueError('qualified landing records and observations disagree')
        self._finalized = True
        return dict(mode=self.mode, mapping=self.mapping, representatives=self.representatives,
                    decisions=self.decisions, attempts=attempts,
                    qualified_discoveries=sum(o.is_new_minimum for o in self.outcomes[1:]),
                    duplicates=sum(o.is_duplicate for o in self.outcomes[1:]),
                    initial_requests=result.initial.evaluation_requests,
                    outer_requests=sum(r.evaluation_requests for r in result.records),
                    total_requests=total, cost_closed=True,
                    entries=[dict(entry_id=e.entry_id, parent_id=e.parent_id,
                                  energy=e.energy, visits=e.visits,
                                  node_trials=e.node_trials, node_successes=e.node_successes,
                                  node_duplicate_failures=e.node_duplicate_failures)
                             for e in self.archive.entries])
