"""Research-only bridge from continuous standalone SSW to existing PAM scoring.

No new acquisition weights. Ordered geometry matching is approximate; this
adapter is limited to unconstrained inputs and is not a checkpoint/session API.
"""
from __future__ import annotations

from pamssw.acquisition import BanditSelector, ProposalOutcome, ProposalScorer
from pamssw.archive import MinimaArchive
from pamssw.fingerprint import structural_descriptor
from pamssw.state import State


class PoolStarterAdapter:
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
