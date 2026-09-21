"""Controller accounting tests; fakes test scheduling, not scientific validity."""
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone import paper_ga
from pamssw.standalone.surface import QuenchResult


def configuration(**changes):
    values = dict(quick_steps=2, generations=0, generation_steps=3, fine_steps=4,
                  ga_candidates=8, regions=1, fine_regions=1, quench_fmax=.01,
                  quench_steps=20, proposal_max_batches=2, proposal_max_cut_attempts=30,
                  proposal_max_pair_attempts=40, partition_max_draws=100,
                  projection_tolerance=1e-8, energy_window=100.)
    values.update(changes)
    return paper_ga.PaperGAConfig(**values)


def geometry(x):
    return Atoms('OHHOHH', positions=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                             [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]]) + x)


def qresult(x, energy, *, converged=True, count=1):
    return QuenchResult(geometry(x), energy, 0. if converged else 1., converged, 1, count, 'true')


def descriptor_fakes(monkeypatch):
    monkeypatch.setattr(paper_ga, 'cluster_descriptor', lambda numbers, positions, *args: float(positions[0, 0]))
    monkeypatch.setattr(paper_ga, 'descriptor_similarity', lambda descriptor, reference, weights: descriptor)


def run(initial, surface, **kwargs):
    return paper_ga.run_ga_ssw(initial, surface, groups=((0, 1, 2), (3, 4, 5)),
                              references=(0., 1., 2.), descriptor_bonds={(1, 1): 1., (1, 8): 1., (8, 8): 1.},
                              descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
                              config=kwargs.pop('config', configuration()), ssw_config=kwargs.pop('ssw_config', object()),
                              rng=np.random.default_rng(3), **kwargs)


def test_ga_direct_quench_passes_ssw_memory(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    seen = []
    def relax(atoms, surface, **kwargs):
        seen.append(kwargs.get('lbfgs_memory'))
        surface.requests += 1
        return qresult(0, 5.)
    monkeypatch.setattr(paper_ga, 'quench', relax)
    ssw = SimpleNamespace(quench_optimizer='safe-lbfgs-total', lbfgs_memory=400)
    run([geometry(0)], surface, ssw_config=ssw)
    assert seen == [400]


@pytest.mark.parametrize('optimizer,expected_direct', [
    ('scipy-lbfgsb', 'scipy-lbfgsb'),
    ('ase-lbfgs-linesearch', 'ase-lbfgs-linesearch'),
    ('ase-lbfgs', paper_ga.BFGS),
])
def test_ga_optimizer_baseline_is_explicit_across_quench_and_walk(monkeypatch, optimizer, expected_direct):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    direct, walks = [], []
    def relax(atoms, surface, **kwargs):
        direct.append(kwargs.get('optimizer'))
        return qresult(0, 5.)
    def walk(atoms, surface, **kwargs):
        walks.append(kwargs['config'].quench_optimizer)
        q = qresult(0, 5.)
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'quench', relax)
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs: ProposalResult(
        (GeneticCandidate(geometry(1), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {}),),
        'target_reached', 1, 0))
    result = run([geometry(0)], surface,
        config=configuration(quick_steps=1, generations=1, generation_steps=1, fine_steps=1),
        ssw_config=SimpleNamespace(quench_optimizer=optimizer, lbfgs_memory=None))
    assert result.status == 'completed'
    assert len(direct) == 2 and all(item == expected_direct for item in direct)
    assert len(walks) == 3 and all(item == optimizer for item in walks)
    assert [s.phase for s in result.stages if s.phase.endswith('quench')] == [
        'initial_quench', 'offspring_quench']


def test_ga_forwards_height_policy_and_budget_to_all_ssw_phases(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda atoms, surface, **kwargs: qresult(0, 5.))
    seen = []
    def walk(atoms, surface, **kwargs):
        seen.append(kwargs)
        q = qresult(0, 5.)
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs: ProposalResult(
        (GeneticCandidate(geometry(1), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {}),),
        'target_reached', 1, 0))
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    policy = ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2)
    ls = object()
    ssw = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='cartesian')
    result = run([geometry(0)], surface,
                 config=configuration(quick_steps=1, generations=1,
                                      generation_steps=1, fine_steps=1),
                 ssw_config=ssw, ls=ls, height_policy=policy,
                 height_update_budget=37)
    assert result.status == 'completed'
    assert [call['steps'] for call in seen] == [1, 1, 1]
    assert all(call['height_policy'] is policy and call['height_update_budget'] == 37
               and call['ls'] is ls for call in seen)


def test_ga_height_policy_validation_precedes_initial_quench(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda *args, **kwargs: pytest.fail('initial quench should not run'))
    ssw = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='cartesian')
    with pytest.raises(TypeError):
        run([geometry(0)], surface, ssw_config=ssw, height_policy=object())
    assert surface.requests == 0


def test_ga_height_policy_eckart_rejected_before_initial_quench(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda *args, **kwargs: pytest.fail('initial quench should not run'))
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    policy = ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2)
    ssw = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='eckart')
    with pytest.raises(NotImplementedError):
        run([geometry(0)], surface, ssw_config=ssw, height_policy=policy)
    assert surface.requests == 0


def test_ga_height_budget_validation_precedes_initial_quench(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda *args, **kwargs: pytest.fail('initial quench should not run'))
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    policy = ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2)
    ssw = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='cartesian')
    with pytest.raises(ValueError, match='height-update budget'):
        run([geometry(0)], surface, ssw_config=ssw, height_policy=policy,
            height_update_budget=0)
    assert surface.requests == 0


def test_ga_offspring_eckart_rejected_before_initial_quench(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda *args, **kwargs: pytest.fail('initial quench should not run'))
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    policy = ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2)
    normal = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='cartesian')
    offspring = paper_ga.SSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
        temperature_K=150., fmax=.01, relax_steps=0, fd_step=1e-4,
        rotation_hvp=12, rotation_tol=.02, cluster_frame='eckart')
    with pytest.raises(NotImplementedError):
        run([geometry(0)], surface, ssw_config=normal,
            offspring_ssw_config=offspring, height_policy=policy)
    assert surface.requests == 0


def test_archives_every_true_converged_walk_landing_and_counts_failures(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=7)
    def relax(atoms, surface, **kwargs):
        surface.requests += 2
        return qresult(0, 5., count=2)
    calls = []
    def walk(atoms, surface, *, steps, **kwargs):
        calls.append(steps)
        surface.requests += 5
        minima = ((qresult(0, 5.), qresult(1, 3.), qresult(2, 4.)) if len(calls) == 1 else
                  (qresult(1, 3.), qresult(3, 2.), qresult(4, -10., converged=False)))
        return SimpleNamespace(initial=minima[0], minima=minima, records=(), evaluation_requests=5)
    monkeypatch.setattr(paper_ga, 'quench', relax)
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    result = run([geometry(0)], surface)
    assert calls == [2, 4]
    assert result.evaluation_requests == 12
    assert sorted(row['energy'] for row in result.archive) == [2., 3., 4., 5.]
    assert len(result.observations) == 7  # Explicit initial quench plus all walk minima.
    failed = [observation for observation in result.observations if not observation.eligible_for_archive]
    assert len(failed) == 1
    assert failed[0].projection is not None
    assert result.status == 'completed_with_failures'


def test_generation_really_quenches_each_child_and_retains_lineage(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    relax_x = []
    memory = []
    def relax(atoms, surface, **kwargs):
        x = float(atoms.positions[0, 0]); relax_x.append(x)
        memory.append(kwargs.get('lbfgs_memory'))
        surface.requests += 1
        return qresult(x, 10. - x)
    def walk(atoms, surface, *, steps, **kwargs):
        surface.requests += 1
        x = float(atoms.positions[0, 0])
        q = qresult(x, 10. - x)
        return SimpleNamespace(initial=q, minima=(q,), records=(), evaluation_requests=1)
    def propose(parents, energies, groups, change_types, rng, **kwargs):
        candidates = tuple(GeneticCandidate(geometry(x), groups, 'test_operator', (0, 1), {'fixture': x}) for x in (3., 4.))
        return ProposalResult(candidates, 'target_reached', 1, 0)
    monkeypatch.setattr(paper_ga, 'quench', relax)
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    monkeypatch.setattr(paper_ga, 'propose_type3', propose)
    result = run([geometry(0), geometry(1), geometry(2)], surface, config=configuration(generations=1),
                 ssw_config=SimpleNamespace(quench_optimizer='safe-lbfgs-total', lbfgs_memory=400))
    assert relax_x == [0., 1., 2., 3., 4.]
    assert memory == [400, 400, 400, 400, 400]
    offspring = [o for o in result.observations if o.phase == 'offspring_quench']
    assert len(offspring) == 2
    assert all(len(o.parent_ids) == 2 for o in offspring)
    assert all(o.operator == 'test_operator' for o in offspring)
    assert result.status == 'completed'


def test_backend_failure_is_returned_with_spent_requests(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    def fail(atoms, surface, **kwargs):
        surface.requests += 1
        raise RuntimeError('backend failed')
    monkeypatch.setattr(paper_ga, 'quench', fail)
    result = run([geometry(0)], surface)
    assert result.status == 'no_eligible_minima'
    assert result.evaluation_requests == 1
    assert not result.archive
    assert result.failures[0].reason == 'RuntimeError: backend failed'


def test_failed_walker_record_and_initial_quench_keep_geometries(monkeypatch):
    from pamssw.standalone.paper_reference import InitialQuenchError
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    def relax(atoms, surface, **kwargs):
        surface.requests += 1
        return qresult(0, 5.)
    count = 0
    def walk(atoms, surface, **kwargs):
        nonlocal count
        count += 1
        surface.requests += 3
        if count == 1:
            q = qresult(1, 6., converged=False, count=3)
            record = SimpleNamespace(status='true_quench_failed', error=None, landing=q)
            return SimpleNamespace(initial=qresult(0, 5.), minima=(qresult(0, 5.),),
                                   records=(record,), evaluation_requests=3, status='completed')
        raise InitialQuenchError(qresult(2, 7., converged=False, count=3))
    monkeypatch.setattr(paper_ga, 'quench', relax)
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    result = run([geometry(0)], surface)
    assert result.evaluation_requests == 7
    assert len(result.observations) == 4
    assert sum(not observation.eligible_for_archive for observation in result.observations) == 2
    assert result.status == 'completed_with_failures'
    assert [stage.status for stage in result.stages][-2:] == ['completed_with_failures', 'failed']


def test_actual_python_walker_and_ase_surface_are_connected_without_binaries():
    from pathlib import Path
    from ase.calculators.calculator import Calculator, all_changes
    from pamssw.standalone.surface import ASESurface
    from pamssw.standalone.paper_reference import SSWConfig
    path = Path(__file__).resolve().parents[2] / 'research/ga_ssw/evidence/staged-water/final-arc/0.arc'
    rows = [line.split() for line in path.read_text().splitlines() if 'CORE' in line]
    atoms = Atoms([r[0] for r in rows], positions=[[float(x) for x in r[1:4]] for r in rows])
    class HarmonicFixture(Calculator):
        # Numerical interface check on supplied coordinates, not a water PES.
        implemented_properties = ['energy', 'forces']
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            delta = self.atoms.positions - target
            self.results = dict(energy=.5 * float((delta * delta).sum()), forces=-delta)
    target = atoms.positions.copy()
    surface = ASESurface(HarmonicFixture())
    bonds = {(1, 1): 1., (1, 8): 1., (8, 8): 1.}
    reference = paper_ga.cluster_descriptor(atoms.numbers, atoms.positions, bonds, 1.2)
    ssw = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1, temperature_K=100.,
                    fmax=.01, relax_steps=2, fd_step=1e-4, rotation_hvp=2, rotation_tol=.01)
    result = paper_ga.run_ga_ssw([atoms], surface, groups=tuple(tuple(range(i, i+3)) for i in range(0, 45, 3)),
                                 references=(reference,) * 3, descriptor_bonds=bonds,
                                 descriptor_weights=(1.,) * 6, neighbor_range=1.2, proposal_bond_limits={},
                                 config=configuration(quick_steps=0, fine_steps=0), ssw_config=ssw,
                                 rng=np.random.default_rng(1))
    assert result.status == 'completed'
    assert len(result.walks) == 2 and len(result.observations) == 3
    assert result.evaluation_requests == surface.requests == 3
    assert not result.physical_validation_performed


def test_looser_walk_certificate_does_not_relax_archive_force_tolerance(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench', lambda *args, **kwargs: qresult(0, 5.))
    weak = QuenchResult(geometry(1), 1., .1, True, 1, 1, 'true')
    monkeypatch.setattr(paper_ga, 'run_ssw', lambda *args, **kwargs: SimpleNamespace(minima=(weak,), records=(), status='completed'))
    result = run([geometry(0)], surface)
    assert len(result.archive) == 1 and result.archive[0]['energy'] == 5.
    assert all(not observation.eligible_for_archive for observation in result.observations[1:])


@pytest.mark.parametrize('energy', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_energy_is_rejected_before_archive(monkeypatch, energy):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda *args, **kwargs: qresult(0, energy))
    result = run([geometry(0)], surface)
    assert len(result.observations) == 1
    assert not result.observations[0].eligible_for_archive
    assert not result.archive
    assert result.status == 'no_eligible_minima'
    assert 'finite-energy' in result.failures[0].reason


def test_global_evaluation_cap_stops_later_phase_and_keeps_archive(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    def evaluate(atoms):
        surface.requests += 1
        return 0., np.zeros((len(atoms), 3))
    surface.evaluate = evaluate
    def paid_quench(atoms, surface, **kwargs):
        surface.evaluate(atoms)
        return qresult(0, 5.)
    def walk(atoms, surface, **kwargs):
        q = paper_ga.quench(atoms, surface, **kwargs)
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'quench', paid_quench)
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    result = run([geometry(0)], surface,
                 config=configuration(quick_steps=1, fine_steps=1), max_evaluations=2)
    assert result.evaluation_requests == surface.requests == 2
    assert result.budget_limit == 2 and result.budget_exhausted
    assert result.status == 'budget_exhausted'
    assert result.archive and result.archive[0]['energy'] == 5.


def test_global_cap_exactly_reached_by_natural_end_is_not_blocked(monkeypatch):
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    def evaluate(atoms):
        surface.requests += 1
        return 0., np.zeros((len(atoms), 3))
    surface.evaluate = evaluate
    def paid_quench(atoms, surface, **kwargs):
        surface.evaluate(atoms)
        return qresult(0, 5.)
    monkeypatch.setattr(paper_ga, 'quench', paid_quench)
    def walker(atoms, surface, **kwargs):
        q = paper_ga.quench(atoms, surface, **kwargs)
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walker)
    result = run([geometry(0)], surface,
                 config=configuration(quick_steps=0, fine_steps=0), max_evaluations=3)
    assert result.evaluation_requests == surface.requests == 3
    assert result.budget_limit == 3 and not result.budget_exhausted
    assert result.status == 'completed'


@pytest.mark.parametrize('element', ['Cu', 'Al'])
def test_emt_initial_quench_budget_stop_preserves_no_paid_archive(element):
    atoms = Icosahedron(element, 2)
    atomic_number = int(atoms.numbers[0])
    bonds = {(atomic_number, atomic_number): 2.8}
    reference = paper_ga.cluster_descriptor(atoms.numbers, atoms.positions, bonds, 1.2)
    from pamssw.standalone.paper_reference import SSWConfig
    surface = paper_ga.ASESurface(EMT()) if hasattr(paper_ga, 'ASESurface') else None
    if surface is None:
        from pamssw.standalone.surface import ASESurface
        surface = ASESurface(EMT())
    config = configuration(quick_steps=0, generations=0, fine_steps=0,
                           quench_steps=20)
    ssw = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                    temperature_K=0., fmax=.1, relax_steps=1, fd_step=.01,
                    rotation_hvp=2, rotation_tol=.1, direction_sampling='global')
    result = paper_ga.run_ga_ssw(
        [atoms], surface, groups=(tuple(range(13)),), references=(reference,) * 3,
        descriptor_bonds=bonds, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=config, ssw_config=ssw,
        rng=np.random.default_rng(1), max_evaluations=0)
    assert result.evaluation_requests == surface.requests == 0
    assert result.status == 'budget_exhausted' and result.budget_exhausted
    assert not result.archive


@pytest.mark.parametrize('element', ['Cu', 'Al'])
def test_emt_quick_budget_stop_keeps_initial_archive(element):
    atoms = Icosahedron(element, 2)
    atomic_number = int(atoms.numbers[0])
    bonds = {(atomic_number, atomic_number): 2.8}
    reference = paper_ga.cluster_descriptor(atoms.numbers, atoms.positions, bonds, 1.2)
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.surface import ASESurface
    config = configuration(quick_steps=1, generations=0, fine_steps=0,
                           quench_steps=100)
    ssw = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                    temperature_K=0., fmax=.1, relax_steps=1, fd_step=.01,
                    rotation_hvp=2, rotation_tol=.1, direction_sampling='global')
    probe_surface = ASESurface(EMT())
    probe = paper_ga.run_ga_ssw(
        [atoms], probe_surface, groups=(tuple(range(13)),), references=(reference,) * 3,
        descriptor_bonds=bonds, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=configuration(quick_steps=0, generations=0,
        fine_steps=0, quench_steps=100), ssw_config=ssw,
        rng=np.random.default_rng(1))
    initial_cost = probe.stages[0].evaluation_requests
    surface = ASESurface(EMT())
    result = paper_ga.run_ga_ssw(
        [atoms], surface, groups=(tuple(range(13)),), references=(reference,) * 3,
        descriptor_bonds=bonds, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=config, ssw_config=ssw,
        rng=np.random.default_rng(1), max_evaluations=initial_cost + 1)
    assert result.evaluation_requests <= initial_cost + 1
    assert result.budget_exhausted and result.status == 'budget_exhausted'
    assert result.archive and result.archive[0]['energy'] == probe.archive[0]['energy']


def test_generation_exact_cap_at_natural_end_is_not_blocked(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    surface.evaluate = lambda atoms: (setattr(surface, 'requests', surface.requests + 1)
                                      or (0., np.zeros((len(atoms), 3))))
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda atoms, surface, **kwargs: (surface.evaluate(atoms), qresult(0, 5.))[1])
    def walk(atoms, surface, *, steps, **kwargs):
        q = paper_ga.quench(atoms, surface, **kwargs)
        return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs: ProposalResult(
        (GeneticCandidate(geometry(1), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {}),),
        'target_reached', 1, 0))
    result = run([geometry(0)], surface,
                 config=configuration(quick_steps=0, generations=1,
                                          generation_steps=1, fine_steps=0), max_evaluations=5)
    assert result.evaluation_requests == surface.requests == 5
    assert not result.budget_exhausted and result.status == 'completed'


def test_next_generation_is_blocked_after_previous_generation_fills_cap(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    surface.evaluate = lambda atoms: (setattr(surface, 'requests', surface.requests + 1)
                                      or (0., np.zeros((len(atoms), 3))))
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda atoms, surface, **kwargs: (surface.evaluate(atoms), qresult(0, 5.))[1])
    monkeypatch.setattr(paper_ga, 'run_ssw', lambda atoms, surface, *, steps, **kwargs:
                        SimpleNamespace(minima=() if not steps else (paper_ga.quench(atoms, surface),),
                                         records=(), status='completed'))
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs: ProposalResult(
        (GeneticCandidate(geometry(1), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {}),),
        'target_reached', 1, 0))
    result = run([geometry(0)], surface,
                 config=configuration(quick_steps=0, generations=2,
                                      generation_steps=1, fine_steps=0), max_evaluations=3)
    assert result.evaluation_requests == surface.requests == 3
    assert result.budget_exhausted and result.status == 'budget_exhausted'


def test_controller_rejects_noncanonical_group_order_before_evaluation():
    import pytest
    surface = SimpleNamespace(requests=0)
    with pytest.raises(ValueError, match='contiguous monomer order'):
        paper_ga.run_ga_ssw([geometry(0)], surface, groups=((3, 4, 5), (0, 1, 2)),
                            references=(0., 1., 2.), descriptor_bonds={}, descriptor_weights=(1.,) * 6,
                            neighbor_range=1., proposal_bond_limits={}, config=configuration(),
                            ssw_config=object(), rng=np.random.default_rng(1))
    assert surface.requests == 0


@pytest.mark.parametrize('quota', [1, 2, 3])
def test_multi_element_zero_operator_quota_rejected_before_pes(quota):
    from ase.collections import g2
    surface = SimpleNamespace(requests=0)
    with pytest.raises(ValueError, match='integer quotas generate no candidates'):
        paper_ga.run_ga_ssw([g2['butadiene']], surface, groups=None,
            references=(), descriptor_bonds={}, descriptor_weights=(1.,) * 6,
            neighbor_range=1.2, proposal_bond_limits={},
            config=configuration(proposal_type=0, ga_candidates=quota, generations=1),
            ssw_config=object(), rng=np.random.default_rng(11))
    assert surface.requests == 0


def test_mutable_group_controller_uses_exact_atomic_lineage(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface=SimpleNamespace(requests=0);seen=[]
    def relax(atoms,surface,**kwargs):
        surface.requests+=1;x=float(atoms.positions[0,0]);return qresult(x,10.-x)
    def walk(atoms,surface,**kwargs):
        surface.requests+=1;x=float(atoms.positions[0,0]);q=qresult(x,10.-x)
        return SimpleNamespace(initial=q,minima=(q,),records=(),evaluation_requests=1)
    def propose(parents,energies,groups,change_types,rng,**kwargs):
        seen.append(tuple(change_types))
        child=GeneticCandidate(geometry(3),groups,'mixed_unit',(None,1),
            {'atom_parent_indices':(0,1,0,1,1,1),'source_atom_indices':tuple(range(6))})
        return ProposalResult((child,),'target_reached',1,0)
    monkeypatch.setattr(paper_ga,'quench',relax)
    monkeypatch.setattr(paper_ga,'run_ssw',walk)
    monkeypatch.setattr(paper_ga,'propose_type3',propose)
    result=run([geometry(0),geometry(1),geometry(2)],surface,
        config=configuration(generations=1),change_types=(1,0))
    assert seen==[(1,0)]
    child=next(o for o in result.observations if o.phase=='offspring_quench')
    assert len(child.parent_ids)==6
    assert child.parent_ids[0]==child.parent_ids[2]
    assert child.parent_ids[1]==child.parent_ids[3]==child.parent_ids[4]==child.parent_ids[5]
    assert child.parent_ids[0]!=child.parent_ids[1]


def test_offspring_ssw_refinement_is_best_only_and_forwards_config(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    direct = []
    def direct_quench(atoms, surface, **kwargs):
        direct.append(1)
        surface.requests += 1
        return qresult(0, 5.)
    monkeypatch.setattr(paper_ga, 'quench', direct_quench)
    child = GeneticCandidate(geometry(1), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 0), {})
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs:
                        ProposalResult((child,), 'target_reached', 1, 0))
    calls = []
    def walker(atoms, surface, *, steps, config, ls, height_policy, height_update_budget, rng, gaussian_policy=None):
        calls.append((steps, config, ls, height_policy, height_update_budget))
        if config is not refined:
            surface.requests += 1
            q = qresult(0, 5.)
            return SimpleNamespace(initial=q, minima=(q,), records=(), status="completed")
        surface.requests += 2
        low = qresult(1, 1.)
        high = qresult(2, 3.)
        return SimpleNamespace(initial=high, minima=(high, low), records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walker)
    refined = paper_ga.SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=700., fmax=.01, relax_steps=2, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=.02)
    normal = paper_ga.SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=700., fmax=.01, relax_steps=2, fd_step=1e-4,
        rotation_hvp=2, rotation_tol=.02)
    result = run([geometry(0), geometry(1), geometry(2)], surface,
        config=configuration(quick_steps=0, generations=1, generation_steps=0,
                             fine_steps=0, offspring_steps=1),
        ssw_config=normal, offspring_ssw_config=refined,
        height_policy=ConservativeNativeHeightPolicy(1., 2., 0, 10., 1.1, 1.2),
        height_update_budget=17)
    assert len(direct) == 3 and result.evaluation_requests == 10
    refined_calls = [call for call in calls if call[1] is refined]
    assert refined_calls == [(1, refined, None, calls[0][3], 17)]
    offspring = [o for o in result.observations if o.phase == 'offspring_ssw']
    assert len(offspring) == 2 and result.archive
    high = next(o for o in offspring if o.result.energy == 3.)
    assert high.eligible_for_archive
    assert high.id not in {row['observation_id'] for row in result.archive}
    assert not any(row['energy'] == 3. for row in result.archive)
    stage = next(s for s in result.stages if s.phase == 'offspring_ssw')
    assert stage.evaluation_requests == 2 and stage.details['selected_observation_id'] == min(
        offspring, key=lambda o: o.result.energy).id
    assert min(row['energy'] for row in result.archive) == 1.


def test_offspring_budget_caught_by_walker_stops_later_candidates(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    def evaluate(atoms):
        surface.requests += 1
        return 0., np.zeros((len(atoms), 3))
    surface.evaluate = evaluate
    def paid_quench(atoms, surface, **kwargs):
        surface.evaluate(atoms)
        x = float(atoms.positions[0, 0])
        return qresult(x, 5. + x)
    monkeypatch.setattr(paper_ga, 'quench', paid_quench)
    child = GeneticCandidate(geometry(9), ((0, 1, 2), (3, 4, 5)), 'fixture', (0, 1), {})
    monkeypatch.setattr(paper_ga, 'propose_type3', lambda *args, **kwargs:
                        ProposalResult((child, child), 'target_reached', 1, 0))
    children = []
    def walker(atoms, surface, **kwargs):
        x = float(atoms.positions[0, 0])
        q = paid_quench(atoms, surface)
        if x != 9:
            return SimpleNamespace(initial=q, minima=(q,), records=(), status='completed')
        children.append(x)
        try:
            surface.evaluate(atoms)
        except paper_ga.BudgetExhausted:
            record = SimpleNamespace(status='evaluation_failed', error='cap', landing=None)
            return SimpleNamespace(initial=q, minima=(q,), records=(record,), status='evaluation_failed')
        raise AssertionError('expected budget boundary inside offspring')
    monkeypatch.setattr(paper_ga, 'run_ssw', walker)
    result = run([geometry(0), geometry(1), geometry(2)], surface,
                 config=configuration(quick_steps=0, generations=1, offspring_steps=1),
                 max_evaluations=7)
    assert children == [9]
    assert result.status == 'budget_exhausted'
    assert result.evaluation_requests == surface.requests == 7
    assert {5., 6., 7.}.issubset({row['energy'] for row in result.archive})
    stage = next(s for s in result.stages if s.phase == 'offspring_ssw')
    assert stage.evaluation_requests == 1 and stage.status == 'completed_with_failures'
    assert stage.details['operator'] == 'fixture'
    assert len(stage.details['parent_ids']) == 2
    assert not any(s.phase == 'generation_short' for s in result.stages)


def test_type3_small_ga_candidates_rejected_before_surface_or_descriptor(monkeypatch):
    surface = SimpleNamespace(requests=13)
    monkeypatch.setattr(paper_ga, 'cluster_descriptor',
                        lambda *args: pytest.fail('descriptor must not run'))
    with pytest.raises(ValueError, match='ga_candidates.*4'):
        paper_ga.run_ga_ssw(
            [geometry(0)], surface, groups=((0, 1, 2), (3, 4, 5)),
            references=(0., 1., 2.), descriptor_bonds={(1, 1): 1., (1, 8): 1., (8, 8): 1.},
            descriptor_weights=(1.,) * 6, neighbor_range=1.2,
            proposal_bond_limits={},
            config=configuration(generations=1, ga_candidates=1),
            ssw_config=object(), rng=np.random.default_rng(9))
    assert surface.requests == 13
