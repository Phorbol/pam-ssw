"""Controller accounting tests; fakes test scheduling, not scientific validity."""
from types import SimpleNamespace

import numpy as np
from ase import Atoms

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
                              config=kwargs.pop('config', configuration()), ssw_config=object(),
                              rng=np.random.default_rng(3), **kwargs)


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
    def relax(atoms, surface, **kwargs):
        x = float(atoms.positions[0, 0]); relax_x.append(x)
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
    result = run([geometry(0), geometry(1), geometry(2)], surface, config=configuration(generations=1))
    assert relax_x == [0., 1., 2., 3., 4.]
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


def test_controller_rejects_noncanonical_group_order_before_evaluation():
    import pytest
    surface = SimpleNamespace(requests=0)
    with pytest.raises(ValueError, match='contiguous monomer order'):
        paper_ga.run_ga_ssw([geometry(0)], surface, groups=((3, 4, 5), (0, 1, 2)),
                            references=(0., 1., 2.), descriptor_bonds={}, descriptor_weights=(1.,) * 6,
                            neighbor_range=1., proposal_bond_limits={}, config=configuration(),
                            ssw_config=object(), rng=np.random.default_rng(1))
    assert surface.requests == 0
