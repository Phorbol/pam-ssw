"""Controller lifecycle tests for fine feedback cycles."""
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone import paper_ga
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.surface import ASESurface
from test_paper_ga import configuration, descriptor_fakes, geometry


def test_fine_minimum_enters_next_cycle_parent_pool(monkeypatch):
    from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
    descriptor_fakes(monkeypatch)
    surface = SimpleNamespace(requests=0)
    monkeypatch.setattr(paper_ga, 'quench',
                        lambda atoms, surface, **kwargs: QuenchResult(
                            atoms.copy(), 5., 0., True, 0, 1, 'true'))
    monkeypatch.setattr(paper_ga, 'partition', lambda rows, *args, **kwargs: [[0]])
    proposal_energies = []
    def propose(parents, energies, *args, **kwargs):
        proposal_energies.append(tuple(energies))
        return ProposalResult((GeneticCandidate(geometry(1.), ((0, 1, 2), (3, 4, 5)),
                                                'fixture', (0, 0), {}),),
                              'target_reached', 1, 0)
    monkeypatch.setattr(paper_ga, 'propose_type3', propose)
    def walk(atoms, surface, *, steps, **kwargs):
        if not steps:
            return SimpleNamespace(minima=(), records=(), status='completed')
        return SimpleNamespace(minima=(QuenchResult(geometry(2.), -5., 0., True, 0, 1, 'true'),),
                               records=(), status='completed')
    monkeypatch.setattr(paper_ga, 'run_ssw', walk)
    result = paper_ga.run_ga_ssw(
        [geometry(0.)], surface, groups=((0, 1, 2), (3, 4, 5)), references=(0., 1., 2.),
        descriptor_bonds={}, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=configuration(quick_steps=0, generations=1,
        generation_steps=0, fine_steps=1, cycles=2), ssw_config=object(),
        rng=np.random.default_rng(3))
    assert len(proposal_energies) == 2
    assert proposal_energies[0] == (5.,)
    assert proposal_energies[1] == (-5.,)
    assert [stage.cycle for stage in result.stages if stage.phase == 'fine'] == [0, 1]


@pytest.mark.parametrize('element', ['Cu', 'Al'])
def test_emt_13_atom_two_cycle_fine_lifecycle(element):
    atoms = Icosahedron(element, 2)
    z = int(atoms.numbers[0])
    bonds = {(z, z): 2.8}
    reference = paper_ga.cluster_descriptor(atoms.numbers, atoms.positions, bonds, 1.2)
    from pamssw.standalone.paper_reference import SSWConfig
    config = configuration(quick_steps=0, generations=0, generation_steps=0,
                           fine_steps=1, quench_steps=100, cycles=2)
    ssw = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
                    temperature_K=0., fmax=.1, relax_steps=1, fd_step=.01,
                    rotation_hvp=2, rotation_tol=.1, direction_sampling='global')
    surface = ASESurface(EMT())
    result = paper_ga.run_ga_ssw(
        [atoms], surface, groups=(tuple(range(13)),), references=(reference,) * 3,
        descriptor_bonds=bonds, descriptor_weights=(1.,) * 6, neighbor_range=1.2,
        proposal_bond_limits={}, config=config, ssw_config=ssw,
        rng=np.random.default_rng(1))
    assert result.evaluation_requests == surface.requests
    assert [stage.cycle for stage in result.stages if stage.phase == 'fine'] == [0, 1]
