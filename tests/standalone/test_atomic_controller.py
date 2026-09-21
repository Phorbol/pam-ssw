"""TYPE0 controller wiring; real-system efficacy is assessed separately."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone import paper_ga
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.ga_operators import GeneticCandidate, ProposalResult
from test_paper_ga import configuration


@pytest.mark.parametrize("cross_region", [False, True])
def test_atomic_controller_composition_lineage_and_safe_quench(monkeypatch, cross_region):
    initial=[Atoms(numbers=z,positions=[[i+j,0,0] for j in range(6)])
             for i,z in enumerate(([29,29,47,29,29,47],[47,29,29,47,29,29],[29,47,29,29,47,29]))]
    surface=SimpleNamespace(requests=0);backends=[];proposal_calls=[]
    monkeypatch.setattr(paper_ga,'cluster_descriptor',lambda n,p,*a:float(p[0,0]))
    monkeypatch.setattr(paper_ga,'descriptor_similarity',lambda d,r,w:d)
    def quench(a,s,**kw):
        backends.append(kw['optimizer']);s.requests+=1
        return QuenchResult(a.copy(),10.-a.positions[0,0],0.,True,1,1,'true')
    monkeypatch.setattr(paper_ga,'quench',quench)
    def walk(a,s,**kw):
        q=quench(a,s,optimizer='safe-lbfgs-total')
        return SimpleNamespace(minima=(q,),records=(),status='completed')
    monkeypatch.setattr(paper_ga,'run_ssw',walk)
    if cross_region:
        calls = []
        def partition(rows, *args, **kwargs):
            calls.append(tuple(row["id"] for row in rows))
            return [[2, 0], [1]] if len(calls) == 1 else [list(range(len(rows)))]
        monkeypatch.setattr(paper_ga, "partition", partition)
    def propose(parents,energies,rng,**kw):
        proposal_calls.append(kw)
        if cross_region:
            assert [a.positions[0, 0] for a in parents] == [0., 2., 1.]
            assert kw["parent_regions"] == [(0, 1), (2,)]
        a=parents[0].copy();a.positions+=5.
        return ProposalResult((GeneticCandidate(a,tuple((i,) for i in range(6)),
            'atomic_crossover',(2,0,1,2,0,1),{}),),'target_reached',1,0)
    monkeypatch.setattr(paper_ga,'propose_type0',propose)
    result=paper_ga.run_ga_ssw(initial,surface,groups=None,references=(0,1,2),
        descriptor_bonds={},descriptor_weights=(1.,)*6,neighbor_range=1.2,
        proposal_bond_limits={},config=replace(configuration(generations=1),proposal_type=0),
        ssw_config=SimpleNamespace(quench_optimizer='safe-lbfgs-total'),rng=np.random.default_rng(3))
    assert result.status=='completed'
    assert set(backends)=={'safe-lbfgs-total'}
    assert result.evaluation_requests==surface.requests
    assert len(proposal_calls)==1
    assert sorted(i for region in proposal_calls[0]['parent_regions'] for i in region)==[0,1,2]
    offspring=[o for o in result.observations if o.phase=='offspring_quench']
    assert len(offspring)==1 and len(offspring[0].parent_ids)==6
    if cross_region:
        assert offspring[0].parent_ids == (1, 0, 2, 1, 0, 2)


def test_atomic_controller_config_validation():
    with pytest.raises(ValueError,match='proposal_type'):
        replace(configuration(),proposal_type=2)
    with pytest.raises(ValueError,match='insertion'):
        replace(configuration(),proposal_max_insertion_attempts=0)


@pytest.mark.parametrize("invalid", [0.0, 3.0, False, np.bool_(False), "0"])
def test_proposal_type_requires_nonboolean_integer(invalid):
    with pytest.raises(ValueError, match="proposal_type"):
        replace(configuration(), proposal_type=invalid)


@pytest.mark.parametrize("numbers", [[29]*10, [29]*4+[47], [29]*6+[47]*5])
def test_invalid_type0_input_fails_before_surface_or_descriptor(monkeypatch, numbers):
    surface = SimpleNamespace(requests=17)
    # Last case checks composition mismatch, with valid size.
    initial = [Atoms(numbers=numbers, positions=np.zeros((len(numbers), 3)))]
    if len(numbers) == 11:
        initial.append(Atoms('Cu11', positions=np.zeros((11, 3))))
    def unexpected(*args, **kwargs):
        pytest.fail("invalid TYPE0 input must fail before descriptor or E/F work")
    monkeypatch.setattr(paper_ga, "cluster_descriptor", unexpected)
    monkeypatch.setattr(paper_ga, "quench", unexpected)
    with pytest.raises(ValueError, match="full mutation|composition"):
        paper_ga.run_ga_ssw(initial, surface, groups=None, references=(0,1,2),
            descriptor_bonds={}, descriptor_weights=(1.,)*6, neighbor_range=1.2,
            proposal_bond_limits={}, config=replace(configuration(generations=1), proposal_type=0),
            ssw_config=object(), rng=np.random.default_rng(3))
    assert surface.requests == 17
