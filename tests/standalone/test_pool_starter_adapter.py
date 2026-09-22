from types import SimpleNamespace as NS
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter
from pamssw.standalone.starter_selection import StarterObservation, StarterPoolSnapshot


def observation(index, x, energy=0.):
    return StarterObservation(index, Atoms('H', positions=[[x, 0, 0]]), energy, 0.)


def minimum(o, requests=1):
    return NS(atoms=o.atoms, energy=o.energy, max_force=0., converged=True,
              evaluation_requests=requests)


def adapter(mode='uniform'):
    return PoolStarterAdapter(mode=mode, energy_tol=.001, rmsd_tol=.1)


class Choose:
    def __init__(self, value): self.value=value; self.sizes=[]
    def integers(self, size): self.sizes.append(size); return self.value


def test_rejected_new_landing_enters_pool_immediately():
    a=adapter(); rng=Choose(1)
    initial, landing=observation(0,0), observation(1,1,-1)
    assert a(StarterPoolSnapshot((initial,landing),0,1,0,3),rng)==1
    assert len(a.archive.entries)==2 and a.mapping==[0,1]
    assert a.archive.entries[0].node_trials==1
    assert a.archive.entries[1].node_trials==0


def test_uniform_is_over_entries_and_current_duplicate_preserves_memory():
    a=adapter(); rng=Choose(0)
    items=(observation(0,0),observation(1,0),observation(2,1))
    assert a(StarterPoolSnapshot(items[:2],1,1,0,3),rng) is None
    rng.value=1
    assert a(StarterPoolSnapshot(items,2,2,1,5),rng) is None
    assert rng.sizes==[1,2] and a.mapping==[0,0,1]
    assert a.decisions[-1]['actual_index']==2


def test_pam_calls_existing_selector_mixture(monkeypatch):
    a=adapter('pam'); calls=[]
    def select(self, archive, rng):
        calls.append(archive); return archive.entries[0]
    monkeypatch.setattr(type(a.selector),'select',select)
    a(StarterPoolSnapshot((observation(0,0),observation(1,1)),1,1,0,3),np.random.default_rng(1))
    assert calls==[a.archive]


def test_finalize_charges_failed_terminal_without_duplicate_or_phantom_trial():
    a=adapter(); initial, landing=observation(0,0),observation(1,1,-1)
    choice=a(StarterPoolSnapshot((initial,landing),0,1,0,3),Choose(1))
    records=(NS(landing=minimum(landing),accepted=False,status='gaussian_limit',evaluation_requests=2,
                starter_selection={'chosen_index':choice}),
             NS(landing=None,accepted=False,status='evaluation_failed',evaluation_requests=4,starter_selection=None))
    r=NS(initial=minimum(initial),minima=(minimum(initial),minimum(landing)),records=records,evaluation_requests=7)
    report=a.finalize(r)
    assert report['total_requests']==7 and report['outer_requests']==6
    assert report['qualified_discoveries']==1 and report['duplicates']==0
    assert [e['node_trials'] for e in report['entries']]==[1,1]
    assert [e['node_duplicate_failures'] for e in report['entries']]==[0,0]
    assert report['attempts'][1]['source_entry']==1


def test_finalize_without_any_callback_and_cost_error():
    o=observation(0,0); a=adapter()
    r=NS(initial=minimum(o),minima=(minimum(o),),records=(),evaluation_requests=2)
    with pytest.raises(ValueError,match='costs'): a.finalize(r)
    r.evaluation_requests=1
    assert a.finalize(r)['entries'][0]['node_trials']==0


def test_constraints_fail_explicitly():
    o=observation(0,0);o.atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError,match='constrained'):
        adapter()(StarterPoolSnapshot((o,),0,None,0,1),Choose(0))


def test_terminal_qualified_landing_before_first_callback_keeps_parent():
    a=adapter(); initial=observation(0,0); landing=observation(1,1,-1)
    record=NS(landing=minimum(landing), accepted=False,
              status='direction_selection_failed', evaluation_requests=2,
              starter_selection=None)
    report=a.finalize(NS(initial=minimum(initial), minima=(minimum(initial),minimum(landing)),
                         records=(record,), evaluation_requests=3))
    assert a.archive.entries[1].parent_id==0
    assert report['qualified_discoveries']==1
    assert [e['node_trials'] for e in report['entries']]==[1,0]


def test_failed_step_between_callbacks_still_charges_actual_source():
    a=adapter(); items=tuple(observation(i,i,-i) for i in range(3))
    first=a(StarterPoolSnapshot(items[:2],0,1,0,3),Choose(1))
    second=a(StarterPoolSnapshot(items,1,2,2,9),Choose(1))
    assert second is None
    assert [e.node_trials for e in a.archive.entries]==[1,2,0]
    records=(NS(landing=minimum(items[1]),accepted=False,status='gaussian_limit',evaluation_requests=2,starter_selection={'chosen_index':first}),
             NS(landing=None,accepted=False,status='quench_failed',evaluation_requests=3,starter_selection=None),
             NS(landing=minimum(items[2]),accepted=False,status='gaussian_limit',evaluation_requests=3,starter_selection={'chosen_index':second}))
    report=a.finalize(NS(initial=minimum(items[0]),minima=tuple(minimum(o) for o in items),records=records,evaluation_requests=9))
    assert [e['node_trials'] for e in report['entries']]==[1,2,0]
    assert report['qualified_discoveries']==2 and report['duplicates']==0
    assert [e['source_entry'] for e in report['attempts']]==[0,1,1]


def test_finalize_does_not_credit_uncommitted_ls_restart_request():
    a=adapter(); initial=observation(0,0); landing=observation(1,1,-1)
    records=(NS(landing=minimum(landing), accepted=False,
                status='starter_selection_failed', evaluation_requests=2,
                starter_selection={'chosen_index':1, 'mc_current_index':0,
                                   'restarted':False, 'ls_reinitialized':False}),
             NS(landing=None, accepted=False, status='evaluation_failed',
                evaluation_requests=3, starter_selection=None))
    report=a.finalize(NS(initial=minimum(initial),
                         minima=(minimum(initial), minimum(landing)),
                         records=records, evaluation_requests=6))
    assert [attempt['source_entry'] for attempt in report['attempts']]==[0,0]
    assert [entry['node_trials'] for entry in report['entries']]==[2,0]
    assert report['attempts'][0]['status']=='starter_selection_failed'
