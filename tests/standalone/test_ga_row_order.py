"""Row identity and real EMT lifecycle checks, not search-efficiency evidence."""
from copy import deepcopy
import pickle

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone import paper_ga
from pamssw.standalone.ga_checkpoint import GACheckpoint
from pamssw.standalone.surface import ASESurface


def inputs(element='Cu'):
    atoms = Icosahedron(element, 2)
    atoms.positions[1] += [.12, -.08, .03]
    bonds = {(int(atoms.numbers[0]), int(atoms.numbers[0])): 2.8}
    refs = [paper_ga.cluster_descriptor(atoms.numbers, atoms.positions * scale, bonds, 1.2)
            for scale in (.95, 1., 1.05)]
    config = paper_ga.PaperGAConfig(
        quick_steps=1, generations=0, generation_steps=1, fine_steps=1,
        ga_candidates=4, regions=1, fine_regions=1, quench_fmax=.03,
        quench_steps=100, proposal_max_batches=1, proposal_max_cut_attempts=30,
        proposal_max_pair_attempts=40, partition_max_draws=100,
        projection_tolerance=1e-8, energy_window=100.)
    ssw = paper_ga.SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=0., fmax=.1, relax_steps=1, fd_step=.01,
        rotation_hvp=2, rotation_tol=.1, direction_sampling='global')
    return dict(initial=[atoms], groups=(tuple(range(13)),), references=refs,
        descriptor_bonds=bonds, descriptor_weights=(1.,)*6, neighbor_range=1.2,
        proposal_bond_limits={}, config=config, ssw_config=ssw, max_evaluations=2000)


def run(kwargs, surface=None, **extra):
    return paper_ga.run_ga_ssw(**kwargs, surface=surface or ASESurface(EMT()),
                              rng=np.random.default_rng(317), **extra)


@pytest.mark.parametrize('element', ['Cu', 'Al'])
def test_full_mode_real_lifecycle_resume_and_cross_mode_rejection(element, tmp_path):
    kwargs = inputs(element)
    references_before = deepcopy(kwargs['references'])
    full = run(kwargs, descriptor_row_order='full_fingerprint')
    partial = run(kwargs, descriptor_row_order='full_fingerprint',
                  checkpoint_callback=lambda state: state.phase == 'quick_complete')
    assert partial.status == 'checkpoint_boundary'
    path = tmp_path / 'ga.chk'
    partial.checkpoint.save(path)
    checkpoint = GACheckpoint.load(path)
    resumed = run(kwargs, descriptor_row_order='full_fingerprint', checkpoint=checkpoint)
    assert full.status == resumed.status
    assert full.evaluation_requests == resumed.evaluation_requests
    assert full.archive and resumed.archive
    from pamssw.standalone.legacy_descriptor import _full_fingerprint_order
    ordered_refs = [_full_fingerprint_order(ref) for ref in kwargs['references']]
    for observation in full.observations:
        atoms = observation.result.atoms
        descriptor = _full_fingerprint_order(paper_ga.cluster_descriptor(
            atoms.numbers, atoms.positions, kwargs['descriptor_bonds'], kwargs['neighbor_range']))
        expected = tuple(paper_ga.descriptor_similarity(descriptor, ref, kwargs['descriptor_weights'])
                         for ref in ordered_refs)
        assert observation.projection == expected
    assert [a['sims'] for a in full.archive] == [a['sims'] for a in resumed.archive]
    assert [a['energy'] for a in full.archive] == [a['energy'] for a in resumed.archive]
    assert kwargs['references'] == references_before
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='contract'):
        run(kwargs, surface, checkpoint=checkpoint)
    assert surface.requests == 0
    legacy = run(kwargs, checkpoint_callback=lambda state: True)
    explicit = run(kwargs, descriptor_row_order='legacy_counts', checkpoint_callback=lambda state: True)
    assert pickle.dumps(legacy.checkpoint.contract) == pickle.dumps(explicit.checkpoint.contract)
    assert 'descriptor_row_order' not in legacy.checkpoint.contract
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='contract'):
        run(kwargs, surface, descriptor_row_order='full_fingerprint', checkpoint=legacy.checkpoint)
    assert surface.requests == 0


def test_invalid_order_rejected_without_calculator_calls():
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='descriptor_row_order'):
        run(inputs(), surface, descriptor_row_order='unknown')
    assert surface.requests == 0


def test_full_fingerprint_ties_and_reference_copy():
    from pamssw.standalone.legacy_descriptor import _full_fingerprint_order
    descriptor = dict(n1=[[1], [1]], n2=[[2], [2]], n3=[[3], [3]],
                      d1=[[.8], [.2]], d2=[.5, .6], d3=[.7, .9])
    before = deepcopy(descriptor)
    result = _full_fingerprint_order(descriptor)
    swapped = {key: list(reversed(value)) for key, value in descriptor.items()}
    assert result == _full_fingerprint_order(swapped)
    assert descriptor == before
    result['n1'][0][0] = 999
    assert descriptor == before
