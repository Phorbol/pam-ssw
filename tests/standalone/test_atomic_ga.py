"""TYPE0 geometry contracts; these tests do not establish GO efficiency."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone.atomic_ga import cut_atoms, build_atomic_pool, cross_atomic_pool
from pamssw.standalone.ga_operators import SamplingExhausted


class DrawStream:
    def __init__(self, values):
        self.values = iter(values)

    def random(self, size=None):
        if size is None:
            return next(self.values)
        return np.array([next(self.values) for _ in range(size)])


def test_original_jar_cut_fixture():
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'type0_cut.json').read_text())
    for case in fixture['cases']:
        atoms = Atoms(numbers=case['numbers'], positions=case['positions'])
        old = atoms.positions.copy()
        cut = cut_atoms(atoms, DrawStream(case['draws']), max_attempts=100)
        assert cut.plane_slope == pytest.approx(case['slope'], abs=1.e-15)
        for kind in ('son', 'daughter'):
            fragment = getattr(cut, kind).atoms
            assert fragment.numbers.tolist() == case[kind]['numbers']
            np.testing.assert_allclose(fragment.positions, case[kind]['positions'], atol=2.e-12, rtol=0.)
        np.testing.assert_array_equal(atoms.positions, old)


def parents():
    atoms = Icosahedron('Cu', 2)
    atoms.numbers[::2] = 47
    copies = [atoms.copy() for _ in range(3)]
    copies[1].positions *= 1.02
    copies[2].positions *= .98
    return copies


def test_alloy_pool_complete_candidates_and_no_aliasing():
    inputs = parents()
    originals = [a.positions.copy() for a in inputs]
    pool = build_atomic_pool(inputs, [-2., -1., 0.], np.random.default_rng(44),
                             max_cut_attempts=100, cuts_per_parent_slot=20)
    assert len(pool.sons) == len(pool.daughters) == 60
    assert len(pool.parent_slots) == 3
    assert pool.cut_attempts >= 60
    rng = np.random.default_rng(46)
    for _ in range(8):
        child = cross_atomic_pool(pool, rng, max_pair_attempts=200)
        assert sorted(child.atoms.numbers) == sorted(inputs[0].numbers)
        assert len(child.parent_indices) == len(child.source_atom_indices) == 13
        assert child.pair_attempts >= 1
        assert child.atoms.calc is None
        assert np.isfinite(child.atoms.positions).all()
        assert child.atoms.pbc.any() == False
        child.atoms.positions[:] = 100.
    for a, old in zip(inputs, originals):
        np.testing.assert_array_equal(a.positions, old)
    assert not np.all(pool.sons[0].atoms.positions == 100.)


def test_replay_and_native_pool_size():
    a = Atoms('Cu4', positions=[[1, 1, 1], [1,-1,-1], [-1,1,-1], [-1,-1,1]])
    p = [a.copy() for _ in range(3)]
    pools = [build_atomic_pool(p, [0, 1, 2], np.random.default_rng(1), max_cut_attempts=100)
             for _ in range(2)]
    assert len(pools[0].sons) == 300  # 3 parent slots * 10^(one element+1)
    outputs = [cross_atomic_pool(pool, np.random.default_rng(3), max_pair_attempts=20)
               for pool in pools]
    np.testing.assert_array_equal(outputs[0].atoms.positions, outputs[1].atoms.positions)


def test_invalid_domains_and_cut_exhaustion():
    p = parents()
    with pytest.raises(ValueError, match='composition'):
        wrong = p[0].copy()
        wrong.numbers[0] = 6
        build_atomic_pool([*p[:2], wrong], [0, 1, 2], np.random.default_rng(1), max_cut_attempts=10)
    with pytest.raises(ValueError, match='Compete'):
        build_atomic_pool(p, [0, 0, 0], np.random.default_rng(1), max_cut_attempts=10)
    for field in ('pbc', 'constraints'):
        a = p[0].copy()
        if field == 'pbc':
            a.pbc = True
        else:
            from ase.constraints import FixAtoms
            a.set_constraint(FixAtoms(indices=[0]))
        with pytest.raises(ValueError):
            cut_atoms(a, np.random.default_rng(1), max_attempts=5)
    with pytest.raises(SamplingExhausted):
        cut_atoms(Atoms('Cu4', positions=np.zeros((4, 3))), np.random.default_rng(1), max_attempts=2)


class JavaRandom:
    """Test-only java.util.Random 48-bit generator, to replay original fixture."""
    def __init__(self, seed):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def bits(self, n):
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.state >> (48 - n)

    def random(self, size=None):
        if size is not None:
            return np.array([self.random() for _ in range(size)])
        return ((self.bits(26) << 27) + self.bits(27)) / float(1 << 53)


def test_original_jar_full_cross_fixture():
    case = json.loads((Path(__file__).parent / 'fixtures' / 'type0_cross.json').read_text())
    check = JavaRandom(case['seed'])
    np.testing.assert_array_equal(check.random(len(case['draws'])), case['draws'])
    inputs = [Atoms(numbers=case['numbers'], positions=np.array(case['positions']) * scale)
              for scale in case['scales']]
    rng = JavaRandom(case['seed'])
    pool = build_atomic_pool(inputs, case['energies'], rng, max_cut_attempts=100)
    assert len(pool.sons) == 3000
    for expected in case['children']:
        child = cross_atomic_pool(pool, rng, max_pair_attempts=1000)
        assert child.atoms.numbers.tolist() == expected['numbers']
        np.testing.assert_allclose(child.atoms.positions, expected['positions'], atol=2.e-12, rtol=0.)


def test_pair_budget_exhaustion_and_singular_cut():
    from pamssw.standalone.atomic_ga import AtomicPool, AtomicGamete
    son = AtomicGamete(Atoms('Cu', positions=[[0, 0, .3]]), 0, (0,))
    daughter = AtomicGamete(Atoms('Cu', positions=[[0, 0, -.3]]), 1, (0,))
    pool = AtomicPool((son,), (daughter,), (29, 47), (0, 1), 1, 2)
    with pytest.raises(SamplingExhausted, match='3 attempts'):
        cross_atomic_pool(pool, np.random.default_rng(1), max_pair_attempts=3)
    with pytest.raises(SamplingExhausted, match='singular'):
        cut_atoms(Atoms('Cu2', positions=[[0, 1, 0], [0, -1, 0]]),
                  DrawStream([.5, .5, .5, .5]), max_attempts=1)
