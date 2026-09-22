import numpy as np
import pytest

from pamssw.standalone.pool_checkpoint import (
    build_pool_state,
    restore_pool_state,
    validate_pool_state,
)


class Selector:
    def __init__(self, contract=None, state=None):
        self.contract = contract or {'identity': 'fixture', 'version': 1,
                                     'config': {'mode': 'uniform'}}
        self.state = state or {'trials': 3}
        self.restored = None

    def checkpoint_contract(self):
        return self.contract

    def export_state(self):
        return self.state

    def restore_state(self, payload):
        self.restored = payload
        self.state = payload


def test_pool_state_is_pure_and_restores_before_mutation():
    selector = Selector()
    rng = np.random.default_rng(4)
    payload = build_pool_state(selector, rng, current_index=1,
                               last_landing_index=2)
    assert payload['current_index'] == 1
    assert payload['last_landing_index'] == 2
    assert payload['contract'] == selector.contract
    validate_pool_state(payload, selector=selector, selector_rng=rng,
                        observation_count=3)
    restored = Selector()
    restore_pool_state(restored, rng, payload, observation_count=3)
    assert restored.restored == selector.state


@pytest.mark.parametrize('bad', [lambda: object(), lambda: {'callback': lambda: None}])
def test_pool_state_rejects_non_pure_selector_data(bad):
    selector = Selector(state=bad()) if callable(bad) else Selector(state=bad)
    with pytest.raises(ValueError, match='pure'):
        build_pool_state(selector, np.random.default_rng(1), current_index=0,
                         last_landing_index=None)


def test_pool_state_validates_contract_rng_and_bounds():
    selector = Selector()
    rng = np.random.default_rng(2)
    payload = build_pool_state(selector, rng, current_index=0,
                               last_landing_index=None)
    with pytest.raises(ValueError, match='current_index'):
        validate_pool_state({**payload, 'current_index': 3}, selector=selector,
                            selector_rng=rng, observation_count=3)
    with pytest.raises(ValueError, match='contract'):
        validate_pool_state(payload, selector=Selector({'identity': 'other'}),
                            selector_rng=rng, observation_count=1)
    with pytest.raises(ValueError, match='bit generator'):
        invalid = {**payload, 'selector_rng_state':
                   {**payload['selector_rng_state'], 'bit_generator': 'MT19937'}}
        validate_pool_state(invalid, selector=selector, selector_rng=rng,
                            observation_count=1)


def test_invalid_rng_state_is_rejected_before_selector_restore():
    selector = Selector()
    payload = build_pool_state(selector, np.random.default_rng(2),
                               current_index=0, last_landing_index=None)
    payload['selector_rng_state']['state']['state'] = 'invalid'
    before = dict(selector.state)
    with pytest.raises(ValueError, match='RNG state'):
        restore_pool_state(selector, np.random.default_rng(2), payload,
                           observation_count=1)
    assert selector.state == before


def test_restore_method_is_required_for_pool_checkpoint():
    class NoRestore:
        def checkpoint_contract(self):
            return {'identity': 'fixture', 'version': 1}

        def export_state(self):
            return {'trials': 0}

    from pamssw.standalone.pool_checkpoint import _require_restore
    with pytest.raises(ValueError, match='restore_state'):
        _require_restore(NoRestore())
