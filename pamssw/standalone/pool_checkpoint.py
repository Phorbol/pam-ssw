"""Pure-data checkpoint helpers for the opt-in starter-selector state."""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np


def _validate_pure(value, path='pool_state'):
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise ValueError(f'{path} must contain only pure numeric arrays')
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f'{path} must contain string keys')
        for key, item in value.items():
            _validate_pure(item, f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_pure(item, f'{path}[{index}]')
        return
    raise ValueError(f'{path} must contain pure data, not {type(value).__name__}')


def _require_contract(selector):
    method = getattr(selector, 'checkpoint_contract', None)
    if not callable(method):
        raise ValueError('checkpointed starter selector requires checkpoint_contract()')
    contract = method()
    _validate_pure(contract, 'contract')
    if not isinstance(contract, dict):
        raise ValueError('selector checkpoint contract must be a pure data dict')
    return deepcopy(contract)


def _require_state(selector):
    method = getattr(selector, 'export_state', None)
    if not callable(method):
        raise ValueError('checkpointed starter selector requires export_state()')
    state = method()
    _validate_pure(state, 'state')
    return deepcopy(state)


def _require_restore(selector):
    if not callable(getattr(selector, 'restore_state', None)):
        raise ValueError('checkpointed starter selector requires restore_state(payload)')


def build_pool_state(selector, selector_rng, *, current_index, last_landing_index):
    """Export a validated, detached selector state payload."""
    if not isinstance(selector_rng, np.random.Generator):
        raise ValueError('checkpointed starter selector requires selector_rng Generator')
    if not isinstance(current_index, Integral) or isinstance(current_index, (bool, np.bool_)):
        raise ValueError('pool_state current_index must be an integer')
    if (last_landing_index is not None and
            (not isinstance(last_landing_index, Integral) or
             isinstance(last_landing_index, (bool, np.bool_)))):
        raise ValueError('pool_state last_landing_index must be an integer or None')
    payload = {
        'current_index': int(current_index),
        'last_landing_index': None if last_landing_index is None else int(last_landing_index),
        'selector_rng_state': deepcopy(selector_rng.bit_generator.state),
        'contract': _require_contract(selector),
        'state': _require_state(selector),
    }
    _validate_pure(payload)
    return payload


def validate_pool_state(payload, *, selector, selector_rng, observation_count):
    """Validate pool state against a fresh compatible selector before PES work."""
    _validate_pure(payload)
    if not isinstance(payload, dict):
        raise ValueError('pool_state must be a pure data dict')
    required = {'current_index', 'last_landing_index', 'selector_rng_state',
                'contract', 'state'}
    if set(payload) != required:
        raise ValueError('pool_state fields are incomplete or unsupported')
    if (not isinstance(observation_count, Integral) or observation_count < 1):
        raise ValueError('pool_state observation count must be positive')
    current = payload['current_index']
    landing = payload['last_landing_index']
    if (not isinstance(current, Integral) or isinstance(current, (bool, np.bool_)) or
            current < 0 or current >= observation_count):
        raise ValueError('pool_state current_index is outside observations')
    if (landing is not None and
            (not isinstance(landing, Integral) or isinstance(landing, (bool, np.bool_)) or
             landing < 0 or landing >= observation_count)):
        raise ValueError('pool_state last_landing_index is outside observations')
    if not isinstance(selector_rng, np.random.Generator):
        raise ValueError('checkpointed starter selector requires selector_rng Generator')
    rng_state = payload['selector_rng_state']
    if (not isinstance(rng_state, dict) or
            rng_state.get('bit_generator') != selector_rng.bit_generator.__class__.__name__):
        raise ValueError('pool_state selector RNG bit generator does not match')
    try:
        temporary = selector_rng.bit_generator.__class__()
        temporary.state = deepcopy(rng_state)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('pool_state selector RNG state is invalid') from error
    if payload['contract'] != _require_contract(selector):
        raise ValueError('pool_state selector contract does not match')
    return True


def restore_pool_state(selector, selector_rng, payload, *, observation_count):
    """Validate then restore selector and selector RNG without partial mutation."""
    validate_pool_state(payload, selector=selector, selector_rng=selector_rng,
                        observation_count=observation_count)
    _require_restore(selector)
    restore = selector.restore_state
    state = deepcopy(payload['state'])
    restore(state)
    selector_rng.bit_generator.state = deepcopy(payload['selector_rng_state'])
    return int(payload['current_index']), payload['last_landing_index']
