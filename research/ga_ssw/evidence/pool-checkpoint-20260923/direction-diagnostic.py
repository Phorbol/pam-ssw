import sys
sys.path.insert(0, 'tests/standalone')
from pathlib import Path
import pytest
import numpy as np
from test_pool_direction_checkpoint import _run, Selector
with pytest.MonkeyPatch.context() as patch:
    result = _run(Path('/tmp/pool-direction-diagnostic.pkl'), Selector(), np.random.default_rng(23), steps=2, monkeypatch=patch)
    print('status', result.status)
    for record in result.records:
        print('record', record.index, record.status, record.error)
        print('climb', record.climb)
