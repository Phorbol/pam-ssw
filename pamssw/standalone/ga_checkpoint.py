"""Trusted local persistence for GA boundaries and optional active walks."""
from dataclasses import dataclass
import copy
import os
import pickle
import tempfile


@dataclass
class GAActiveWalk:
    """Selected work and one completed SSW outer-step continuation point."""

    phase: str
    cycle: int
    generation: int
    queue: tuple
    cursor: int
    steps: int
    ssw_checkpoint: object
    walk_start_requests: int
    incomplete_proposal: bool = False


@dataclass
class GACheckpoint:
    """A restartable GA state captured at a completed safe boundary.

    The calculator, surface and user callbacks are deliberately absent.  A
    caller must provide compatible objects again when resuming a checkpoint.
    Pickle is intended for trusted local files only.
    """

    version: int
    phase: str
    cycle: int
    generation: int
    archive: list
    observations: list
    failures: list
    stages: list
    walks: list
    validation_attempted: bool
    rng_state: dict
    config: object
    ssw_config: object
    offspring_ssw_config: object
    evaluation_requests: int
    max_evaluations: int | None
    contract: dict
    active_walk: GAActiveWalk | None = None

    VERSION = 2

    def __post_init__(self):
        if self.version not in (1, self.VERSION):
            raise ValueError(f'unsupported GA checkpoint version: {self.version}')

    def clone(self):
        return copy.deepcopy(self)

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path)) or '.'
        fd, temporary = tempfile.mkstemp(prefix='.ga-checkpoint-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        if not isinstance(state, cls):
            raise TypeError('file does not contain a GACheckpoint')
        if state.version not in (1, cls.VERSION):
            raise ValueError(f'unsupported GA checkpoint version: {state.version}')
        return state
