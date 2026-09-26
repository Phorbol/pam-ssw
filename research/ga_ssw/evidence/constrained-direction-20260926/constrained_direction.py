"""Full-geometry direction memory on a fixed active Cartesian chart."""
import numpy as np
from .recovered_direction import RecoveredDirectionController, RecoveredDirectionSettings


class ConstrainedDirectionLifecycle:
    """Lift chart positions for selection; restrict generated vectors for CBD.

    Fixed atoms remain geometric references. The controller's active projector
    replaces free-cluster rigid-motion removal; no auxiliary PES is introduced.
    """
    def __init__(self, settings, active_mask):
        if not isinstance(settings, RecoveredDirectionSettings):
            raise TypeError('recovered_direction must be RecoveredDirectionSettings')
        self.rotation_settings = settings
        self.controller = RecoveredDirectionController(settings, active_mask=active_mask)
        self.initialized = False

    @staticmethod
    def clean(atoms):
        clean = atoms.copy()
        clean.set_constraint()
        clean.calc = None
        return clean

    def initialize(self, source, minimum, rng):
        self.controller.initialize(self.clean(source), self.clean(minimum), rng)
        self.initialized = True

    def restore(self, state):
        self.controller.restore_checkpoint_state(state)
        self.initialized = True

    def checkpoint_state(self):
        return self.controller.checkpoint_state() if self.initialized else None

    def propose(self, current, reduced, work, *, first, rng):
        atoms = self.clean(reduced.atoms(work))
        result = (self.controller.begin_escape(self.clean(current), atoms, rng)
                  if first else self.controller.update_direction(atoms, rng))
        active = reduced.chart.active_indices
        vector = np.asarray(result.direction)[active].ravel().copy()
        return vector, result.release_all, self.controller.diagnostics

    def save_center(self, reduced, work):
        self.controller.save_gaussian_center(reduced.atoms(work))

    def observe(self, landing, rng):
        self.controller.observe_landing(self.clean(landing), rng)
