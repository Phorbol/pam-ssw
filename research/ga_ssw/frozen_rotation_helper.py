"""Shared construction of the fixed-frame, frozen-LS rotation callback."""
import numpy as np
from pamssw.standalone.cluster_frame import ClusterFrame


def make_frozen_rotation_evaluator(center, anchor, softening, surface):
    """Return (center, raw_anchor, projected_anchor, callback) for rotation.

    The callback applies ClusterFrame positions, evaluates the physical cached
    surface plus the frozen LS term, then projects the complete force. It never
    adds Gaussian terms and never calls a calculator itself.
    """
    center = center.copy()
    frame = ClusterFrame(center)
    raw_anchor = np.asarray(anchor, dtype=float).copy()
    projected_anchor = frame.project(raw_anchor)
    projected_anchor /= np.linalg.norm(projected_anchor)

    def evaluate(candidate):
        candidate = candidate.copy()
        candidate.positions = frame.positions(candidate.positions)
        energy, forces = surface.evaluate(candidate)
        de, df = softening.evaluate(candidate)
        return float(energy + de), frame.project(np.asarray(forces) + np.asarray(df))

    return center, raw_anchor, projected_anchor, evaluate
