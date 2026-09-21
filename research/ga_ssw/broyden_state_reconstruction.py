"""Research-only numerical reconstruction of the BRZERO4 history state.

This is a raw fixed-G0 history action, not a CBD/rotation driver.  The caller
owns direction normalization, rotation-specific control, and all PES evaluation.
This state implements prefix trimming and minimal-history spectral reset for
the tested raw branch; it does not implement the separate force-step angle gate.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg.lapack import dggev

from .broyden_history_reconstruction import HistoryAction, reconstruct_history_action


@dataclass(frozen=True)
class BroydenUpdate:
    action: HistoryAction
    dropped: int
    spectral_max: float
    restarted: bool = False

    @property
    def x(self):
        return self.action.x


class BroydenState:
    """Stateful fixed-G0 Broyden history arithmetic for research comparisons."""

    def __init__(self, g0, *, weight, metric, history_limit, spectral_limit):
        self.g0 = self._vector(g0, "g0")
        self.weight = self._finite_scalar(weight, "weight")
        if metric not in {"native_block_sum", "euclidean"}:
            raise ValueError("metric must be 'native_block_sum' or 'euclidean'")
        self.metric = metric
        if isinstance(history_limit, bool) or not isinstance(history_limit, (int, np.integer)):
            raise ValueError("history_limit must be a positive integer")
        if history_limit <= 0:
            raise ValueError("history_limit must be a positive integer")
        self.history_limit = int(history_limit)
        self.spectral_limit = self._finite_scalar(spectral_limit, "spectral_limit")
        if self.spectral_limit <= 0:
            raise ValueError("spectral_limit must be positive")
        self.x_previous = None
        self.force_previous = None
        self.df = np.empty((self.g0.size, 0), dtype=float)
        self.u = np.empty_like(self.df)
        self.z = np.empty_like(self.df)
        self.last_update = None

    @staticmethod
    def _finite_scalar(value, name):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError(f"{name} must be a finite scalar")
        return float(array)

    @staticmethod
    def _vector(value, name, size=None):
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a nonempty finite vector")
        if size is not None and array.size != size:
            raise ValueError(f"{name} has incompatible size")
        return array.copy()

    def _spectral_max(self, df, u):
        c = df.T @ u
        gp = np.triu(c, 1) - np.tril(c)
        sp = df.T @ (self.g0[:, None] * df)
        alphar, alphai, beta, _vl, _vr, _work, info = dggev(gp, sp, compute_vl=0, compute_vr=0)
        if info != 0:
            raise RuntimeError(f"DGGEV failure outside recovered branch: info={info}")
        ratios = np.empty_like(alphar, dtype=complex)
        nonzero = beta != 0.0
        ratios[nonzero] = 1.0 + (alphar[nonzero] + 1j * alphai[nonzero]) / beta[nonzero]
        ratios[~nonzero] = self.spectral_limit
        return float(np.max(np.abs(ratios))) if ratios.size else np.inf

    def step(self, x, force):
        """Advance one raw Broyden step and return its numerical update."""
        x = self._vector(x, "x", self.g0.size)
        force = self._vector(force, "force", self.g0.size)
        if self.x_previous is None:
            self.x_previous = x.copy()
            self.force_previous = force.copy()
            action = HistoryAction(
                a=np.empty((0, 0)), beta=np.empty((0, 0)), z=np.empty((self.g0.size, 0)),
                x=x + self.g0 * force, residual=0.0,
            )
            self.last_update = BroydenUpdate(action, 0, 0.0)
            return self.last_update

        delta_force = force - self.force_previous
        if self.metric == "native_block_sum":
            if delta_force.size % 3:
                raise ValueError("native_block_sum requires ndim divisible by 3")
            blocks = delta_force.reshape(-1, 3).sum(axis=1)
            normalizer = float(np.sqrt(np.dot(blocks, blocks)))
        else:
            normalizer = float(np.linalg.norm(delta_force))
        if normalizer == 0.0 or not np.isfinite(normalizer):
            raise ValueError("zero or nonfinite force-difference normalizer")
        new_df = delta_force / normalizer
        new_u = self.g0 * new_df + (x - self.x_previous) / normalizer

        old_m = self.df.shape[1]
        spectral_max = np.inf
        for dropped in range(old_m + 1):
            candidate_df = np.column_stack((self.df[:, dropped:], new_df))
            candidate_u = np.column_stack((self.u[:, dropped:], new_u))
            if candidate_df.shape[1] >= self.history_limit:
                continue
            spectral_max = self._spectral_max(candidate_df, candidate_u)
            if not spectral_max < self.spectral_limit:
                continue
            old_z = self.z[:, dropped:]
            candidate_z_old = np.column_stack((old_z, np.zeros(self.g0.size)))
            action = reconstruct_history_action(
                candidate_df, candidate_u, candidate_z_old, x, force, self.g0,
                weight=self.weight, metric=self.metric,
            )
            self.df = candidate_df
            self.u = candidate_u
            self.z = action.z.copy()
            self.x_previous = x.copy()
            self.force_previous = force.copy()
            self.last_update = BroydenUpdate(action, dropped, spectral_max)
            return self.last_update
        # Original raw branch: failed final prefix sets INI=-1 and returns
        # through initialization in this same call (0x6ff8b1 -> 0x6f8d41).
        self.df = np.empty((self.g0.size, 0))
        self.u = np.empty_like(self.df)
        self.z = np.empty_like(self.df)
        self.x_previous = x.copy()
        self.force_previous = force.copy()
        action = HistoryAction(a=np.empty((0, 0)), beta=np.empty((0, 0)),
                               z=self.z.copy(), x=x + self.g0 * force, residual=0.)
        self.last_update = BroydenUpdate(action, old_m + 1, spectral_max, restarted=True)
        return self.last_update

    @property
    def history_size(self):
        return self.df.shape[1]
