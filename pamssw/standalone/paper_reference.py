"""Independent cluster SSW/LS-SSW, following the published outer algorithm.

SSW: Shang & Liu, JCTC 2013, DOI 10.1021/ct301010b, p1840 steps 1-8.
Height: BP-CBD, DOI 10.1021/ct300250h, p2218 forward-force equation.
LS: DOI 10.1021/acs.jctc.4c01081, eqs 11-15.

Explicit numerical differences: single-sided finite-difference Ritz or dimer rotation
instead of native Broyden; ASE LBFGS; conventional Metropolis without the
release's NSAME trapping schedule. Climb lower-energy exit uses true energy
at the modified minimum (an interpretation of SSW step 5). This module is
a paper-level reference, NOT execution parity with the uploaded release.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import pickle
from pathlib import Path
from copy import deepcopy
import os
import tempfile

import numpy as np
from ase import units
from ase.optimize import LBFGS, LBFGSLineSearch

from .cluster_frame import ClusterFrame, ClusterFrameDomainError
from .direction import paper_biased_direction
from .gaussian import ProjectedGaussian
from .ls_cycle import LSCycleError, prepare_ls_step
from .ls_prequench import LSPrequenchSettings, validate_prequench, normalize_prequench_settings
from .softening import FrozenBondSoftening, LSResponseState, _energy_filter
from .surface import SurfaceCalculator, QuenchResult, quench
from .cluster_reconnection import reconnect_clusters
from .native_mc import NativeMCSettings, NativeMCState, native_metropolis
from .starter_selection import snapshot_from_minima, validate_starter_index


@dataclass(frozen=True)
class SSWConfig:
    width: float                 # Angstrom, Gaussian width and translation.
    rotation_bias: float | None  # eV/Angstrom², fixed rank-one curvature.
    max_gaussians: int           # H, hard climbing budget from SSW step 5.
    temperature_K: float         # MC strategy temperature, not dynamics.
    fmax: float                  # eV/Angstrom, largest atom-force norm.
    relax_steps: int             # Per-quench optimizer iteration budget.
    fd_step: float               # Angstrom, dimer separation.
    rotation_hvp: int            # Finite-difference rotation budget.
    rotation_tol: float          # eV/Angstrom², numerical eigen residual.
    forward_force: float = .1    # eV/Angstrom, BP-CBD 2012 p2218 setting.
    direction_sampling: str = 'paper'  # 'global'/'isotropic' are explicit reference variants.
    rotation_solver: str = 'ritz'  # Explicit numerical alternative: 'dimer'.
    cluster_frame: str = 'cartesian'  # 'eckart'/'direction_only' require free-cluster symmetry.
    quench_optimizer: str = 'ase-lbfgs'  # Optional existing PAM numerical Safe-total backend.

    lbfgs_memory: int | None = None  # Safe-total only; None retains10 pairs, storage O(m*3N).
    bias_stage_steps: int | None = None  # Opt-in cap for biased Safe-total stages only.
    bias_fmax: float | None = None  # Opt-in biased-stage force threshold only.
    pre_rotation_hvp: int | None = None  # Experimental two-stage presweep.
    rotation_exit_policy: str = 'force'  # 'force_or_budget' permits evaluated budget exits.

    def __post_init__(self):
        from pamssw.relax import _validate_lbfgs_memory
        _validate_lbfgs_memory(self.lbfgs_memory, self.quench_optimizer)
        if self.quench_optimizer not in ('ase-lbfgs', 'ase-lbfgs-linesearch', 'scipy-lbfgsb', 'safe-lbfgs-total'):
            raise ValueError('quench_optimizer must be ase-lbfgs, ase-lbfgs-linesearch, scipy-lbfgsb or safe-lbfgs-total')
        if self.quench_optimizer == 'scipy-lbfgsb' and self.cluster_frame == 'eckart':
            raise NotImplementedError('scipy-lbfgsb does not support the eckart quench frame')
        if self.bias_stage_steps is not None:
            if (isinstance(self.bias_stage_steps, (bool, np.bool_)) or
                    not isinstance(self.bias_stage_steps, (int, np.integer)) or
                    self.bias_stage_steps <= 0):
                raise ValueError('bias_stage_steps must be a positive integer or None')
            if self.quench_optimizer != 'safe-lbfgs-total':
                raise ValueError('bias_stage_steps requires safe-lbfgs-total')
        if self.bias_fmax is not None and (not np.isfinite(self.bias_fmax) or self.bias_fmax <= 0):
            raise ValueError('bias_fmax must be positive and finite when set')
        if self.quench_optimizer == 'safe-lbfgs-total' and self.cluster_frame == 'eckart':
            raise NotImplementedError('safe-lbfgs-total does not support the eckart quench frame')
        if self.cluster_frame not in ('cartesian', 'eckart', 'direction_only', 'translation_only'):
            raise ValueError('cluster_frame must be cartesian, eckart, direction_only or translation_only')
        if self.rotation_solver not in ('ritz', 'dimer', 'broyden-euclidean'):
            raise ValueError('rotation_solver must be ritz, dimer or broyden-euclidean')
        if self.rotation_exit_policy not in ('force', 'force_or_budget'):
            raise ValueError('rotation_exit_policy must be force or force_or_budget')
        if self.direction_sampling not in ('paper', 'global', 'isotropic'):
            raise ValueError('direction_sampling must be paper, global or isotropic')
        if self.pre_rotation_hvp is not None:
            if (isinstance(self.pre_rotation_hvp, (bool, np.bool_)) or
                    not isinstance(self.pre_rotation_hvp, (int, np.integer)) or
                    self.pre_rotation_hvp < 1):
                raise ValueError('pre_rotation_hvp must be a positive integer or None')
            if self.rotation_bias is not None:
                raise ValueError('pre_rotation_hvp requires rotation_bias=None')
        elif self.rotation_bias is None:
            raise ValueError('rotation_bias=None requires pre_rotation_hvp')
        for key in ('width', 'fmax', 'fd_step',
                    'rotation_tol', 'forward_force'):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.rotation_bias is not None and (not np.isfinite(self.rotation_bias) or self.rotation_bias <= 0):
            raise ValueError('rotation_bias must be positive and finite when set')
        if not np.isfinite(self.temperature_K) or self.temperature_K < 0:
            raise ValueError('temperature_K must be finite and nonnegative')
        rotation_hvp_lower = 1 if self.rotation_solver == 'broyden-euclidean' else 2
        for key, lower in (('max_gaussians', 1), ('relax_steps', 0), ('rotation_hvp', rotation_hvp_lower)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f'{key} must be an integer >= {lower}')
        if self.pre_rotation_hvp is not None:
            main_min_hvp = 1 if self.rotation_solver in ('dimer', 'broyden-euclidean') else 2
            if self.rotation_hvp < 2 + main_min_hvp:
                raise ValueError('rotation_hvp must cover presweep and main solver minimum budgets')
            if self.pre_rotation_hvp > self.rotation_hvp - main_min_hvp - 1:
                raise ValueError('pre_rotation_hvp leaves no HVP for the main stage')


@dataclass(frozen=True)
class LSSettings:
    bond_energies: dict          # Explicit element-pair standard bond energies, eV.
    bond_lengths: dict           # Explicit pair neighbor cutoffs, Angstrom.
    target_per_atom: float       # Target true pre-quench response, eV/atom.
    initial_fraction: float = .03
    xi: float = .2              # Dimensionless fraction of frozen bond length.
    learning_rate: float = 1.8  # Paper eq15, not universal optimal parameters.
    energy_filter: dict | tuple | None = None  # Independent paper LS energy multipliers.
    prequench: LSPrequenchSettings | None = None

    def __post_init__(self):
        validate_prequench(self.prequench)
        object.__setattr__(self, 'energy_filter', _energy_filter(self.energy_filter))


def _validate_height_policy_options(config, height_policy, height_update_budget):
    """Validate shared height-policy inputs before any caller-owned PES work."""
    if height_policy is None:
        return
    from .native_height_policy import ConservativeNativeHeightPolicy
    from .minimal_angle_height import MinimalAngleHeightPolicy
    if not isinstance(height_policy, (ConservativeNativeHeightPolicy, MinimalAngleHeightPolicy)):
        raise TypeError('supported explicit height policy required')
    if config.cluster_frame == 'eckart':
        raise NotImplementedError('native-derived height policy has no eckart section contract')
    if (isinstance(height_update_budget, (bool, np.bool_)) or
            not isinstance(height_update_budget, (int, np.integer)) or
            height_update_budget < 1):
        raise ValueError('positive integer numerical height-update budget required')


@dataclass(frozen=True)
class SSWStep:
    index: int
    status: str
    accepted: bool
    climb: tuple
    landing: object
    energy_response: object
    evaluation_requests: int
    error: object = None
    last_atoms: object = None
    initial_direction: object = None
    ls_update: object = None
    cluster_reconnection: object = None
    ls_preparation: object = None
    mc_telemetry: object = None
    starter_selection: object = None


@dataclass(frozen=True)
class BiasStageQuenchOutcome:
    """Result returned by the explicit experimental biased-quench adapter."""
    relaxed: QuenchResult
    stage_stopped: bool
    release_all: bool
    diagnostics: dict


@dataclass(frozen=True)
class SSWResult:
    initial: QuenchResult
    current: object
    best: object
    minima: tuple
    records: tuple
    evaluation_requests: int
    status: str
    checkpoint: object = None
    identity_view: object = None


@dataclass
class SSWCheckpoint:
    """Trusted local snapshot at a completed fixed-cell outer attempt.

    Pickle is intentionally used only for local, trusted files.  Calculators
    are stripped from every ASE ``Atoms`` snapshot; callers must provide a
    freshly constructed ``surface`` when loading.  A checkpoint whose status
    is an error is retained for diagnosis but is never resumable.
    """
    initial: object
    current: object
    current_energy: float
    best: object
    minima: tuple
    records: tuple
    frozen: object
    response: object
    config: SSWConfig
    ls: object
    height_policy: object
    gaussian_policy: object
    height_update_budget: int
    reconnect_distance: object
    rng_state: object
    evaluation_requests: int
    next_index: int
    status: str
    schema_version: int = 1
    identity_view: object = None
    mc_settings: object = None
    native_mc_state: object = None
    recovered_rotation: object = None
    recovered_direction_state: object = None
    pool_state: object = None


def _checkpoint_copy(value):
    """Copy a value while removing calculator instances from ASE snapshots."""
    from ase import Atoms
    if isinstance(value, SSWCheckpoint):
        return SSWCheckpoint(
            _checkpoint_copy(value.initial), _checkpoint_copy(value.current), value.current_energy,
            _checkpoint_copy(value.best), _checkpoint_copy(value.minima), _checkpoint_copy(value.records),
            _checkpoint_copy(value.frozen), _checkpoint_copy(value.response), value.config,
            _checkpoint_copy(value.ls), _checkpoint_copy(value.height_policy),
            _checkpoint_copy(value.gaussian_policy), value.height_update_budget,
            value.reconnect_distance, deepcopy(value.rng_state), value.evaluation_requests,
            value.next_index, value.status, value.schema_version,
            _checkpoint_copy(getattr(value, 'identity_view', None)),
            _checkpoint_copy(getattr(value, 'mc_settings', None)),
            _checkpoint_copy(getattr(value, 'native_mc_state', None)),
            _checkpoint_copy(getattr(value, 'recovered_rotation', None)),
            _checkpoint_copy(getattr(value, 'recovered_direction_state', None)),
            _checkpoint_copy(getattr(value, 'pool_state', None)))
    if isinstance(value, Atoms):
        result = value.copy()
        result.calc = None
        return result
    if isinstance(value, QuenchResult):
        return QuenchResult(_checkpoint_copy(value.atoms), value.energy, value.max_force,
                            value.converged, value.optimizer_steps,
                            value.evaluation_requests, value.surface,
                            _checkpoint_copy(value.optimizer_telemetry))
    if isinstance(value, SSWStep):
        return SSWStep(value.index, value.status, value.accepted,
                       _checkpoint_copy(value.climb), _checkpoint_copy(value.landing),
                       value.energy_response, value.evaluation_requests, value.error,
                       _checkpoint_copy(value.last_atoms), _checkpoint_copy(value.initial_direction),
                       _checkpoint_copy(value.ls_update), _checkpoint_copy(value.cluster_reconnection),
                       _checkpoint_copy(getattr(value, 'ls_preparation', None)),
                       _checkpoint_copy(getattr(value, 'mc_telemetry', None)),
                       _checkpoint_copy(getattr(value, 'starter_selection', None)))
    if isinstance(value, LSResponseState):
        return deepcopy(value)
    if isinstance(value, dict):
        return {key: _checkpoint_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_copy(item) for item in value)
    if is_dataclass(value):
        return replace(value, **{field.name: _checkpoint_copy(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    return deepcopy(value)


def save_ssw_checkpoint(path, checkpoint):
    """Save a trusted local SSW/LS checkpoint; calculators are never pickled."""
    if not isinstance(checkpoint, SSWCheckpoint):
        raise TypeError('checkpoint must be SSWCheckpoint')
    snapshot = _checkpoint_copy(checkpoint)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_ssw_checkpoint(checkpoint):
    if not isinstance(checkpoint, SSWCheckpoint):
        raise TypeError('checkpoint must be SSWCheckpoint')
    if checkpoint.schema_version not in (1, 2, 3, 4, 5):
        raise ValueError(f'unsupported SSW checkpoint schema {checkpoint.schema_version!r}')
    pool_state = getattr(checkpoint, 'pool_state', None)
    if checkpoint.schema_version == 5 and pool_state is None:
        raise ValueError('schema 5 checkpoint requires pool state')
    if checkpoint.schema_version < 5 and pool_state is not None:
        raise ValueError('pool state requires checkpoint schema 5')
    if pool_state is not None:
        from .pool_checkpoint import _validate_pure
        _validate_pure(pool_state)
    saved_mc = getattr(checkpoint, 'mc_settings', None)
    saved_mc_state = getattr(checkpoint, 'native_mc_state', None)
    if (saved_mc is None) != (saved_mc_state is None):
        raise ValueError('checkpoint native MC settings and state must be paired')
    if checkpoint.schema_version == 2:
        if not isinstance(getattr(checkpoint, 'mc_settings', None), NativeMCSettings):
            raise ValueError('schema 2 checkpoint requires native MC settings')
        if not isinstance(getattr(checkpoint, 'native_mc_state', None), NativeMCState):
            raise ValueError('schema 2 checkpoint requires native MC state')
    if checkpoint.schema_version in (3, 4, 5) and saved_mc is not None:
        if not isinstance(checkpoint.mc_settings, NativeMCSettings):
            raise ValueError('schema 3 checkpoint has invalid native MC settings')
        if not isinstance(getattr(checkpoint, 'native_mc_state', None), NativeMCState):
            raise ValueError('schema 3 checkpoint with MC requires native MC state')
    saved_rotation = getattr(checkpoint, 'recovered_rotation', None)
    if checkpoint.schema_version < 3 and saved_rotation is not None:
        raise ValueError('recovered rotation requires checkpoint schema 3')
    if checkpoint.schema_version == 3 or saved_rotation is not None:
        from .recovered_rotation import RecoveredRotationSettings
        if not isinstance(saved_rotation, RecoveredRotationSettings):
            raise ValueError('schema 3 checkpoint requires typed recovered rotation settings')
    saved_direction = getattr(checkpoint, 'recovered_direction_state', None)
    if checkpoint.schema_version < 4 and saved_direction is not None:
        raise ValueError('recovered direction requires checkpoint schema 4')
    if checkpoint.schema_version == 4 or saved_direction is not None:
        from .recovered_direction import RecoveredDirectionCheckpointState
        if not isinstance(saved_direction, RecoveredDirectionCheckpointState):
            raise ValueError('schema 4 checkpoint requires typed recovered direction state')
    if checkpoint.schema_version >= 4 and saved_direction is not None and saved_rotation is not None:
        raise ValueError('checkpoint cannot contain both recovered rotation and direction state')
    initial_failure = checkpoint.status == 'ls_initialization_failed'
    indices = tuple(r.index for r in checkpoint.records)
    expected = (-1,) if initial_failure else tuple(range(checkpoint.next_index))
    if checkpoint.next_index < 0 or indices != expected:
        raise ValueError('checkpoint records are not contiguous from index zero')
    accounted = checkpoint.initial.evaluation_requests + sum(r.evaluation_requests for r in checkpoint.records)
    if checkpoint.evaluation_requests < 0 or checkpoint.evaluation_requests != accounted:
        raise ValueError('checkpoint evaluation request count disagrees with records')


def load_ssw_checkpoint(path):
    """Load a trusted local SSW/LS checkpoint written by save_ssw_checkpoint."""
    with Path(path).open('rb') as handle:
        checkpoint = pickle.load(handle)
    _validate_ssw_checkpoint(checkpoint)
    return checkpoint


def _policy_signature(value):
    return pickle.dumps(normalize_prequench_settings(value), protocol=4)


class InitialQuenchError(RuntimeError):
    def __init__(self, result):
        super().__init__('initial true quench did not reach requested force tolerance')
        self.result = result


def sample_initial_direction(atoms, rng, *, mode):
    """2013 eqs 1-2: normalized Maxwell direction plus lambda bond-formation.

Paper constants: lambda uniform [0.1,1.5], pair separation >3 Angstrom.
    The local vector is the unnormalized coordinate swap of eq2, evaluated in
    the paper's Angstrom convention. 'global' omits the pair term explicitly;
    it is useful for systems with no eligible pair and is not a silent fallback.
    'isotropic' draws the same Cartesian normal shape without mass scaling;
    the existing rotation-frame projection, when requested, remains separate.
    """
    if mode == 'isotropic':
        direction = rng.normal(size=(len(atoms), 3))
        return direction / np.linalg.norm(direction)
    global_direction = rng.normal(size=(len(atoms), 3)) / np.sqrt(atoms.get_masses()[:, None])
    global_direction /= np.linalg.norm(global_direction)
    if mode == 'global':
        return global_direction
    if mode != 'paper':
        raise ValueError('unknown direction sampling mode')
    eligible = [(i, j) for i in range(len(atoms)) for j in range(i+1, len(atoms))
                if np.linalg.norm(atoms.positions[j]-atoms.positions[i]) > 3.]
    if not eligible:
        raise ValueError('paper direction requires an atom pair separated by more than 3 Angstrom')
    i, j = eligible[int(rng.integers(len(eligible)))]
    local_direction = np.zeros_like(global_direction)
    local_direction[i] = atoms.positions[j] - atoms.positions[i]
    local_direction[j] = -local_direction[i]
    direction = global_direction + rng.uniform(.1, 1.5) * local_direction
    return direction / np.linalg.norm(direction)


def _initialize_ls_state(current, ls):
    """Build the LS frozen potential and response controller for ``current``."""
    from .ls_native_reference import NativeLSSettings, NativeLSRuntime
    if isinstance(ls, NativeLSSettings):
        response = NativeLSRuntime(current, ls)
        return response.frozen, response
    from .periodic_softening import FrozenPeriodicBondSoftening
    softening_type = FrozenPeriodicBondSoftening if current.pbc.all() else FrozenBondSoftening
    frozen = softening_type.from_atoms(
        current, bond_energies=ls.bond_energies, bond_lengths=ls.bond_lengths,
        initial_fraction=ls.initial_fraction, xi=ls.xi,
        energy_filter=ls.energy_filter)
    return frozen, LSResponseState(ls.target_per_atom, learning_rate=ls.learning_rate)


def _prepare_pool_restart(selected, *, ls, recovered_direction, rng):
    """Prepare all state for an explicit pool jump before committing it."""
    frozen = response = None
    if ls is not None:
        frozen, response = _initialize_ls_state(selected, ls)
    controller = None
    if recovered_direction is not None:
        from .recovered_direction import RecoveredDirectionController
        controller = RecoveredDirectionController(recovered_direction)
        controller.initialize(selected, selected, rng)
    return frozen, response, controller


def run_ssw(atoms, surface, *, steps, config, rng, ls=None, height_policy=None, gaussian_policy=None,
            height_update_budget=1000, reconnect_distance=None, checkpoint=None,
            checkpoint_path=None, structure_matcher=None, bias_quench_adapter=None,
            recovered_direction=None, recovered_rotation=None, mc=None,
            starter_selector=None, selector_rng=None):
    """Run independent fixed-cell SSW; optional LS uses frozen image bonds for full PBC.

``steps`` counts additional outer attempts. To resume, explicitly pass a
    loaded ``checkpoint`` and the same settings and RNG bit-generator type;
    positions come from the checkpoint. ``checkpoint_path`` is an output file,
    atomically replaced after each completed attempt (never an implicit input).
    The surface must use the same potential/settings; its counter is not reset.
    Failed terminal states are diagnostic snapshots, not resumable boundaries.

``bias_quench_adapter`` is an explicit experimental hook replacing only the
biased quench. It receives the original quench arguments plus a ``context``
dict with fixed keys and must return ``BiasStageQuenchOutcome``. It cannot be
combined with checkpointing; the default path remains the direct ``quench``
call. ``gaussian_index`` is zero-based and ``steps`` remains the number of
additional outer attempts.

``recovered_direction`` explicitly selects experimental Run5/Q-off local
generation, stage displacement updates and recovered CBD. It requires a free
nonperiodic cluster with ``cluster_frame='direction_only'``. Outer-boundary
checkpointing uses schema 4 and restores pair/group/marker and main RNG. Its settings replace SSWConfig's rotation_solver, rotation_bias,
rotation_hvp and rotation_tol; fd_step and rotation_exit_policy still apply.
Startup selection is a stated Python contract, not full native trajectory
parity. Gaussian/quench/MC rules remain those of this shared driver.

``recovered_rotation`` selects the stateless recovered-CBD rotation stages
while retaining the ordinary global/paper direction proposal and outer
Gaussian/landing lifecycle. It requires explicit ``RecoveredRotationSettings``
and is checkpointed as schema 3; it cannot be combined with
``recovered_direction`` or ``pre_rotation_hvp``.

``starter_selector(snapshot, selector_rng)`` optionally chooses an index from
all force-certified observations, including MC-rejected landings. It runs only
after a successful true landing and completed state updates. Return None to
retain the MC-selected current observation. A different index explicitly
restarts direction selection from that stored geometry without another quench.
The controller persists across other outer steps. The snapshot owns copies;
its cost includes initialization and failed work. A different selected
observation reinitializes LS and recovered-direction state transactionally;
failed restart preparation leaves the prior current/LS state in place and is
recorded as a terminal selection failure. Selection errors otherwise propagate.
``selector_rng`` must be an independent Generator. With checkpointing, the
selector must expose the pure-data checkpoint contract methods. ``accepted`` remains the MC decision;
``starter_selection`` separately records the actual next-starter decision.

No native program is called. Atoms/its calculator are not changed. The
returned minima include rejected MC landings, all force-converged on the
true surface; they are NOT deduplicated or positive-Hessian certified.
    Optional height_policy selects the explicitly conservative native-derived
    stage-height/history policy; its weights freeze throughout each quench.
    height_update_budget is a numerical loop ceiling, not an accepted fallback.
Failed rotations/quenches consume an outer step and stay in records,
    never enter minima. A failed LS strength update retains any already valid
    landing. Calculator exceptions propagate; there is no alternate potential.
    Explicit rotation_exit_policy="force_or_budget" allows a finite evaluated
    direction at its numerical request limit to enter Gaussian climbing. It
    remains marked unconverged; subspace exhaustion and numerical errors are
    not budget exits. True landing force qualification is unchanged.
Random velocity directions use normal components divided by sqrt(mass),
then normalize; the Maxwell temperature factor cancels. Global rigid modes
are not projected out by default (external ASE potentials may break these
symmetries). Explicit cluster_frame="eckart" restricts each escape to a fixed
linear section, including full modified-energy quenching. This requires an
isolated free cluster; it is an experimental formulation, not native parity.
Initial and final true quenches remain unrestricted and use full atomic forces.
With cluster_frame="direction_only", the direction/HVP projector is rebuilt at
each Gaussian center, while biased quenching stays unrestricted Cartesian. This
follows the recovered native rigid tangent span, not exact CBD execution parity.
Three-dimensional PBC uses explicit translation_only/global sampling: continuous
unwrapped Cartesian positions and fixed cell throughout each escape, projecting
only global translations in direction refinement. No rotation removal, MIC bias,
or cell optimization is performed. Physical E/F retain ASE PBC; the path bias
lives on the continuous coordinate lift and must not be evaluated after wrapping.
"""
    if checkpoint is not None:
        _validate_ssw_checkpoint(checkpoint)
        if checkpoint.status != 'completed':
            raise ValueError(f'cannot resume terminal checkpoint with status {checkpoint.status!r}')
    if starter_selector is None and selector_rng is not None:
        raise ValueError('selector_rng requires starter_selector')
    if starter_selector is not None:
        if not callable(starter_selector):
            raise TypeError('starter_selector must be callable or None')
        if not isinstance(selector_rng, np.random.Generator):
            raise ValueError('starter_selector requires an independent selector_rng Generator')
        if selector_rng is rng or selector_rng.bit_generator is rng.bit_generator:
            raise ValueError('selector_rng must be independent and must not share the main rng or bit_generator')
    pool_resume = checkpoint is not None and getattr(checkpoint, 'pool_state', None) is not None
    pool_checkpoint = starter_selector is not None and (checkpoint_path is not None or pool_resume)
    if pool_resume and starter_selector is None:
        raise ValueError('pool checkpoint resume requires starter_selector')
    if pool_checkpoint or pool_resume:
        from .pool_checkpoint import _require_contract, _require_restore, _require_state
        _require_contract(starter_selector)
        _require_state(starter_selector)
        _require_restore(starter_selector)
    if checkpoint is not None and getattr(checkpoint, 'pool_state', None) is None and starter_selector is not None:
        raise ValueError('checkpoint lacks pool selector state')
    if mc is not None and not isinstance(mc, NativeMCSettings):
        raise TypeError('mc must be NativeMCSettings or None')
    if mc is not None and config.temperature_K <= 0:
        raise ValueError('native MC requires positive temperature_K')
    if checkpoint is not None and recovered_rotation is None:
        recovered_rotation = getattr(checkpoint, 'recovered_rotation', None)
    if checkpoint is not None and recovered_direction is None:
        saved_direction = getattr(checkpoint, 'recovered_direction_state', None)
        if saved_direction is not None:
            recovered_direction = saved_direction.settings
    direction_controller = None
    if recovered_direction is not None:
        from .recovered_direction import (RecoveredDirectionSettings,
                                           RecoveredDirectionCheckpointState,
                                           RecoveredDirectionController)
        if not isinstance(recovered_direction, RecoveredDirectionSettings):
            raise TypeError('recovered_direction must be RecoveredDirectionSettings')
        if checkpoint is not None and getattr(checkpoint, 'recovered_direction_state', None) is None:
            raise ValueError('checkpoint lacks recovered direction state')
        if checkpoint is not None:
            saved_direction = checkpoint.recovered_direction_state
            if not isinstance(saved_direction, RecoveredDirectionCheckpointState):
                raise ValueError('checkpoint recovered direction state is invalid')
            if saved_direction.settings != recovered_direction:
                raise ValueError('checkpoint recovered direction settings do not match requested settings')
            if any(i < 0 or i >= len(atoms) for i in saved_direction.pair):
                raise ValueError('checkpoint recovered direction pair does not match input atom count')
            if len(saved_direction.group) != len(atoms):
                raise ValueError('checkpoint recovered direction group does not match input atom count')
            # Pickle bypasses dataclass construction; validate the saved values
            # before restoring RNG or entering the PES, including zero-step resumes.
            _checkpoint_copy(saved_direction).validate_for_atom_count(len(atoms))
        if atoms.pbc.any() or atoms.constraints or config.cluster_frame != 'direction_only':
            raise ValueError('recovered direction requires a free nonperiodic cluster and direction_only frame')
        if config.pre_rotation_hvp is not None:
            raise ValueError('recovered direction owns its PreRot stages; do not combine pre_rotation_hvp')
        direction_controller = RecoveredDirectionController(recovered_direction)
    if recovered_rotation is not None:
        from .recovered_rotation import RecoveredRotationSettings
        if not isinstance(recovered_rotation, RecoveredRotationSettings):
            raise TypeError('recovered_rotation must be RecoveredRotationSettings')
        if recovered_direction is not None:
            raise ValueError('recovered_rotation and recovered_direction are mutually exclusive')
        if config.pre_rotation_hvp is not None:
            raise ValueError('recovered_rotation and pre_rotation_hvp are mutually exclusive')
        if atoms.constraints:
            raise NotImplementedError('recovered_rotation requires unconstrained atoms')
    from .minimal_angle_height import MinimalAngleHeightPolicy
    if reconnect_distance is not None:
        # Validate the optional geometry contract before the initial PES call.
        reconnect_clusters(atoms, reconnect_distance, repair=False)
    _validate_height_policy_options(config, height_policy, height_update_budget)
    if gaussian_policy is not None:
        from .pam_gaussian import PAMCurvatureGaussian
        if not isinstance(gaussian_policy, PAMCurvatureGaussian):
            raise TypeError('gaussian_policy must be PAMCurvatureGaussian')
        if height_policy is not None:
            raise ValueError('gaussian_policy and height_policy are mutually exclusive')
        if config.cluster_frame == 'eckart':
            raise NotImplementedError('PAM Gaussian policy has no eckart section contract')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    if bias_quench_adapter is not None and not callable(bias_quench_adapter):
        raise TypeError('bias_quench_adapter must be callable or None')
    if bias_quench_adapter is not None and (checkpoint is not None or checkpoint_path is not None):
        raise ValueError('bias_quench_adapter cannot be combined with checkpoint or checkpoint_path')
    if atoms.constraints:
        raise NotImplementedError('standalone SSW currently requires unconstrained atoms')
    if atoms.pbc.any():
        if not atoms.pbc.all():
            raise NotImplementedError('periodic SSW currently requires three-dimensional PBC')
        if (config.cluster_frame != 'translation_only' or
                config.direction_sampling not in ('global', 'isotropic')):
            raise ValueError('periodic SSW requires translation_only frame and global or isotropic direction sampling')
        from .periodic_geometry import FixedCellTranslationFrame
        FixedCellTranslationFrame(atoms)  # Validate before calculator requests.
    elif config.cluster_frame == 'translation_only':
        raise ValueError('translation_only is the explicit periodic SSW geometry')
    masses = atoms.get_masses()
    if not len(atoms) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('finite positive masses required')
    if checkpoint is not None:
        saved_mc = getattr(checkpoint, 'mc_settings', None) is not None
        if mc is None and saved_mc:
            raise ValueError('native MC checkpoint requires native MC settings')
        if mc is not None:
            if checkpoint.schema_version not in (2, 3, 4, 5) or getattr(checkpoint, 'native_mc_state', None) is None:
                raise ValueError('checkpoint lacks native MC state')
            if getattr(checkpoint, 'mc_settings', None) != mc:
                raise ValueError('checkpoint native MC settings do not match requested settings')
    if structure_matcher is not None and not callable(structure_matcher):
        raise TypeError('structure_matcher must be callable or None')
    if checkpoint is not None and ((getattr(checkpoint, 'identity_view', None) is None) != (structure_matcher is None)):
        raise ValueError('checkpoint identity view requires the same structure_matcher enablement')
    begin = surface.requests
    quench_optimizer = {'ase-lbfgs': LBFGS, 'ase-lbfgs-linesearch': LBFGSLineSearch}.get(
        config.quench_optimizer, config.quench_optimizer)
    from .ls_native_reference import NativeLSRuntime
    if ls is not None:
        from .ls_prequench import validate_prequench_exit_policy
        validate_prequench_exit_policy(getattr(ls, 'prequench', None),
                                       optimizer=quench_optimizer)
    if checkpoint is not None:
        if checkpoint.config != config:
            raise ValueError('checkpoint SSWConfig does not match requested config')
        if getattr(checkpoint, 'recovered_rotation', None) != recovered_rotation:
            raise ValueError('checkpoint recovered rotation settings do not match requested settings')
        if _policy_signature(checkpoint.ls) != _policy_signature(ls):
            raise ValueError('checkpoint LS settings do not match requested settings')
        if checkpoint.height_update_budget != height_update_budget or checkpoint.reconnect_distance != reconnect_distance:
            raise ValueError('checkpoint policy settings do not match requested settings')
        if _policy_signature(checkpoint.height_policy) != _policy_signature(height_policy):
            raise ValueError('checkpoint height policy does not match requested policy')
        if _policy_signature(checkpoint.gaussian_policy) != _policy_signature(gaussian_policy):
            raise ValueError('checkpoint Gaussian policy does not match requested policy')
        cp_atoms = checkpoint.current
        if tuple(cp_atoms.numbers) != tuple(atoms.numbers) or tuple(cp_atoms.pbc) != tuple(atoms.pbc):
            raise ValueError('checkpoint current composition/PBC does not match input atoms')
        if not np.array_equal(cp_atoms.cell.array, atoms.cell.array):
            raise ValueError('checkpoint current cell does not match input atoms')
        if not np.array_equal(cp_atoms.get_masses(), atoms.get_masses()):
            raise ValueError('checkpoint current masses do not match input atoms')
        bit_name = checkpoint.rng_state.get('bit_generator')
        if bit_name != rng.bit_generator.__class__.__name__:
            raise ValueError('checkpoint RNG bit generator does not match requested RNG')
        initial = _checkpoint_copy(checkpoint.initial)
        current = _checkpoint_copy(checkpoint.current)
        current_energy = float(checkpoint.current_energy)
        best = _checkpoint_copy(checkpoint.best)
        minima = list(_checkpoint_copy(checkpoint.minima))
        records = list(_checkpoint_copy(checkpoint.records))
        frozen = _checkpoint_copy(checkpoint.frozen)
        response = _checkpoint_copy(checkpoint.response)
        ls = _checkpoint_copy(checkpoint.ls)
        rng.bit_generator.state = deepcopy(checkpoint.rng_state)
        native_mc_state = _checkpoint_copy(getattr(checkpoint, 'native_mc_state', None))
        start_index = checkpoint.next_index
        prior_requests = checkpoint.evaluation_requests
    else:
        initial = quench(atoms, surface, fmax=config.fmax,
                         steps=config.relax_steps, optimizer=quench_optimizer, lbfgs_memory=config.lbfgs_memory)
        if not initial.converged:
            raise InitialQuenchError(initial)
        current = initial.atoms.copy()
        current_energy = initial.energy
        best = initial
        minima = [initial]
        records = []
        native_mc_state = NativeMCState() if mc is not None else None
        frozen = response = None
        start_index = 0
        prior_requests = 0
    pool_last_landing_index = None
    if pool_resume:
        from .pool_checkpoint import restore_pool_state
        current_observation_index, pool_last_landing_index = restore_pool_state(
            starter_selector, selector_rng, checkpoint.pool_state,
            observation_count=len(minima))
    else:
        current_observation_index = 0
    last_landing_index = pool_last_landing_index
    if direction_controller is not None:
        if checkpoint is None:
            direction_controller.initialize(atoms, current, rng)
        else:
            direction_controller.restore_checkpoint_state(checkpoint.recovered_direction_state)
    identity_view = None if checkpoint is None else _checkpoint_copy(getattr(checkpoint, 'identity_view', None))
    if structure_matcher is not None and identity_view is None:
        from .minimum_identity import MinimumIdentityView, update_identity_view
        identity_view = MinimumIdentityView([], [], [], 0, 0)
        update_identity_view(identity_view, minima, structure_matcher)
    pool_checkpoint_enabled = pool_checkpoint or pool_resume

    def checkpoint_pool_state():
        if not pool_checkpoint_enabled:
            return None
        from .pool_checkpoint import build_pool_state
        return build_pool_state(
            starter_selector, selector_rng,
            current_index=current_observation_index,
            last_landing_index=last_landing_index)

    if checkpoint is None and ls is not None:
        try:
            frozen, response = _initialize_ls_state(current, ls)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            # Initialization has already paid for and certified a true minimum.
            # Preserve it and its requests when the explicit LS domain is absent.
            record = SSWStep(-1, 'ls_initialization_failed', False, (), None,
                             None, 0, str(error), current.copy())
            terminal = 'ls_initialization_failed'
            cp = None
            if checkpoint_path is not None:
                cp = SSWCheckpoint(_checkpoint_copy(initial), _checkpoint_copy(current), current_energy,
                    _checkpoint_copy(best), tuple(_checkpoint_copy(minima)), (record,), None, None,
                    config, _checkpoint_copy(ls), _checkpoint_copy(height_policy),
                    _checkpoint_copy(gaussian_policy), height_update_budget, reconnect_distance,
                    deepcopy(rng.bit_generator.state), surface.requests - begin, 0, terminal,
                    identity_view=_checkpoint_copy(identity_view),
                    schema_version=(5 if pool_checkpoint_enabled else (4 if recovered_direction is not None else (3 if recovered_rotation is not None else (2 if mc is not None else 1)))),
                    mc_settings=_checkpoint_copy(mc),
                    native_mc_state=_checkpoint_copy(native_mc_state),
                    recovered_rotation=_checkpoint_copy(recovered_rotation),
                    recovered_direction_state=(None if direction_controller is None else _checkpoint_copy(direction_controller.checkpoint_state())),
                    pool_state=_checkpoint_copy(checkpoint_pool_state()))
                save_ssw_checkpoint(checkpoint_path, cp)
            return SSWResult(initial, current.copy(), best.atoms.copy(), tuple(minima),
                             (record,), surface.requests-begin, terminal, cp, identity_view)
    run_status = 'completed'
    checkpoint_result = checkpoint if checkpoint is not None else None
    checkpoint_enabled = checkpoint is not None or checkpoint_path is not None
    for index in range(start_index, start_index + steps):
        before = surface.requests
        work = current.copy()
        climb = []
        landing = prepared = None
        status = 'gaussian_limit'
        error_message = None
        accepted = False
        mc_telemetry = None
        soft_terms = () if frozen is None else (frozen,)
        if frozen is not None:
            try:
                prepared = prepare_ls_step(work, surface, softening=frozen,
                    prequench=getattr(ls, "prequench", None),
                    fmax=config.fmax, steps=config.relax_steps, optimizer=quench_optimizer, lbfgs_memory=config.lbfgs_memory)
                work = prepared.atoms.copy()
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                failed = getattr(error, 'result', None)
                records.append(SSWStep(index, 'ls_prequench_failed', False, (),
                    failed, None, surface.requests - before, str(error),
                    work.copy() if failed is None else failed.atoms.copy()))
                run_status = 'ls_prequench_failed'
                if checkpoint_enabled:
                    checkpoint_result = SSWCheckpoint(
                        _checkpoint_copy(initial), _checkpoint_copy(current), current_energy,
                        _checkpoint_copy(best), tuple(_checkpoint_copy(minima)),
                        tuple(_checkpoint_copy(records)), _checkpoint_copy(frozen), _checkpoint_copy(response),
                        config, _checkpoint_copy(ls), _checkpoint_copy(height_policy),
                        _checkpoint_copy(gaussian_policy), height_update_budget, reconnect_distance,
                        deepcopy(rng.bit_generator.state), prior_requests + surface.requests - begin,
                        index + 1, run_status, identity_view=_checkpoint_copy(identity_view),
                        schema_version=(5 if pool_checkpoint_enabled else (4 if direction_controller is not None else (3 if recovered_rotation is not None else (2 if mc is not None else 1)))),
                        mc_settings=_checkpoint_copy(mc),
                        native_mc_state=_checkpoint_copy(native_mc_state),
                        recovered_rotation=_checkpoint_copy(recovered_rotation),
                        recovered_direction_state=(None if direction_controller is None else _checkpoint_copy(direction_controller.checkpoint_state())),
                        pool_state=_checkpoint_copy(checkpoint_pool_state()))
                    if checkpoint_path is not None:
                        save_ssw_checkpoint(checkpoint_path, checkpoint_result)
                break
        # LS section 2.4: generate the geometric proposal after the soft quench.
        # Global mass-weighted sampling is unchanged; pair membership/vectors
        # must refer to the actual structure entering direction refinement.
        if direction_controller is None:
            anchor = sample_initial_direction(work, rng, mode=config.direction_sampling)
        else:
            anchor = np.zeros_like(work.positions)  # Populated inside the recorded stage.
        frame = None
        if config.cluster_frame == "eckart":
            frame = ClusterFrame(work)
            anchor = frame.project(anchor)
            anchor_norm = np.linalg.norm(anchor)
            if not np.isfinite(anchor_norm) or anchor_norm <= np.finfo(float).eps * anchor.size:
                raise ValueError("initial direction has no resolvable internal component")
            anchor /= anchor_norm
        initial_anchor = anchor.copy()
        terms = list(soft_terms)
        rotation_frame = frame

        def rotation_surface(candidate):
            if rotation_frame is not None and direction_controller is None:
                candidate = candidate.copy()
                candidate.positions = rotation_frame.positions(candidate.positions)
            energy, forces = surface.evaluate(candidate)
            for term in soft_terms:
                de, df = term.evaluate(candidate)
                energy += de
                forces += df
            if rotation_frame is not None and direction_controller is None:
                forces = rotation_frame.project(forces)
            return energy, forces

        experimental_two_stage = config.pre_rotation_hvp is not None
        if config.rotation_solver == 'dimer':
            from .dimer import paper_dimer_direction
            solve_direction = paper_dimer_direction
        elif config.rotation_solver == 'ritz':
            solve_direction = paper_biased_direction
        else:
            from .broyden_direction import paper_broyden_direction
            solve_direction = paper_broyden_direction
        for gaussian_index in range(config.max_gaussians):
            event = None
            stage_before = surface.requests
            try:
                if direction_controller is not None:
                    if gaussian_index:
                        generated_direction = direction_controller.update_direction(work, rng)
                    else:
                        generated_direction = direction_controller.begin_escape(current, work, rng)
                    if generated_direction.release_all:
                        status = 'stage_release'
                        climb.append(dict(index=gaussian_index, status='direction_zero_release',
                            recovered_direction=direction_controller.diagnostics,
                            requests=surface.requests-stage_before))
                        break
                    anchor = generated_direction.direction.copy()
                    if gaussian_index == 0:
                        initial_anchor = anchor.copy()
                rotation_anchor = anchor
                if config.cluster_frame in ('direction_only', 'translation_only'):
                    rotation_frame = (FixedCellTranslationFrame(work) if config.cluster_frame == 'translation_only'
                                      else ClusterFrame(work))
                    rotation_anchor = rotation_frame.project(anchor)
                    norm = np.linalg.norm(rotation_anchor)
                    if not np.isfinite(norm) or norm <= np.finfo(float).eps * rotation_anchor.size:
                        raise ValueError('anchor has no resolvable internal component')
                    rotation_anchor /= norm
                if direction_controller is not None:
                    from .recovered_cbd import recovered_cbd_direction
                    settings = recovered_direction
                    mode = recovered_cbd_direction(work, rotation_anchor,
                        fd_step=config.fd_step, max_force_calls=settings.max_force_calls,
                        pre_rotmax=settings.pre_rotmax, rotmax=settings.rotmax,
                        pre_ftol=settings.pre_ftol, ftol=settings.ftol, metric=settings.metric,
                        evaluate=rotation_surface, project=rotation_frame.project)
                    actual_anchor = mode.bias_reference
                    actual_a = mode.rotation_weight if mode.stage == 'CBD_biasedRot' else 0.
                elif recovered_rotation is not None:
                    from .recovered_cbd import recovered_cbd_direction
                    mode = recovered_cbd_direction(
                        work, rotation_anchor,
                        fd_step=config.fd_step,
                        max_force_calls=recovered_rotation.max_force_calls,
                        pre_rotmax=recovered_rotation.pre_rotmax,
                        rotmax=recovered_rotation.rotmax,
                        pre_ftol=recovered_rotation.pre_ftol,
                        ftol=recovered_rotation.ftol,
                        metric=recovered_rotation.metric,
                        evaluate=rotation_surface,
                        project=(None if rotation_frame is None else rotation_frame.project))
                    actual_anchor = mode.bias_reference
                    actual_a = mode.rotation_weight if mode.stage == 'CBD_biasedRot' else 0.
                elif experimental_two_stage:
                    from .staged_direction import two_stage_dimer_direction
                    mode = two_stage_dimer_direction(work, rotation_anchor,
                        fd_step=config.fd_step, max_hvp=config.rotation_hvp,
                        pre_rotation_hvp=config.pre_rotation_hvp,
                        tol=config.rotation_tol, evaluate=rotation_surface,
                        main_solver=config.rotation_solver)
                    actual_anchor = mode.pre.direction.reshape(work.positions.shape)
                    actual_a = mode.bias_curvature
                else:
                    mode = solve_direction(work, rotation_anchor,
                        rotation_bias=config.rotation_bias, fd_step=config.fd_step,
                        max_hvp=config.rotation_hvp, tol=config.rotation_tol,
                        evaluate=rotation_surface)
                    actual_anchor = rotation_anchor
                    actual_a = config.rotation_bias
                stage_diagnostics = None
                if direction_controller is not None:
                    stage_diagnostics = dict(
                        recovered_direction=dict(direction_controller.diagnostics,
                            route=generated_direction.local_route,
                            proposal=generated_direction.direction.tolist()),
                        recovered_rotation=dict(stage=mode.stage, stage_complete=mode.stage_complete,
                            real_curvature=mode.real_curvature, trace=mode.trace))
                elif recovered_rotation is not None:
                    stage_diagnostics = dict(
                        recovered_rotation=dict(stage=mode.stage, stage_complete=mode.stage_complete,
                            real_curvature=mode.real_curvature, trace=mode.trace,
                            bias_reference=mode.bias_reference.tolist(),
                            rotation_weight=float(mode.rotation_weight)))
                if experimental_two_stage:
                    stage_diagnostics = dict(
                        pre_rotation=dict(direction=mode.pre.direction.tolist(), curvature=mode.pre.curvature,
                            residual=mode.pre.residual_norm, hvp_calls=mode.pre.hvp_calls,
                            force_calls=mode.pre.force_calls, converged=mode.pre.converged,
                            stop_reason=getattr(mode.pre, 'stop_reason', 'unspecified')),
                        main_rotation=dict(direction=mode.main.direction.tolist(), curvature=mode.main.curvature,
                            residual=mode.main.residual_norm, hvp_calls=mode.main.hvp_calls,
                            force_calls=mode.main.force_calls, converged=mode.main.converged,
                            stop_reason=getattr(mode.main, 'stop_reason', 'unspecified')),
                        actual_anchor=actual_anchor.tolist(), actual_rotation_bias=float(actual_a))
                rotation_stop = getattr(mode, 'stop_reason', 'unspecified')
                rotation_budget_released = (not mode.converged and
                    config.rotation_exit_policy == 'force_or_budget' and
                    (rotation_stop == 'budget_exhausted' or
                     ((direction_controller is not None or recovered_rotation is not None) and
                      rotation_stop in ('rotation_limit', 'force_budget'))))
                if rotation_budget_released:
                    direction = np.asarray(mode.direction)
                    if (direction.shape != work.positions.shape or
                            not np.isfinite(direction).all() or np.linalg.norm(direction) == 0 or
                            not np.isfinite(mode.curvature) or
                            not np.isfinite(mode.residual_norm) or mode.residual_norm < 0):
                        raise ValueError('budget rotation returned invalid evaluated direction or certificate')
                if not mode.converged and not rotation_budget_released:
                    status = 'rotation_failed'
                    failed_event = dict(index=gaussian_index,
                                      rotation_solver=('recovered-cbd' if (direction_controller is not None or recovered_rotation is not None) else config.rotation_solver), residual=mode.residual_norm,
                                      force_requests=mode.force_calls,
                                      rotation_stop_reason=rotation_stop, rotation_converged=False,
                                      rotation_budget_released=False)
                    if stage_diagnostics is not None: failed_event.update(stage_diagnostics)
                    climb.append(failed_event)
                    break
                center = work.positions.copy()
                background_force = None
                force_parallel = None
                height_preparation = None
                policy_data = None
                if gaussian_policy is not None:
                    history = tuple(terms[len(soft_terms):])
                    policy_data = gaussian_policy.choose(mode=mode, anchor=actual_anchor,
                        center=center, terms=history, base_width=config.width,
                        rotation_bias=actual_a)
                    width = policy_data['width']; weight = policy_data['weight']
                    displaced = work.copy(); displaced.positions += width * mode.direction
                    if frame is not None: displaced.positions = frame.positions(displaced.positions)
                    terms.append(ProjectedGaussian(center, mode.direction, width, weight))
                elif height_policy is None:
                    displaced = work.copy(); displaced.positions += config.width * mode.direction
                    if frame is not None: displaced.positions = frame.positions(displaced.positions)
                    background = SurfaceCalculator(surface, terms=terms); displaced.calc = background
                    background_force = displaced.get_forces()
                    force_parallel = float(np.sum(background_force * mode.direction))
                    width = config.width
                    weight = (config.forward_force - force_parallel) * config.width * math.exp(.5)
                    if not np.isfinite(weight) or weight <= 0:
                        status = 'nonpositive_height'
                        failed_event = dict(index=gaussian_index, weight=weight,
                                          background_forward_force=force_parallel)
                        if stage_diagnostics is not None: failed_event.update(stage_diagnostics)
                        climb.append(failed_event)
                        break
                    terms.append(ProjectedGaussian(center, mode.direction, config.width, weight))
                else:
                    from .native_height_policy import FrozenHeightGaussian
                    displaced = work.copy(); displaced.positions += config.width * mode.direction
                    if frame is not None: displaced.positions = frame.positions(displaced.positions)
                    background = SurfaceCalculator(surface, terms=terms); displaced.calc = background
                    background_force = displaced.get_forces()
                    force_parallel = float(np.sum(background_force * mode.direction))
                    width = config.width
                    history = tuple(FrozenHeightGaussian(t.center.ravel(), t.direction.ravel(), t.sigma, t.weight)
                                    for t in terms[len(soft_terms):])
                    if isinstance(height_policy, MinimalAngleHeightPolicy):
                        event = dict(index=gaussian_index, status='height_preparing',
                            center=center.tolist(), direction=mode.direction.tolist(), width=config.width,
                            height_input=dict(history=history, point=displaced.positions.ravel().tolist(),
                                background_force=background_force.ravel().tolist()),
                            requests=surface.requests-stage_before)
                        if stage_diagnostics is not None:
                            event.update(stage_diagnostics)
                        climb.append(event)
                        height_preparation = height_policy.prepare(history,
                            center=center.ravel(), direction=mode.direction.ravel(), width=config.width,
                            point=displaced.positions.ravel(), background_force=background_force.ravel())
                        event['height_preparation'] = height_preparation
                        weight = height_preparation.weight
                        if height_preparation.status == 'already_forward_satisfied':
                            status = 'nonpositive_height'
                            event.update(status=status, weight=0., requests=surface.requests-stage_before)
                            break
                    else:
                        height_preparation = None
                    # mode.curvature includes -a*(n dot anchor)^2 from the
                    # analytic rotation-only rank-one bias. Remove it without
                    # another PES request; the finite-secant error remains.
                    if not isinstance(height_policy, MinimalAngleHeightPolicy):
                        curvature = mode.curvature + actual_a * float(np.sum(mode.direction * actual_anchor))**2
                        height_preparation = height_policy.prepare(history,
                            center=center.ravel(), direction=mode.direction.ravel(), width=config.width,
                            point=displaced.positions.ravel(), background_force=background_force.ravel(),
                            curvature=curvature,
                            curvature_scope=('physical PES; rotation-only bias excluded' if not soft_terms
                                             else 'physical PES plus frozen LS; rotation-only bias excluded'),
                            max_updates=height_update_budget)
                    terms = list(soft_terms) + [ProjectedGaussian(t.center.reshape(center.shape),
                        t.direction.reshape(center.shape), t.width, t.weight) for t in height_preparation.terms]
                    weight = (height_preparation.weight if isinstance(height_policy, MinimalAngleHeightPolicy)
                              else height_preparation.final_weight)
                # Persist prepared height/history before optimizer/evaluator work:
                # a failed quench must not erase which objective was attempted.
                if event is None:
                    event = {}
                    climb.append(event)
                event.update(index=gaussian_index, rotation_solver=('recovered-cbd' if (direction_controller is not None or recovered_rotation is not None) else config.rotation_solver), cluster_frame=config.cluster_frame, center=center.tolist(),
                             direction=mode.direction.tolist(), weight=weight,
                             width=width, rotation_residual=mode.residual_norm,
                             rotation_stop_reason=rotation_stop, rotation_converged=bool(mode.converged),
                             rotation_budget_released=bool(rotation_budget_released),
                             rotation_force_requests=mode.force_calls, actual_anchor=actual_anchor.tolist(),
                             actual_rotation_bias=float(actual_a),
                             status='height_prepared', requests=surface.requests-stage_before)
                if policy_data is not None:
                    event['gaussian_policy'] = policy_data
                    event['curvature_scope'] = ('physical PES plus frozen LS' if soft_terms else 'physical PES')
                if stage_diagnostics is not None: event.update(stage_diagnostics)
                if direction_controller is not None:
                    direction_controller.save_gaussian_center(center)
                if height_preparation is not None:
                    event['height_preparation'] = height_preparation
                stage_steps = config.relax_steps if config.bias_stage_steps is None else config.bias_stage_steps
                bias_fmax = config.fmax if config.bias_fmax is None else config.bias_fmax
                if bias_quench_adapter is None:
                    relaxed = quench(displaced, surface, fmax=bias_fmax,
                                     steps=stage_steps, terms=terms, optimizer=quench_optimizer,
                                     frame=frame, lbfgs_memory=config.lbfgs_memory)
                    adapter_outcome = None
                else:
                    context = {
                        'current': current.copy(), 'current_energy': float(current_energy),
                        'best_energy': float(best.energy), 'outer_index': int(index),
                        'gaussian_index': int(gaussian_index), 'center': center.copy(),
                        'soft_terms': tuple(soft_terms), 'max_gaussians': int(config.max_gaussians),
                    }
                    adapter_outcome = bias_quench_adapter(
                        displaced, surface, fmax=bias_fmax, steps=stage_steps,
                        terms=tuple(terms), optimizer=quench_optimizer, frame=frame,
                        lbfgs_memory=config.lbfgs_memory, context=context)
                    if not isinstance(adapter_outcome, BiasStageQuenchOutcome):
                        raise TypeError('bias_quench_adapter must return BiasStageQuenchOutcome')
                    if not isinstance(adapter_outcome.stage_stopped, (bool, np.bool_)) or not isinstance(adapter_outcome.release_all, (bool, np.bool_)):
                        raise TypeError('bias_quench_adapter outcome flags must be bool')
                    if adapter_outcome.release_all and not adapter_outcome.stage_stopped:
                        raise ValueError('release_all requires stage_stopped')
                    if not isinstance(adapter_outcome.diagnostics, dict):
                        raise TypeError('bias_quench_adapter diagnostics must be a dict')
                    relaxed = adapter_outcome.relaxed
                    if not isinstance(relaxed, QuenchResult):
                        raise TypeError('bias_quench_adapter relaxed must be QuenchResult')
                    if (tuple(relaxed.atoms.numbers) != tuple(displaced.numbers) or
                            tuple(relaxed.atoms.pbc) != tuple(displaced.pbc) or
                            not np.array_equal(relaxed.atoms.cell.array, displaced.cell.array)):
                        raise ValueError('bias_quench_adapter changed fixed-cell composition/PBC/cell')
                    if (not np.isfinite(relaxed.energy) or not np.isfinite(relaxed.max_force) or
                            relaxed.max_force < 0 or
                            not np.isfinite(relaxed.atoms.positions).all()):
                        raise ValueError('bias_quench_adapter returned non-finite evaluated result')
                event.update(biased_energy=relaxed.energy, force_certificate=relaxed.surface,
                             max_force=relaxed.max_force, quench_requests=relaxed.evaluation_requests,
                             optimizer_telemetry=relaxed.optimizer_telemetry,
                             termination_reason=(None if relaxed.optimizer_telemetry is None
                                                 else relaxed.optimizer_telemetry.termination_reason),
                             status='converged' if relaxed.converged else 'biased_quench_failed',
                             requests=surface.requests-stage_before)
                budget_stop = (config.bias_stage_steps is not None and
                               relaxed.optimizer_telemetry is not None and
                               relaxed.optimizer_telemetry.termination_reason == 'maxiter' and
                               np.isfinite(relaxed.energy) and np.isfinite(relaxed.max_force) and
                               np.isfinite(relaxed.atoms.positions).all())
                if config.bias_stage_steps is not None:
                    event['stage_stop_reason'] = 'iteration_budget' if budget_stop else ('force_converged' if relaxed.converged else 'failure')
                    if budget_stop:
                        event['status'] = 'stage_budget'
                if adapter_outcome is not None:
                    event['stage_stop_reason'] = 'adapter' if adapter_outcome.stage_stopped else None
                    event['release_all'] = bool(adapter_outcome.release_all)
                    event['diagnostics'] = dict(adapter_outcome.diagnostics)
                    if adapter_outcome.stage_stopped:
                        if adapter_outcome.release_all:
                            status = 'stage_release'
                        else:
                            # The event records the stage stop; the outer
                            # attempt remains eligible for the normal landing
                            # after the Gaussian loop.
                            status = 'gaussian_limit'
                        event['status'] = 'stage_release' if adapter_outcome.release_all else 'stage_stopped'
                if config.bias_fmax is not None:
                    event['bias_fmax'] = config.bias_fmax
                work = relaxed.atoms.copy()
                if (not relaxed.converged and not budget_stop and
                        not (adapter_outcome is not None and adapter_outcome.stage_stopped)):
                    status = 'biased_quench_failed'
                    break
                true_energy, _ = surface.evaluate(work)
                event['true_energy'] = true_energy
                event['requests'] = surface.requests-stage_before
                if adapter_outcome is not None and adapter_outcome.release_all:
                    break
                if true_energy < current_energy:
                    status = 'lower_true_energy'
                    break
            except ClusterFrameDomainError as error:
                status = 'cluster_frame_failed'
                error_message = str(error)
                if event is None:
                    climb.append(dict(index=gaussian_index, cluster_frame=config.cluster_frame,
                                      error=error_message, requests=surface.requests-stage_before))
                else:
                    event.update(status=status, error=error_message, requests=surface.requests-stage_before)
                break
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, StopIteration) as error:
                status = run_status = 'evaluation_failed'
                error_message = f'climb: {type(error).__name__}: {error}'
                if event is None:
                    climb.append(dict(index=gaussian_index, error=error_message, requests=surface.requests-stage_before))
                else:
                    event.update(status=status, error=error_message, requests=surface.requests-stage_before)
                break
        cluster_reconnection = None
        if status in ('gaussian_limit', 'lower_true_energy', 'stage_release') and reconnect_distance is not None and climb:
            # Keep `work` and the diagnostic last_atoms untouched; only this
            # copy is handed to the unbiased landing quench.
            cluster_reconnection = reconnect_clusters(work, reconnect_distance, repair=True)
            landing_input = cluster_reconnection.atoms
        else:
            landing_input = work
        if status in ('gaussian_limit', 'lower_true_energy', 'stage_release'):
            try:
                landing = quench(landing_input, surface, fmax=config.fmax,
                                 steps=config.relax_steps, optimizer=quench_optimizer, lbfgs_memory=config.lbfgs_memory)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                status = run_status = 'evaluation_failed'
                error_message = f'true_quench: {type(error).__name__}: {error}'
        if landing is not None:
            if not landing.converged:
                status = 'true_quench_failed'
            else:
                minima.append(landing)
                last_landing_index = len(minima) - 1
                if landing.energy < best.energy:
                    best = landing
                selection_failed = False
                if direction_controller is not None:
                    # Selection precedes MC: retain it even when the landing is rejected.
                    try:
                        direction_controller.observe_landing(landing.atoms, rng)
                    except (ValueError, RuntimeError, FloatingPointError, StopIteration) as error:
                        selection_failed = True
                        status = run_status = 'direction_selection_failed'
                        error_message = f'landing_selection: {type(error).__name__}: {error}'
                if not selection_failed:
                    delta = landing.energy - current_energy
                    if mc is None:
                        accepted = delta <= 0 or (config.temperature_K > 0 and
                            rng.random() < math.exp(-delta / (units.kB * config.temperature_K)))
                    else:
                        uniform = rng.random()
                        try:
                            decision = native_metropolis(
                                current_energy, landing.energy, config.temperature_K,
                                energy_tol=mc.energy_tol, maxtrap=mc.maxtrap,
                                state=native_mc_state, uniform=uniform)
                            native_mc_state = decision.state
                            accepted = decision.accepted
                            mc_telemetry = decision
                        except (ValueError, OverflowError, TypeError) as error:
                            status = run_status = 'mc_failed'
                            error_message = f'mc: {type(error).__name__}: {error}'
                            mc_telemetry = {'error': error_message, 'uniform': uniform}
                if accepted:
                    current = landing.atoms.copy()
                    current_energy = landing.energy
                    current_observation_index = last_landing_index
        energy_response = None if prepared is None else prepared.energy_response
        update_record = None
        if prepared is not None and run_status != 'mc_failed':
            try:
                update_kwargs = dict(
                    energy_before=prepared.energy_before, energy_after=prepared.energy_after,
                    bond_energies=ls.bond_energies, bond_lengths=ls.bond_lengths)
                if isinstance(response, NativeLSRuntime):
                    update_kwargs.update(
                        prequench_exit_policy=getattr(getattr(ls, 'prequench', None), 'exit_policy', 'force'),
                        prequench_qualification=prepared.qualification,
                        prequench_telemetry=prepared.soft_quench.optimizer_telemetry)
                frozen = response.update(frozen, current, **update_kwargs)
                update_record = getattr(response, "last_update", None)
            except ValueError as error:
                # The paper finite domain has ended, e.g. negative amplitudes.
                # Preserve the completed work and stop, without amplitude clipping.
                status = 'ls_update_failed'
                run_status = 'ls_update_failed'
                error_message = str(error)
        ls_preparation = None if prepared is None else {
            'exit_policy': getattr(getattr(ls, 'prequench', None), 'exit_policy', 'force'),
            'qualification': prepared.qualification,
            'soft_quench': prepared.soft_quench,
            'true_energy_before': prepared.energy_before,
            'true_energy_after': prepared.energy_after,
            'evaluation_requests': prepared.evaluation_requests,
        }
        starter_selection = None
        if (starter_selector is not None and run_status == 'completed' and
                landing is not None and landing.converged):
            snapshot = snapshot_from_minima(
                minima, current_index=current_observation_index,
                last_landing_index=last_landing_index, step=index,
                cost=prior_requests + surface.requests - begin)
            chosen = validate_starter_index(
                starter_selector(snapshot, selector_rng), len(snapshot.observations))
            starter_selection = {
                'mc_current_index': current_observation_index,
                'chosen_index': chosen,
                'last_landing_index': last_landing_index,
                'step': index,
                'cost': snapshot.cost,
                'restarted': bool(chosen is not None and chosen != current_observation_index),
                'restart_failed': False,
                'ls_reinitialized': False,
                'direction_reinitialized': False,
            }
            if chosen is not None and chosen != current_observation_index:
                selected = minima[chosen]
                try:
                    restart_frozen, restart_response, restart_controller = _prepare_pool_restart(
                        selected.atoms, ls=ls, recovered_direction=recovered_direction,
                        rng=rng)
                except (ValueError, RuntimeError, FloatingPointError,
                        np.linalg.LinAlgError, StopIteration) as error:
                    if ls is None:
                        raise
                    status = run_status = 'starter_selection_failed'
                    error_message = f'pool_restart: {type(error).__name__}: {error}'
                    starter_selection.update(
                        restarted=False, restart_failed=True,
                        ls_reinitialized=False, direction_reinitialized=False,
                        error=error_message)
                else:
                    current = selected.atoms.copy()
                    current_energy = float(selected.energy)
                    current_observation_index = chosen
                    if ls is not None:
                        frozen, response = restart_frozen, restart_response
                    if direction_controller is not None:
                        direction_controller = restart_controller
                    starter_selection.update(
                        restarted=True, restart_failed=False,
                        ls_reinitialized=ls is not None,
                        direction_reinitialized=direction_controller is not None)
        records.append(SSWStep(index, status, bool(accepted), tuple(climb), landing,
                              energy_response, surface.requests - before,
                              error_message, work.copy(), initial_anchor, update_record,
                              cluster_reconnection, ls_preparation, mc_telemetry,
                              starter_selection))
        if identity_view is not None:
            from .minimum_identity import update_identity_view
            update_identity_view(identity_view, minima, structure_matcher)
        if checkpoint_enabled:
            checkpoint_result = SSWCheckpoint(
            _checkpoint_copy(initial), _checkpoint_copy(current), current_energy,
            _checkpoint_copy(best), tuple(_checkpoint_copy(minima)),
            tuple(_checkpoint_copy(records)), _checkpoint_copy(frozen),
            _checkpoint_copy(response), config, _checkpoint_copy(ls),
            _checkpoint_copy(height_policy), _checkpoint_copy(gaussian_policy),
            height_update_budget, reconnect_distance, deepcopy(rng.bit_generator.state),
                prior_requests + surface.requests - begin, index + 1, run_status,
                identity_view=_checkpoint_copy(identity_view),
                schema_version=(5 if pool_checkpoint_enabled else (4 if direction_controller is not None else (3 if recovered_rotation is not None else (2 if mc is not None else 1)))),
                mc_settings=_checkpoint_copy(mc),
                native_mc_state=_checkpoint_copy(native_mc_state),
                recovered_rotation=_checkpoint_copy(recovered_rotation),
                recovered_direction_state=(None if direction_controller is None else _checkpoint_copy(direction_controller.checkpoint_state())),
                pool_state=_checkpoint_copy(checkpoint_pool_state()))
            if checkpoint_path is not None:
                save_ssw_checkpoint(checkpoint_path, checkpoint_result)
        if run_status != 'completed':
            break
    if checkpoint_enabled and not records and checkpoint is None:
        checkpoint_result = SSWCheckpoint(_checkpoint_copy(initial), _checkpoint_copy(current), current_energy,
            _checkpoint_copy(best), tuple(_checkpoint_copy(minima)), tuple(), _checkpoint_copy(frozen),
            _checkpoint_copy(response), config, _checkpoint_copy(ls), _checkpoint_copy(height_policy),
            _checkpoint_copy(gaussian_policy), height_update_budget, reconnect_distance,
            deepcopy(rng.bit_generator.state), prior_requests + surface.requests - begin, 0, run_status,
            identity_view=_checkpoint_copy(identity_view),
            schema_version=(5 if pool_checkpoint_enabled else (4 if direction_controller is not None else (3 if recovered_rotation is not None else (2 if mc is not None else 1)))),
            mc_settings=_checkpoint_copy(mc),
            native_mc_state=_checkpoint_copy(native_mc_state),
            recovered_rotation=_checkpoint_copy(recovered_rotation),
            recovered_direction_state=(None if direction_controller is None else _checkpoint_copy(direction_controller.checkpoint_state())),
            pool_state=_checkpoint_copy(checkpoint_pool_state()))
        if checkpoint_path is not None:
            save_ssw_checkpoint(checkpoint_path, checkpoint_result)
    if checkpoint_path is not None and steps == 0 and checkpoint is not None:
        save_ssw_checkpoint(checkpoint_path, checkpoint_result)
    final_checkpoint = checkpoint_result if checkpoint_enabled else None
    return SSWResult(initial, current.copy(), best.atoms.copy(), tuple(minima),
                     tuple(records), prior_requests + surface.requests - begin, run_status,
                     final_checkpoint, identity_view)


def run_ls_ssw(atoms, surface, *, steps, config, rng, ls,
               reconnect_distance=None, height_policy=None, gaussian_policy=None,
               height_update_budget=1000, checkpoint=None, checkpoint_path=None,
               structure_matcher=None, bias_quench_adapter=None, mc=None):
    """Explicit LS entry point, sharing the independent paper SSW lifecycle."""
    if not isinstance(ls, LSSettings):
        raise TypeError('ls must be LSSettings with explicit bond tables and response target')
    kwargs = dict(reconnect_distance=reconnect_distance,
                  height_policy=height_policy,
                  gaussian_policy=gaussian_policy,
                  height_update_budget=height_update_budget,
                  checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                  structure_matcher=structure_matcher, mc=mc)
    if bias_quench_adapter is not None:
        kwargs['bias_quench_adapter'] = bias_quench_adapter
    return run_ssw(atoms, surface, steps=steps, config=config, rng=rng, ls=ls, **kwargs)
