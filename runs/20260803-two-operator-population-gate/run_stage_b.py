#!/usr/bin/env python3
"""Run the frozen direct-displacement versus H8 SSW paired gate."""

from __future__ import annotations

from copy import deepcopy
import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
OBSERVABILITY_PATH = (
    REPO_ROOT / "runs/20260801-uphill-action-observability/run_gate.py"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


@dataclass(frozen=True)
class StarterArtifact:
    system: str
    context: str
    path: Path
    sha256: str


def load_module(path: Path, name: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


observability = load_module(OBSERVABILITY_PATH, "_two_operator_observability")
protocol = load_module(PROTOCOL_PATH, "_two_operator_protocol_runner")


def file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_calculator(system: str, cuo_resources):
    return observability.base.source.source._calculator(system, cuo_resources)


def starter_artifacts() -> tuple[StarterArtifact, ...]:
    hashes = {
        ("c60", "bootstrap"): (
            "8c0ee78c143f9e285397a0fd5f6e2a0313b278811c8157afea30316f2b94fdc3"
        ),
        ("pdo", "bootstrap"): (
            "97efa24720cd1cf9942e417f736b8988b123bacb61915a04a3825c0ba89925f4"
        ),
        ("cuo", "bootstrap"): (
            "df3ee1d666d02ab0e3229d2aef210f56f9715ef9fbe3df56bbe714503b4a89a2"
        ),
        ("c60", "h8_best"): (
            "15c3d6bf022382bf4da22144accb639cf72980c3c55e89ff658f336b86827424"
        ),
        ("pdo", "h8_best"): (
            "b1aa09190b2e110e2ba2c6cd3264d234d148ec588155ad4ca29379c1343dcd4c"
        ),
        ("cuo", "h8_best"): (
            "1d3a732ec470a8fe455950f604f70023bcc5cf6edae042f62970fe030175f771"
        ),
    }
    rows = []
    for system in protocol.SYSTEMS:
        rows.append(
            StarterArtifact(
                system=system,
                context="bootstrap",
                path=REPO_ROOT
                / (
                    "runs/20260801-uphill-action-observability/confirm-output/"
                    f"{system}/seed-00000049/bootstrap/minimum.xyz"
                ),
                sha256=hashes[(system, "bootstrap")],
            )
        )
        rows.append(
            StarterArtifact(
                system=system,
                context="h8_best",
                path=REPO_ROOT
                / (
                    "runs/20260802-fixed-h4-h8-equal-budget/output/"
                    f"{system}/seed-00000049/h8/best_minimum.xyz"
                ),
                sha256=hashes[(system, "h8_best")],
            )
        )
    return tuple(rows)


def _starter_artifact(system: str, context: str) -> StarterArtifact:
    matches = [
        item
        for item in starter_artifacts()
        if item.system == system and item.context == context
    ]
    if len(matches) != 1:
        raise ValueError("starter artifact is not uniquely defined")
    return matches[0]


def load_locked_starter(system: str, context: str, cuo_resources):
    from pamssw.io import read_state

    artifact = _starter_artifact(system, context)
    if file_sha256(artifact.path) != artifact.sha256:
        raise RuntimeError(f"starter SHA256 drifted: {artifact.path}")
    coordinates = read_state(artifact.path)
    template = observability.base.source.source._load_state(system, cuo_resources)
    if not np.array_equal(coordinates.numbers, template.numbers):
        raise RuntimeError("starter atom ordering drifted from the frozen template")
    if coordinates.pbc != template.pbc:
        raise RuntimeError("starter PBC drifted from the frozen template")
    if coordinates.cell is None or template.cell is None:
        if coordinates.cell is not None or template.cell is not None:
            raise RuntimeError("starter cell rank drifted from the frozen template")
    elif not np.allclose(
        coordinates.cell,
        template.cell,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("starter cell drifted from the frozen template")
    return template.with_flat_positions(coordinates.positions.reshape(-1))


def build_config(system: str, case_directory: Path, *, seed: int):
    base = observability.build_config(
        system,
        Path(case_directory),
        seed=int(seed),
        force_budget=protocol.SSW_AND_SHARED_PREFIX_CAP,
    )
    return replace(
        base,
        max_trials=1,
        max_force_evals=protocol.SSW_AND_SHARED_PREFIX_CAP,
        max_steps_per_walk=8,
        direction_selection_mode="discrete",
        direction_ranking_mode="static_score",
        local_softening_scope="oracle",
        write_relaxation_trajectories=False,
        direction_diagnostics_enabled=False,
        proposal_pool_size=1,
    )


def direct_positions(positions, direction, sigma):
    return np.asarray(positions, dtype=float) + float(sigma) * np.asarray(
        direction,
        dtype=float,
    ).reshape((-1, 3))


def split_pair_cost(*, shared_direction_fe, direct_counts, ssw_counts):
    direct = sum(int(value) for value in direct_counts.values())
    ssw_exclusive = {str(key): int(value) for key, value in ssw_counts.items()}
    if ssw_exclusive.get("direction_oracle", 0) < int(shared_direction_fe):
        raise ValueError("shared direction cost exceeds SSW direction ledger")
    ssw_exclusive["direction_oracle"] -= int(shared_direction_fe)
    ssw = sum(ssw_exclusive.values())
    return {
        "shared_direction_force_evaluations": int(shared_direction_fe),
        "direct_force_evaluations": direct,
        "ssw_force_evaluations": ssw,
        "pair_force_evaluations": int(shared_direction_fe) + direct + ssw,
    }


def selected_cases(*, systems, starters, seeds):
    system_set = set(systems)
    starter_set = set(starters)
    seed_set = {int(seed) for seed in seeds}
    if not system_set.issubset(protocol.SYSTEMS):
        raise ValueError("unknown selected system")
    if not starter_set.issubset(protocol.STARTERS):
        raise ValueError("unknown selected starter")
    if not seed_set.issubset(protocol.SEEDS):
        raise ValueError("unknown selected seed")
    return tuple(
        item
        for item in protocol.case_matrix()
        if item.system in system_set
        and item.starter_context in starter_set
        and item.seed in seed_set
    )


def validate_pair_record(pair) -> None:
    system = str(pair["system"])
    starter_context = str(pair["starter_context"])
    seed = int(pair["seed"])
    if (system, starter_context, seed) not in {
        (item.system, item.starter_context, item.seed)
        for item in protocol.case_matrix()
    }:
        raise ValueError("pair key is outside the preregistered cohort")
    artifact = _starter_artifact(system, starter_context)
    if str(pair["starter_sha256"]) != artifact.sha256:
        raise ValueError("pair starter SHA256 drifted")
    if int(pair["fixed_atom_count"]) != protocol.FIXED_ATOM_COUNTS[system]:
        raise ValueError("pair fixed-atom count drifted")

    actions = list(pair["actions"])
    if len(actions) != 2 or {row["operator_family"] for row in actions} != set(
        protocol.FAMILIES
    ):
        raise ValueError("pair does not contain exactly one action per family")
    for action in actions:
        if (
            action["system"],
            action["starter_context"],
            int(action["seed"]),
        ) != (system, starter_context, seed):
            raise ValueError("action key does not match pair key")
        protocol.validate_row(action)

    starter_counts = {
        str(key): int(value)
        for key, value in pair["starter_validation_purpose_counts"].items()
    }
    if starter_counts.get("unattributed", 0) != 0:
        raise ValueError("starter validation contains unattributed work")
    shared_direction = int(pair["shared_direction_force_evaluations"])
    if shared_direction < 0:
        raise ValueError("shared direction force evaluations are negative")
    expected_total = (
        sum(starter_counts.values())
        + shared_direction
        + sum(int(action["force_evaluations"]) for action in actions)
    )
    if int(pair["pair_force_evaluations"]) != expected_total:
        raise ValueError("pair force ledger does not close")
    if expected_total > protocol.PAIR_SUBMISSION_CAP:
        raise ValueError("pair exceeded its preregistered force cap")
    if pair.get("direction_sha256") is None and not bool(
        pair.get("paired_input_censored", False)
    ):
        raise ValueError("uncensored pair has no direction hash")


def counts_delta(after, before) -> dict[str, int]:
    after_map = after.as_dict()
    before_map = before.as_dict()
    delta = {
        name: int(after_map[name]) - int(before_map[name])
        for name in after_map
    }
    if any(value < 0 for value in delta.values()):
        raise ValueError("evaluation counters moved backwards")
    return delta


def without_shared_direction(counts, shared_direction_fe) -> dict[str, int]:
    result = {str(key): int(value) for key, value in counts.items()}
    result["direction_oracle"] -= int(shared_direction_fe)
    if result["direction_oracle"] < 0:
        raise ValueError("shared direction cost exceeds action direction cost")
    return result


def landing_action_row(
    *,
    system,
    starter_context,
    seed,
    family,
    starter_state,
    starter_energy,
    landing,
    walker,
    counts,
    config,
    wall_time_s,
    budget_censored=False,
    basin_label_mode="archive",
):
    from pamssw.archive import MinimaArchive
    from pamssw.relax import has_force_convergence_certificate

    counts = {str(key): int(value) for key, value in counts.items()}
    if basin_label_mode not in {"archive", "geometry_primary"}:
        raise ValueError("unknown basin label mode")
    if landing is None:
        return {
            "system": system,
            "starter_context": starter_context,
            "seed": int(seed),
            "operator_family": family,
            "certified": False,
            "same_starter_basin": False,
            "geometry_valid": False,
            "fragmented": False,
            "budget_censored": bool(budget_censored),
            "landing_delta_eV": None,
            "starter_landing_rmsd_A": None,
            "starter_landing_rmsd_finite": None,
            "archive_same_starter_basin": None,
            "basin_label_mode": basin_label_mode,
            "improved_global_best": False,
            "force_evaluations": sum(counts.values()),
            "wall_time_s": float(wall_time_s),
            "purpose_counts": counts,
        }

    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    starter_entry = archive.add(starter_state, float(starter_energy), parent_id=None)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=starter_entry.entry_id,
    )
    geometry_valid = bool(walker.geometry_validator.is_valid_state(landing.state))
    fragmented = bool(walker._is_fragmented_cluster(starter_state, landing.state))
    certified = bool(
        geometry_valid
        and not fragmented
        and has_force_convergence_certificate(landing, config.quench_fmax)
    )
    starter_landing_rmsd = MinimaArchive._rmsd(starter_state, landing.state)
    starter_landing_rmsd_finite = bool(np.isfinite(starter_landing_rmsd))
    archive_same_starter_basin = bool(
        landing_entry.entry_id == starter_entry.entry_id
    )
    same_starter_basin = (
        archive_same_starter_basin
        if basin_label_mode == "archive"
        else starter_landing_rmsd <= float(config.dedup_rmsd_tol)
    )
    return {
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "operator_family": family,
        "certified": certified,
        "same_starter_basin": bool(same_starter_basin),
        "archive_same_starter_basin": archive_same_starter_basin,
        "starter_landing_rmsd_A": (
            float(starter_landing_rmsd)
            if starter_landing_rmsd_finite
            else None
        ),
        "starter_landing_rmsd_finite": starter_landing_rmsd_finite,
        "basin_label_mode": basin_label_mode,
        "geometry_valid": geometry_valid,
        "fragmented": fragmented,
        "budget_censored": bool(budget_censored),
        "landing_delta_eV": float(landing.energy) - float(starter_energy),
        "improved_global_best": bool(float(landing.energy) < float(starter_energy)),
        "force_evaluations": sum(counts.values()),
        "wall_time_s": float(wall_time_s),
        "purpose_counts": counts,
    }


def serialize_pair(
    *,
    system,
    starter_context,
    seed,
    starter_state,
    starter_energy,
    direction,
    sigma,
    ssw_trace,
    ssw_landing,
    ssw_walker,
    ssw_counts,
    direct_landing,
    direct_walker,
    direct_counts,
    config,
    starter_artifact,
    starter_validation_counts,
    ssw_budget_censored=False,
    direct_budget_censored=False,
    starter_validation_wall_time_s=0.0,
    initial_direction_wall_time_s=0.0,
    ssw_wall_time_s=0.0,
    direct_wall_time_s=0.0,
    starter_preparation_mode="single_point_certificate",
    basin_label_mode="archive",
):
    shared_direction_fe = int(
        ssw_trace.steps[0].direction_oracle_force_evaluations
    )
    ssw_exclusive = without_shared_direction(ssw_counts, shared_direction_fe)
    actions = [
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="direct",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=direct_landing,
            walker=direct_walker,
            counts=direct_counts,
            config=config,
            wall_time_s=direct_wall_time_s,
            budget_censored=direct_budget_censored,
            basin_label_mode=basin_label_mode,
        ),
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="ssw",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=ssw_landing,
            walker=ssw_walker,
            counts=ssw_exclusive,
            config=config,
            wall_time_s=max(
                0.0,
                float(ssw_wall_time_s) - float(initial_direction_wall_time_s),
            ),
            budget_censored=ssw_budget_censored,
            basin_label_mode=basin_label_mode,
        ),
    ]
    for action in actions:
        action["fully_loaded_force_evaluations"] = (
            int(action["force_evaluations"]) + shared_direction_fe
        )
        action["fully_loaded_wall_time_s"] = (
            float(action["wall_time_s"]) + float(initial_direction_wall_time_s)
        )
        protocol.validate_row(action)

    return {
        "schema_version": 1,
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "starter_path": str(starter_artifact.path.relative_to(REPO_ROOT)),
        "starter_sha256": starter_artifact.sha256,
        "fixed_atom_count": int(np.count_nonzero(starter_state.fixed_mask)),
        "starter_preparation_mode": starter_preparation_mode,
        "basin_label_mode": basin_label_mode,
        "direction_sha256": sha256(
            np.asarray(direction, dtype=np.float64).tobytes()
        ).hexdigest(),
        "execution_sigma": float(sigma),
        "shared_direction_force_evaluations": shared_direction_fe,
        "shared_initial_direction_wall_time_s": float(
            initial_direction_wall_time_s
        ),
        "starter_validation_purpose_counts": {
            str(key): int(value)
            for key, value in starter_validation_counts.items()
        },
        "starter_validation_wall_time_s": float(starter_validation_wall_time_s),
        "actions": actions,
        "pair_force_evaluations": (
            sum(int(value) for value in starter_validation_counts.values())
            + shared_direction_fe
            + sum(int(action["force_evaluations"]) for action in actions)
        ),
        "pair_wall_time_s": (
            float(starter_validation_wall_time_s)
            + float(ssw_wall_time_s)
            + float(direct_wall_time_s)
        ),
        "effective_config": asdict(config),
    }


def write_prefix_censored_pair(
    *,
    system,
    starter_context,
    seed,
    starter_artifact,
    starter_state,
    starter_energy,
    starter_validation_counts,
    ssw_counts,
    ssw_walker,
    config,
    output_directory,
    starter_validation_wall_time_s,
    prefix_wall_time_s,
    starter_preparation_mode="single_point_certificate",
    basin_label_mode="archive",
):
    from pamssw.io import write_state

    zero_counts = {name: 0 for name in ssw_counts}
    prefix_budget_exhausted = bool(ssw_walker.calculator.exhausted())
    actions = [
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="direct",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=None,
            walker=ssw_walker,
            counts=zero_counts,
            config=config,
            wall_time_s=0.0,
            budget_censored=prefix_budget_exhausted,
            basin_label_mode=basin_label_mode,
        ),
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="ssw",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=None,
            walker=ssw_walker,
            counts=ssw_counts,
            config=config,
            wall_time_s=prefix_wall_time_s,
            budget_censored=prefix_budget_exhausted,
            basin_label_mode=basin_label_mode,
        ),
    ]
    for action in actions:
        action["fully_loaded_force_evaluations"] = int(
            action["force_evaluations"]
        )
        action["fully_loaded_wall_time_s"] = float(action["wall_time_s"])
        protocol.validate_row(action)
    row = {
        "schema_version": 1,
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "starter_path": str(starter_artifact.path.relative_to(REPO_ROOT)),
        "starter_sha256": starter_artifact.sha256,
        "fixed_atom_count": int(np.count_nonzero(starter_state.fixed_mask)),
        "starter_preparation_mode": starter_preparation_mode,
        "basin_label_mode": basin_label_mode,
        "direction_sha256": None,
        "execution_sigma": None,
        "paired_input_censored": True,
        "shared_direction_force_evaluations": 0,
        "shared_initial_direction_wall_time_s": 0.0,
        "starter_validation_purpose_counts": {
            str(key): int(value)
            for key, value in starter_validation_counts.items()
        },
        "starter_validation_wall_time_s": float(starter_validation_wall_time_s),
        "actions": actions,
        "pair_force_evaluations": (
            sum(int(value) for value in starter_validation_counts.values())
            + sum(int(value) for value in ssw_counts.values())
        ),
        "pair_wall_time_s": (
            float(starter_validation_wall_time_s) + float(prefix_wall_time_s)
        ),
        "effective_config": asdict(config),
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_state(output_directory / "starter.xyz", starter_state)
    write_json(output_directory / "pair.json", row)
    return row


def run_pair(
    *,
    system: str,
    starter_context: str,
    seed: int,
    output_directory: Path,
    cuo_resources,
    shared_bootstrap_true_quench: bool = False,
    basin_label_mode: str = "archive",
):
    from pamssw.accounting import BudgetExceeded, EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.coordinates import CartesianCoordinates, TangentVector
    from pamssw.io import write_state
    from pamssw.walker import SurfaceWalker

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact = _starter_artifact(system, starter_context)
    starter = load_locked_starter(system, starter_context, cuo_resources)
    fixed_count = int(np.count_nonzero(starter.fixed_mask))
    if fixed_count != protocol.FIXED_ATOM_COUNTS[system]:
        raise RuntimeError("locked starter constraint count drifted")

    config = build_config(system, output_directory, seed=seed)
    ssw = SurfaceWalker(
        calculator=make_calculator(system, cuo_resources),
        config=config,
        softening_enabled=True,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )

    starter_validation_started = perf_counter()
    if shared_bootstrap_true_quench:
        starter_result = ssw.relax_true_minimum(
            starter,
            trajectory_name="shared-starter-bootstrap",
            quench_purpose=EvaluationPurpose.STARTER_TRUE_QUENCH,
        )
        starter_state = starter_result.state
        starter_energy = float(starter_result.energy)
        starter_max_force = float(starter_result.gradient_norm)
        starter_preparation_mode = "shared_true_quench"
    else:
        with ssw.calculator.purpose(EvaluationPurpose.STARTER_TRUE_QUENCH):
            starter_evaluation = ssw.calculator.evaluate(starter)
        starter_state = starter
        starter_energy = float(starter_evaluation.energy)
        movable_forces = np.linalg.norm(
            -starter_evaluation.gradient,
            axis=1,
        )[starter.movable_mask]
        starter_max_force = (
            float(np.max(movable_forces)) if movable_forces.size else 0.0
        )
        starter_preparation_mode = "single_point_certificate"
    starter_validation_wall_time_s = perf_counter() - starter_validation_started
    if starter_max_force > config.quench_fmax:
        raise RuntimeError(
            f"locked starter lost its force certificate: {starter_max_force}"
        )
    starter_entry = archive.add(starter_state, starter_energy, parent_id=None)
    starter_validation_counts = ssw.calculator.snapshot().as_dict()

    captured = {}
    original_choose = ssw.oracle.choose_direction
    original_record_control = ssw._record_uphill_control

    def capture_choice(*args, **kwargs):
        direction_started = perf_counter()
        choice = original_choose(*args, **kwargs)
        if "choice" not in captured:
            captured["choice"] = deepcopy(choice)
            captured["direction_started"] = direction_started
        return choice

    def capture_control(*args, **kwargs):
        if "initial_direction_wall_time_s" not in captured:
            captured["initial_direction_wall_time_s"] = (
                perf_counter() - captured["direction_started"]
            )
        return original_record_control(*args, **kwargs)

    ssw.oracle.choose_direction = capture_choice
    ssw._record_uphill_control = capture_control
    prefix_traces = []
    continuations = []
    before_ssw = ssw.calculator.snapshot()
    prefix_started = perf_counter()
    try:
        ssw._walk_candidate_from_seed(
            starter_state,
            archive,
            ssw.step_target_controller.target(archive),
            trial_index=0,
            proposal_index=0,
            seed_entry_id=starter_entry.entry_id,
            trace_sink=prefix_traces,
            pause_after_step=0,
            continuation_sink=continuations,
        )
    except BudgetExceeded:
        prefix_wall_time_s = perf_counter() - prefix_started
        return write_prefix_censored_pair(
            system=system,
            starter_context=starter_context,
            seed=seed,
            starter_artifact=artifact,
            starter_state=starter_state,
            starter_energy=starter_energy,
            starter_validation_counts=starter_validation_counts,
            ssw_counts=counts_delta(ssw.calculator.snapshot(), before_ssw),
            ssw_walker=ssw,
            config=config,
            output_directory=output_directory,
            starter_validation_wall_time_s=starter_validation_wall_time_s,
            prefix_wall_time_s=prefix_wall_time_s,
            starter_preparation_mode=starter_preparation_mode,
            basin_label_mode=basin_label_mode,
        )
    prefix_wall_time_s = perf_counter() - prefix_started
    if (
        len(prefix_traces) != 1
        or len(prefix_traces[0].steps) != 1
        or len(continuations) != 1
        or "choice" not in captured
        or "initial_direction_wall_time_s" not in captured
    ):
        raise RuntimeError("SSW arm did not expose one complete shared input")
    first_step = prefix_traces[0].steps[0]
    direction = np.asarray(captured["choice"].direction, dtype=float)
    sigma = float(first_step.executed_sigma)

    direct_config = replace(config, max_force_evals=protocol.DIRECT_ACTION_CAP)
    direct = SurfaceWalker(
        calculator=make_calculator(system, cuo_resources),
        config=direct_config,
        softening_enabled=False,
    )
    direct_state = CartesianCoordinates.from_state(starter_state).displace(
        TangentVector(direction),
        sigma,
    )
    direct_budget_censored = False
    direct_started = perf_counter()
    if not direct.geometry_validator.is_valid_state(direct_state):
        direct_landing = None
    else:
        try:
            direct_landing = direct.relax_true_minimum(
                direct_state,
                trajectory_name="direct-landing",
            )
        except BudgetExceeded:
            direct_landing = None
            direct_budget_censored = bool(direct.calculator.exhausted())
    direct_wall_time_s = perf_counter() - direct_started

    traces = []
    ssw_budget_censored = False
    continuation_started = perf_counter()
    try:
        escape = ssw._walk_candidate_from_seed(
            starter_state,
            archive,
            trial_index=0,
            proposal_index=0,
            seed_entry_id=starter_entry.entry_id,
            trace_sink=traces,
            continuation=continuations[0],
        )
        ssw_landing = ssw.relax_true_minimum(
            escape,
            trajectory_name="ssw-landing",
        )
    except BudgetExceeded:
        ssw_landing = None
        ssw_budget_censored = bool(ssw.calculator.exhausted())
        if not traces:
            traces = prefix_traces
    continuation_wall_time_s = perf_counter() - continuation_started
    after_ssw = ssw.calculator.snapshot()

    row = serialize_pair(
        system=system,
        starter_context=starter_context,
        seed=seed,
        starter_state=starter_state,
        starter_energy=starter_energy,
        direction=direction,
        sigma=sigma,
        ssw_trace=traces[0],
        ssw_landing=ssw_landing,
        ssw_walker=ssw,
        ssw_counts=counts_delta(after_ssw, before_ssw),
        direct_landing=direct_landing,
        direct_walker=direct,
        direct_counts=direct.calculator.snapshot().as_dict(),
        config=config,
        starter_artifact=artifact,
        starter_validation_counts=starter_validation_counts,
        ssw_budget_censored=ssw_budget_censored,
        direct_budget_censored=direct_budget_censored,
        starter_validation_wall_time_s=starter_validation_wall_time_s,
        initial_direction_wall_time_s=float(
            captured["initial_direction_wall_time_s"]
        ),
        ssw_wall_time_s=prefix_wall_time_s + continuation_wall_time_s,
        direct_wall_time_s=direct_wall_time_s,
        starter_preparation_mode=starter_preparation_mode,
        basin_label_mode=basin_label_mode,
    )
    write_state(output_directory / "starter.xyz", starter_state)
    if ssw_landing is not None:
        write_state(output_directory / "ssw-landing.xyz", ssw_landing.state)
    if direct_landing is not None:
        write_state(output_directory / "direct-landing.xyz", direct_landing.state)
    write_json(output_directory / "pair.json", row)
    return row


def _case_directory(output_directory: Path, case) -> Path:
    return (
        Path(output_directory)
        / case.system
        / case.starter_context
        / f"seed-{case.seed:08d}"
    )


def _display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_campaign(
    *,
    output_directory: Path,
    systems,
    starters,
    seeds,
    max_force_evaluations: int,
    cuo_resources,
    shared_bootstrap_true_quench: bool = False,
    basin_label_mode: str = "archive",
):
    if (
        isinstance(max_force_evaluations, bool)
        or not isinstance(max_force_evaluations, int)
        or max_force_evaluations <= 0
    ):
        raise ValueError("max_force_evaluations must be a positive integer")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(systems=systems, starters=starters, seeds=seeds)
    if not cases:
        raise ValueError("campaign selection is empty")

    rows = []
    consumed = 0
    for case in cases:
        case_directory = _case_directory(output_directory, case)
        pair_path = case_directory / "pair.json"
        if pair_path.is_file():
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            validate_pair_record(pair)
            expected_preparation = (
                "shared_true_quench"
                if shared_bootstrap_true_quench
                else "single_point_certificate"
            )
            if pair.get(
                "starter_preparation_mode",
                "single_point_certificate",
            ) != expected_preparation or pair.get(
                "basin_label_mode",
                "archive",
            ) != basin_label_mode:
                raise RuntimeError("existing pair execution mode does not match request")
        else:
            if max_force_evaluations - consumed < protocol.PAIR_SUBMISSION_CAP:
                raise RuntimeError(
                    "remaining campaign budget cannot reserve the next pair"
                )
            pair = run_pair(
                system=case.system,
                starter_context=case.starter_context,
                seed=case.seed,
                output_directory=case_directory,
                cuo_resources=cuo_resources,
                shared_bootstrap_true_quench=shared_bootstrap_true_quench,
                basin_label_mode=basin_label_mode,
            )
            validate_pair_record(pair)
        consumed += int(pair["pair_force_evaluations"])
        if consumed > max_force_evaluations:
            raise RuntimeError("campaign exceeded its force-evaluation budget")
        rows.append(pair)
        write_json(
            output_directory / "campaign.json",
            {
                "schema_version": 1,
                "max_force_evaluations": max_force_evaluations,
                "force_evaluations": consumed,
                "completed_pairs": len(rows),
                "expected_pairs": len(cases),
                "starter_preparation_mode": (
                    "shared_true_quench"
                    if shared_bootstrap_true_quench
                    else "single_point_certificate"
                ),
                "basin_label_mode": basin_label_mode,
                "cases": [
                    {
                        "system": row["system"],
                        "starter_context": row["starter_context"],
                        "seed": int(row["seed"]),
                        "pair_force_evaluations": int(
                            row["pair_force_evaluations"]
                        ),
                        "pair_path": _display_path(
                            _case_directory(
                                output_directory,
                                protocol.CaseSpec(
                                    row["system"],
                                    row["starter_context"],
                                    int(row["seed"]),
                                ),
                            )
                            / "pair.json"
                        ),
                    }
                    for row in rows
                ],
            },
        )
    return {
        "schema_version": 1,
        "max_force_evaluations": max_force_evaluations,
        "force_evaluations": consumed,
        "completed_pairs": len(rows),
        "expected_pairs": len(cases),
        "starter_preparation_mode": (
            "shared_true_quench"
            if shared_bootstrap_true_quench
            else "single_point_certificate"
        ),
        "basin_label_mode": basin_label_mode,
        "pairs": rows,
    }


def require_stage_a_admission(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Stage A evidence is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != "ADMIT_FRESH_STAGE_B":
        raise RuntimeError("Stage A did not admit fresh Stage B execution")
    if int(payload.get("new_force_evaluations", -1)) != 0:
        raise RuntimeError("Stage A evidence does not certify zero new FE")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=protocol.SYSTEMS,
        default=list(protocol.SYSTEMS),
    )
    parser.add_argument(
        "--starters",
        nargs="+",
        choices=protocol.STARTERS,
        default=list(protocol.STARTERS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=protocol.SEEDS,
        default=list(protocol.SEEDS),
    )
    parser.add_argument(
        "--max-force-evaluations",
        type=int,
        default=protocol.MAX_FORCE_EVALUATIONS,
    )
    parser.add_argument(
        "--stage-a-evidence",
        type=Path,
        default=RUN_ROOT / "output/stage-a.json",
    )
    parser.add_argument(
        "--shared-bootstrap-true-quench",
        action="store_true",
        help="true-quench each shared starter once before branching",
    )
    parser.add_argument(
        "--basin-label-mode",
        choices=("archive", "geometry_primary"),
        default="archive",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    require_stage_a_admission(args.stage_a_evidence)
    source = observability.base.source.source
    if "cuo" not in args.systems:
        result = run_campaign(
            output_directory=args.output,
            systems=tuple(args.systems),
            starters=tuple(args.starters),
            seeds=tuple(args.seeds),
            max_force_evaluations=args.max_force_evaluations,
            cuo_resources=None,
            shared_bootstrap_true_quench=args.shared_bootstrap_true_quench,
            basin_label_mode=args.basin_label_mode,
        )
    else:
        with TemporaryDirectory(prefix="pamssw-two-operator-cuo-") as temporary:
            cuo_resources = source._materialize_cuo_resources(
                source.CUO_ARCHIVE_PATH,
                Path(temporary) / "resources",
            )
            result = run_campaign(
                output_directory=args.output,
                systems=tuple(args.systems),
                starters=tuple(args.starters),
                seeds=tuple(args.seeds),
                max_force_evaluations=args.max_force_evaluations,
                cuo_resources=cuo_resources,
                shared_bootstrap_true_quench=args.shared_bootstrap_true_quench,
                basin_label_mode=args.basin_label_mode,
            )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "pairs"
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
