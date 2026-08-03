#!/usr/bin/env python3
"""Run the cumulative- versus newest-Gaussian full-action gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_PATH = (
    REPO_ROOT
    / "runs"
    / "20260803-two-operator-population-gate"
    / "run_stage_b.py"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


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


protocol = load_module(PROTOCOL_PATH, "_bias_history_action_protocol")


def source_module():
    return load_module(SOURCE_PATH, "_bias_history_source_runner")


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def split_pair_cost(
    *,
    starter_counts,
    shared_prefix_counts,
    arm_counts,
):
    starter = sum(int(value) for value in starter_counts.values())
    shared = sum(int(value) for value in shared_prefix_counts.values())
    exclusive = {
        family: sum(int(value) for value in counts.values())
        for family, counts in arm_counts.items()
    }
    return {
        "starter_force_evaluations": starter,
        "shared_prefix_force_evaluations": shared,
        "exclusive_force_evaluations": exclusive,
        "fully_loaded_force_evaluations": {
            family: shared + cost for family, cost in exclusive.items()
        },
        "pair_force_evaluations": starter + shared + sum(exclusive.values()),
    }


def _history_walker_class(family: str):
    from pamssw.walker import SurfaceWalker

    if family == "cumulative":
        class CumulativeWalker(SurfaceWalker):
            bias_history_mode = "cumulative"

            def _active_walk_biases(self, biases):
                active = super()._active_walk_biases(biases)
                self.bias_history_observations.append(
                    (len(biases), len(active))
                )
                return active

        return CumulativeWalker
    if family == "newest_only":
        class NewestOnlyWalker(SurfaceWalker):
            bias_history_mode = "newest_only"

            def _active_walk_biases(self, biases):
                active = tuple(biases[-1:])
                self.bias_history_observations.append(
                    (len(biases), len(active))
                )
                return active

        return NewestOnlyWalker
    raise ValueError(f"unknown bias-history family: {family}")


def _make_history_walker(family, *, calculator, config):
    cls = _history_walker_class(family)
    walker = cls(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    walker.bias_history_observations = []
    return walker


def _trace_summary(trace, walker) -> dict:
    observations = [
        {"stored": int(stored), "active": int(active)}
        for stored, active in walker.bias_history_observations
    ]
    return {
        "termination_reason": trace.termination_reason if trace is not None else None,
        "completed_microsteps": len(trace.steps) if trace is not None else 0,
        "bias_history_mode": walker.bias_history_mode,
        "bias_history_observations": observations,
        "history_reduction_active": any(
            item["active"] < item["stored"] for item in observations
        ),
    }


def _run_arm(
    *,
    family,
    system,
    starter_state,
    archive,
    continuation,
    config,
    cuo_resources,
):
    from pamssw.accounting import BudgetExceeded

    src = source_module()
    walker = _make_history_walker(
        family,
        calculator=src.make_calculator(system, cuo_resources),
        config=config,
    )
    traces = []
    landing = None
    budget_censored = False
    started = perf_counter()
    try:
        escape = walker._walk_candidate_from_seed(
            starter_state,
            archive,
            trial_index=0,
            proposal_index=0,
            seed_entry_id=0,
            continuation=deepcopy(continuation),
            trace_sink=traces,
        )
        landing = walker.relax_true_minimum(
            escape,
            trajectory_name=f"{family}-landing",
        )
    except BudgetExceeded:
        budget_censored = bool(walker.calculator.exhausted())
    wall_time_s = perf_counter() - started
    trace = traces[0] if traces else None
    return {
        "walker": walker,
        "landing": landing,
        "counts": walker.calculator.snapshot().as_dict(),
        "wall_time_s": wall_time_s,
        "budget_censored": budget_censored,
        "trace": trace,
        "trace_summary": _trace_summary(trace, walker),
    }


def _case_directory(output: Path, case) -> Path:
    return (
        Path(output)
        / case.system
        / case.starter_context
        / f"seed-{case.seed:08d}"
    )


def validate_pair(pair) -> None:
    if pair.get("starter_preparation_mode") != "shared_true_quench":
        raise ValueError("pair lacks shared true-quench preparation")
    if pair.get("basin_label_mode") != "geometry_primary":
        raise ValueError("pair lacks geometry-primary labels")
    actions = pair.get("actions", [])
    if {row["operator_family"] for row in actions} != set(protocol.FAMILIES):
        raise ValueError("pair does not contain both history families")
    for row in actions:
        protocol.validate_row(row)
    expected = (
        sum(int(value) for value in pair["starter_validation_purpose_counts"].values())
        + sum(int(value) for value in pair["shared_prefix_purpose_counts"].values())
        + sum(int(row["force_evaluations"]) for row in actions)
    )
    if expected != int(pair["pair_force_evaluations"]):
        raise ValueError("pair purpose ledger does not close")


def run_pair(
    *,
    system: str,
    starter_context: str,
    seed: int,
    output_directory: Path,
    cuo_resources,
):
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.io import write_state
    from pamssw.walker import SurfaceWalker

    src = source_module()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact = src._starter_artifact(system, starter_context)
    locked = src.load_locked_starter(system, starter_context, cuo_resources)
    if int(np.count_nonzero(locked.fixed_mask)) != protocol.FIXED_ATOM_COUNTS[system]:
        raise RuntimeError("locked starter constraint count drifted")
    base_config = replace(
        src.build_config(system, output_directory, seed=seed),
        max_force_evals=src.protocol.SSW_AND_SHARED_PREFIX_CAP,
    )

    prefix_walker = SurfaceWalker(
        calculator=src.make_calculator(system, cuo_resources),
        config=base_config,
        softening_enabled=True,
    )
    validation_started = perf_counter()
    starter_result = prefix_walker.relax_true_minimum(
        locked,
        trajectory_name="shared-starter-bootstrap",
        quench_purpose=EvaluationPurpose.STARTER_TRUE_QUENCH,
    )
    starter_state = starter_result.state
    starter_energy = float(starter_result.energy)
    validation_wall_time_s = perf_counter() - validation_started
    if float(starter_result.gradient_norm) > base_config.quench_fmax:
        raise RuntimeError("locked starter lost its force certificate")
    archive = MinimaArchive(
        energy_tol=base_config.dedup_energy_tol,
        rmsd_tol=base_config.dedup_rmsd_tol,
        max_prototypes=base_config.max_prototypes,
    )
    starter_entry = archive.add(starter_state, starter_energy, parent_id=None)
    validation_counts = prefix_walker.calculator.snapshot().as_dict()

    captured = {}
    original_choose = prefix_walker.oracle.choose_direction

    def capture_choice(*args, **kwargs):
        choice = original_choose(*args, **kwargs)
        if "choice" not in captured:
            captured["choice"] = deepcopy(choice)
        return choice

    prefix_walker.oracle.choose_direction = capture_choice
    before_prefix = prefix_walker.calculator.snapshot()
    prefix_started = perf_counter()
    continuations = []
    prefix_traces = []
    prefix_walker._walk_candidate_from_seed(
        starter_state,
        archive,
        prefix_walker.step_target_controller.target(archive),
        trial_index=0,
        proposal_index=0,
        seed_entry_id=starter_entry.entry_id,
        trace_sink=prefix_traces,
        pause_after_step=0,
        continuation_sink=continuations,
    )
    prefix_wall_time_s = perf_counter() - prefix_started
    prefix_counts = src.counts_delta(
        prefix_walker.calculator.snapshot(),
        before_prefix,
    )
    if len(continuations) != 1 or len(prefix_traces) != 1 or "choice" not in captured:
        raise RuntimeError("shared step-0 continuation was not captured")
    if len(prefix_traces[0].steps) != 1:
        raise RuntimeError("shared prefix did not complete exactly one microstep")
    direction = np.asarray(captured["choice"].direction, dtype=np.float64)
    sigma = float(prefix_traces[0].steps[0].executed_sigma)

    arms = {
        family: _run_arm(
            family=family,
            system=system,
            starter_state=starter_state,
            archive=archive,
            continuation=continuations[0],
            config=base_config,
            cuo_resources=cuo_resources,
        )
        for family in protocol.FAMILIES
    }
    action_rows = []
    shared_prefix_fe = sum(int(value) for value in prefix_counts.values())
    for family in protocol.FAMILIES:
        arm = arms[family]
        row = src.landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family=family,
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=arm["landing"],
            walker=arm["walker"],
            counts=arm["counts"],
            config=base_config,
            wall_time_s=arm["wall_time_s"],
            budget_censored=arm["budget_censored"],
            basin_label_mode="geometry_primary",
        )
        row["fully_loaded_force_evaluations"] = (
            int(row["force_evaluations"]) + shared_prefix_fe
        )
        row["fully_loaded_wall_time_s"] = (
            float(row["wall_time_s"]) + prefix_wall_time_s
        )
        row["trace"] = arm["trace_summary"]
        protocol.validate_row(row)
        action_rows.append(row)

    costs = split_pair_cost(
        starter_counts=validation_counts,
        shared_prefix_counts=prefix_counts,
        arm_counts={family: arms[family]["counts"] for family in protocol.FAMILIES},
    )
    pair = {
        "schema_version": 1,
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "starter_path": str(artifact.path.relative_to(REPO_ROOT)),
        "starter_sha256": artifact.sha256,
        "fixed_atom_count": int(np.count_nonzero(starter_state.fixed_mask)),
        "starter_preparation_mode": "shared_true_quench",
        "basin_label_mode": "geometry_primary",
        "direction_sha256": sha256(direction.tobytes()).hexdigest(),
        "execution_sigma": sigma,
        "starter_validation_purpose_counts": validation_counts,
        "shared_prefix_purpose_counts": prefix_counts,
        "shared_prefix_force_evaluations": costs["shared_prefix_force_evaluations"],
        "starter_validation_wall_time_s": validation_wall_time_s,
        "shared_prefix_wall_time_s": prefix_wall_time_s,
        "actions": action_rows,
        "pair_force_evaluations": costs["pair_force_evaluations"],
        "pair_wall_time_s": (
            validation_wall_time_s
            + prefix_wall_time_s
            + sum(float(arms[family]["wall_time_s"]) for family in protocol.FAMILIES)
        ),
        "effective_config": asdict(base_config),
    }
    validate_pair(pair)
    write_state(output_directory / "starter.xyz", starter_state)
    for family in protocol.FAMILIES:
        if arms[family]["landing"] is not None:
            write_state(
                output_directory / f"{family}-landing.xyz",
                arms[family]["landing"].state,
            )
    write_json(output_directory / "pair.json", pair)
    return pair


def run_campaign(*, output, systems, starters, seeds, max_force_evaluations, cuo_resources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        case
        for case in protocol.case_matrix()
        if case.system in systems
        and case.starter_context in starters
        and case.seed in seeds
    ]
    consumed = 0
    pairs = []
    for case in selected:
        path = _case_directory(output, case) / "pair.json"
        if path.is_file():
            pair = json.loads(path.read_text(encoding="utf-8"))
            validate_pair(pair)
        else:
            if max_force_evaluations - consumed < protocol.PAIR_SUBMISSION_CAP:
                raise RuntimeError("remaining campaign budget cannot reserve next pair")
            pair = run_pair(
                system=case.system,
                starter_context=case.starter_context,
                seed=case.seed,
                output_directory=path.parent,
                cuo_resources=cuo_resources,
            )
        consumed += int(pair["pair_force_evaluations"])
        if consumed > max_force_evaluations:
            raise RuntimeError("campaign exceeded force-evaluation budget")
        pairs.append(pair)
        write_json(
            output / "campaign.json",
            {
                "schema_version": 1,
                "execution_commit": git_commit(),
                "max_force_evaluations": int(max_force_evaluations),
                "force_evaluations": consumed,
                "completed_pairs": len(pairs),
                "expected_pairs": len(selected),
                "starter_preparation_mode": "shared_true_quench",
                "basin_label_mode": "geometry_primary",
            },
        )
    return pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--systems", nargs="+", choices=protocol.SYSTEMS, default=list(protocol.SYSTEMS))
    parser.add_argument("--starters", nargs="+", choices=protocol.STARTERS, default=list(protocol.STARTERS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=protocol.SEEDS, default=list(protocol.SEEDS))
    parser.add_argument("--max-force-evaluations", type=int, default=protocol.MAX_FORCE_EVALUATIONS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = source_module()
    if "cuo" not in args.systems:
        pairs = run_campaign(
            output=args.output,
            systems=set(args.systems),
            starters=set(args.starters),
            seeds=set(args.seeds),
            max_force_evaluations=args.max_force_evaluations,
            cuo_resources=None,
        )
    else:
        source = src.observability.base.source.source
        with TemporaryDirectory(prefix="pamssw-bias-history-cuo-") as temporary:
            resources = source._materialize_cuo_resources(
                source.CUO_ARCHIVE_PATH,
                Path(temporary) / "resources",
            )
            pairs = run_campaign(
                output=args.output,
                systems=set(args.systems),
                starters=set(args.starters),
                seeds=set(args.seeds),
                max_force_evaluations=args.max_force_evaluations,
                cuo_resources=resources,
            )
    print(json.dumps({"completed_pairs": len(pairs), "force_evaluations": sum(int(pair["pair_force_evaluations"]) for pair in pairs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
