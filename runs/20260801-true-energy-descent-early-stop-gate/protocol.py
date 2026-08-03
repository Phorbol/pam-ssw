"""Pure protocol for the G-E0 true-energy descent stopping audit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_NEW_FORCE_EVALUATIONS = 10_000
MAX_KERNEL_WALL_TIME_S = 300.0


def accepted_step_indices(case: Mapping[str, Any]) -> tuple[int, ...]:
    reached = int(case["reached_macro_steps"])
    attempted = int(case["attempted_macro_steps"])
    if reached < 0 or attempted < reached:
        raise ValueError("invalid reached/attempted macro-step counts")
    return tuple(range(1, reached + 1))


def project_manifest_case(
    case: Mapping[str, Any],
    case_dir: Path,
    *,
    root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for step in accepted_step_indices(case):
        path = case_dir / "macro_checkpoints" / (
            f"step{step:03d}_checkpoint.xyz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "step": step,
                "checkpoint_path": path.relative_to(root).as_posix(),
                "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def first_descent_crossing(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any] | None:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    for expected, row in enumerate(rows, start=1):
        if int(row["step"]) != expected:
            raise ValueError("accepted endpoint steps must be consecutive")
        delta = row.get("checkpoint_delta_eV")
        if delta is not None and float(delta) < -float(tolerance):
            return dict(row)
    return None


def classify_tradeoff(
    crossing_landing_delta_eV: float | None,
    terminal_landing_delta_eV: float | None,
    tolerance: float,
) -> str:
    if (
        crossing_landing_delta_eV is None
        or terminal_landing_delta_eV is None
    ):
        return "UNLEARNABLE"
    crossing = float(crossing_landing_delta_eV)
    terminal = float(terminal_landing_delta_eV)
    if terminal > crossing + float(tolerance):
        return "AVOIDED_OVERSHOOT"
    if terminal < crossing - float(tolerance):
        return "FORGONE_DEEPER_TERMINAL"
    return "ENERGY_EQUIVALENT"
