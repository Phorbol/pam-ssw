#!/usr/bin/env python3
"""Render deterministic trace curves for the reviewed history-capacity ledger."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))

from analyze_ablation import ARMS, SEEDS, SYSTEMS, load_validated_ledger


SYSTEM_LABELS = {"c60": "C60", "pdo": "PdO"}
ARM_LABELS = {
    "safe-total-gradient-history10": "History 10",
    "safe-total-gradient-history0": "History 0",
}
ARM_LINESTYLES = {
    "safe-total-gradient-history10": "-",
    "safe-total-gradient-history0": "--",
}
TASK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)
METRICS = (
    ("total_energy_eV", "Total biased energy (eV)", "energy"),
    (
        "active_max_total_force_eV_per_A",
        "Active max total force (eV/Å)",
        "force",
    ),
)
SVG_METADATA = {
    "Creator": "pamssw deterministic history-capacity plotter",
    "Date": "2026-07-27T00:00:00+00:00",
}


def _explicit_finalization_mask(row: Mapping[str, Any]) -> np.ndarray:
    """Use the reviewed trace rule: the last N callback-observed records."""

    records = row["trace_records"]
    finalization_count = int(row["telemetry"]["explicit_finalization_calls"])
    accepted = np.asarray([record["accepted_state"] for record in records], dtype=bool)
    if finalization_count > int(np.count_nonzero(accepted)):
        raise ValueError(
            f"{row['system']}/{row['task_id']}/{row['arm_id']}: "
            "explicit finalizations exceed callback-observed records"
        )
    mask = np.zeros(len(records), dtype=bool)
    if finalization_count:
        mask[np.flatnonzero(accepted)[-finalization_count:]] = True
    return mask


def _task_legend_handles(Line2D: Any, system: str) -> list[Any]:
    return [
        Line2D(
            [],
            [],
            color=TASK_COLORS[index],
            linewidth=2.0,
            label=f"{system}-seed-{seed}-bias-1",
        )
        for index, seed in enumerate(SEEDS)
    ]


def render_plot(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "pamssw-history-capacity-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "path.simplify": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "pamssw-safe-history-capacity-v1",
        }
    ):
        figure, axes = plt.subplots(
            len(METRICS),
            len(SYSTEMS),
            figsize=(13.0, 9.2),
            sharex="col",
            squeeze=False,
        )
        figure.set_gid("safe-history-capacity-curves")

        for column, system in enumerate(SYSTEMS):
            system_rows = [row for row in rows if row["system"] == system]
            for metric_index, (field, ylabel, metric_id) in enumerate(METRICS):
                axis = axes[metric_index][column]
                axis.set_gid(f"panel-{system}-{metric_id}")
                if metric_index == 0:
                    axis.set_title(SYSTEM_LABELS[system], fontsize=12)
                axis.set_ylabel(ylabel)
                if metric_index == len(METRICS) - 1:
                    axis.set_xlabel("Evaluation index")
                if metric_id == "force":
                    axis.set_yscale("log")
                axis.grid(True, alpha=0.20, linewidth=0.6)
                axis.set_axisbelow(True)

                for row in system_rows:
                    records = row["trace_records"]
                    x = np.asarray(
                        [record["evaluation_index"] for record in records],
                        dtype=float,
                    )
                    values = np.asarray([record[field] for record in records], dtype=float)
                    accepted = np.asarray(
                        [record["accepted_state"] for record in records],
                        dtype=bool,
                    )
                    finalization = _explicit_finalization_mask(row)
                    color = TASK_COLORS[SEEDS.index(row["seed"])]
                    arm_id = row["arm_id"]
                    arm_short = ARM_LABELS[arm_id].lower().replace(" ", "")
                    artist_prefix = (
                        f"{system}-{row['task_id']}-{arm_short}-{metric_id}"
                    )

                    (exact_line,) = axis.plot(
                        x,
                        values,
                        color=color,
                        linestyle=ARM_LINESTYLES[arm_id],
                        linewidth=0.85,
                        alpha=0.58,
                        zorder=1,
                    )
                    exact_line.set_gid(f"all-exact-{artist_prefix}")

                    callback_points = axis.scatter(
                        x[accepted],
                        values[accepted],
                        color=color,
                        marker="o",
                        s=5.0,
                        linewidths=0.0,
                        alpha=0.88,
                        zorder=2,
                    )
                    callback_points.set_gid(f"callback-observed-{artist_prefix}")

                    if np.any(~accepted):
                        nonaccepted_points = axis.scatter(
                            x[~accepted],
                            values[~accepted],
                            color=color,
                            marker="x",
                            s=14.0,
                            linewidths=0.65,
                            zorder=3,
                        )
                        nonaccepted_points.set_gid(
                            f"callback-nonaccepted-{artist_prefix}"
                        )

                    if np.any(finalization):
                        finalization_points = axis.scatter(
                            x[finalization],
                            values[finalization],
                            facecolors="none",
                            edgecolors=color,
                            marker="D",
                            s=28.0,
                            linewidths=0.9,
                            zorder=4,
                        )
                        finalization_points.set_gid(
                            f"explicit-finalization-{artist_prefix}"
                        )

        figure.suptitle(
            "Safe L-BFGS history-capacity traces",
            fontsize=14,
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.075,
            right=0.985,
            top=0.94,
            bottom=0.34,
            hspace=0.14,
            wspace=0.18,
        )

        task_legends = []
        for column, system in enumerate(SYSTEMS):
            task_legend = figure.legend(
                handles=_task_legend_handles(Line2D, system),
                title=f"Task ID (color) — {SYSTEM_LABELS[system]}",
                loc="upper left",
                bbox_to_anchor=(0.02 + 0.49 * column, 0.315),
                ncol=2,
                fontsize=7,
                title_fontsize=8,
                frameon=False,
                handlelength=1.6,
                columnspacing=1.0,
            )
            task_legend.set_gid(f"legend-task-id-{system}")
            task_legends.append(task_legend)

        arm_handles = [
            Line2D(
                [],
                [],
                color="black",
                linestyle=ARM_LINESTYLES[arm_id],
                linewidth=1.2,
                label=ARM_LABELS[arm_id],
            )
            for arm_id, _ in ARMS
        ]
        arm_legend = figure.legend(
            handles=arm_handles,
            title="Arm (line style)",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.105),
            ncol=1,
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )
        arm_legend.set_gid("legend-arm-line-style")

        semantic_handles = [
            Line2D(
                [],
                [],
                color="black",
                linewidth=0.85,
                alpha=0.58,
                label="All exact evaluations",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markersize=3,
                linestyle="None",
                label=(
                    "Callback-observed accepted_state "
                    "(not optimizer acceptance rule)"
                ),
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="x",
                markersize=4,
                linestyle="None",
                label="Callback-nonaccepted evaluation",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="D",
                markerfacecolor="none",
                markersize=4,
                linestyle="None",
                label="Explicit finalization recheck",
            ),
        ]
        semantic_legend = figure.legend(
            handles=semantic_handles,
            title="Trace semantics",
            loc="upper left",
            bbox_to_anchor=(0.24, 0.105),
            ncol=2,
            fontsize=7,
            title_fontsize=8,
            frameon=False,
            columnspacing=1.2,
        )
        semantic_legend.set_gid("legend-trace-semantics")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(
                output_path,
                format="svg",
                metadata=SVG_METADATA,
            )
        finally:
            plt.close(figure)
        svg = output_path.read_text(encoding="utf-8")
        output_path.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        rows = load_validated_ledger(arguments.ledger_dir)
        render_plot(rows, arguments.output)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"plot failed: {error}", file=sys.stderr)
        return 2
    print(f"wrote plot={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
