#!/usr/bin/env python3
"""Zero-PES geometry-only coverage audit of archived periodic minima."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

from ase.io import read
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

REPO = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity")
FIXED = REPO / "research/ga_ssw/evidence/fixed-cell-ssw-bh-20260921"
ALOH = REPO / "research/ga_ssw/evidence/aloh-fixed-cell-ssw-bh-20260921"
OUT = REPO / "research/ga_ssw/evidence/search-coverage-reaudit-20260923"
ARMS = [
    (case, method, seed, root / f"{case}-{method}-seed{seed}")
    for case, root in (("brookite48", FIXED), ("aloh2", ALOH), ("aloh3", ALOH))
    for method in ("ssw", "bh")
    for seed in (11, 29)
]
TOLERANCES = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}
ADAPTOR = AseAtomsAdaptor()


def pymatgen_version():
    for distribution in ("pymatgen", "pymatgen-core"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "unknown"


def make_matcher(values):
    return StructureMatcher(
        **values,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )


def load_structures(path):
    frames = read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    result = []
    for frame in frames:
        if not frame.pbc.all():
            raise ValueError(f"expected fully periodic frame in {path}")
        result.append(ADAPTOR.get_structure(frame))
    return result


def match_summary(frames, initial, matcher):
    # Deterministic first-representative assignment in archived frame order.
    representatives = []
    representative_indices = []
    group_ids = []
    group_sizes = []
    for frame_index, frame in enumerate(frames):
        assigned = None
        for group_index, representative in enumerate(representatives):
            if matcher.fit(representative, frame):
                assigned = group_index
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append(frame)
            representative_indices.append(frame_index)
            group_sizes.append(0)
        group_ids.append(assigned)
        group_sizes[assigned] += 1

    matches_initial = [bool(matcher.fit(initial, frame)) for frame in frames]
    initial_groups = sorted({group_ids[i] for i, hit in enumerate(matches_initial) if hit})
    first_is_initial = bool(frames and matcher.fit(initial, frames[0]))
    return {
        "groups": len(representatives),
        "representative_frame_indices_zero_based": representative_indices,
        "group_sizes": group_sizes,
        "initial_match_frame_indices_diagnostic_zero_based": [i for i, hit in enumerate(matches_initial) if hit],
        "first_frame_matches_initial_reference": first_is_initial,
        "initial_match_frame_count_including_first": sum(matches_initial) if first_is_initial else None,
        "initial_match_fraction_including_first": (sum(matches_initial) / len(frames) if first_is_initial else None),
        "subsequent_frame_count": len(frames) - 1 if first_is_initial else None,
        "subsequent_initial_match_count": sum(matches_initial[1:]) if first_is_initial else None,
        "subsequent_initial_match_fraction": (sum(matches_initial[1:]) / (len(frames) - 1)
                                              if first_is_initial and len(frames) > 1 else
                                              0.0 if first_is_initial else None),
        "groups_with_initial_match": len(initial_groups),
    }


def analyze_arm(case, method, seed, directory):
    minima_path = directory / "minima.extxyz"
    initial_path = directory / "initial.extxyz"
    summary_path = directory / "summary.json"
    row = {
        "case": case,
        "method": method,
        "seed": seed,
        "run_directory": str(directory),
        "minima_path": str(minima_path),
        "initial_reference_path": str(initial_path),
    }
    try:
        summary = json.loads(summary_path.read_text())
        row["archived_run_status"] = summary.get("status")
        row["archived_execution_class"] = summary.get("execution_class")
        row["archived_minima_count"] = summary.get("minima")
        initial_frames = load_structures(initial_path)
        if len(initial_frames) != 1:
            raise ValueError(f"expected exactly one initial reference, found {len(initial_frames)}")
        frames = load_structures(minima_path)
        if not frames:
            raise ValueError("minima.extxyz contains no frames")
        initial = initial_frames[0]
        composition = initial.composition.reduced_formula
        if any(frame.composition != initial.composition for frame in frames):
            raise ValueError("minima composition differs from initial reference")
        row.update(
            status="analyzed",
            stored_minimum_frame_count=len(frames),
            initial_reference_composition=composition,
            initial_cell_volume_A3=float(initial.volume),
            tolerance_results={},
        )
        for name, values in TOLERANCES.items():
            row["tolerance_results"][name] = match_summary(frames, initial, make_matcher(values))
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    return row


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "analysis": "zero-PES retrospective periodic-geometry coverage audit",
        "protocol": str(OUT / "protocol.md"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pymatgen_version": pymatgen_version(),
        "matcher": {
            "class": "pymatgen.analysis.structure_matcher.StructureMatcher",
            "primitive_cell": False,
            "scale": False,
            "attempt_supercell": False,
            "comparator": "ElementComparator",
            "grouping": "first matching representative, input frame order; order-sensitive geometric proxy, not basin identity",
            "tolerances": TOLERANCES,
        },
        "arms": [],
    }
    json_path = OUT / "results.json"
    for case, method, seed, directory in ARMS:
        arm = analyze_arm(case, method, seed, directory)
        result["arms"].append(arm)
        result["elapsed_seconds"] = time.monotonic() - started
        json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{case} {method} seed={seed}: {arm['status']} frames={arm.get('stored_minimum_frame_count')}", flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result)
    errors = [arm for arm in result["arms"] if arm["status"] != "analyzed"]
    return 1 if errors else 0


def write_report(result):
    lines = [
        "# Fixed-cell search coverage re-audit",
        "",
        "Retrospective geometry-only analysis of archived periodic minima. These are reused search outputs, not new independent searches. Group counts are tolerance-dependent geometric proxies, not strict basin counts.",
        "",
        "| Case | Method | Seed | Frames | Groups tight / broad | Initial match all tight / broad | Later recurrence tight / broad | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in result["arms"]:
        tight = arm.get("tolerance_results", {}).get("tight", {})
        broad = arm.get("tolerance_results", {}).get("broad", {})
        if arm["status"] == "analyzed":
            all_tight = (f"{tight['initial_match_frame_count_including_first']}/{arm['stored_minimum_frame_count']} ({tight['initial_match_fraction_including_first']:.3f})"
                         if tight["first_frame_matches_initial_reference"] else "unresolved")
            all_broad = (f"{broad['initial_match_frame_count_including_first']}/{arm['stored_minimum_frame_count']} ({broad['initial_match_fraction_including_first']:.3f})"
                         if broad["first_frame_matches_initial_reference"] else "unresolved")
            later_tight = (f"{tight['subsequent_initial_match_count']}/{tight['subsequent_frame_count']} ({tight['subsequent_initial_match_fraction']:.3f})"
                           if tight["first_frame_matches_initial_reference"] else "unresolved")
            later_broad = (f"{broad['subsequent_initial_match_count']}/{broad['subsequent_frame_count']} ({broad['subsequent_initial_match_fraction']:.3f})"
                           if broad["first_frame_matches_initial_reference"] else "unresolved")
            lines.append(f"| {arm['case']} | {arm['method']} | {arm['seed']} | {arm['stored_minimum_frame_count']} | {tight['groups']} / {broad['groups']} | {all_tight} / {all_broad} | {later_tight} / {later_broad} | {arm.get('archived_run_status')} |")
        else:
            lines.append(f"| {arm['case']} | {arm['method']} | {arm['seed']} | — | — | — | — | — | ERROR: {arm.get('error')} |")
    lines.extend([
        "",
        "Initial-return figures compare against the archived first quenched structure. All-frame numerators include frame 0; later-frame numerators exclude it. Tolerance order in the paired columns is tight / broad.",
        "",
        f"Pymatgen {result['pymatgen_version']}; elapsed analysis time {result.get('elapsed_seconds', float('nan')):.1f} s. No model or calculator was loaded.",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
