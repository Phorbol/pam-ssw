#!/usr/bin/env python3
"""Offline audit of C60 native MC semantics and Python outer attempts.

This consumes archived LASP ``Minimum found`` rows and Python ``result.json``
files only.  It does not infer the native RNG stream, heating trace, or any
causal search effect.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from ase import units

from pamssw.standalone.native_mc import NativeMCState, native_metropolis


LINE = re.compile(
    r"^\s*Minimum found\s+(?P<event>\d+)\s+(?P<parent>\d+)\s+"
    r"(?P<initial>[-+]?\d+(?:\.\d+)?)\s+(?P<candidate>[-+]?\d+(?:\.\d+)?)\s+"
    r"\S+\s+(?P<temperature>[-+]?\d+(?:\.\d+)?)\s+(?P<decision>[TF])\s+"
)

R_GAS = 8.314
FARADAY = 96485.0
TEMPERATURE_K = 150.0
FMAX = 0.03


def probability(delta_eV: float, temperature_K: float) -> float:
    if delta_eV <= 0:
        return 1.0
    return math.exp(-delta_eV / (units.kB * temperature_K))


def native_probability(delta_eV: float) -> float:
    if delta_eV <= 0:
        return 1.0
    return math.exp(((delta_eV / 20.0) * FARADAY / -R_GAS) / TEMPERATURE_K)


def read_native(path: Path, seed: int) -> dict:
    events = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = LINE.match(line)
        if not match:
            continue
        row = match.groupdict()
        event = int(row["event"])
        initial = float(row["initial"])
        candidate = float(row["candidate"])
        delta = candidate - initial
        printed_temperature = float(row["temperature"])
        if printed_temperature != TEMPERATURE_K:
            raise ValueError(f"unexpected printed temperature {printed_temperature} at {path}:{line_number}")
        # Explicit state-zero, positive maxtrap cross-check: negative exponent
        # gives no heating, while retaining the recovered native arithmetic.
        native = native_metropolis(
            initial,
            candidate,
            TEMPERATURE_K,
            energy_tol=0.0,
            maxtrap=1,
            state=NativeMCState(0),
            uniform=0.0,
        )
        events.append(
            {
                "event": event,
                "parent": int(row["parent"]),
                "line": line_number,
                "initial_energy_eV": initial,
                "candidate_energy_eV": candidate,
                "delta_eV": delta,
                "printed_temperature_K": printed_temperature,
                "printed_decision": row["decision"],
                "uphill": delta > 0.0,
                "conventional_150K_probability": probability(delta, TEMPERATURE_K),
                "native_no_heating_probability": native_probability(delta),
                "native_crosscheck_probability": native.acceptance_probability,
                "native_crosscheck_effective_temperature_K": native.effective_temperature_K,
            }
        )
    if not events:
        raise ValueError(f"no Minimum found rows in {path}")
    proposals = [row for row in events if row["event"] > 0]
    uphill = [row for row in proposals if row["uphill"]]
    accepted_uphill = [row for row in uphill if row["printed_decision"] == "T"]
    mismatches = [
        row for row in proposals
        if abs(row["native_no_heating_probability"] - row["native_crosscheck_probability"]) > 1e-14
    ]
    return {
        "seed": seed,
        "source": str(path),
        "event_count_including_initial": len(events),
        "proposal_count_excluding_initial": len(proposals),
        "initial_event_count": len(events) - len(proposals),
        "uphill_event_count": len(uphill),
        "accepted_uphill_event_count": len(accepted_uphill),
        "observed_accepted_event_count": sum(row["printed_decision"] == "T" for row in proposals),
        "observed_rejected_event_count": sum(row["printed_decision"] == "F" for row in proposals),
        "max_delta_eV_all_proposals": max(row["delta_eV"] for row in proposals),
        "max_delta_eV_uphill": max((row["delta_eV"] for row in uphill), default=0.0),
        "native_probability_crosscheck_mismatches": len(mismatches),
        "events": events,
    }


def read_python(path: Path, seed: int) -> dict:
    result = json.loads(path.read_text())
    initial = result["initial"]
    records = result["records"]
    initial_qualified = bool(initial.get("max_force", math.inf) <= FMAX)
    completed = [record for record in records if record.get("landing") is not None]
    qualified_candidates = [
        record for record in completed
        if record["landing"].get("max_force", math.inf) <= FMAX
    ]
    return {
        "seed": seed,
        "source": str(path),
        "outer_attempt_count_from_result": len(records),
        "outer_completed_count": len(completed),
        "outer_failed_count": len(records) - len(completed),
        "initial_qualified_count": int(initial_qualified),
        "candidate_qualified_count": len(qualified_candidates),
        "initial_plus_candidate_qualified_count": int(initial_qualified) + len(qualified_candidates),
        "initial_energy_eV": initial.get("energy"),
        "initial_max_force_eV_per_A": initial.get("max_force"),
        "record_status_counts": {
            status: sum(record.get("status") == status for record in records)
            for status in sorted({record.get("status") for record in records})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, default=Path("research/ga_ssw/evidence/c60-random-native-development-20260917"))
    parser.add_argument("--python-root", type=Path, default=Path("research/ga_ssw/evidence/c60-random-python-development-20260917"))
    parser.add_argument("--output", type=Path, default=Path("research/ga_ssw/evidence/c60-mc-semantics-20260918/analysis-v2.json"))
    args = parser.parse_args()

    native = [
        read_native(args.native_root / f"seed{seed}" / "lasp.out", seed)
        for seed in (17093, 17094)
    ]
    python = [
        read_python(args.python_root / f"seed{seed}-paper" / "result.json", seed)
        for seed in (17093, 17094)
    ]
    ase_equivalent_temperature = 20.0 * R_GAS * TEMPERATURE_K / (FARADAY * units.kB)
    all_proposals = [row for case in native for row in case["events"] if row["event"] > 0]
    all_uphill = [row for row in all_proposals if row["uphill"]]
    output = {
        "scope": "offline semantics audit; no native RNG, heating trace, or counterfactual causal claim",
        "constants": {
            "temperature_K": TEMPERATURE_K,
            "gas_constant_J_per_mol_K": R_GAS,
            "faraday_C_per_mol": FARADAY,
            "fmax_threshold_eV_per_A": FMAX,
            "native_divisor": 20.0,
            "ase_kB_eV_per_K": units.kB,
            "native_equivalent_ASE_temperature_K": ase_equivalent_temperature,
        },
        "native": native,
        "aggregate_native": {
            "event_count_including_initial": sum(case["event_count_including_initial"] for case in native),
            "proposal_count_excluding_initial": len(all_proposals),
            "uphill_event_count": len(all_uphill),
            "accepted_uphill_event_count": sum(row["printed_decision"] == "T" for row in all_uphill),
            "max_delta_eV_all_proposals": max(row["delta_eV"] for row in all_proposals),
            "max_delta_eV_uphill": max(row["delta_eV"] for row in all_uphill),
            "conventional_150K_probability_min": min(probability(row["delta_eV"], TEMPERATURE_K) for row in all_uphill),
            "native_no_heating_probability_min": min(native_probability(row["delta_eV"]) for row in all_uphill),
        },
        "python_original_paper": python,
        "limitations": [
            "The two event-0 rows are initial records, not proposals; 36 later rows are actual proposals.",
            "Printed T/F is reported as an observed decision; no native random uniforms are archived.",
            "Python counts are returned candidates/outer attempts from result.json and are not distinct basin counts.",
            "Python qualification counts use the archived result.json max_force fields; they are self-reported result fields, not a fresh independent force recomputation.",
            "The no-heating native probability uses state=0, positive maxtrap=1, and explicit energy_tol=0.0; it does not assume the historical runtime parameters.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "native_events": output["aggregate_native"]["event_count_including_initial"],
        "native_proposals": output["aggregate_native"]["proposal_count_excluding_initial"],
        "uphill": output["aggregate_native"]["uphill_event_count"],
        "accepted_uphill": output["aggregate_native"]["accepted_uphill_event_count"],
        "equivalent_temperature_K": ase_equivalent_temperature,
    }, indent=2))


if __name__ == "__main__":
    main()
