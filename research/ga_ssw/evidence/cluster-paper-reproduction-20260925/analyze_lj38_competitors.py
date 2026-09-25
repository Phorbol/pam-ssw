"""Classify saved LJ38 best structures against qualified OPTIM SLM by geometry only."""
import json
from pathlib import Path

from ase.io import read
from analyze_lj_pilot import geometry

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912"
)
CASES = (
    ("runs/lj38-seed25092501", "runs"),
    ("runs/lj38-seed25092502", "runs"),
    ("paper-direction-runs/lj38-seed25092501", "paper-direction-runs"),
    ("paper-direction-runs/lj38-seed25092502", "paper-direction-runs"),
)
def main():
    ref = read(REFERENCE_DIR / "optim-finish.extxyz")
    qualification_path = REFERENCE_DIR / "optim-qualification.json"
    qualification = json.loads(qualification_path.read_text())
    ref_properties = next(p for p in qualification["properties"] if p["file"] == "optim-finish")
    rows = []
    for relative, arm in CASES:
        path = HERE / relative / "best.extxyz"
        row = {"arm": arm, "case": path.parent.name, "input": str(path.relative_to(HERE))}
        try:
            atoms = read(path)
            matched = geometry(atoms, ref)
            row.update({"atom_count": len(atoms), **matched})
            row["classification"] = "match" if matched["geometry_match"] else "inconclusive"
            row["interpretation"] = (
                "positive RMS match supports geometric identity to reference"
                if matched["geometry_match"] else
                "no match found; capped or exhausted graph mappings, so nonmatch is inconclusive"
            )
        except Exception as exc:
            row.update({"geometry_match": None, "classification": "failure", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    result = {
        "scope": "Four saved best geometries vs independently qualified OPTIM second-lowest minimum; geometry only, zero potential evaluations.",
        "reference": {
            "file": str((REFERENCE_DIR / "optim-finish.extxyz")),
            "energy_eV": ref_properties["energy"],
            "fmax_eV_per_A": ref_properties["fmax"],
            "qualification": str(qualification_path),
            "classification": "second-lowest minimum (SLM); optim-odata is the lower GM endpoint",
        },
        "geometry_method": {
            "implementation": "analyze_lj_pilot.geometry",
            "potential_evaluations": 0,
        },
        "limitations": [
            "A capped search without a match is inconclusive, not proof of distinct minima.",
            "Matching the SLM endpoint does not establish full funnel connectivity or a path between minima.",
        ],
        "rows": rows,
    }
    target = HERE / "lj38-competitor-geometry.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
