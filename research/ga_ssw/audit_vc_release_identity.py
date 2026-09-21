"""Zero-PES structural identity audit for certified numerical-release landings."""
import json
from pathlib import Path
from ase import Atoms
from pamssw.standalone.periodic_ga_reference import pymatgen_identity


def _atoms(d):
    return Atoms(numbers=d["numbers"], positions=d["positions"],
                cell=d["cell"], pbc=d["pbc"])


def main(root="research/ga_ssw/fe7c3-vc-numerical-release"):
    root = Path(root)
    rows = []
    for p in sorted((root / "comparison").glob("*/result.json")):
        d = json.loads(p.read_text())
        landings = d.get("landings") or []
        checks = (d.get("fresh") or {}).get("checks") or []
        initial = landings[0] if landings else None
        for check in checks:
            i = int(check.get("index", -1))
            if not (0 <= i < len(landings)) or int(landings[i].get("index", -1)) < 0:
                continue
            landing = landings[i]
            rows.append({"id": p.parent.name, "landing_index": int(landing["index"]),
                         "accepted": bool(landing.get("accepted", False)),
                         "energy": landing.get("energy"),
                         "energy_delta_from_initial": landing.get("energy") - initial.get("energy"),
                         "fresh_certified": check.get("certified") is True,
                         "initial": _atoms(initial["atoms"]),
                         "terminal": _atoms(landing["atoms"])})
    tolerances = {"strict": (.05, .1, 2.), "default": (.2, .3, 5.),
                  "loose": (.3, .5, 10.)}
    matchers = {k: pymatgen_identity(ltol=a, stol=b, angle_tol=c)
                for k, (a, b, c) in tolerances.items()}
    initial_identity = {k: [bool(fn(r["initial"], r["terminal"])) for r in rows]
                        for k, fn in matchers.items()}
    terminal_identity = {k: [[bool(fn(a["terminal"], b["terminal"]))
                              for b in rows] for a in rows]
                         for k, fn in matchers.items()}
    out_rows = [{k: v for k, v in r.items() if k not in ("initial", "terminal")}
                for r in rows]
    out = {"status": "audited", "zero_pes": True, "count": len(rows),
           "tolerances": tolerances, "rows": out_rows,
           "terminal_vs_initial_identity": initial_identity,
           "terminal_identity": terminal_identity,
           "interpretation": "Approximate matcher sensitivity; identity is not a strict basin or phase proof."}
    path = root / "identity-summary.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
