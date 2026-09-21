"""Zero-PES identity and continuous-geometry audit of no-bias endpoints."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[2]))
from pamssw.standalone.periodic_ga_reference import pymatgen_identity

ROOT = Path(__file__).parent / "fe7c3-no-bias-counterfactual"

def atoms(d):
    from ase import Atoms
    return Atoms(numbers=d["numbers"], positions=d["positions"], cell=d["cell"], pbc=d["pbc"])

def centered_cartesian_component_rms(a, b):
    x, y = a.positions.copy(), b.positions.copy()
    x -= x.mean(0); y -= y.mean(0)
    return float(np.sqrt(np.mean((x-y)**2)))

def same_label_nonaffine_rms(initial, terminal):
    """Periodic same-label displacement, measured in the initial cell.

    Fractional displacements use the minimum-image representative, are mapped
    through the initial cell, and then have their translational component
    removed before the per-atom Euclidean RMS is calculated.
    """
    cell = np.asarray(initial.cell.array, dtype=float)
    s0 = np.asarray(initial.get_scaled_positions(wrap=False), dtype=float)
    s1 = np.asarray(terminal.get_scaled_positions(wrap=False), dtype=float)
    ds = s1 - s0
    from ase.geometry import find_mic
    # Componentwise fractional wrapping is not generally minimum-image in a
    # skew cell; use ASE's lattice-aware minimum-image calculation.
    dr, _ = find_mic(ds @ cell, cell, pbc=initial.pbc)
    dr -= dr.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(dr * dr, axis=1))))

def collect():
    rows=[]
    for p in sorted((ROOT/"results").glob("*/result.json")):
        d=json.loads(p.read_text()); o=d["origin"]; q=d["quench"]["evaluation"]["atoms"]
        initial=o["initial"]["atoms"]
        fresh_by_label = {x.get("label"): x for x in d.get("fresh", [])}
        fi, ft = fresh_by_label.get("initial", {}), fresh_by_label.get("terminal", {})
        rows.append({"id":o["id"],"solver":o["solver"],"seed":o["seed"],"attempt":o["attempt"],"initial":atoms(initial),"terminal":atoms(q),"status":d["status"],"energy":d["quench"]["evaluation"]["energy"],"origin_energy":o["initial"].get("energy"),"fresh_initial_energy":fi.get("energy"),"fresh_initial_certified":bool(fi.get("certified", False)),"fresh_terminal_energy":ft.get("energy"),"fresh_terminal_certified":bool(ft.get("certified", False)),"fresh_terminal_present":bool(ft),"quench_certified":bool(d["quench"].get("certificate", {}).get("certified", False)),"requests":d.get("requests",0),"quench_requests":d.get("quench_requests",0),"fresh_count":len(d.get("fresh",[]))})
    return rows

def main():
    rows=collect(); tolerances={"strict":(.05,.1,2.),"default":(.2,.3,5.),"loose":(.3,.5,10.)}
    try:
        matcher = {name: pymatgen_identity(ltol=x, stol=y, angle_tol=z)
                   for name, (x, y, z) in tolerances.items()}
        identities = {
            name: [[bool(fn(a["terminal"], b["terminal"])) for b in rows]
                   for a in rows]
            for name, fn in matcher.items()
        }
        initial_identities = {
            name: [bool(fn(a["terminal"], a["initial"])) for a in rows]
            for name, fn in matcher.items()
        }
        identity_status="available"
    except ModuleNotFoundError as error:
        identities=None; initial_identities=None
        identity_status=f"unavailable: {error}"
    out=[]
    for a in rows:
        cell0=a["initial"].cell.array; cell1=a["terminal"].cell.array
        out.append({"id":a["id"],"solver":a["solver"],"seed":a["seed"],"attempt":a["attempt"],"status":a["status"],"energy":a["energy"],"origin_energy":a["origin_energy"],"delta_energy":None if a["origin_energy"] is None else a["energy"]-a["origin_energy"],"fresh_initial_energy":a["fresh_initial_energy"],"fresh_initial_certified":a["fresh_initial_certified"],"fresh_terminal_energy":a["fresh_terminal_energy"],"fresh_terminal_certified":a["fresh_terminal_certified"],"fresh_terminal_present":a["fresh_terminal_present"],"quench_certified":a["quench_certified"],"centered_cartesian_component_rms_A":centered_cartesian_component_rms(a["terminal"],a["initial"]),"same_label_nonaffine_rms_A":same_label_nonaffine_rms(a["initial"],a["terminal"]),"cell_delta_frobenius_A":float(np.linalg.norm(cell1-cell0)),"cell_relative_frobenius":float(np.linalg.norm(cell1-cell0)/np.linalg.norm(cell0))})
    payload={"status":"audited","zero_pes":True,"count":len(rows),"accounting":{"reported_total_requests":sum(a["requests"] for a in rows),"reported_quench_requests":sum(a["quench_requests"] for a in rows),"fresh_checks":sum(a["fresh_count"] for a in rows)},"tolerances":tolerances,"identity_status":identity_status,"terminal_vs_initial_identity":initial_identities,"terminal_identity":identities,"rows":out,"interpretation":"Approximate identity sensitivity only; matcher result is not a strict basin or phase identity proof. Continuous metrics remain valid when the optional pymatgen backend is unavailable."}
    p=ROOT/"identity-report.json";p.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");print(p)
if __name__=="__main__":main()
