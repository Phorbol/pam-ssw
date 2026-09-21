"""Offline identity audit for the two strict replicated XXXII endpoints."""
from pathlib import Path
import json
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from pamssw.standalone.rc_topology import read_rigid_topology

ROOT = Path(__file__).resolve().parents[2]
EVID = ROOT / "research/ga_ssw/evidence/xxxii-replicated-endpoint-quench"
OUT = ROOT / "research/ga_ssw/evidence/xxxii-endpoint-identity-audit"

def atoms_from_final(row):
    x = row["final"]
    return Atoms(numbers=x["numbers"], positions=x["positions"], cell=x["cell"], pbc=x["pbc"])

def mic_vector(a, i, j):
    d, _ = find_mic(a.positions[j] - a.positions[i], a.cell, a.pbc)
    return np.asarray(d, float)

def dihedral(a, q):
    return float(a.get_dihedral(*q, mic=True))

def topology_joints(t):
    adj = {i: set() for i in range(t.natoms)}
    for i, j in t.bonds: adj[i].add(j); adj[j].add(i)
    rows = []
    for comp in t.components:
        bodies = comp["bodies"]
        for k in range(1, len(bodies)):
            axis = tuple(comp["joints"][k])
            parent = set(bodies[comp["parents"][k]])
            child = set(bodies[k])
            for i,j in (axis,axis[::-1]):
                left = sorted(adj[i] & (parent-child))
                right = sorted(adj[j] & (child-parent))
                if left and right:
                    q = (left[0],i,j,right[0])
                    assert all(v in adj[u] for u,v in zip(q[:-1],q[1:]))
                    rows.append(q)
                    break
            else:
                raise ValueError(f'no bonded parent-to-child dihedral for {axis}')
    return rows

def main():
    data = json.loads((EVID / "result.json").read_text())
    atoms = [atoms_from_final(x) for x in sorted(data["endpoints"], key=lambda x: x["endpoint"])]
    topo = read_rigid_topology(EVID / "rigidbody", EVID / "blist", natoms=172)
    joints = topology_joints(topo)
    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    for a in atoms:
        niggli = a.cell.niggli_reduce()[0]
        cells.append({"volume_A3": float(a.get_volume()),
                      "cell_singular_values_A": np.linalg.svd(a.cell.array, compute_uv=False).tolist(),
                      "cell_lengths_A": a.cell.lengths().tolist(), "cell_angles_deg": a.cell.angles().tolist(),
                      "niggli_lengths_A": niggli.lengths().tolist(), "niggli_angles_deg": niggli.angles().tolist()})
    bonds = []
    for i, j in topo.bonds:
        d = [float(np.linalg.norm(mic_vector(a, i, j))) for a in atoms]
        bonds.append({"i": i, "j": j, "initial_A": d[0], "final_A": d[1], "delta_A": d[1]-d[0], "relative": d[1]/d[0]-1.0})
    torsions = []
    for q in joints:
        vals = [dihedral(a, q) for a in atoms]; delta = (vals[1]-vals[0]+180.) % 360. - 180.
        torsions.append({"indices": q, "initial_deg": vals[0], "final_deg": vals[1], "delta_deg": delta})
    matcher = {"available": False, "settings": {"scale": False, "primitive_cell": True, "attempt_supercell": True, "allow_subset": False, "comparator": "SpeciesComparator", "sweeps": [[0.1,0.15,2.0],[0.2,0.3,5.0],[0.3,0.5,10.0]]}, "result": "not_run: pymatgen unavailable in audit interpreter"}
    try:
        from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
        from pymatgen.io.ase import AseAtomsAdaptor
        matcher["available"] = True; matcher["result"] = []
        for ltol, stol, angle_tol in matcher["settings"]["sweeps"]:
            sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                                  scale=False, primitive_cell=True,
                                  attempt_supercell=True, allow_subset=False,
                                  comparator=SpeciesComparator())
            matcher["result"].append({"ltol": ltol, "stol": stol, "angle_tol_deg": angle_tol,
                "fit": bool(sm.fit(AseAtomsAdaptor.get_structure(atoms[0]),
                                    AseAtomsAdaptor.get_structure(atoms[1])))})
    except ImportError:
        pass
    result = {"status": "offline_complete", "source_result": str(EVID / "result.json"),
              "topology": {"bonds": len(topo.bonds), "components": [len(x["bodies"]) for x in topo.components], "explicit_joint_torsions": len(joints)},
              "cell": cells, "bond_lengths": bonds, "torsions": torsions, "matcher": matcher,
              "joint_method": "ASE get_dihedral(mic=True), bonded 4-atom paths oriented from parent-exclusive neighbor to child-exclusive neighbor; all three edges checked against explicit topology", "limits": ["same-species topology and periodic lattice descriptors only", "no arbitrary Cartesian RMSD across skew cells", "does not prove true deduplication, phase identity, chemical stability, or Hessian stability"]}
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "script.py").write_text(Path(__file__).read_text())
    rel = np.array([x["relative"] for x in bonds]); da = np.array([x["delta_A"] for x in bonds]); dt = np.array([x["delta_deg"] for x in torsions])
    lines = ["# XXXII strict endpoint identity audit", "", "Offline only; no PES or refinement was run.", "", f"Volumes: endpoint 0 {cells[0]['volume_A3']:.10f} Å^3; endpoint 1 {cells[1]['volume_A3']:.10f} Å^3; ratio {cells[1]['volume_A3']/cells[0]['volume_A3']:.8f}.", f"Topology: {len(topo.bonds)} explicit bonds, four 43-atom components, {len(joints)} explicit joint torsions.", "", "Cell and Niggli descriptors:"]
    lines += [f"- endpoint {i}: singular values {np.round(r['cell_singular_values_A'],6).tolist()}, Niggli lengths {np.round(r['niggli_lengths_A'],6).tolist()}, angles {np.round(r['niggli_angles_deg'],6).tolist()}" for i, r in enumerate(cells)]
    lines += ["", f"All 180 explicit topology bonds: Δ length range [{da.min():.6g}, {da.max():.6g}] Å; relative range [{rel.min():.6g}, {rel.max():.6g}].", f"The 28 topology joint torsions: wrapped Δ range [{dt.min():.6g}, {dt.max():.6g}] degrees; values are per joint in result.json.", "", "StructureMatcher settings recorded from the existing contract: scale=False, primitive_cell=True, attempt_supercell=True, allow_subset=False and (.1,.15,2), (.2,.3,5), (.3,.5,10) diagnostic sweeps. The actually available matcher results are recorded in result.json; absence is reported explicitly.", "", "These measurements support only supplied species/topology and periodic packing descriptors. They cannot certify duplicate identity, chemical stability, or stable phase. The 1.46% volume change and differing lattice basis require treating the endpoints as distinct numerical minima unless an explicitly executed matcher qualifies them as the same."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")

if __name__ == '__main__': main()
