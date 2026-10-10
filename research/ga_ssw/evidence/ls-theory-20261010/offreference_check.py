#!/usr/bin/env python3
"""Off-reference finite-difference checks for FrozenShapeBias; analytic LS only."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
from pamssw.standalone.softening import FrozenBondSoftening

shape_path = ROOT / "research/ga_ssw/shape_ls_probe.py"
spec = importlib.util.spec_from_file_location("shape_ls_probe_offreference", shape_path)
shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape)

EVIDENCE = ROOT / "research/ga_ssw/evidence"
OUT = ROOT / "research/ga_ssw/evidence/ls-theory-20261010/shape-probe-offreference-check.json"
CASES = [
    ("C4H6", EVIDENCE / "c4h6-ls-isomer-transfer-20261007/qualified-1662853/representative-1/refined.extxyz",
     dict(bond_energies=HC_BOND_ENERGIES,
          bond_lengths={k:v+0.1 for k,v in HC_BOND_LENGTHS.items()}, initial_fraction=0.03)),
    ("C60", EVIDENCE / "c60-local-defect-20260925/qualification/isomer-2/final.extxyz",
     dict(bond_energies={(6,6):3.61}, bond_lengths={(6,6):1.64}, initial_fraction=0.03, xi=0.2)),
]


def case_check(name, source, config):
    atoms = read(source)
    frozen = FrozenBondSoftening.from_atoms(atoms, **config)
    bias = shape.FrozenShapeBias(frozen, atoms)
    n = len(atoms)
    x0 = atoms.positions.copy()
    centered = x0-x0.mean(axis=0)
    rng = np.random.default_rng(61010)
    base_perturb = rng.normal(size=(n,3)); base_perturb -= base_perturb.mean(axis=0)
    base_perturb *= 0.01/np.linalg.norm(base_perturb)
    xoff = 1.17*centered + base_perturb
    off = atoms.copy(); off.positions[:] = xoff
    relative = xoff-xoff.mean(axis=0)
    radius = float(np.linalg.norm(relative))
    radius_ratio = radius/bias.radius
    alpha = bias.radius/radius
    energy, force = bias.evaluate(off)
    grad = -force.reshape(-1)
    hess = bias.hessian(off)
    rng_d = np.random.default_rng(61010)
    checks=[]
    eps=1.0e-5
    for idx in range(3):
        direction=rng_d.normal(size=(n,3)); direction-=direction.mean(axis=0)
        direction=direction.reshape(-1); direction/=np.linalg.norm(direction)
        xp=off.copy(); xm=off.copy()
        xp.positions[:]=(xoff.reshape(-1)+eps*direction).reshape(n,3)
        xm.positions[:]=(xoff.reshape(-1)-eps*direction).reshape(n,3)
        ep,fp=bias.evaluate(xp); em,fm=bias.evaluate(xm)
        dE_fd=(ep-em)/(2*eps)
        dE_analytic=float(grad@direction)
        hv_fd=(-fp.reshape(-1)+fm.reshape(-1))/(2*eps)
        hv_analytic=hess@direction
        checks.append({"direction":idx,
          "energy_directional_analytic_eV_A":dE_analytic,
          "energy_directional_fd_eV_A":float(dE_fd),
          "energy_directional_abs_error_eV_A":float(abs(dE_fd-dE_analytic)),
          "hvp_relative_error":float(np.linalg.norm(hv_fd-hv_analytic)/max(1e-15,np.linalg.norm(hv_fd))),
          "hvp_max_abs_error_eV_A2":float(np.max(np.abs(hv_fd-hv_analytic)))})
    return {"system":name,"input":str(source),"pair_count":len(frozen.pairs),
      "seed":61010,"base_perturbation_norm_A":float(np.linalg.norm(base_perturb)),
      "base_perturbation_center_of_mass_norm_A":float(np.linalg.norm(base_perturb.mean(axis=0))),
      "scale_applied_to_centered_reference":1.17,"R0_A":bias.radius,"Roff_A":radius,
      "Roff_over_R0":radius_ratio,"alpha_R0_over_R":alpha,
      "Wshape_eV":float(energy),"gradient_norm_eV_A":float(np.linalg.norm(grad)),
      "hessian_symmetry_residual":float(np.linalg.norm(hess-hess.T)),
      "Hx_plus_g_norm_eV_A":float(np.linalg.norm(hess@relative.reshape(-1)+grad)),
      "directional_fd_step_A":eps,"directional_checks":checks}

rows=[case_check(name,source,config) for name,source,config in CASES]
result={"scope":"off-reference analytic FrozenShapeBias derivative checks only; no Calculator/PES/optimization",
 "formula":"Wshape(x)=W(R0*C*x/||C*x||), with center projector C",
 "systems":rows,
 "acceptance_checks":{
   "all_Roff_differ_from_R0":all(abs(r["Roff_over_R0"]-1)>0.1 for r in rows),
   "max_energy_directional_abs_error_eV_A":max(c["energy_directional_abs_error_eV_A"] for r in rows for c in r["directional_checks"]),
   "max_hvp_relative_error":max(c["hvp_relative_error"] for r in rows for c in r["directional_checks"]),
   "max_Hx_plus_g_norm_eV_A":max(r["Hx_plus_g_norm_eV_A"] for r in rows)},
 "interpretation":"Checks validate chain-rule implementation away from R=R0 for these frozen biases; they do not establish optimization behavior or scientific benefit."}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
