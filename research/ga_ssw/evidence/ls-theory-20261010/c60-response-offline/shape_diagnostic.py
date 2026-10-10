#!/usr/bin/env python3
"""Offline Hessian/gradient diagnostic for radial-shape-normalized frozen LS."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from pamssw.standalone.softening import FrozenBondSoftening

EVIDENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925"
NPZ = EVIDENCE / "curvature/isomer-2.npz"
GEOM = EVIDENCE / "qualification/isomer-2/final.extxyz"
OUT = Path(__file__).resolve().parent / "shape-result.json"

# Reuse the project's exact analytic Cartesian pair stiffness implementation.
probe_path = ROOT / "research/ga_ssw/probe_ls_response.py"
spec = importlib.util.spec_from_file_location("probe_ls_response_offline", probe_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

z = np.load(NPZ, allow_pickle=False)
atoms = read(GEOM)
xraw = atoms.positions.copy()
x = xraw - xraw.mean(axis=0)
X = np.asarray(z["positions"])
Q = np.asarray(z["internal_basis"])
Hraw = np.asarray(z["raw_hessian"])
assert np.max(np.abs(xraw-X)) == 0.0
assert np.max(np.abs(Q.T@Q-np.eye(Q.shape[1]))) < 2e-12
n = len(atoms)
dim = 3*n
R0 = float(np.linalg.norm(x))
R = R0
alpha = R0/R

table_e, table_l, xi, fraction = 3.61, 1.64, 0.2, 0.03
soft = FrozenBondSoftening.from_atoms(
    atoms, bond_energies={(6,6): table_e}, bond_lengths={(6,6): table_l},
    initial_fraction=fraction, xi=xi)
# This construction has y=x at the reference. Keep it explicit for the formula.
def shape_map(pos):
    centered = pos - pos.mean(axis=0)
    radius = float(np.linalg.norm(centered))
    return (R0/radius) * centered

y = shape_map(xraw)
work = atoms.copy()
work.positions[:] = y
W0, force0 = soft.evaluate(work)
g = -force0.reshape(-1)
radial, transverse = probe.stiffness(soft, work)
K = radial + transverse
Hcart = (Hraw + Hraw.T) * 0.5
H = Q.T @ Hcart @ Q

# Translation projector C and scale-shape projector P=C-ee^T.
translation = np.zeros((dim, dim))
for k in range(3):
    v = np.zeros((n,3)); v[:,k] = 1.0/np.sqrt(n)
    t = v.reshape(-1)
    translation += np.outer(t,t)
C = np.eye(dim)-translation
e = x.reshape(-1)/R0
P = C - np.outer(e,e)
assert np.linalg.norm(P@P-P, ord=2) < 1e-12

gshape = alpha * P @ g
Hshape = (alpha**2 * P @ K @ P
          - alpha/R * (np.outer(e, P@g) + np.outer(P@g, e) + float(e@g)*P))
Hshape = (Hshape + Hshape.T)/2

# Fixed-W linear susceptibilities in the same archived physical-Hessian chart.
gq = Q.T @ g
gsq = Q.T @ gshape
chi = float(gq @ np.linalg.solve(H, gq))
chi_shape = float(gsq @ np.linalg.solve(H, gsq))
val_v, mode_v = np.linalg.eigh(H)
mode0 = mode_v[:,0]
Kq = Q.T @ K @ Q
Hshapeq = Q.T @ Hshape @ Q
base_biased = np.linalg.eigvalsh(H + Kq)
shape_biased = np.linalg.eigvalsh(H + Hshapeq)

# Compare response norms and instantaneous Hessian effect along lowest original V mode.
base_mode_bias = float(mode0 @ Kq @ mode0)
shape_mode_bias = float(mode0 @ Hshapeq @ mode0)

# Analytic W_shape energy and gradient at arbitrary Cartesian coordinates.
def energy_grad_shape(pos):
    pos = np.asarray(pos).reshape(n,3)
    centered = pos - pos.mean(axis=0)
    radius = float(np.linalg.norm(centered))
    a = R0/radius
    ylocal = a*centered
    local = atoms.copy(); local.positions[:] = ylocal
    energy, force = soft.evaluate(local)
    grad_w = -force.reshape(-1)
    elocal = centered.reshape(-1)/radius
    plocal = C - np.outer(elocal,elocal)
    grad = a*plocal@grad_w
    return energy, grad

def hshape_apply(pos, direction):
    pos = np.asarray(pos).reshape(n,3)
    centered = pos-pos.mean(axis=0)
    radius = float(np.linalg.norm(centered)); a=R0/radius
    elocal=centered.reshape(-1)/radius
    plocal=C-np.outer(elocal,elocal)
    ylocal=a*centered
    local=atoms.copy(); local.positions[:]=ylocal
    _, flocal=soft.evaluate(local)
    glocal=-flocal.reshape(-1)
    rr,tt=probe.stiffness(soft,local)
    klocal=rr+tt
    pg=plocal@glocal
    # Directional derivative of grad_shape, using the prescribed Hessian identity.
    return (a*a*plocal@klocal@plocal@direction
            -a/radius*(elocal*(pg@direction)+pg*(elocal@direction)
                       +float(elocal@glocal)*(plocal@direction)))

# Deterministic directional checks for energy->gradient and gradient->HVP.
rng=np.random.default_rng(61010)
fd_checks=[]
for k in range(3):
    v=rng.normal(size=(n,3)); v-=v.mean(axis=0); v=v.reshape(-1); v/=np.linalg.norm(v)
    eps=2e-5
    ep,_=energy_grad_shape(xraw.reshape(-1)+eps*v)
    em,_=energy_grad_shape(xraw.reshape(-1)-eps*v)
    _,gp=energy_grad_shape((xraw.reshape(-1)+eps*v).reshape(n,3))
    _,gm=energy_grad_shape((xraw.reshape(-1)-eps*v).reshape(n,3))
    efd=(ep-em)/(2*eps); ean=float(gshape@v)
    hvfd=(gp-gm)/(2*eps); hvan=hshape_apply(xraw.reshape(-1),v)
    fd_checks.append({"direction": k,
        "energy_directional_derivative_analytic": ean,
        "energy_directional_derivative_fd": float(efd),
        "energy_directional_abs_error": float(abs(efd-ean)),
        "hvp_relative_error": float(np.linalg.norm(hvfd-hvan)/max(1e-15,np.linalg.norm(hvfd))),
        "hvp_max_abs_error_eV_A2": float(np.max(np.abs(hvfd-hvan)))})

result={
 "scope":"offline analytic frozen-LS shape map W(R0*x/||x||); no Calculator/PES/optimizer",
 "inputs":{"geometry":str(GEOM),"Hessian":str(NPZ),"stiffness_source":str(probe_path),
           "softening_source":str(ROOT/"pamssw/standalone/softening.py")},
 "definition":{"x":"centered Cartesian coordinates","R0_A":R0,"R_A":R,"alpha":alpha,
   "pair_cutoff_A":table_l,"bond_energy_eV":table_e,"initial_fraction":fraction,"xi":xi,
   "frozen_pair_count":len(soft.pairs),"W_shape_at_reference_eV":W0,
   "translation_projector":"C=I-T T^T, T contains three normalized translations",
   "P":"C-e e^T, e=x/R"},
 "response":{"original":{"cartesian_gradient_norm_eV_A":float(np.linalg.norm(g)),
                             "internal_gradient_norm_eV_A":float(np.linalg.norm(gq)),
                             "chi_eV":chi},
            "shape":{"cartesian_gradient_norm_eV_A":float(np.linalg.norm(gshape)),
                     "internal_gradient_norm_eV_A":float(np.linalg.norm(gsq)),
                     "chi_eV":chi_shape,
                     "radial_gradient_component_removed_eV_A":float(abs(e@g)),
                     "x_dot_gradient_eV":float(x.reshape(-1)@gshape)}},
 "curvature":{"H_V_projected_eigenvalue_min_eV_A2":float(val_v[0]),
               "H_V_plus_K_original_LS_min_eigenvalue_eV_A2":float(base_biased[0]),
               "H_V_plus_Hshape_min_eigenvalue_eV_A2":float(shape_biased[0]),
               "original_V_soft_mode_LS_curvature_shift_eV_A2":base_mode_bias,
               "original_V_soft_mode_shape_curvature_shift_eV_A2":shape_mode_bias,
               "original_V_soft_mode_total_curvature_before_eV_A2":float(val_v[0]),
               "original_V_soft_mode_total_after_original_LS_eV_A2":float(val_v[0]+base_mode_bias),
               "original_V_soft_mode_total_after_shape_LS_eV_A2":float(val_v[0]+shape_mode_bias),
               "Hessian_shape_identity_residual_norm_eV_A":float(np.linalg.norm(Hshape@x.reshape(-1)+gshape)),
               "shape_Hessian_symmetry_residual_norm_eV_A2":float(np.linalg.norm(Hshape-Hshape.T))},
 "directional_finite_difference_checks":fd_checks,
 "interpretation_limits":[
   "This tests an analytically defined radial shape-normalized W at the original geometry only; it does not optimize the modified surface.",
   "The physical V Hessian is the archived h=0.01 A central-force-difference matrix; it is not a dual-step Hessian certificate.",
   "Instantaneous Hessian spectra do not establish a stationary point because the modified gradient is generally nonzero.",
   "No conclusion about absent expansion, finite-amplitude behavior, or improved search efficiency follows from this local diagnostic."]}
OUT.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
