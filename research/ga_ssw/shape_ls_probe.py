"""Research-only radial-shape pullback of a frozen local-softening bias.

This wrapper changes only the soft bias geometry. It has no physical calculator,
optimizer, controller, checkpoint, or public-core integration.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stiffness_function():
    path = ROOT / "research/ga_ssw/probe_ls_response.py"
    spec = importlib.util.spec_from_file_location("_shape_ls_probe_response", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stiffness helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stiffness


_STIFFNESS = None


def _stiffness():
    global _STIFFNESS
    if _STIFFNESS is None:
        _STIFFNESS = _stiffness_function()
    return _STIFFNESS


class FrozenShapeBias:
    """Apply a FrozenBondSoftening potential to fixed-radius centered shape.

    ``evaluate(atoms)`` returns ``(energy_eV, forces_eV_per_A)``. The wrapped
    frozen potential retains responsibility for validating atom numbers, cell,
    and PBC. This wrapper additionally requires a free, isolated cluster.
    """

    def __init__(self, frozen_bias, reference_atoms: Atoms):
        self._validate_domain(reference_atoms)
        self._bias = frozen_bias
        self._reference = reference_atoms.copy()
        relative = self._reference.positions - self._reference.positions.mean(axis=0)
        self._radius0 = float(np.linalg.norm(relative))
        if not np.isfinite(self._radius0) or self._radius0 <= 0:
            raise ValueError("reference centered radius must be finite and positive")
        # Validate reference species/order, cell and PBC through the wrapped API.
        self._bias.evaluate(self._reference)
        if tuple(map(int, self._reference.numbers)) != self._bias.numbers:
            raise ValueError("reference atom identity/order differs from frozen bias")

    @staticmethod
    def _validate_domain(atoms: Atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        if len(atoms) < 2 or atoms.pbc.any() or atoms.constraints:
            raise ValueError("shape bias requires a free nonperiodic unconstrained cluster")
        if not np.isfinite(atoms.positions).all():
            raise ValueError("positions must be finite")
        relative = atoms.positions - atoms.positions.mean(axis=0)
        radius = float(np.linalg.norm(relative))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("centered radius must be finite and positive")

    @property
    def pairs(self):
        return self._bias.pairs

    @property
    def reference_distances(self):
        return self._bias.reference_distances

    @property
    def strengths(self):
        return self._bias.strengths

    @property
    def xi(self):
        return self._bias.xi

    @property
    def radius0(self):
        return self._radius0

    @property
    def radius(self):
        """Frozen reference radius R0 used by the shape map."""
        return self._radius0

    def _mapped_atoms(self, atoms: Atoms):
        self._validate_domain(atoms)
        # soft.evaluate below is the authority for fixed species/order/cell/PBC.
        work = atoms.copy()
        relative = atoms.positions - atoms.positions.mean(axis=0)
        radius = float(np.linalg.norm(relative))
        work.positions[:] = (self._radius0 / radius) * relative
        return work, relative, radius

    def evaluate(self, atoms: Atoms):
        work, _, _ = self._mapped_atoms(atoms)
        energy, force_y = self._bias.evaluate(work)
        relative = atoms.positions - atoms.positions.mean(axis=0)
        radius = float(np.linalg.norm(relative))
        e = relative.reshape(-1) / radius
        grad_y = -force_y.reshape(-1)
        # P v = centered(v) - e(e.v); translations are projected out first.
        grad_x = (self._radius0 / radius) * (grad_y - e * float(e @ grad_y))
        force_x = -grad_x.reshape((-1, 3))
        return float(energy), force_x

    def hessian(self, atoms: Atoms):
        """Return full Cartesian Hessian of W(R0*x/||x||), including chain terms."""
        work, relative, radius = self._mapped_atoms(atoms)
        # This call both performs the frozen-bias identity/cell/PBC validation
        # and obtains the gradient at the mapped geometry; no calculator is used.
        _, force_y = self._bias.evaluate(work)
        grad_y = -force_y.reshape(-1)
        stiffness = _stiffness()
        radial, transverse = stiffness(self._bias, work)
        k = radial + transverse
        n = len(atoms)
        e = relative.reshape(-1) / radius
        alpha = self._radius0 / radius
        translation = np.zeros((3*n, 3*n))
        for axis in range(3):
            t = np.zeros((n, 3))
            t[:, axis] = 1.0 / np.sqrt(n)
            t = t.reshape(-1)
            translation += np.outer(t, t)
        c = np.eye(3*n) - translation
        p = c - np.outer(e, e)
        pg = p @ grad_y
        h = (alpha**2 * p @ k @ p
             - alpha / radius * (np.outer(e, pg) + np.outer(pg, e)
                                 + float(e @ grad_y) * p))
        return (h + h.T) * 0.5


def _load_stiffness_for_tests():
    return _stiffness()


def _directional_checks(bias, atoms, *, seed=0):
    rng = np.random.default_rng(seed)
    x = atoms.positions.copy()
    e0, f0 = bias.evaluate(atoms)
    g0 = -f0.reshape(-1)
    h = bias.hessian(atoms)
    n = len(atoms)
    checks = []
    for i in range(3):
        direction = rng.normal(size=(n, 3))
        direction -= direction.mean(axis=0)
        direction = direction.reshape(-1)
        direction /= np.linalg.norm(direction)
        eps = 1.0e-5
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[:] = (x.reshape(-1) + eps*direction).reshape(n, 3)
        minus.positions[:] = (x.reshape(-1) - eps*direction).reshape(n, 3)
        ep, fp = bias.evaluate(plus)
        em, fm = bias.evaluate(minus)
        energy_fd = (ep-em)/(2*eps)
        energy_analytic = float(g0@direction)
        hv_fd = (-fp.reshape(-1) + fm.reshape(-1))/(2*eps)
        hv_analytic = h@direction
        checks.append({
            "direction": i,
            "energy_directional_abs_error_eV_A": float(abs(energy_fd-energy_analytic)),
            "hvp_relative_error": float(np.linalg.norm(hv_fd-hv_analytic)/max(1e-15,np.linalg.norm(hv_fd))),
            "hvp_max_abs_error_eV_A2": float(np.max(np.abs(hv_fd-hv_analytic))),
        })
    relative = x-x.mean(axis=0)
    scale = 1.37
    moved = atoms.copy()
    moved.positions[:] = scale*relative + np.array([2.3, -1.7, 0.8])
    es, _ = bias.evaluate(moved)
    return {
        "pair_count": len(bias.pairs),
        "reference_energy_eV": float(e0),
        "scaled_translated_energy_eV": float(es),
        "scale_translation_energy_abs_difference_eV": float(abs(es-e0)),
        "gradient_norm_eV_A": float(np.linalg.norm(g0)),
        "hessian_symmetry_residual": float(np.linalg.norm(h-h.T)),
        "Hx_plus_g_norm_eV_A": float(np.linalg.norm(h@relative.reshape(-1)+g0)),
        "directional_checks": checks,
    }


def run_formula_checks(output: Path | None = None):
    """Run bounded analytic-only formula checks on archived C4H6 and C60 inputs."""
    import json
    from ase.io import read
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.softening import FrozenBondSoftening

    evidence = ROOT / "research/ga_ssw/evidence"
    cases = [
        ("C4H6", evidence / "c4h6-ls-isomer-transfer-20261007/qualified-1662853/representative-1/refined.extxyz",
         FrozenBondSoftening, dict(bond_energies=HC_BOND_ENERGIES,
             bond_lengths={k:v+0.1 for k,v in HC_BOND_LENGTHS.items()}, initial_fraction=0.03)),
        ("C60", evidence / "c60-local-defect-20260925/qualification/isomer-2/final.extxyz",
         FrozenBondSoftening, dict(bond_energies={(6,6):3.61}, bond_lengths={(6,6):1.64},
             initial_fraction=0.03, xi=0.2)),
    ]
    rows = []
    for name, source, cls, config in cases:
        atoms = read(source)
        frozen = cls.from_atoms(atoms, **config)
        wrapper = FrozenShapeBias(frozen, atoms)
        rows.append({"system":name, "input":str(source), **_directional_checks(wrapper, atoms,
                    seed=61010 if name == "C4H6" else 61011)})
    result = {
        "scope":"formula/API check of radial shape pullback for analytic frozen LS only; no Calculator, PES, optimization, or scientific-benefit claim",
        "systems":rows,
        "acceptance_checks":{
            "scale_translation_energy_invariance_max_abs_eV":max(r["scale_translation_energy_abs_difference_eV"] for r in rows),
            "Hx_plus_g_max_norm_eV_A":max(r["Hx_plus_g_norm_eV_A"] for r in rows),
            "max_energy_gradient_directional_abs_error_eV_A":max(c["energy_directional_abs_error_eV_A"] for r in rows for c in r["directional_checks"]),
            "max_hvp_relative_error":max(c["hvp_relative_error"] for r in rows for c in r["directional_checks"]),
        },
        "interpretation":"Passing checks validate the implementation's analytic identities on these inputs only; they establish neither search benefit nor physical efficacy.",
    }
    if output is not None:
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="No-PES analytic checks for FrozenShapeBias")
    parser.add_argument("--check", action="store_true", help="run bounded C4H6/C60 analytic formula checks")
    parser.add_argument("--output", type=Path, default=ROOT/"research/ga_ssw/evidence/ls-theory-20261010/shape-probe-formula-check.json")
    args = parser.parse_args()
    if not args.check:
        parser.error("pass --check to run the analytic-only formula checks")
    print(json.dumps(run_formula_checks(args.output), indent=2))
