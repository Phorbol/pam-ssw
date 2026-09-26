"""S2 formula and image accumulation checks on saved physical geometries; zero PES."""
import json
from pathlib import Path
import sys
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.special import sph_harm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.ga_ssw.s2_reference import center_neighbors_value_gradient
from research.ga_ssw.s1_reference import radial_value_derivative

SOURCE = ROOT.parent / 'ga-ssw-behavior-parity/research/ga_ssw/evidence'
CASES = {
 'c60_defect': SOURCE / 'mh1-native-ls-equal-budget-20260920/c60_17093-first-native_ls/result.json',
 'rutile12': SOURCE / 'tio2-fixed-ritz-staged-holdout-20260912/rutile-seed11-fixed_ritz_a100/result.json',
}
CUTOFF, EPSILON, EXPONENT = 6.48507, 1.e-5, -1


def direct_harmonic_value(vectors, degree):
    radii = np.linalg.norm(vectors, axis=1)
    radial = np.array([radial_value_derivative(r, CUTOFF, EXPONENT, EPSILON)[0] for r in radii])
    theta = np.arctan2(vectors[:, 1], vectors[:, 0])
    polar = np.arccos(np.clip(vectors[:, 2] / radii, -1., 1.))
    moments = [np.sum(radial * sph_harm(m, degree, theta, polar))
               for m in range(-degree, degree + 1)]
    return float(np.linalg.norm(moments))


def main():
    rows = []
    rng = np.random.default_rng(26092642)
    for label, source in CASES.items():
        atoms = Atoms(**json.loads(source.read_text())['initial']['atoms'])
        ii, jj, shifts = neighbor_list('ijS', atoms, CUTOFF)
        # One representative of each species; not center selection for a walker.
        centers = [int(np.flatnonzero(atoms.numbers == z)[0]) for z in np.unique(atoms.numbers)]
        for center in centers:
            mask = ii == center
            neighbors, images = jj[mask], shifts[mask]
            def vectors(positions):
                return positions[neighbors] + images @ atoms.cell.array - positions[center]
            base = vectors(atoms.positions)
            for degree in (2, 4):
                value, central, gradient = center_neighbors_value_gradient(
                    base, degree, exponent=EXPONENT, cutoff=CUTOFF, epsilon=EPSILON)
                full = np.zeros_like(atoms.positions)
                np.add.at(full, neighbors, gradient)
                full[center] += central
                probe = rng.normal(size=full.shape); probe /= np.linalg.norm(probe)
                h = 1.e-5
                direct = direct_harmonic_value(base, degree)
                fd = (direct_harmonic_value(vectors(atoms.positions + h*probe), degree)
                    - direct_harmonic_value(vectors(atoms.positions - h*probe), degree)) / (2*h)
                analytic = float(np.sum(full * probe))
                row = dict(case=label, source=str(source), center=center, species=int(atoms.numbers[center]),
                    degree=degree, exponent=EXPONENT, cutoff_A=CUTOFF, guard_A=EPSILON,
                    image_neighbors=len(neighbors), nonzero_images=int(np.any(images != 0, axis=1).sum()),
                    self_images=int((neighbors == center).sum()), value=value, direct_value=direct,
                    analytic_directional_derivative=analytic, finite_difference=fd,
                    gradient_error=abs(fd-analytic), translation_residual=float(np.linalg.norm(full.sum(axis=0))))
                row['passed'] = bool(np.isclose(value, direct, rtol=1.e-10, atol=1.e-12)
                    and np.isclose(analytic, fd, rtol=1.e-5, atol=1.e-8)
                    and row['translation_residual'] < 1.e-11)
                rows.append(row)
    output = dict(scope='implementation check on physical geometries; no PES or search-efficacy evidence',
                  pes_calls=0, seed=26092642, finite_difference_A=1.e-5, rows=rows,
                  passed=all(r['passed'] for r in rows))
    Path(__file__).with_name('geometry-check.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
    if not output['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
