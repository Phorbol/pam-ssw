"""TYPE2 whole-molecule proposals in the independent three-stage VC-GA flow.

Molecules are disjoint atom groups, distinct from overlapping RC rigid bodies.
Physical relaxation may change molecular geometry; proposals preserve each
parent molecule internally. Identity and oracle budgets remain caller-owned.
"""
from .molecular_periodic_ga import propose_type2, _molecular_lift
from .periodic_ga_reference import run_periodic_ga


def run_molecular_periodic_ga(initial, surface, *, molecules,
                              image_shifts=None, **kwargs):
    """Quick exploration, TYPE2 offspring walks, periodic routing and fine search.

    ``image_shifts`` follows propose_type2's per-parent convention. For evolving
    archives, unwrap initial molecules first and omit it: fixed parent-indexed
    image arrays cannot describe a changing archive.
    """
    if kwargs.get('fixed_cell', False):
        raise ValueError('TYPE2 molecular periodic GA does not support fixed-cell mode')
    if image_shifts is not None:
        raise ValueError('unwrap initial molecules before evolving-archive GA')
    initial=tuple(initial)
    molecules=tuple(tuple(group) for group in molecules)
    for seed in initial:
        _molecular_lift(seed,molecules)
    def factory(parents, energies, regions, rng, *, config, bond_limits):
        return propose_type2(parents, energies, molecules, rng,
            min_ga=config.min_ga, bond_limits=bond_limits,
            max_batches=config.max_batches,
            max_cut_attempts=config.max_cut_attempts,
            max_pair_attempts=config.max_pair_attempts)
    return run_periodic_ga(initial, surface, proposal_factory=factory, **kwargs)
