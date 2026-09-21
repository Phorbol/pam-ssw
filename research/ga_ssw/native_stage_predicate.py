"""Research-only scalar reconstruction of the closed native climb gates.

This is not a native lifecycle implementation.  It exposes only predicates
whose operands and strict comparisons are recovered from
``ssw_fixlat_mp_climb_convg_``.  The caller must provide the native optimizer
force maximum and masks; no missing ``Allopt`` or status semantics are guessed.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NativeStagePredicate:
    max_force: float
    climb_stopf: float
    max_excursion: float
    base_energy: float
    initial_energy: float
    saved_gaussian_energy: float
    force_stop: bool
    e_limit_stop: bool
    energy_lower: bool
    saved_energy_stop: bool
    step_over: bool
    final_ng_stop: bool
    allstop: bool
    ng: int
    climbstep: int
    budget: int
    budget_initial: int

    @property
    def known_stop(self) -> bool:
        """Whether one of the independently closed native gates is true."""
        return (self.force_stop or self.e_limit_stop or self.energy_lower or
                self.saved_energy_stop or self.step_over or self.final_ng_stop)


def closed_native_stage_predicate(
    *,
    max_force: float,
    climb_stopf: float,
    base_energy: float,
    initial_energy: float,
    saved_gaussian_energy: float,
    maxe_height: float,
    maxe_height_gm: float,
    e_maxlimit: float,
    f_maxlimit: float,
    e_maxlimit_gm: float,
    para_ng: int,
    ng: int,
    climbstep: int,
    ngaus_relax: int,
    ngaus_relax_ini: int,
    multi_pes: bool = False,
    energy_margin: float = 0.1,
    saved_energy_margin: float = 1.0,
) -> NativeStagePredicate:
    """Evaluate the recovered scalar gates with explicit native operands.

    ``max_force`` is the maximum absolute component of the optimizer-side
    ``fa`` array seen by the callee.  ``base_energy`` and ``initial_energy``
    are respectively object ``+0x1ac8`` (``tene0``) and object ``+0x1ac0``
    (``energy0``).  ``base_energy`` is deliberately not called *bare*: the
    caller evidence says the saved incoming energy may already include LS or
    another upstream term.  The strict comparisons mirror ``comisd`` /
    ``cmpltsd`` and signed ``cmp`` plus ``cmovg`` in the archived assembly.
    ``multi_pes`` models the recovered ``~control+0x1bc`` suppression of the
    initial lower-energy bit.

    The native routine encodes these predicates into control-word bits.  This
    helper returns the recovered scalar predicates and an `allstop` summary;
    it intentionally does not reproduce the original word representation.
    The defaults ``0.1`` and ``1.0`` are the recovered scalar margins; callers
    overriding them are running a scalar sensitivity check, not claiming
    native-default parity.
    """
    values = (max_force, climb_stopf, base_energy, initial_energy,
              saved_gaussian_energy, maxe_height, maxe_height_gm,
              e_maxlimit, f_maxlimit, e_maxlimit_gm, energy_margin,
              saved_energy_margin)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("native stage scalar inputs must be finite")
    if (ng < 0 or climbstep < 0 or ngaus_relax < 0 or ngaus_relax_ini < 0
            or para_ng < 0):
        raise ValueError("native stage counters must be nonnegative")
    if energy_margin < 0:
        raise ValueError("energy_margin must be nonnegative")
    if saved_energy_margin < 0:
        raise ValueError("saved_energy_margin must be nonnegative")

    force_stop = float(max_force) < float(climb_stopf)
    max_excursion = max(float(maxe_height),
                        float(base_energy) - float(initial_energy))
    e_limit_stop = int(ng) != 1 and (
        max_excursion > float(e_maxlimit)
        or float(max_force) > float(f_maxlimit)
        or float(maxe_height_gm) > float(e_maxlimit_gm)
    )
    energy_lower = (not multi_pes) and float(base_energy) < (
        float(initial_energy) - float(energy_margin))
    saved_energy_stop = int(ng) != 1 and float(base_energy) < (
        float(saved_gaussian_energy) - float(saved_energy_margin))
    budget = ngaus_relax_ini if ng == 1 else ngaus_relax
    step_over = int(climbstep) > int(budget)
    final_ng_stop = int(ng) == int(para_ng)
    allstop = e_limit_stop or energy_lower or final_ng_stop
    return NativeStagePredicate(
        max_force=float(max_force), climb_stopf=float(climb_stopf),
        max_excursion=max_excursion,
        base_energy=float(base_energy),
        initial_energy=float(initial_energy),
        saved_gaussian_energy=float(saved_gaussian_energy),
        force_stop=force_stop, energy_lower=energy_lower,
        e_limit_stop=e_limit_stop, saved_energy_stop=saved_energy_stop,
        step_over=step_over, ng=int(ng), climbstep=int(climbstep),
        budget=int(budget), budget_initial=int(ngaus_relax_ini),
        final_ng_stop=final_ng_stop, allstop=allstop,
    )
