"""Native-derived LS settings/controller on the independent Python SSW driver.

Caller convention: one-based completed outer attempts (including MC rejects and
failed climbs with a completed soft prequench). Update uses the selected current
seed's new bond count. This convention is explicit and NOT proven native caller
parity. The selected prequench exit policy is recorded with each response
update; iteration-cap release is an explicit independent Python policy, not
native caller parity. The exact known table transitions are implemented.
"""
from dataclasses import dataclass
from copy import deepcopy
from .ls_prequench import LSPrequenchSettings, validate_prequench
from .native_ls import (initialize_native_ls,freeze_native_ls,NativeLSCycleState,
                        response_mev_per_atom,_bonds)


@dataclass(frozen=True)
class NativeLSSettings:
    bond_energies: dict
    bond_lengths: dict
    scale: float = 5.
    energy_filter: dict | None = None
    length_filter: dict | None = None
    atom_filter: tuple | None = None
    amp_c: float = 2.
    length_tolerance: float = .1
    target_mev_per_atom: float = 20.
    eta: float = .005
    max_change: float = .01
    frequency: int = 10
    presteps: int = 100
    cycle: int = 100
    ratio: float = 1.100000023841858
    lselfadapt: bool = True
    bond_geometry: str = 'native-mic'
    prequench: LSPrequenchSettings | None = None

    def __post_init__(self):
        validate_prequench(self.prequench)
        if self.bond_geometry not in ('native-mic','periodic-images'):
            raise ValueError('bond_geometry must be native-mic or periodic-images')


class NativeLSRuntime:
    def __init__(self,atoms,settings):
        self.settings=deepcopy(settings)
        initial=initialize_native_ls(atoms,bond_energies=settings.bond_energies,
            bond_lengths=settings.bond_lengths,scale=settings.scale,
            energy_filter=settings.energy_filter,length_filter=settings.length_filter,
            atom_filter=settings.atom_filter,amp_c=settings.amp_c,
            length_tolerance=settings.length_tolerance,bond_geometry=settings.bond_geometry)
        self.frozen=initial.potential;self.lengths=initial.lengths
        self.state=NativeLSCycleState(initial.table,initial.bond_count,len(atoms),
            lselfadapt=settings.lselfadapt,cycle=settings.cycle,ratio=settings.ratio,
            frequency=settings.frequency,presteps=settings.presteps,
            target_mev_per_atom=settings.target_mev_per_atom,eta=settings.eta,max_change=settings.max_change)
        self.steps=0;self.last_update=None

    def update(self,current,next_atoms,*,energy_before,energy_after,**unused):
        current._validate_atoms(next_atoms)
        s=self.settings
        response=response_mev_per_atom(energy_before,energy_after,len(next_atoms))
        pairs,_=_bonds(next_atoms,self.lengths,s.length_tolerance,s.bond_geometry)
        state=deepcopy(self.state)
        event=state.advance(self.steps+1,measured_response_mev_per_atom=response,new_bond_count=len(pairs))
        frozen=freeze_native_ls(next_atoms,state.table,self.lengths,
            atom_filter=s.atom_filter,amp_c=s.amp_c,length_tolerance=s.length_tolerance,
            bond_geometry=s.bond_geometry)
        event.update(caller_convention='completed_outer_attempts_selected_current',
            prequench_exit_policy=getattr(getattr(s, 'prequench', None), 'exit_policy', 'force'),
            prequench_qualification=unused.get('prequench_qualification', 'unknown_not_forwarded'),
            prequench_telemetry=deepcopy(unused.get('prequench_telemetry')),
            observed_response_mev_per_atom=response,bond_count=len(pairs))
        self.state=state;self.steps+=1;self.last_update=event;self.frozen=frozen
        return frozen


def run_native_ls_ssw(atoms,surface,*,steps,config,rng,ls,
                      height_policy=None,gaussian_policy=None,height_update_budget=1000,
                      checkpoint=None, checkpoint_path=None, structure_matcher=None, mc=None):
    """Complete LS prequench→SSW climb→bare quench→MC→native-derived update.

    This entry supports all-mobile atoms. Frozen bonds use the selected
    native-MIC or independent periodic-images convention; the latter does not
    claim native neighbor parity. Constraints use the separate constrained
    entry. No binary dependency.
    """
    # Numerical lbfgs_memory is supplied through the shared SSWConfig, including prequench.
    if not isinstance(ls,NativeLSSettings):raise TypeError('ls must be NativeLSSettings')
    from .paper_reference import run_ssw
    return run_ssw(atoms,surface,steps=steps,config=config,rng=rng,ls=ls,
                   height_policy=height_policy,
                   gaussian_policy=gaussian_policy,
                   height_update_budget=height_update_budget,
                   checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                   structure_matcher=structure_matcher, mc=mc)
