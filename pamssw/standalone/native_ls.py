"""Recovered release LS arithmetic, independent of binary runtime.

Not the full native softmode-cycle controller. Defaults below are recovered ELF
values, not general physical optima. B is the element-pair table; effective pair
A=amp_c*B*m_i*m_j. Responses/targets use meV/atom, energies eV, lengths Angstrom.
Paper FrozenBondSoftening/LSResponseState behavior is deliberately unchanged.
"""
from dataclasses import dataclass
import math
import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from .softening import FrozenBondSoftening, _geometry, _table
from .periodic_softening import FrozenPeriodicBondSoftening

# Exact promoted release lookup values; only these elements are supplied here.
HC_BOND_ENERGIES={(1,1):4.526579856872559,(1,6):4.29817008972168,(6,6):3.4468400478363037}
HC_BOND_LENGTHS={(1,1):.7400000095367432,(1,6):1.090000033378601,(6,6):1.5399999618530273}

# Explicit opt-in raw lookup tables, before LS normalization/filtering.
# Original ELF leaf-function returns and H2O/CH3OH initialization-prefix checks:
# research/ga_ssw/evidence/native-ls-pair-table-hco-20260912/
# research/ga_ssw/evidence/native-ls-initialization-hco-20260912/
# These are release reference parameters, not fitted GFN2/MLIP bond strengths.
HCO_BOND_ENERGIES={**HC_BOND_ENERGIES, (1,8):4.817279815673828,
                   (6,8):3.384550094604492, (8,8):1.515779972076416}
HCO_BOND_LENGTHS={**HC_BOND_LENGTHS, (1,8):.9599999785423279,
                  (6,8):1.4299999475479126, (8,8):1.4800000190734863}
# Explicit opt-in Ti/O lookup values recovered from the pinned GA-SSW ELF.
# These are release reference parameters for reproduction, not physical
# optima and not a production default.
TIO_BOND_ENERGIES={(22,22):3.6298000812530518,(22,8):3.6298000812530518,
                   (8,8):1.515779972076416}
TIO_BOND_LENGTHS={(22,22):1.899999976158142,(22,8):1.9429999589920044,
                  (8,8):1.4800000190734863}
REFERENCE_CC_ENERGY=3.4468400478363037


def _count(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _finite(value,name,positive=False):
    value=float(value)
    if not math.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError(f'{name} must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def _nonnegative_table(table):
    # Reuse canonical key/conflict checks while explicitly allowing zero B.
    values={}
    for key,value in table.items():
        value=_finite(value,'B')
        canonical=next(iter(_table({key:1.})))
        if canonical in values and values[canonical]!=value:raise ValueError('conflicting pair table')
        values[canonical]=value
    if not values:raise ValueError('nonempty pair table required')
    return values


def _periodic_bonds(atoms,lengths,tolerance):
    """Enumerate canonical periodic image records using the native cutoff."""
    if len(atoms)<1 or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError('finite nonempty periodic geometry required')
    if atoms.constraints: raise NotImplementedError('native frozen-atom counting branch is not recovered')
    if np.linalg.det(atoms.cell.array)<=0: raise ValueError('periodic cell must have positive volume')
    if any(atoms.pbc) and np.any(np.linalg.norm(atoms.cell.array,axis=1)[atoms.pbc]==0):
        raise ValueError('periodic cell vectors must be nonzero')
    lengths=_table(lengths); tolerance=_finite(tolerance,'length tolerance')
    cutoff=np.nextafter(max(lengths.values())+tolerance,np.inf)
    i,j,shifts,distances=neighbor_list('ijSd',atoms,cutoff,self_interaction=False)
    records={}
    for a,b,shift,distance in zip(i,j,shifts,distances):
        key=tuple(sorted((int(atoms.numbers[a]),int(atoms.numbers[b]))))
        if key not in lengths:raise ValueError(f'missing length for {key}')
        distance=float(distance)
        if not math.isfinite(distance) or distance<=0:raise ValueError('positive finite pair distance required')
        if not distance < lengths[key]+tolerance:continue
        shift=tuple(map(int,shift))
        canonical=min((int(a),int(b),*shift),(int(b),int(a),*(-x for x in shift)))
        records[canonical]=distance
    if not records:raise ValueError('zero bond-count branch is unsupported')
    ordered=sorted(records)
    return tuple((key[:2],key[2:],records[key]) for key in ordered)


def _bonds(atoms,lengths,tolerance,bond_geometry='native-mic'):
    if bond_geometry not in ('native-mic','periodic-images'):raise ValueError('bond_geometry must be native-mic or periodic-images')
    if bond_geometry=='periodic-images' and np.any(atoms.pbc):
        records=_periodic_bonds(atoms,lengths,tolerance)
        return tuple(pair for pair,shift,distance in records),tuple(distance for pair,shift,distance in records)
    _geometry(atoms)
    if atoms.constraints:raise NotImplementedError('native frozen-atom counting branch is not recovered')
    lengths=_table(lengths);tolerance=_finite(tolerance,'length tolerance')
    pairs=[];distances=[]
    for i in range(len(atoms)-1):
        for j in range(i+1,len(atoms)):
            key=tuple(sorted((int(atoms.numbers[i]),int(atoms.numbers[j]))))
            if key not in lengths:raise ValueError(f'missing length for {key}')
            _,distance=find_mic(atoms.positions[j]-atoms.positions[i],atoms.cell,atoms.pbc)
            distance=float(distance)
            if not math.isfinite(distance) or distance<=0:raise ValueError('positive finite pair distance required')
            if distance<lengths[key]+tolerance: # native strict comparison
                pairs.append((i,j));distances.append(distance)
    if not pairs:raise ValueError('zero bond-count branch is unsupported')
    return tuple(pairs),tuple(distances)


def effective_amplitude(table_value,*,amp_c=2.,filter_i=1.,filter_j=1.):
    """A in eV: multiplication order matches the recovered scalar instructions."""
    return (_finite(amp_c,'amp_c')*_finite(table_value,'B')
            *_finite(filter_i,'atom filter')*_finite(filter_j,'atom filter'))


def freeze_native_ls(atoms, table, lengths, *, atom_filter=None, amp_c=2.,
                     length_tolerance=.1, bond_geometry='native-mic'):
    """Freeze current MIC pair distances, strict bond list and recovered A.

    Zero atom_filter removes amplitude, not the bond from N_b. Atom_filter is
    distinct from native fixatom, whose subset-counting path is unsupported.
    ASE MIC is a geometric implementation, not arbitrary-cell native parity.
    """
    table=_nonnegative_table(table)
    filters=np.ones(len(atoms)) if atom_filter is None else np.asarray(atom_filter,dtype=float)
    if filters.shape!=(len(atoms),) or not np.isfinite(filters).all() or np.any(filters<0):
        raise ValueError('one finite nonnegative filter per atom required')
    if bond_geometry not in ('native-mic','periodic-images'):raise ValueError('bond_geometry must be native-mic or periodic-images')
    periodic=bond_geometry=='periodic-images' and np.any(atoms.pbc)
    records=_periodic_bonds(atoms,lengths,length_tolerance) if periodic else None
    pairs,distances=(_bonds(atoms,lengths,length_tolerance) if records is None else
                     (tuple(r[0] for r in records),tuple(r[2] for r in records)))
    strengths=[]
    for i,j in pairs:
        key=tuple(sorted((int(atoms.numbers[i]),int(atoms.numbers[j]))))
        if key not in table:raise ValueError(f'missing B for {key}')
        strengths.append(effective_amplitude(table[key],amp_c=amp_c,filter_i=filters[i],filter_j=filters[j]))
    if periodic:
        numbers=tuple(map(int,atoms.numbers)); cell=tuple(tuple(map(float,row)) for row in atoms.cell.array); pbc=tuple(map(bool,atoms.pbc))
    else:
        numbers,cell,pbc=_geometry(atoms)
    if not periodic:return FrozenBondSoftening(numbers,cell,pbc,pairs,distances,tuple(strengths),xi=.2)
    return FrozenPeriodicBondSoftening(numbers,cell,pbc,pairs,
        tuple(r[1] for r in records),distances,tuple(strengths),xi=.2)


@dataclass(frozen=True)
class NativeLSInitialization:
    table: dict
    lengths: dict
    bond_count: int
    potential: FrozenBondSoftening | FrozenPeriodicBondSoftening
    bond_geometry: str = 'native-mic'


def initialize_native_ls(atoms, *, bond_energies, bond_lengths, scale=5.,
                         energy_filter=None, length_filter=None, atom_filter=None,
                         amp_c=2., length_tolerance=.1, bond_geometry='native-mic'):
    """No-custom-file normal initialization, explicitly supplied lookup tables.

    B_ab=D_ab*f_ab*scale*float64(float32(N)*float32(.02))/(Nb*D_CC).
    D_CC is the recovered reference constant even if carbon is absent. Positive
    reference lengths and all-mobile atoms only. No empirical radius fallback.
    """
    if bond_geometry not in ('native-mic','periodic-images'):raise ValueError('bond_geometry must be native-mic or periodic-images')
    energies=_table(bond_energies);lengths=_table(bond_lengths)
    scale=_finite(scale,'scale',positive=True)
    ef={k:1. for k in energies} if energy_filter is None else _nonnegative_table(energy_filter)
    lf={k:1. for k in lengths} if length_filter is None else _table(length_filter)
    if set(ef)!=set(energies) or set(lf)!=set(lengths):raise ValueError('filter keys must match supplied tables')
    lengths={k:v*lf[k] for k,v in lengths.items()}
    pairs,_=_bonds(atoms,lengths,length_tolerance,bond_geometry);nb=len(pairs)
    scale_n=float(np.float32(len(atoms))*np.float32(.02))
    table={k:v*ef[k]*scale*scale_n/(nb*REFERENCE_CC_ENERGY) for k,v in energies.items()}
    potential=freeze_native_ls(atoms,table,lengths,atom_filter=atom_filter,amp_c=amp_c,
                               length_tolerance=length_tolerance,bond_geometry=bond_geometry)
    return NativeLSInitialization(table,lengths,nb,potential,bond_geometry)


def response_mev_per_atom(energy_before,energy_after,natoms):
    """True PES response; this does not certify prequench convergence."""
    natoms=_count(natoms,'natoms')
    if not np.isfinite([energy_before,energy_after]).all():raise ValueError('finite true energies required')
    return 1000.*(float(energy_after)-float(energy_before))/natoms


def normal_update_due(step,*,frequency=10,presteps=100):
    """Normal-branch predicate ONLY; does not decide periodic save/restore.

    Step zero never updates. Thereafter update at frequency multiples OR while
    step <= presteps. A caller must independently establish the normal branch.
    """
    step=_count(step,'step',0);frequency=_count(frequency,'frequency');presteps=_count(presteps,'presteps',0)
    return step!=0 and (step%frequency==0 or step<=presteps)


@dataclass(frozen=True)
class NativeLSUpdate:
    table: dict
    q: float
    response_mev_per_atom: float


def update_native_table(table,*,natoms,old_bond_count,new_bond_count,
                        response_mev_per_atom,target_mev_per_atom=20.,
                        eta=.005,max_change=.01,branch='normal'):
    """Recovered normal table arithmetic; no hidden schedule or clipping.

    eta converts the stored meV/atom response to a dimensionless table update
    together with N/Nb. max_change has B's eV units. Q controls the response
    term, not the separate old/new bond normalization. Negative output is an
    unsupported state (explicit error), not silently clipped to positive.
    Periodic table save/restore and failed-prequench eligibility are unrecovered.
    """
    if branch!='normal':raise NotImplementedError('native periodic save/restore branch is not recovered')
    table=_nonnegative_table(table);natoms=_count(natoms,'natoms')
    old=_count(old_bond_count,'old_bond_count');new=_count(new_bond_count,'new_bond_count')
    eta=_finite(eta,'eta');cap=_finite(max_change,'max_change',positive=True)
    response=float(response_mev_per_atom);target=float(target_mev_per_atom)
    if not np.isfinite([response,target]).all():raise ValueError('finite meV/atom response and target required')
    delta=response-target
    q=max(1.,max(v*eta*natoms*abs(delta)/new for v in table.values())/cap)
    updated={k:v*old/new-v*eta*natoms*delta/(new*q) for k,v in table.items()}
    if not np.isfinite(q) or any(not np.isfinite(v) or v<0 for v in updated.values()):
        raise ValueError('nonfinite/negative native update outside supported normal path')
    return NativeLSUpdate(updated,q,response)


@dataclass
class NativeLSCycleState:
    """Explicit independent controller for recovered table-state transitions.

    Callers supply the native-style integer step, associated measured response
    and NEW geometry's bond count. This class does not infer accepted-seed
    association, execute prequench, or determine numerical-failure eligibility.
    Iteration-limit response may be supplied deliberately; backend failures
    must not be silently converted into measured energies.

    lselfadapt=False represents the recovered nonadaptive table cycle, which
    uses step%cycle without presteps and performs no response updates. Defaults
    correspond to recovered static values, including inactive cycling at 1.1.
    """
    table: dict
    old_bond_count: int
    natoms: int
    lselfadapt: bool = True
    cycle: int = 100
    ratio: float = 1.100000023841858
    frequency: int = 10
    presteps: int = 100
    target_mev_per_atom: float = 20.
    eta: float = .005
    max_change: float = .01
    response: float = 0.
    note_table: dict | None = None
    note_bond_count: int | None = None
    note_response_integer: int | None = None

    def __post_init__(self):
        self.table=_nonnegative_table(self.table)
        self.old_bond_count=_count(self.old_bond_count,'old_bond_count')
        self.natoms=_count(self.natoms,'natoms');self.cycle=_count(self.cycle,'cycle')
        self.frequency=_count(self.frequency,'frequency');self.presteps=_count(self.presteps,'presteps',0)
        if not isinstance(self.lselfadapt,(bool,np.bool_)):raise ValueError('lselfadapt must be boolean')
        if not np.isfinite([self.ratio,self.response,self.target_mev_per_atom]).all():raise ValueError('finite controller parameters required')
        self.eta=_finite(self.eta,'eta');self.max_change=_finite(self.max_change,'max_change',positive=True)
        if self.note_table is not None:
            self.note_table=_nonnegative_table(self.note_table)
            if set(self.note_table)!=set(self.table):raise ValueError('note/live table keys must match')
        # Fortran NINT for the finite normal domain, not Python round-to-even.
        x=self.cycle*self.ratio
        if not np.isfinite(x):raise ValueError('nonfinite soft interval')
        self.nsoftstep=math.trunc(x+(0.5 if x>=0 else -0.5))
        if not -(2**31)<=self.nsoftstep<2**31:raise ValueError('soft interval outside native int32 range')

    def advance(self,step,*,measured_response_mev_per_atom,new_bond_count):
        """Apply one caller-identified invocation; commit only if all valid.

        Returned transition lists save_zero/restore/normal_update in execution
        order. No action at step zero; it is the already-initialized boundary.
        A restore without a supplied or previously saved note raises explicitly.
        Repeated/out-of-order calls are not inferred or silently corrected.
        """
        step=_count(step,'step',0)
        new_bond_count=_count(new_bond_count,'new_bond_count')
        response=float(measured_response_mev_per_atom)
        if not np.isfinite(response):raise ValueError('finite measured response required')
        table=dict(self.table);old=self.old_bond_count
        note=None if self.note_table is None else dict(self.note_table)
        note_n=self.note_bond_count;note_r=self.note_response_integer
        actions=[];q=None
        active=0<self.nsoftstep<self.cycle
        phase=None
        if step and active and (not self.lselfadapt or step>self.presteps):
            phase=(step-(self.presteps if self.lselfadapt else 0))%self.cycle
            if phase==self.nsoftstep:
                note=dict(table);table={k:0. for k in table}
                if self.lselfadapt:
                    trunc=math.trunc(response)
                    if not -(2**31)<=trunc<2**31:raise ValueError('response note outside native int32 range')
                    note_n=old;note_r=trunc
                actions.append('save_zero')
            elif phase==0:
                if note is None:raise ValueError('restore requires saved note table')
                table=dict(note)
                if self.lselfadapt:
                    if note_n is None or note_r is None:raise ValueError('restore requires saved count/response notes')
                    old=_count(note_n,'note_bond_count')
                    if isinstance(note_r,bool) or not isinstance(note_r,(int,np.integer)) or not -(2**31)<=note_r<2**31:
                        raise ValueError('invalid native int32 response note')
                    response=float(note_r)
                actions.append('restore')
        if self.lselfadapt and normal_update_due(step,frequency=self.frequency,presteps=self.presteps):
            result=update_native_table(table,natoms=self.natoms,old_bond_count=old,
                new_bond_count=new_bond_count,response_mev_per_atom=response,
                target_mev_per_atom=self.target_mev_per_atom,eta=self.eta,max_change=self.max_change)
            table=result.table;q=result.q;old=new_bond_count;actions.append('normal_update')
        # step zero does not consume/overwrite the previously initialized state.
        if step:
            self.table=table;self.old_bond_count=old;self.response=response
            self.note_table=note;self.note_bond_count=note_n;self.note_response_integer=note_r
        return dict(step=step,phase=phase,nsoftstep=self.nsoftstep,actions=tuple(actions),
            table=dict(self.table),old_bond_count=self.old_bond_count,response=self.response,q=q)
