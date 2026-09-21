# C60 native fragmentation controls: bounded audit

## Source facts

The local SI snapshot
(`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/ct4c01081_si_001.txt`)
explicitly includes `SSW.globalcompress 0.0001`,
`SSW.vapor_cri 1.7`, and `SSW.Ratio_Local 50` in both the C60 LS-SSW input
(§7.3, lines 250–260) and the C60 SSW input (§7.4, lines 275–283), using
`SSW.NG 12`, `ftol 0.05`, and `ds_atom 0.6`. Thus these options were present in
the published C60 input examples. This does not prove that every option is
active in the archived Python C60 runs or identify the exact C60 trajectory
configuration.

The inspected ELF has symbols
`ssw_commsub_mp_check_vapor_new_` at `0x5950f0`,
`ssw_fixlat_mp_allopt_judge_converg_` at `0x5cf6f0`, and
`newssw_basics_mp_compress_mode_` at `0x6e36f0` (the saved
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/lasp-symbols.txt`).
In the saved judge disassembly, the vapor branch compares the parameter field
at `para+0x2de60` with a constant, requires local step `> 50`, calls
`check_vapor_new_`, and compares the returned value with `vapor_cri`.
The nearby byte test at `para+0x2dad8` is the separate `lts_extra` condition,
not a vapor enable byte.
If that comparison is exceeded it writes `-1` to the judge result and follows
the normal stop path (`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/judge-convergence.asm`, `0x5cfb64–0x5cfbd5`).
This judge call passes logical zero (`rcx=0x4a45894`), so it takes the
non-mutating detection branch. It is not itself a coordinate compression or MC
rejection action.

## 2026-09-11 recovered coordinate-modifying branch

The new [full criterion audit](native-vapor-criterion.md) supersedes the former
statement that only a conditional stop was known. The function connects finite
coordinate pairs using strict `distance < vapor_cri`. Its fourth argument selects
a detection branch or an in-place coordinate-translation branch. In the latter,
the original instructions compute a translation of the form
`NSAVE * (|NSAVE| - 0.7*vapor_cri) / |NSAVE|` and add it to selected coordinates.
The exact member selection and returned scalar are still being recovered; this
is not yet a complete Python reconnection specification.

The `Allopt` branch in `ssw_fixlat_mp_set_status_` calls this function at
`0x5c2857` with `rcx=0x4a45654`, whose logical value is true (`0xffffffff`).
This establishes a native state transition that can modify coordinates, absent
from the current independent walker. The following indirect action at
`0x5c28e1` and energy/force refresh ordering remain open. The complete evidence
is in `research/ga_ssw/evidence/native-vapor-allopt-caller.asm` and
`native-vapor-criterion.asm`.

This finding does not establish `globalcompress` or `Ratio_Local` semantics,
nor prove that either parameter was active in an archived Python trajectory.
Their parser/selection consumers have now been partly recovered separately in
[native-cluster-control-selection.md](native-cluster-control-selection.md):
`globalcompress` selects a mode via a random comparison, while `Ratio_Local`
scales an intermediate random coefficient. The final direction action remains open.

## Relation to the C60 archives

The old and new C60 records both reach a final biased entry whose last
successful bias has a C58+C2-like separation, but their reported gaps differ
(old 3.773 Å, new 6.925 Å). Those are trajectory observations, not proof that
the native vapor guard was enabled or that it would have acted. The current
Python archive has no native vapor/global-compress/Ratio_Local state field, so
it cannot claim to reproduce those controls. Its initial fresh check being
connected does not qualify the later failed stage.

## Boundary and next falsifiable check

No fragmentation constraint or threshold is added here. Next: execute the
isolated original geometry routine with controlled disconnected components;
close member selection, output scalar and downstream refresh; only then specify
an independent optional implementation. A same-MACE ordinary SSW control is
also needed before attributing the observed C60 fragmentation to LS. Neither the
paper's input threshold nor the recovered factor 0.7 is a universally justified
chemical distance or a default for mixtures, crystals or reactive fragments.
