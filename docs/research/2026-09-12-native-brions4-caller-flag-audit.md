# Native BRIONS4 → BRZERO4 caller and flag audit

This is a bounded static audit of the frozen native executable.  It does not
start LASP, bypass its protection logic, or evaluate a PES.

* ELF: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`
* SHA256: `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`
* Disassembly: `/tmp/lasp-dis.asm`

## ABI and actual SSW caller

The DWARF declarations identify the module routines as follows.

| routine | register arguments | stack arguments (in order) | entry |
|---|---|---|---:|
| `broyden_module_mp_brions4_` | `nions,pos,for,a,fact,hist_len` | `iout,iu6,iniangle,langle,infor,rotmode` | `0x6f6440` |
| `broyden_module_mp_brzero4_` | `ndim,x1,f1,g01,ini,iu6` | `iout,iniangle,norder,langle,rotmode` | `0x6f6c00` |

The production `newssw_basics_mp_rotate_dimer_` path calls BRIONS4 at
`0x6e639a`.  Its call-site construction (`0x6e635a–0x6e6396`) gives this
mapping, preserving the pointer/address distinction:

| BRIONS4 formal | production source at call site | what is established |
|---|---|---|
| `pos` | `%r14` | rotation position workspace |
| `for` | `0x10(%rbp)` | force pointer supplied to `rotate_dimer` |
| `fact` | `0x50(%rbp)` | original caller FACT, passed in `%r8`; later retry scaling uses saved FACT1 |
| `hist_len` | `&(-0x40(%rbp))` | local value set from decremented `%esi` (`0x6e6366–0x6e636e`); existing follow-up identifies it as outer `rotnum-1` |
| `iu6` | `&(-0x44(%rbp))` | local is assigned `-1` on the shown retry/entry paths (`0x6e60b8`, `0x6e632f`) |
| `iniangle` | `&(-0x50(%rbp))` | scalar copied from `-0xd8(%rbp)` at `0x6e5e8e–0x6e5e98`; `-0xd8` is formed by the preceding dot-product/normalization paths (`0x6e5850–0x6e5b61`) as `(sum(r9*r12)-sum(arg[0x10]*r12))/input[0x58]` on the main path, and is not a fixed `.5` |
| `langle` | `&(-0x3c(%rbp))` | pointer passed; value not recovered in this bounded slice |
| `infor` | `&(-0x38(%rbp))` | pointer passed; contents not recovered here |
| `rotmode` | `&(-0x48(%rbp))` | local is initialized to `-1` at `0x6e5817`; no later store appears before the traced call |
| `iout` | `%r8 = 0x4a4cb98` (`__NLITPACK_0.0.14`) | packed literal address; its first 32-bit word is 6 (the remaining packed words are descriptor data) |

The `nions` and `a` arguments are loaded indirectly from the local pointer
held around `0x6e6371–0x6e6388`; this report does not relabel those Fortran
allocatable descriptors as scalar values.  The packed literal at `0x4a4cb98`
is the bytes `06 00 00 00 38 04 02 00 00 00 00 00 09 01 02 00` (followed by
`00 00 00 00 30 01 02 00 00 00 00 00`).  Its first word is therefore the
integer output unit 6; the rest is packed descriptor data consumed by the
diagnostic write path (`0x6f8c3a–0x6f8c7b`).  BRIONS4 computes `ndim=3*nions`
(`0x6f6b35–0x6f6b39`) and fills its saved G0 array with the saved STEP
(`0x6f69fd–0x6f6b33`); the initial STEP is 1 according to the adjacent static
record.

BRIONS4 forwards to BRZERO4 at `0x6f6b8d`.  The wrapper establishes
`RDI=&ndim` (`0x6f6b55`), `RCX=&G0` (`0x6f6acd`), `R8=&ini`
(`0x6f6b5d`), and `R9=[BRIONS4 iu6]` (`0x6f6b84`).  Its stack copies are:

```
BRZERO4 iout     <- BRIONS4 iout       [0x6f6b71]
BRZERO4 iniangle <- BRIONS4 iniangle   [0x6f6b75]
BRZERO4 norder   <- BRIONS4 infor      [0x6f6b7a]
BRZERO4 langle   <- BRIONS4 langle     [0x6f6b7f]
BRZERO4 rotmode  <- BRIONS4 rotmode    [0x6f6b88]
```

The wrapper sets `ini` to `-1` when its internal BRIONS4 iteration becomes 1,
and to 0 otherwise (`0x6f6b3c–0x6f6b4a`).  Thus this is a first-internal-
iteration initialization flag, separate from the caller's `iu6` local.  In
BRZERO4, the forwarded `norder` pointer is overwritten with integer 1 at
`0x6f8c0e–0x6f8c19` before the later diagnostic/output branch.  No read of
`0x28(%rbp)` (the BRZERO4 `langle` slot) appears in the inspected BRZERO4
body, whereas `0x30(%rbp)` (`rotmode`) is dereferenced at `0x6fea49` and
`0x6fee94`.  This establishes both use of `rotmode` and the traced production value `-1`.
The high-level reset/retry interpretation is also documented in
`docs/research/native-rotation-followup.md`, but this table is the direct
assembly mapping.

## Probe versus production flags

`research/ga_ssw/probe_native_broyden_full.py:98–101` packs the BRZERO4 call as
`ini=initial`, `iu6=-1`, `iout=-1`, `iniangle=.5`, `norder=0`, `langle=0`, and
`rotmode=0`.  The comparison is therefore:

| field | isolated probe | production rotate_dimer | conclusion |
|---|---|---|---|
| `ini` | explicit `initial` per probe call | `-1` only when BRIONS4 ITER becomes 1; otherwise 0 | differs in control generation; probe can exercise the value but does not reproduce caller timing |
| `iu6` | fixed `-1` | local pointer observed as `-1` on the traced paths | value agrees in this path; broader branch coverage unknown |
| `iout` | pointer containing `-1` | packed literal at `0x4a4cb98`, first word 6 | production selects output unit 6; probe selects unit -1, relevant only when diagnostic output executes |
| `iniangle` | fixed `.5` | runtime scalar from `-0xd8(%rbp)` | not the same source; main-path machine formula is shown above |
| `norder` | fixed integer 0 | production initially forwards `infor`, then writes integer 1 through it at `0x6f8c0e–0x6f8c19` | effective value is 1 in this path; pre-call 0 alone is not a numerical mismatch |
| `langle` | fixed integer 0 | pointer to local `-0x3c(%rbp)`; no store was found before the call and no BRZERO4 read of `0x28(%rbp)` was found | value/role remains unknown, but it is not observed to affect this BRZERO4 body |
| `rotmode` | fixed integer 0 | pointer to local `-0x48(%rbp)`, initialized `-1` at `0x6e5817`; BRZERO4 reads `0x30(%rbp)` at `0x6fea49`/`0x6fee94` | differs: production value is -1 on this path |

Consequently, the full BRZERO4 arithmetic probe is evidence for the numerical
body under its explicitly supplied arguments, while it is not a native
`rotate_dimer` call-equivalence test.  In particular, `.5/0/0` must not be
reported as the native default angle/mode tuple, and the packed `iout` literal
must not be replaced by `-1` in a prospective integration without resolving
its ABI role.

## Remaining boundary

This trace closes the register/stack forwarding and the first-iteration
`ini` assignment.  It does not recover the values written into production
`-0x3c` or `-0x38`, nor all alternate `rotate_dimer` branches (the
routine has explicit size/branch paths before `0x6e633f`).  It does not recover the semantic contents of the packed descriptor beyond
its output-unit word, or establish whether `-0x3c`/`infor` values matter in
other branches.  These remaining boundaries are sufficient to block calling the current fixed-flag probe a full
native BRIONS4/BRZERO4 integration; no production integration change is
justified by this audit alone.
