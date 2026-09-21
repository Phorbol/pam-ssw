# Standalone quench memory audit

Static grep of `pamssw/standalone` found all direct `quench` and
`safe_lbfgs` consumers carrying the explicit `lbfgs_memory` value or the
documented `None` default:

- fixed-cell TYPE0/TYPE3 paths in `paper_ga.py` and `paper_reference.py`;
- constrained TYPE4 path in `constrained_reference.py` and its
  `surface_ga_reference.py` wrapper;
- periodic TYPE1 and TYPE2/3/4-related wrappers using the shared SSW config;
- fixed-cell LS cycle entry/exit in `ls_cycle.py`;
- fixed-cell block path in `block_ssw.py`;
- variable-cell `vc_reference.py` and `rc_vc_reference.py`;
- reduced-coordinate RC path in `rc_reference.py`.

`rc_vc_reference.py` conditionally omits the keyword when the value is
`None`, which intentionally preserves the backend default; its non-`None`
path passes the value. No remaining definite memory-forwarding omission was
found in this bounded scan.

Algorithm conclusion: the implemented change is a numerical-capacity and
interface-consistency option for optimizer history size, with `None` retaining
the established default trajectory. The native pre-quench disassembly only
established caller ordering and did not prove either a `xa/fa` defect or a new
algorithmic benefit. It therefore supports no claim of native-derived search
gain and does not justify a new mechanism or extra PES evaluation.
