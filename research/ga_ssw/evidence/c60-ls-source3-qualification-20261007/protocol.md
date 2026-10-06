# New local C60 LS target: input qualification only

Question: does a previously unused published C60 isomer remain a distinct,
connected fullerene stationary candidate on our MH-1/omol PES after ordinary
initial quench? This gates a possible later LS-only local bond-rearrangement
comparison, not a random-start global search. The four-arm random C60
direction panel1664068 is unchanged; its new clouds will not be reused for
an LS parameter selection campaign.

Scientific motivation: LS2024, DOI
[10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081),
section2.2 discusses local bond rearrangement in non-Ih C60 and section3.2
examines fullerene targets. Its C60 Table1 gain is modest compared to C70;
this qualification cannot establish the paper's global efficiency claim.
Fulltext215.txt and SIct4c01081_si_001.txt are in
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/`.

Source geometry: Bin Liu, Jirui Jin and Mingjie Liu, *Mapping
structure-property relationships in fullerene systems: a computational study
from C20 to C60*, npj Computational Materials10,227 (2024), DOI
[10.1038/s41524-024-01410-7](https://www.nature.com/articles/s41524-024-01410-7).
Use only `c60/c60-iso-3_opt.xyz` from the already downloaded author archive
`literature/c60-defect-input-20260925/41524_2024_1410_MOESM3_ESM.zip`
under the above archive root. Save its bytes/member identity and corresponding
CSV row in MOESM2. This is author isomer#3, not proven a single Stone-Wales
neighbor or a PBE/SSW-paper input. Prior #1/#2 experiments stay historical;
do not substitute them or select another archive member if this gate fails.

Boundary: isolated60C,50Angstrom diagonal storage cell,pbc=False;
translate the source center of mass to(25,25,25) without rotation or
distortion. Validate the actual serialized input before calculation.
Read-only geometry screening found a cage distinct from #1/#2; numerical
qualification has not been performed. Retain all three graph-cutoff checks
(1.64,1.70,1.80Angstrom) and compare graph identities by isomorphism.

Backend: cached MH-1 a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47,
omol,float64,CUDA,cueq/oeq off, one thread, deterministic settings inherited
from the current C60 panel. Qualified Ih reference is
`research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-1/final.extxyz`.
Record a fresh E/F for this unchanged reference; no new reference quench.

Run only ordinary `run_ssw(steps=0)`, with the existing C60 protocol's
Safe-total/history500,relax_steps1000,fmax.03eV/Angstrom and direction_only
frame. LS, Gaussian climbing, direction rotation and MC moves are absent.
No new default, optimizer or persistent-state format. Exact source revision,
import paths, actual script/used helper snapshots and effective config must
be recorded. Reuse existing counted-surface and graph primitives, rather
than copy another budget framework into the core.

Competing explanations: MH-1 retains this source topology as a force-qualified
candidate with a distinct Ih target, enabling a local LS-only experiment; or
ordinary quench already repairs/destroys its graph, making that experiment
ill-defined. Numerical convergence alone is not the decision criterion.
Require finite input/endpoint, initial convergence and coldfmax<=.03,
a fullerene graph at all three cutoffs, source graph identity retained at
all three cutoffs, and nonisomorphism to Ih. Also report relative energy;
if it lies within the Ih+.01eV window, do not silently relabel it a useful
energy-repair case. No Hessian/TS/barrier or model-global-minimum claim.

Bound: one input,<=3000 paid initial-quench E/F and120sec input deadline,
one independentcold for a certified candidate and one referencecold,
total<=3002 requests;240sec cooperative process deadline, oneV100/5min.
No retry/continuation, input replacement, LS search or geometry refinement
on failure. Preserve source/raw input, failed endpoint if returned, final
candidate, full qualification result and raw cost ledgers. This protocol
authorizes only the eligibility gate within the already authorized bounded
mainline development; a later fixed LS protocol must separately define its
factor, sources, stopping rules and total-cost criterion before submission.
