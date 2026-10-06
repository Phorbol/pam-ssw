# LASP explicit ranseed: bounded interface qualification

Why: old native labels17093/17094 identify input clouds, not internal RNG.
Plans explicitly say this; actual native startup seeds differ. The archived
parser and DWARF expose `ranseed`/`init_random_seed(ranseed)`, but prior runs
used default-666. A same-input reference pair must not confuse job labels
with controlled native random state.

Probe three serial native processes with the identical qualified C60#3
50Angstrom periodic ARC: requested ranseed26100791 twice,26100792 once.
Use global `ranseed` as exposed by the archived allkeys listing. Do not guess
other options if rejected. Use SSWsteps2/MaxOptstep10 solely to expose early
direction requests; a synthetic quadratic E/F callback centered on the input
provides deterministic responses. No MACE, real physical PES, reference
energy, global-target or performance test is involved.

Each process<=30seconds and<=8synthetic E/F attempts; three processes<=24
synthetic E/F attempts total. One CPU-MISC allocation<=5minutes, one thread;
no GPU or automatic retries/extensions. Preserve all outputs, denied
requests, source/input snapshots, applied key values and process cleanup.
Only this job's own process tree is subject to its preset supervisor.

Readout: does allkeys retain each requested value? Is the printed internal
seed deterministic for repeat input? Do repeat runs expose the same early
coordinate prefix, and does the other seed expose a different perturbation?
Compare transmission coordinates at1e-4Angstrom per component; also record
exact maximum differences and first nonzero displacement. Identical short
prefixes for different seeds can mean deterministic startup logic and do not
disprove a later RNG effect. Missing/denied callbacks are retained, not
replaced with selected successful streams.

Decision: a supported, repeated seed and matching short prefix permits a
separately frozen seeded native reference. It does not establish long-run
bitwise reproducibility or statistical independence. A rejected/ignored key
leaves native runs explicitly unseeded; do not call two job IDs a controlled
pair or expand this probe into full RNG decompilation. Other mainline case
tests continue unchanged.
