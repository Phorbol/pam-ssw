# Explicit LASP ranseed: qualified short-prefix control

2026-10-07. CPU1664773, frozen source
`c3c7e55a9002f0933a281de8edb67847a8e4a118`; see [protocol](protocol.md)
and [raw/derived run](run-1664773/).

Three separate native processes consumed the same C60#3 input and synthetic
quadratic forces: ranseed26100791 twice, then26100792. Every allkeys log
retained the requested positive seed. The repeated seed produced identical
coordinates for all8 observed requests; the other seed shared requests1–2
but differed from request3 onward, by up to.105746Angstrom/component.
The first displaced request is deterministic in this prefix. Printed
internal startup seed values were absent for positive inputs, so no claim
is based on an unobserved printed value.

Costs:24 successful synthetic E/F requests,21 synthetic calculate calls
(ASE caching),3 quota denials,0 MACE or physical-PES calls. Each native
process exited29 at its expected callback quota; supervisors returned
normally with no cleanup survivors. Whole probe elapsed4.88s.

The root independently checked all archived source/input hashes, the three
8x60x3 coordinate arrays, their same/different-seed comparisons, allkeys
flags and exact request/calculate/denial counters. This supports using
explicit ranseed for a separately frozen native reference. It does not
establish long-run bitwise reproducibility or statistical independence.
Older17093/17094 native labels only designate input-cloud seeds and remain
uncontrolled internal-RNG runs. No expanded RNG decompilation is needed.
