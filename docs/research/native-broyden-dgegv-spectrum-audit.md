# BRZERO4: two spectral consumers, including history control

Root-verified 2026-09-12. This replaces the earlier incomplete interpretation
based only on references to the named EIGENVAL array.

The DGEGV call is N/N (no eigenvectors), A=GP, B=SP. The EIGENVAL store uses
Re(alpha/beta)+1 and is later printed. This does NOT imply that DGEGV is
only diagnostic: raw AUXC and AUXBET have a second arithmetic consumer.

At 0x6fbf60--0x6fbfb7 the code computes abs(1+alpha/beta), including the
imaginary component, accumulates its sum and maximum. The exact constant
at 0x4a4d080 is the complex pair (1.0,0.0); it is NOT (1,1).
At 0x6fbfea the maximum is compared with 0x4a4d0f8 = 10000000.0, NOT zero.
These constants were read using ELF load-segment virtual-address offsets.
The packed (1,1) used in the earlier two-real-entry output loop must not be
confused with a complex shift.

For active order below50 and maximum strictly below1e7, the branch at
0x6fbff2 accepts the current trial (0x6ff8c3). Otherwise the code can increase
the trial history-removal count at0x6fc002 and repeat the matrix/eigenvalue
calculation via0x6faf72. There the tested order is original order minus the
trial count; GP/SP are copied and shifted before the next DGEGV. The eventual
nonzero count reaches history shifting via0x6fc11a--0x6fc180. The check at
0x6fc064 is an output-unit guard, NOT the numerical deletion criterion.
Zero beta forces the maximum to1e7 in this consumer; exceptional/nonfinite
behavior and the minimum-history restart branch remain separately bounded.

## Actual dynamic control trace

`research/ga_ssw/evidence/native-broyden-full-probes-20260912/n9-seed29-spectral.json`
executes BRZERO4 and its inverse routines from the uploaded ELF. Only DGEGV
is numerically replaced by an actual SciPy DGGEV solve of its real input
matrices; allocation/copy/printing are isolated runtime hooks. This is a
mixed numerical oracle, not full original-LAPACK instruction parity.

| History order | Trial removal count | Original-instruction computed maximum |
|---:|---:|---:|
| 1 | 0 | 2.6226933180 |
| 2 | 0 | 756.0612580682 |
| 3 | 0 | 189674.480399538 |
| 4 | 0 | 27592684.13490168 |
| 3 | 1 | 17.703580342840336 |

The fourth addition triggers one-history removal. The returned DF columns
exactly retain the previous second and third columns; the oldest is removed.
The read-only instrumentation leaves all step outputs identical to the
uninstrumented run `n9-seed29.json`. The mixed LAPACK boundary matters,
especially near a threshold or an ill-conditioned generalized problem.

This establishes a genuine history-control use of the generalized spectrum,
not a lowest-mode eigenvector solver. The 1e7 bound is a recovered numerical
constant, not a universally derived physical tolerance. The meanings and
signs of GP/GMAT must be understood before applying this rule to a new
Cartesian Broyden implementation. In particular, neither this maximum nor
the printed real values may automatically be called a Hessian condition number.
