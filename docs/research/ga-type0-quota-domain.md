# TYPE0 integer quotas and useful input domain

Root checked uploaded decompiled sources under
/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/decompiled/sgn/:
`ga_Interface/TYPE0.java:39-59`, `ga_cluster_cell/Mutate.java:84-110`, and
`ga_cluster_cell/Compete.java:41-54`. The Java scheduling source **is available**;
a prior agent statement that only primitive probes were available was incorrect.

For requested count G, each batch has G//4 crossover children. Region zero uses
mutation n=G//2; other regions n=G//8. A multi-element region contributes
3*(n//4) exchange, four*(n//8) disturbance, and n//8 reinsertion children.
For a single region:

| G | Cross | Exchange | Four disturbance types total | Reinsertion | Total/batch |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 4 | 1 | 0 | 0 | 0 | 1 |
| 8 | 2 | 3 | 0 | 0 | 5 |
| 16 | 4 | 6 | 4 | 1 | 15 |

The independent controller now rejects multi-element TYPE0 G<4 with generations>0
before any PES calls. This is a static zero-output condition, not a fitted search
parameter or altered genetic operator. Pure-element mutations include +1 terms,
so this rejection must not apply to the Cu/Al single-element G=1 tests.
The lower-level legacy proposal retains its existing reconstructed behavior.

The reconstructed Compete domain requires >2 parents and positive energy span.
Java computes T=-(Emax-Emin)*1000*2625/(log(2/N)*8.314); the Python restriction is
an explicit finite, positive-temperature domain, **not an explicit Java guard**.
The N=2 infinite-temperature limit could be defined as a new independent policy,
but is not silently introduced here. Do not reject a small initial population
solely on this basis: quick walks can grow the archive before actual competition.

The two-isomer C4H6 development run therefore provides quick/fine/LS evidence,
not a completed genetic-stage validation. Its failed proposal is retained.
A separate fixed protocol uses all five ASE G2 C4H6 isomers, G=4, up to four
batches. It tests crossover integration; no mutation is claimed at that quota.
It changes the declared population/quota to meet existing operator requirements,
not force tolerances or the physical objective, and is not pooled as a fair
performance comparison against the two-isomer run.
