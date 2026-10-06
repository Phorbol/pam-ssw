# LJ crystal reference qualification: phase ordering is model-dependent

CPU1664166, executed source76a75cc, completed8s. Fixed
[protocol](protocol.md); raw `run-1664166/{fcc,hcp}-rc*/` holds initial/final
structures, full E/F/stress ledgers, fresh checks and source snapshots;
`run-1664166/summary.json` is the authoritative aggregate. No global search.

Root independently reconciled every request and calculate-call delta and
checked all eight numerical certificates and paid gradient checks. Cost:
173 reference E/F/stress requests +8cold =181 requests,164 actual calculator
calls; no failed evaluations. Eight terminal/cold force and stress checks
pass. Hcp rc2.7 BFGS returned precision_loss although its cold stress is
1.50e-7eV/Angstrom³, below the predeclared1e-5 threshold; this optimizer exit
is retained separately from the qualified endpoint. Other seven exits
reported convergence. Qualification does not establish full Hessian stability.

| rc/sigma | Cold Ehcp−Efcc (eV/atom) | Lower of these two relaxed families |
|---|---:|---|
| 2.7 | +0.0076297548 | fcc |
| 3 | −0.0031910091 | hcp |
| 6 | −0.0006624705 | hcp |
| 10 | −0.0008747882 | hcp |

The existing rc2.7 ASE bulk smoke therefore cannot use hcp as its assumed
lowest-energy target. A missed hcp on that model is not evidence of a VC
walker defect. This is a model boundary, not a reason to retune the walker.
The observed signs support the literature warning about cutoff-dependent
phase ordering; they do not show which model reproduces the SSW paper.

The 2014 VC paper quotes an hcp/fcc gap of6.51e-4 epsilon/atom but does not
give the cutoff in available section3.2. The rc6 result is numerically close;
that does not identify the paper's potential or justify selecting rc6 to
manufacture agreement. Neither hcp nor fcc is proven the GM among other
stacking sequences. Higher-cutoff energy convergence was not established.

Decision: retain this as reference/model qualification, with no new runtime
default. Before a paper-rate comparison, resolve the interaction convention
or explicitly limit the next search to its declared ASE-model target.
The random LJ128 cube has documented side5.553sigma; the vacancy-start
geometry remains missing. Do not replace the paper's LJ128 vacancy state by
a guessed127-atom cube, or automatically launch8×250 trajectories.
The ongoing C60 direction panel and prospective LS input gate keep priority.

Sources: [SSW-crystal2014](https://doi.org/10.1039/c4cp01485e), section3.2
and Fig4, local author fulltext cited in protocol; Pártay et al.,
[Polytypism in the ground state structure of the Lennard-Jonesium](https://arxiv.org/abs/1705.01751)
(2017), DOI10.1039/C7CP02923C; [official ASE LJ source](https://docs.ase-lib.org/_modules/ase/calculators/lj.html).
