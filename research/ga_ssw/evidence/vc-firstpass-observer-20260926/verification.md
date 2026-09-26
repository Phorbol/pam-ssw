# First common-certificate passage: observational records only

The full VC baseline bridge discarded accepted-iterate information already
returned by ASE/SciPy adapters. Native termination can exceed the first common
certificate, so terminal cost alone cannot attribute optimizer efficiency.

The bridge now records first_common_qualified_accepted_index (zero includes
the initial point), first_common_qualified_request, the corresponding criterion,
and requests_after_first_common_qualified. If no accepted point passes, all four
are null. The last quantity is native terminal requests minus first passage;
it is not a prediction of savings in an alternative SSW trajectory.

For true cell relaxation the cached physical force/stress criterion is compared
to 1; biased calls use the existing common gradient norm/gtol. A biased pass is
not a physical force/stress certificate. A transient accepted-point pass does
not change terminal success/failure, native stopping, returned state, random
state, or any public search/checkpoint contract. Observation runs only when
records are requested and makes no oracle calls. No old trajectory is reclassified.

CPU job1503321 ran:

```sh
python -m pytest -q tests/research/test_vc_lbfgs_baseline_runner.py tests/research/test_vc_e2e_optimizer_panel.py
```

10 tests passed in5.99s, one existing pynvml deprecation warning. The retained
tests.sbatch supplies the actual environment and group account. The root reviewed
the diff, actual output and complete Gaussian-path assertion; git diff --check
passed. Tests include no-hit/None-callback cases, identical native output and
oracle counts, cached true-cell certificates on Cu/EMT, and a full Gaussian-biased
Cu/EMT call. Earlier diagnostic failures1503279/1503287 are retained: the original
9-HVP smoke reached an outer record but failed rotation before invoking a biased
optimizer. The dedicated observer test keeps seed7 and all tolerances and uses
the existing100-HVP production budget to exercise that call. It does not alter
the original smoke or any scientific experiment settings. The earlier bare-PES
bridge check was not a Gaussian-path check and is not counted as one.

This is instrumentation qualification, not new evidence of search performance.
