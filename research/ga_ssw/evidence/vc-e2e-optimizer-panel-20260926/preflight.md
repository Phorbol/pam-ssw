# Preflight record

All actual EMT evaluations ran on CPU-MISC, not a login node.
1500293: expected module-not-found red before worker implementation; 0.93s.
1500318: 5 passed, 1 failed in4.47s; test summed the fresh_started charged=None
value as a number. Root corrected the assertion to bool-normalize that tri-state.
1500329: identical failure2.14s because the root edit command accidentally used
the base checkout cwd and failed before editing. Unnecessary repeated CPU check;
retained as an execution error, not new algorithm evidence.
1500333: after that fix, 5 passed/1 failed3.58s: the baseline adapter's certificate
contains NumPy arrays, but the research worker serialized only the core result.
Root reused existing serial() for adapter_calls; no numerical algorithm changed.
1500336: targeted worker and existing adapter tests all6passed2.13s, exit0.
One existing pynvml deprecation warning, no scientific interpretation.

Command: python -m pytest tests/research/test_vc_e2e_optimizer_panel.py
 tests/research/test_vc_lbfgs_baseline_runner.py -q
Environment: mace_env, PYTHONNOUSERSITE=1, OMP/OPENBLAS threads1.
The worker test checks a real Cu4 fcc EMT outer attempt with each implementation,
exact search/fresh ledger accounting, context restoration, record-derived fresh
endpoints, and one-request cap handling. This is implementation evidence only.
Failed test outputs and all six-test outputs remain alongside this record.
