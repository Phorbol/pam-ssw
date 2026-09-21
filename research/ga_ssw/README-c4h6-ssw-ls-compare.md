# Bounded C4H6 SSW / LS comparison

This prepared runner starts from the ASE G2 `butadiene` structure and compares
the public `run_ssw`, `run_ls_ssw` and `run_native_ls_ssw` entry points for
seeds 11 and 29.  It uses the existing staged Ritz protocol (`pre_rotation_hvp=5`,
`rotation_hvp=100`, `max_gaussians=25`, width 0.1, 150 K, inner force 0.1,
outer force 0.01, 400 relaxation steps) and the LS tables/settings copied from
`run_fixed_ga_staged_integration.py`.

Each arm has a 40,000 search E/F request cap and 120-second wall bound. Every
successful or failed search call is retained in `evaluations.jsonl`; cap or
wall denials are separately recorded and do not increment calculator requests.
Every returned landing is checked with a fresh cold GFN2-xTB calculator.
LS update/response events remain in `result.json` and are counted in summaries.

The directory is created without PES calls unless `--execute` is supplied. The
paper's C4H6 oracle is DFT GGA-PBE. This runner deliberately uses GFN2-xTB as
an algorithm/coverage comparison, so its energies and ordering are not a
numerical reproduction of the paper (the backend can rank isomers differently).
The 400 value is an outer-step limit under a request cap, not a claim to have
reproduced 400 paper minima. Two seeds and one initial isomer cannot establish
an efficiency or scientific LS advantage claim.

Executed evidence: `evidence/c4h6-ls-reaction-coverage-20260912`.
Six wall-capped arms,131177 search +129 fresh requests; all fresh force checks
pass. Full interpretation: `docs/research/2026-09-12-c4h6-ls-reaction-coverage.md`.
