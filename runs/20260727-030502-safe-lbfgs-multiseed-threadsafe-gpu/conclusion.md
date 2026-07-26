# Proposal-relaxation conclusion

## Scope

This stage isolated one question: whether changing only the biased-PES
proposal-relaxation backend reduces exact force evaluations and improves the
resulting equal-budget SSW search.

It did not change the direction generator, starter selector, Gaussian bias,
true quench, archive, uniform policy, or force budgets.

## Fixed-task evidence

On 16 optimizer-neutral one-bias tasks with a shared 400-step observation cap:

- safe total-gradient L-BFGS satisfied the `0.05 eV/A` certificate on 16/16;
- FIRE satisfied it on 14/16;
- safe L-BFGS used fewer exact evaluator calls than FIRE on 16/16;
- median safe/FIRE call ratios were 0.542 on C60 and 0.491 on PdO.

This is not a same-endpoint acceleration result. Several safe-L-BFGS endpoints
were structurally and energetically different from FIRE. The strongest valid
claim is therefore task-level certificate efficiency on the same initial
biased-PES objective.

The FIRE-conditioned cumulative-bias stress set was smaller than planned:
C60 produced six tasks and PdO two, because the reference walks often
terminated before reaching bias counts 2 or 4. Safe L-BFGS used fewer calls on
7/8 captured tasks, but again frequently reached different endpoints; one C60
task required more calls than FIRE. The stress runner did not persist the full
cumulative-bias task definitions, so this secondary stratum is not independently
reconstructable and is not used for the primary optimizer-neutral claim.

## Equal-budget end-to-end evidence

The final matrix used one frozen, certified bootstrap state per system, paired
seeds 42/43/44, a uniform policy, two ThreadPool workers, 1000 evaluations per
action, and a 3000-evaluation campaign budget.

Every arm was benchmark-eligible with closed purpose-resolved accounting.

### C60

Safe-minus-FIRE best-energy-drop differences were:

- seed 42: `-0.939 eV`;
- seed 43: `-2.447 eV`;
- seed 44: `-0.308 eV`.

Median best-energy drop decreased from `5.583 eV` with FIRE to `4.643 eV` with
safe L-BFGS. Safe L-BFGS nevertheless increased archive coverage on seeds 43
and 44: unique-minimum counts were `[5, 8, 7]` versus FIRE's `[5, 5, 5]`.

### PdO

Safe-minus-FIRE best-energy-drop differences were:

- seed 42: `+0.797 eV`;
- seed 43: `+2.186 eV`;
- seed 44: `+1.272 eV`.

Median best-energy drop increased from `2.483 eV` with FIRE to `3.803 eV`.
Unique-minimum counts were `[10, 7, 9]` versus FIRE's `[7, 6, 6]`.

## Decision

Do not change the global default from FIRE to safe L-BFGS.

Safe L-BFGS is a real, positive proposal-policy candidate on this fixed PdO
benchmark and an effective stationary-point solver on the frozen objectives,
but the C60 regression proves that fewer local evaluator calls do not imply
better global exploration. It must remain an explicit, independently ablatable
action rather than a hidden replacement or a fixed heuristic mixture.

FIRE2 and bias-separated L-BFGS do not advance. No optimizer parameter sweep is
justified by this evidence.

The next learning stage should condition posterior credit on the full action
`(starter context, direction source, proposal backend)` while retaining a
nonzero optimizer-neutral exploration probability. UCB-like and Thompson
sampling policies should be compared on the same logged actions; neither is
promoted from mathematical form alone.

## ThreadPool adapter result

The first multi-seed execution exposed one MACE/e3nn lazy-initialization race:
the second thread failed on its first charged starter evaluation with
`NameError: module is not installed as a submodule`.

The worker now serializes only the first real evaluation of each independent
calculator under a shared lock. It adds no evaluation and leaves subsequent
force calls parallel. The exact failing arm then completed 10/10 attempts with
zero worker errors and closed accounting.

## Evidence boundary

This stage proves execution and accounting behavior for one RTX 3060,
MACE-OMAT-0-small float32, C60, and one fixed PdO slab. Three end-to-end seeds
are a survivor gate, not a significance result. It does not prove canonical
sampling, optimizer-independent endpoint equivalence, transfer to other PES
models, or a generally superior optimizer.

## Immutable summary hashes

- production-cap one-bias replay:
  `62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04`
- shared-cap-400 one-bias replay:
  `a11cd8ee1a1dae9cc71cac0038008149b1c0f0cb67ae72ceba8afc821cbdf370`
- FIRE-conditioned multi-bias stress:
  `ba51e6fe22ccb2fa963ff0ce7217841fb9b990f068a53e00c07609e68fe934bf`
- ThreadPool failing-arm verification:
  `bdbd8fcfaa092e5b7279f5f01d6de4ec9ac80bfdafa449d7edebf5729dd79f0a`
- final thread-safe multi-seed matrix:
  `4ebfca48adba404c20fde1728186d4e3d3f9a0aae0c5e8bdedfd745fb54ebd93`
