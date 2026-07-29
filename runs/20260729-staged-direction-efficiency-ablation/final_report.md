# Staged C60 direction-efficiency campaign

## 1. Verified execution and accounting

- Executed proposals: 84.
- Total force evaluations: 38278.
- Purpose ledger: `{"biased_proposal_relax": 21497, "bootstrap_true_quench": 0, "direction_oracle": 8832, "escape_true_pes_check": 785, "landing_true_quench": 7080, "post_relax_validation": 84, "starter_true_quench": 0, "unattributed": 0}`.
- Generation wall time: 544.712005 s.
- Strict-quench wall time: 122.291778 s.
- Total measured wall time: 667.003783 s.
- Every executed case has a strict terminal certificate and zero unattributed force evaluations.

## 2. Momentum causal result

- Decision: `unproven_retained`.
- Stable sets: `{"momentum_off": [["plateau_accepted", 42], ["plateau_accepted", 43]], "momentum_on": [["plateau_accepted", 44]]}`.

## 3. Candidate-count result and FE savings

- Decision: `reduced`.
- Retained settings: `{"enable_momentum_candidate": true, "max_steps_per_walk": 8, "oracle_candidates": 4, "proposal_relax_steps": 80}`.
- Candidate-count arm total FE: `{"k12": 6073, "k4": 4727, "k8": 4982}`.

## 4. Bias-step result and FE savings

- Decision: `baseline_retained`.
- Retained settings: `{"enable_momentum_candidate": true, "max_steps_per_walk": 8, "oracle_candidates": 4, "proposal_relax_steps": 80}`.
- Bias-step arm total FE: `{"b5": 4890, "b8": 5263}`.

## 5. Stage-L gate

- Decision: `not_entered`.
- Measured gate: `{"entered": false, "proposal_relax_calls": 87, "proposal_relax_cap_hit_fraction": 0.12643678160919541, "proposal_relax_cap_hits": 11, "proposal_relax_force_fraction": 0.631768953068592, "retained_arm": "b8"}`.

## 6. Continuous landing-energy table

| stage | arm | starter | seed | repeat | delta_eV | meaningful |
|---|---|---|---:|---:|---:|---|
| momentum | momentum_on | intermediate_accepted | 42 | 0 | 0.000061035 | false |
| momentum | momentum_off | intermediate_accepted | 42 | 0 | 0.000122070 | false |
| momentum | momentum_off | intermediate_accepted | 42 | 1 | 0.000122070 | false |
| momentum | momentum_on | intermediate_accepted | 42 | 1 | 0.000030518 | false |
| momentum | momentum_on | intermediate_accepted | 43 | 0 | 5.415649414 | false |
| momentum | momentum_off | intermediate_accepted | 43 | 0 | 0.707916260 | false |
| momentum | momentum_off | intermediate_accepted | 43 | 1 | 0.708007812 | false |
| momentum | momentum_on | intermediate_accepted | 43 | 1 | 4.599090576 | false |
| momentum | momentum_on | intermediate_accepted | 44 | 0 | 5.846069336 | false |
| momentum | momentum_off | intermediate_accepted | 44 | 0 | 6.586395264 | false |
| momentum | momentum_off | intermediate_accepted | 44 | 1 | 6.586364746 | false |
| momentum | momentum_on | intermediate_accepted | 44 | 1 | 5.320526123 | false |
| momentum | momentum_on | plateau_accepted | 42 | 0 | 0.761962891 | false |
| momentum | momentum_off | plateau_accepted | 42 | 0 | -3.351837158 | true |
| momentum | momentum_off | plateau_accepted | 42 | 1 | -3.351684570 | true |
| momentum | momentum_on | plateau_accepted | 42 | 1 | 0.761993408 | false |
| momentum | momentum_on | plateau_accepted | 43 | 0 | -2.626800537 | true |
| momentum | momentum_off | plateau_accepted | 43 | 0 | -3.752532959 | true |
| momentum | momentum_off | plateau_accepted | 43 | 1 | -3.752441406 | true |
| momentum | momentum_on | plateau_accepted | 43 | 1 | 2.096984863 | false |
| momentum | momentum_on | plateau_accepted | 44 | 0 | -9.030456543 | true |
| momentum | momentum_off | plateau_accepted | 44 | 0 | 5.136077881 | false |
| momentum | momentum_off | plateau_accepted | 44 | 1 | 6.037292480 | false |
| momentum | momentum_on | plateau_accepted | 44 | 1 | -9.030456543 | true |
| candidate_count | k4 | intermediate_accepted | 42 | 0 | 4.599151611 | false |
| candidate_count | k8 | intermediate_accepted | 42 | 0 | 0.000000000 | false |
| candidate_count | k12 | intermediate_accepted | 42 | 0 | 0.000091553 | false |
| candidate_count | k12 | intermediate_accepted | 42 | 1 | 0.000091553 | false |
| candidate_count | k8 | intermediate_accepted | 42 | 1 | 0.000122070 | false |
| candidate_count | k4 | intermediate_accepted | 42 | 1 | 4.599029541 | false |
| candidate_count | k4 | intermediate_accepted | 43 | 0 | 3.237609863 | false |
| candidate_count | k8 | intermediate_accepted | 43 | 0 | 0.000000000 | false |
| candidate_count | k12 | intermediate_accepted | 43 | 0 | 5.415588379 | false |
| candidate_count | k12 | intermediate_accepted | 43 | 1 | 5.415618896 | false |
| candidate_count | k8 | intermediate_accepted | 43 | 1 | 0.000091553 | false |
| candidate_count | k4 | intermediate_accepted | 43 | 1 | 4.613189697 | false |
| candidate_count | k4 | intermediate_accepted | 44 | 0 | 3.521270752 | false |
| candidate_count | k8 | intermediate_accepted | 44 | 0 | 5.320526123 | false |
| candidate_count | k12 | intermediate_accepted | 44 | 0 | 4.740814209 | false |
| candidate_count | k12 | intermediate_accepted | 44 | 1 | 4.740814209 | false |
| candidate_count | k8 | intermediate_accepted | 44 | 1 | 4.740753174 | false |
| candidate_count | k4 | intermediate_accepted | 44 | 1 | 4.740753174 | false |
| candidate_count | k4 | plateau_accepted | 42 | 0 | -2.404693604 | true |
| candidate_count | k8 | plateau_accepted | 42 | 0 | 2.383850098 | false |
| candidate_count | k12 | plateau_accepted | 42 | 0 | 0.761993408 | false |
| candidate_count | k12 | plateau_accepted | 42 | 1 | 0.792724609 | false |
| candidate_count | k8 | plateau_accepted | 42 | 1 | 2.383972168 | false |
| candidate_count | k4 | plateau_accepted | 42 | 1 | -2.404785156 | true |
| candidate_count | k4 | plateau_accepted | 43 | 0 | 3.084259033 | false |
| candidate_count | k8 | plateau_accepted | 43 | 0 | 3.084228516 | false |
| candidate_count | k12 | plateau_accepted | 43 | 0 | 4.399932861 | false |
| candidate_count | k12 | plateau_accepted | 43 | 1 | 4.399810791 | false |
| candidate_count | k8 | plateau_accepted | 43 | 1 | 3.083984375 | false |
| candidate_count | k4 | plateau_accepted | 43 | 1 | 3.084075928 | false |
| candidate_count | k4 | plateau_accepted | 44 | 0 | -9.030303955 | true |
| candidate_count | k8 | plateau_accepted | 44 | 0 | -9.030426025 | true |
| candidate_count | k12 | plateau_accepted | 44 | 0 | -9.030334473 | true |
| candidate_count | k12 | plateau_accepted | 44 | 1 | -9.030456543 | true |
| candidate_count | k8 | plateau_accepted | 44 | 1 | -9.030487061 | true |
| candidate_count | k4 | plateau_accepted | 44 | 1 | -9.030303955 | true |
| bias_steps | b5 | intermediate_accepted | 42 | 0 | 0.000061035 | false |
| bias_steps | b8 | intermediate_accepted | 42 | 0 | 4.598968506 | false |
| bias_steps | b8 | intermediate_accepted | 42 | 1 | 4.598999023 | false |
| bias_steps | b5 | intermediate_accepted | 42 | 1 | -0.000183105 | false |
| bias_steps | b5 | intermediate_accepted | 43 | 0 | 0.000000000 | false |
| bias_steps | b8 | intermediate_accepted | 43 | 0 | 7.137908936 | false |
| bias_steps | b8 | intermediate_accepted | 43 | 1 | -1.195007324 | true |
| bias_steps | b5 | intermediate_accepted | 43 | 1 | -0.000122070 | false |
| bias_steps | b5 | intermediate_accepted | 44 | 0 | 3.521301270 | false |
| bias_steps | b8 | intermediate_accepted | 44 | 0 | 3.521087646 | false |
| bias_steps | b8 | intermediate_accepted | 44 | 1 | 4.740631104 | false |
| bias_steps | b5 | intermediate_accepted | 44 | 1 | 3.521209717 | false |
| bias_steps | b5 | plateau_accepted | 42 | 0 | -2.464538574 | true |
| bias_steps | b8 | plateau_accepted | 42 | 0 | -0.747253418 | true |
| bias_steps | b8 | plateau_accepted | 42 | 1 | -2.464477539 | true |
| bias_steps | b5 | plateau_accepted | 42 | 1 | -2.464538574 | true |
| bias_steps | b5 | plateau_accepted | 43 | 0 | 3.084075928 | false |
| bias_steps | b8 | plateau_accepted | 43 | 0 | 3.084014893 | false |
| bias_steps | b8 | plateau_accepted | 43 | 1 | 3.084136963 | false |
| bias_steps | b5 | plateau_accepted | 43 | 1 | 3.084075928 | false |
| bias_steps | b5 | plateau_accepted | 44 | 0 | -9.030456543 | true |
| bias_steps | b8 | plateau_accepted | 44 | 0 | -9.030456543 | true |
| bias_steps | b8 | plateau_accepted | 44 | 1 | -9.030303955 | true |
| bias_steps | b5 | plateau_accepted | 44 | 1 | -9.030334473 | true |

## 7. Source composition and selected-source table

| stage | arm | evaluated sources | selected sources |
|---|---|---|---|
| momentum | momentum_off | `{"bond": 108, "random": 540}` | `{"bond": 27, "random": 27}` |
| momentum | momentum_on | `{"bond": 182, "momentum": 79, "random": 831}` | `{"bond": 7, "momentum": 74, "random": 10}` |
| candidate_count | k12 | `{"bond": 180, "momentum": 78, "random": 822}` | `{"bond": 6, "momentum": 74, "random": 10}` |
| candidate_count | k4 | `{"bond": 182, "momentum": 77, "random": 97}` | `{"bond": 12, "momentum": 71, "random": 6}` |
| candidate_count | k8 | `{"bond": 164, "momentum": 68, "random": 408}` | `{"bond": 8, "momentum": 64, "random": 8}` |
| bias_steps | b5 | `{"bond": 122, "momentum": 48, "random": 70}` | `{"bond": 12, "momentum": 42, "random": 6}` |
| bias_steps | b8 | `{"bond": 183, "momentum": 78, "random": 99}` | `{"bond": 10, "momentum": 73, "random": 7}` |

## 8. Invalid, damage, and non-convergence taxonomy

- Fragmented outcomes: 0.
- True-quench fallback outcomes: 6.
- Non-certified terminal outcomes: 0.

## 9. Posterior-feasibility checklist

- `two_fixed_direction_families_with_five_meaningful_each`: false.
- `two_starter_classes_with_positive_outcomes`: true.
- `complete_action_context_cost_certificate_records`: true.
- `held_out_residual_signal_demonstrated`: false.
- `posterior_ready`: false.

## 10. Proven, unproven, and next experiment

- Proven here: the closed cost accounting, the Stage-M null/mixed result, the K reduction gate, the B retention gate, and the measured Stage-L skip.
- Unproven: causal credit for a fixed direction family and held-out value beyond the non-learning controls.
- Next justified experiment: collect terminal labels for fixed direction families under identical starters and costs, then test held-out residual prediction before introducing any posterior selector.
