# C4H6 MH-1 lifecycle pilot analysis

This audits costs, completion, minima geometry, and fresh checks. It does not rank methods or establish reaction accuracy.

Graph class IDs are assigned within each arm and are not comparable across arms; compare each arm's class count only.

| Arm | Result | Attempts | No converged landing | E/F requests | Denials | Search wall (s) | Mean all-attempt requests | Graph classes | Fresh complete | Accounting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ssw | completed | 12 | 0 | 7651 | 0 | 111.30617321841419 | 636.8333333333334 | 3 | True | True |

## ssw

Fresh checks: 13/13; missing=[], duplicate=[], out-of-range=[].
E/F ledger: search=7651, failed calls=0; denials=0 are retained but not charged as calls.
No-converged-landing costs (numerical landing outcome, separate from climb status and MC acceptance): []. Censored attempt: {'present': False, 'indices': [], 'boundary': None, 'records': []}.
400 outer-attempt request scenario: {'outer_attempts': 400, 'search_requests_estimate': 254742.33333333334, 'mean_basis': 'all non-censored attempts, including attempts without a converged landing', 'censored_attempt_excluded': True, 'not_a_request_cap_or_authorization': True}. It includes all ordinary attempts in the mean, excludes the final budget-censored attempt, and is not 400 minima or paper reproduction.
Core `energy_response` is reported as eV/atom; native `observed_response_mev_per_atom` is reported as meV/atom. analysis.json retains outer status, climb stop reason, numerical landing success, MC acceptance, and raw LS updates as separate fields.

| Minimum | Energy eV | Fmax eV/Å | Graph class | Components | Formulas | Butadiene raw CCCC torsion deg | Fresh qualified |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | -4243.921327009705 | 0.02528770158278401 | 0 | 1 | ['C4H6'] | 180.0 | True |
| 1 | -4243.921026925712 | 0.021222302809936065 | 0 | 1 | ['C4H6'] | 182.63116567175902 | True |
| 2 | -4243.837585693271 | 0.014699105027671413 | 0 | 1 | ['C4H6'] | 332.6023969748301 | True |
| 3 | -4243.836590346465 | 0.026511148942411272 | 0 | 1 | ['C4H6'] | 328.69592587015495 | True |
| 4 | -4237.247913731882 | 0.023859648917099147 | 2 | 2 | ['C4H5', 'H'] | None | True |
| 5 | -4243.517878046277 | 0.020405250828279 | 1 | 1 | ['C4H6'] | None | True |
| 6 | -4243.836784322543 | 0.025192348148069303 | 0 | 1 | ['C4H6'] | 30.45827385449521 | True |
| 7 | -4243.921583584442 | 0.022800525909040775 | 0 | 1 | ['C4H6'] | 180.19981624380335 | True |
| 8 | -4243.921527662881 | 0.02521452401196937 | 0 | 1 | ['C4H6'] | 180.5281991045307 | True |
| 9 | -4243.834841600169 | 0.019355121216148775 | 0 | 1 | ['C4H6'] | 34.26224597274964 | True |
| 10 | -4243.921390641957 | 0.02684354261159804 | 0 | 1 | ['C4H6'] | 181.37703297069922 | True |
| 11 | -4243.837402244858 | 0.020018746533583227 | 0 | 1 | ['C4H6'] | 28.50218746223835 | True |
| 12 | -4243.517743699287 | 0.018194642158398867 | 1 | 1 | ['C4H6'] | None | True |

| paper_ls | completed | 12 | 0 | 7512 | 0 | 107.52979988884181 | 625.25 | 4 | True | True |

## paper_ls

Fresh checks: 13/13; missing=[], duplicate=[], out-of-range=[].
E/F ledger: search=7512, failed calls=0; denials=0 are retained but not charged as calls.
No-converged-landing costs (numerical landing outcome, separate from climb status and MC acceptance): []. Censored attempt: {'present': False, 'indices': [], 'boundary': None, 'records': []}.
400 outer-attempt request scenario: {'outer_attempts': 400, 'search_requests_estimate': 250109.0, 'mean_basis': 'all non-censored attempts, including attempts without a converged landing', 'censored_attempt_excluded': True, 'not_a_request_cap_or_authorization': True}. It includes all ordinary attempts in the mean, excludes the final budget-censored attempt, and is not 400 minima or paper reproduction.
Core `energy_response` is reported as eV/atom; native `observed_response_mev_per_atom` is reported as meV/atom. analysis.json retains outer status, climb stop reason, numerical landing success, MC acceptance, and raw LS updates as separate fields.

| Minimum | Energy eV | Fmax eV/Å | Graph class | Components | Formulas | Butadiene raw CCCC torsion deg | Fresh qualified |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | -4243.921327009705 | 0.025287701582644744 | 2 | 1 | ['C4H6'] | 180.0 | True |
| 1 | -4243.837460970483 | 0.01422859803484499 | 2 | 1 | ['C4H6'] | 25.23052597077166 | True |
| 2 | -4243.837386403296 | 0.02898657060020104 | 2 | 1 | ['C4H6'] | 332.4108174611014 | True |
| 3 | -4243.921295204165 | 0.02694934275256891 | 2 | 1 | ['C4H6'] | 179.6620898421021 | True |
| 4 | -4243.919491991056 | 0.02755024697953766 | 2 | 1 | ['C4H6'] | 184.50722033157618 | True |
| 5 | -4237.904478937565 | 0.012147478536767328 | 0 | 2 | ['C2H3', 'C2H3'] | None | True |
| 6 | -4243.837205569069 | 0.029887592798922804 | 2 | 1 | ['C4H6'] | 28.735188042116782 | True |
| 7 | -4243.837280801516 | 0.01996658017542376 | 2 | 1 | ['C4H6'] | 332.9663956153099 | True |
| 8 | -4243.837142341091 | 0.02024414869038092 | 2 | 1 | ['C4H6'] | 26.465052475261032 | True |
| 9 | -4243.8373934528045 | 0.02620227737024634 | 2 | 1 | ['C4H6'] | 332.10233797322826 | True |
| 10 | -4243.921342128983 | 0.018120776498302075 | 2 | 1 | ['C4H6'] | 181.0812741270124 | True |
| 11 | -4240.214402115817 | 0.023312599716904664 | 1 | 2 | ['C2H4', 'C2H2'] | None | True |
| 12 | -4240.9074000675355 | 0.029695890539053357 | 3 | 1 | ['C4H6'] | None | True |

| native_ls | completed | 12 | 0 | 8108 | 0 | 114.8572949739173 | 674.9166666666666 | 1 | True | True |

## native_ls

Fresh checks: 13/13; missing=[], duplicate=[], out-of-range=[].
E/F ledger: search=8108, failed calls=0; denials=0 are retained but not charged as calls.
No-converged-landing costs (numerical landing outcome, separate from climb status and MC acceptance): []. Censored attempt: {'present': False, 'indices': [], 'boundary': None, 'records': []}.
400 outer-attempt request scenario: {'outer_attempts': 400, 'search_requests_estimate': 269975.6666666666, 'mean_basis': 'all non-censored attempts, including attempts without a converged landing', 'censored_attempt_excluded': True, 'not_a_request_cap_or_authorization': True}. It includes all ordinary attempts in the mean, excludes the final budget-censored attempt, and is not 400 minima or paper reproduction.
Core `energy_response` is reported as eV/atom; native `observed_response_mev_per_atom` is reported as meV/atom. analysis.json retains outer status, climb stop reason, numerical landing success, MC acceptance, and raw LS updates as separate fields.

| Minimum | Energy eV | Fmax eV/Å | Graph class | Components | Formulas | Butadiene raw CCCC torsion deg | Fresh qualified |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | -4243.921327009705 | 0.025287701582607628 | 0 | 1 | ['C4H6'] | 180.0 | True |
| 1 | -4243.836626034933 | 0.015253604901869184 | 0 | 1 | ['C4H6'] | 31.470596281734526 | True |
| 2 | -4243.837474471128 | 0.024422614391183987 | 0 | 1 | ['C4H6'] | 333.70327020077843 | True |
| 3 | -4243.688051509608 | 0.021801889526744028 | 0 | 1 | ['C4H6'] | 103.59284515262699 | True |
| 4 | -4243.921567066227 | 0.029471781497987938 | 0 | 1 | ['C4H6'] | 180.44619625720765 | True |
| 5 | -4243.687625565294 | 0.015429457224962296 | 0 | 1 | ['C4H6'] | 260.2590015035146 | True |
| 6 | -4243.837584941563 | 0.020584903035136756 | 0 | 1 | ['C4H6'] | 27.13247020050159 | True |
| 7 | -4243.9215994313 | 0.02343585341616018 | 0 | 1 | ['C4H6'] | 179.80392426170877 | True |
| 8 | -4243.837424496488 | 0.02541047067123533 | 0 | 1 | ['C4H6'] | 27.984658253467384 | True |
| 9 | -4243.921296436569 | 0.026361308717596368 | 0 | 1 | ['C4H6'] | 181.3497183152577 | True |
| 10 | -4243.921656537218 | 0.021439748281327074 | 0 | 1 | ['C4H6'] | 180.08750283973802 | True |
| 11 | -4243.921038304698 | 0.02567937280128445 | 0 | 1 | ['C4H6'] | 180.9981773559256 | True |
| 12 | -4243.837596249936 | 0.016759119397710916 | 0 | 1 | ['C4H6'] | 27.37255625547243 | True |

Mean request cost uses every non-censored recorded outer attempt, including attempts without a converged landing. Conditional landing-success cost is reported separately in analysis.json. A `gaussian_limit` stop with a converged landing is a numerical force-quench success, not a Hessian-certified minimum or automatic failure. Search wall ends before fresh checks; wall-per-attempt is amortized, not a performance claim. The 400-attempt scenario is not a new cap or authorization.
