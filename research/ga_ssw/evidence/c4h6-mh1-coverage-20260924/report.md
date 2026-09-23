# C4H6 MH-1 fixed-protocol coverage audit

No method ranking is computed. The common prefix is fixed at 200000 charged E/F requests; arms below it are censored. One element-labeled graph mapping is shared across all six arms. Class counts include the initial structure; repeated observations do not add classes. Connected and fragmented structures are summarized separately; all per-minimum geometry and per-attempt history is in analysis.json.

| Seed | Arm | Prefix reached | Minima in prefix | Connected classes | Fragmented classes | Fragmented frames | Accepted prefix landings | Outer records | Returned minima | Requests | Calculator calls | Search wall (s) | Protocol complete | Budget censored |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 61 | ssw | True | 305 | 5 | 2 | 6 | 105 | 400 | 401 | 263213 | 233237 | 4839.105394165032 | True (True) | False (True) |
| 61 | paper_ls | True | 316 | 6 | 7 | 21 | 121 | 400 | 401 | 253732 | 222381 | 4621.135831703432 | True (True) | False (True) |
| 61 | native_ls | True | 307 | 10 | 3 | 7 | 97 | 400 | 401 | 260522 | 229190 | 4761.957396512851 | True (True) | False (True) |
| 67 | ssw | True | 309 | 5 | 4 | 5 | 93 | 400 | 400 | 258156 | 228260 | 4908.029665577691 | True (True) | False (True) |
| 67 | paper_ls | True | 314 | 8 | 7 | 17 | 119 | 400 | 401 | 254484 | 223132 | 4747.56138569396 | True (True) | False (True) |
| 67 | native_ls | True | 309 | 8 | 6 | 12 | 92 | 400 | 401 | 259049 | 227726 | 4804.113238970749 | True (True) | False (True) |

## ssw-seed61

Accounting reconciles=True; fresh coverage=401/401; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=401, converged landings=400, accepted converged landings=140, charged requests=263213, calculator calls=233237, wall=4839.105394165032 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=305 (noninitial=304), fresh-qualified minima=305, fresh-unqualified/missing=0, connected classes=5, fragmented classes=2, fragmented frames=6, torsion cos-sign regions +/−/0=154/116/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 16, 'first_request': 3965, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 6}, {'class_id': 12, 'first_request': 5877, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 9}]; first fragmented class costs (up to 3): [{'class_id': 24, 'first_request': 33797, 'connected': False, 'component_formulas': ['C4H5', 'H'], 'minimum_index': 53}, {'class_id': 7, 'first_request': 42101, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 65}].
No-converged-landing records: 0; first records (status and charged cost): []; full list remains in analysis.json.
LS response observations=0; native update records=0; core last response=None eV/atom; native last observed response=None meV/atom; native action counts={}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.

## paper_ls-seed61

Accounting reconciles=True; fresh coverage=401/401; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=401, converged landings=400, accepted converged landings=150, charged requests=253732, calculator calls=222381, wall=4621.135831703432 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=316 (noninitial=315), fresh-qualified minima=316, fresh-unqualified/missing=0, connected classes=6, fragmented classes=7, fragmented frames=21, torsion cos-sign regions +/−/0=137/128/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 16, 'first_request': 5211, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 8}, {'class_id': 21, 'first_request': 10380, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 16}]; first fragmented class costs (up to 3): [{'class_id': 7, 'first_request': 34673, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 55}, {'class_id': 4, 'first_request': 48214, 'connected': False, 'component_formulas': ['C4H4', 'H2'], 'minimum_index': 76}, {'class_id': 2, 'first_request': 101153, 'connected': False, 'component_formulas': ['C4H4', 'H2'], 'minimum_index': 159}].
No-converged-landing records: 0; first records (status and charged cost): []; full list remains in analysis.json.
LS response observations=400; native update records=0; core last response=0.700021911085696 eV/atom; native last observed response=None meV/atom; native action counts={}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.

## native_ls-seed61

Accounting reconciles=True; fresh coverage=401/401; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=401, converged landings=400, accepted converged landings=131, charged requests=260522, calculator calls=229190, wall=4761.957396512851 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=307 (noninitial=306), fresh-qualified minima=307, fresh-unqualified/missing=0, connected classes=10, fragmented classes=3, fragmented frames=7, torsion cos-sign regions +/−/0=147/117/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 11, 'first_request': 9427, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 14}, {'class_id': 21, 'first_request': 10871, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 16}]; first fragmented class costs (up to 3): [{'class_id': 7, 'first_request': 20231, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 30}, {'class_id': 0, 'first_request': 37613, 'connected': False, 'component_formulas': ['C4H4', 'H2'], 'minimum_index': 55}, {'class_id': 1, 'first_request': 129910, 'connected': False, 'component_formulas': ['C4H4', 'H2'], 'minimum_index': 198}].
No-converged-landing records: 0; first records (status and charged cost): []; full list remains in analysis.json.
LS response observations=400; native update records=400; core last response=0.4703941536489765 eV/atom; native last observed response=470.3941536489765 meV/atom; native action counts={'normal_update': 130}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.

## ssw-seed67

Accounting reconciles=True; fresh coverage=400/400; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=400, converged landings=399, accepted converged landings=120, charged requests=258156, calculator calls=228260, wall=4908.029665577691 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=309 (noninitial=308), fresh-qualified minima=309, fresh-unqualified/missing=0, connected classes=5, fragmented classes=4, fragmented frames=5, torsion cos-sign regions +/−/0=164/105/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 16, 'first_request': 10508, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 17}, {'class_id': 21, 'first_request': 63810, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 98}]; first fragmented class costs (up to 3): [{'class_id': 8, 'first_request': 20830, 'connected': False, 'component_formulas': ['C2H2', 'C2H4'], 'minimum_index': 33}, {'class_id': 7, 'first_request': 73367, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 112}, {'class_id': 24, 'first_request': 144960, 'connected': False, 'component_formulas': ['C4H5', 'H'], 'minimum_index': 223}].
No-converged-landing records: 1; first records (status and charged cost): [{'index': 369, 'status': 'nonpositive_height', 'evaluation_requests': 204, 'error': None}]; full list remains in analysis.json.
LS response observations=0; native update records=0; core last response=None eV/atom; native last observed response=None meV/atom; native action counts={}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.

## paper_ls-seed67

Accounting reconciles=True; fresh coverage=401/401; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=401, converged landings=400, accepted converged landings=151, charged requests=254484, calculator calls=223132, wall=4747.56138569396 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=314 (noninitial=313), fresh-qualified minima=314, fresh-unqualified/missing=0, connected classes=8, fragmented classes=7, fragmented frames=17, torsion cos-sign regions +/−/0=138/128/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 16, 'first_request': 10660, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 17}, {'class_id': 12, 'first_request': 16923, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 27}]; first fragmented class costs (up to 3): [{'class_id': 23, 'first_request': 40440, 'connected': False, 'component_formulas': ['C4H5', 'H'], 'minimum_index': 64}, {'class_id': 7, 'first_request': 42988, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 68}, {'class_id': 2, 'first_request': 77198, 'connected': False, 'component_formulas': ['C4H4', 'H2'], 'minimum_index': 121}].
No-converged-landing records: 0; first records (status and charged cost): []; full list remains in analysis.json.
LS response observations=400; native update records=0; core last response=0.6998576730850801 eV/atom; native last observed response=None meV/atom; native action counts={}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.

## native_ls-seed67

Accounting reconciles=True; fresh coverage=401/401; result source=full; issues=[].
Endpoint: status=completed, records=400, returned minima=401, converged landings=400, accepted converged landings=122, charged requests=259049, calculator calls=227726, wall=4804.113238970749 s, protocol_complete=True, budget_censored=False.
Common prefix: reached=True, minima=309 (noninitial=308), fresh-qualified minima=309, fresh-unqualified/missing=0, connected classes=8, fragmented classes=6, fragmented frames=12, torsion cos-sign regions +/−/0=156/96/0; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.
First connected class costs (up to 3): [{'class_id': 10, 'first_request': 9, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 0}, {'class_id': 16, 'first_request': 2541, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 4}, {'class_id': 12, 'first_request': 7728, 'connected': True, 'component_formulas': ['C4H6'], 'minimum_index': 12}]; first fragmented class costs (up to 3): [{'class_id': 7, 'first_request': 24841, 'connected': False, 'component_formulas': ['C2H3', 'C2H3'], 'minimum_index': 38}, {'class_id': 24, 'first_request': 28595, 'connected': False, 'component_formulas': ['C4H5', 'H'], 'minimum_index': 44}, {'class_id': 8, 'first_request': 63567, 'connected': False, 'component_formulas': ['C2H2', 'C2H4'], 'minimum_index': 98}].
No-converged-landing records: 0; first records (status and charged cost): []; full list remains in analysis.json.
LS response observations=400; native update records=400; core last response=0.4697722399665508 eV/atom; native last observed response=469.77223996655084 meV/atom; native action counts={'normal_update': 130}.
Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate.
