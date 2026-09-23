# Rotation-priority follow-up

Best-so-far curves use the initial true minimum and converged true landings; failed records retain their cumulative request cost without adding an energy. Gaussian-center distances are direct Cartesian distances in the saved fixed cell, without minimum-image wrapping. Missing terminal centers are not inferred.

The 1e-8 eV equality is used only to map the final stored best back to its earliest record index. It is not an energy-ranking or faster-discovery criterion; energy differences below MLIP accuracy are not interpreted as meaningful.

| Case | Policy | Records | Final best (eV) | Earliest record index |
|---|---|---:|---:|---:|
| aloh3 | ritz | 14 | -177.56743385376285 | 1 |
| aloh3 | recovered | 23 | -177.56738689710477 | 6 |
| aloh3 | Ritz~recovered best geometry | — | — | {'tight': True, 'broad': True} |

### aloh3 / ritz: cost-energy curve

| Record | Cumulative requests | Best-so-far energy (eV) | Event | Qualified landing |
|---:|---:|---:|---|---|
| None | 14 | -177.3293050620191 | initial | True |
| 0 | 77 | -177.56717351223392 | lower_true_energy | True |
| 1 | 945 | -177.56743385376285 | gaussian_limit | True |
| 2 | 1910 | -177.56743385376285 | gaussian_limit | True |
| 3 | 2979 | -177.56743385376285 | gaussian_limit | True |
| 4 | 4096 | -177.56743385376285 | gaussian_limit | True |
| 5 | 5018 | -177.56743385376285 | gaussian_limit | True |
| 6 | 5967 | -177.56743385376285 | gaussian_limit | True |
| 7 | 6916 | -177.56743385376285 | gaussian_limit | True |
| 8 | 7929 | -177.56743385376285 | gaussian_limit | True |
| 9 | 8829 | -177.56743385376285 | gaussian_limit | True |
| 10 | 9755 | -177.56743385376285 | gaussian_limit | True |
| 11 | 10913 | -177.56743385376285 | gaussian_limit | True |
| 12 | 11830 | -177.56743385376285 | gaussian_limit | True |
| 13 | 12000 | -177.56743385376285 | evaluation_failed | False |

### aloh3 / recovered: cost-energy curve

| Record | Cumulative requests | Best-so-far energy (eV) | Event | Qualified landing |
|---:|---:|---:|---|---|
| None | 14 | -177.32930506201916 | initial | True |
| 0 | 71 | -177.566952554733 | lower_true_energy | True |
| 1 | 583 | -177.56695612909118 | gaussian_limit | True |
| 2 | 1122 | -177.56695612909118 | gaussian_limit | True |
| 3 | 1759 | -177.56695612909118 | gaussian_limit | True |
| 4 | 2397 | -177.56695612909118 | gaussian_limit | True |
| 5 | 2883 | -177.56695612909118 | gaussian_limit | True |
| 6 | 3382 | -177.56738689710477 | gaussian_limit | True |
| 7 | 3922 | -177.56738689710477 | gaussian_limit | True |
| 8 | 4404 | -177.56738689710477 | gaussian_limit | True |
| 9 | 4935 | -177.56738689710477 | gaussian_limit | True |
| 10 | 5430 | -177.56738689710477 | gaussian_limit | True |
| 11 | 6115 | -177.56738689710477 | gaussian_limit | True |
| 12 | 6660 | -177.56738689710477 | gaussian_limit | True |
| 13 | 7211 | -177.56738689710477 | gaussian_limit | True |
| 14 | 7736 | -177.56738689710477 | gaussian_limit | True |
| 15 | 8240 | -177.56738689710477 | gaussian_limit | True |
| 16 | 8801 | -177.56738689710477 | gaussian_limit | True |
| 17 | 9387 | -177.56738689710477 | gaussian_limit | True |
| 18 | 9942 | -177.56738689710477 | gaussian_limit | True |
| 19 | 10478 | -177.56738689710477 | gaussian_limit | True |
| 20 | 11022 | -177.56738689710477 | gaussian_limit | True |
| 21 | 11657 | -177.56738689710477 | gaussian_limit | True |
| 22 | 12000 | -177.56738689710477 | evaluation_failed | False |

| brookite48 | ritz | 17 | -427.59133148575665 | 15 |
| brookite48 | recovered | 30 | -427.5912955267992 | 22 |
| brookite48 | Ritz~recovered best geometry | — | — | {'tight': True, 'broad': True} |

### brookite48 / ritz: cost-energy curve

| Record | Cumulative requests | Best-so-far energy (eV) | Event | Qualified landing |
|---:|---:|---:|---|---|
| None | 4 | -427.59036474751946 | initial | True |
| 0 | 719 | -427.59036474751946 | gaussian_limit | True |
| 1 | 1414 | -427.5912905410551 | gaussian_limit | True |
| 2 | 2141 | -427.5912905410551 | gaussian_limit | True |
| 3 | 2829 | -427.5912905410551 | gaussian_limit | True |
| 4 | 3555 | -427.5912905410551 | gaussian_limit | True |
| 5 | 4314 | -427.5912905410551 | gaussian_limit | True |
| 6 | 5052 | -427.5912905410551 | gaussian_limit | True |
| 7 | 5741 | -427.5912905410551 | gaussian_limit | True |
| 8 | 6456 | -427.5912905410551 | gaussian_limit | True |
| 9 | 7168 | -427.5912905410551 | gaussian_limit | True |
| 10 | 7880 | -427.5912905410551 | gaussian_limit | True |
| 11 | 8592 | -427.5912905410551 | gaussian_limit | True |
| 12 | 9344 | -427.5913107894915 | gaussian_limit | True |
| 13 | 10070 | -427.5913107894915 | gaussian_limit | True |
| 14 | 10771 | -427.5913107894915 | gaussian_limit | True |
| 15 | 11479 | -427.59133148575665 | gaussian_limit | True |
| 16 | 12000 | -427.59133148575665 | evaluation_failed | False |

### brookite48 / recovered: cost-energy curve

| Record | Cumulative requests | Best-so-far energy (eV) | Event | Qualified landing |
|---:|---:|---:|---|---|
| None | 4 | -427.59036474751946 | initial | True |
| 0 | 412 | -427.590820305591 | gaussian_limit | True |
| 1 | 784 | -427.590820305591 | gaussian_limit | True |
| 2 | 1144 | -427.5911092754902 | gaussian_limit | True |
| 3 | 1564 | -427.5911092754902 | gaussian_limit | True |
| 4 | 2008 | -427.5911092754902 | gaussian_limit | True |
| 5 | 2483 | -427.5911092754902 | gaussian_limit | True |
| 6 | 2895 | -427.5911092754902 | gaussian_limit | True |
| 7 | 3359 | -427.5911092754902 | gaussian_limit | True |
| 8 | 3791 | -427.59120794151613 | gaussian_limit | True |
| 9 | 4148 | -427.59120794151613 | gaussian_limit | True |
| 10 | 4546 | -427.59120794151613 | gaussian_limit | True |
| 11 | 4935 | -427.59120794151613 | gaussian_limit | True |
| 12 | 5318 | -427.5912475280529 | gaussian_limit | True |
| 13 | 5804 | -427.5912475280529 | gaussian_limit | True |
| 14 | 6174 | -427.5912475280529 | gaussian_limit | True |
| 15 | 6510 | -427.5912475280529 | gaussian_limit | True |
| 16 | 6876 | -427.5912475280529 | gaussian_limit | True |
| 17 | 7321 | -427.5912475280529 | gaussian_limit | True |
| 18 | 7714 | -427.5912475280529 | gaussian_limit | True |
| 19 | 8129 | -427.5912475280529 | gaussian_limit | True |
| 20 | 8624 | -427.5912475280529 | gaussian_limit | True |
| 21 | 9017 | -427.5912475280529 | gaussian_limit | True |
| 22 | 9396 | -427.5912955267992 | gaussian_limit | True |
| 23 | 9825 | -427.5912955267992 | gaussian_limit | True |
| 24 | 10284 | -427.5912955267992 | gaussian_limit | True |
| 25 | 10720 | -427.5912955267992 | gaussian_limit | True |
| 26 | 11139 | -427.5912955267992 | gaussian_limit | True |
| 27 | 11528 | -427.5912955267992 | gaussian_limit | True |
| 28 | 11983 | -427.5912955267992 | gaussian_limit | True |
| 29 | 12000 | -427.5912955267992 | evaluation_failed | False |

