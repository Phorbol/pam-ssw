# U-T1 fixed-reference macro uphill-target gate

Compare the current archive-scaled macro target with the configured fixed
0.8 eV reference while retaining every within-walk SSW mechanism. C60, PdO and
CuO use a shared bootstrap, Metropolis starter chain and exactly 20,000 FE per
arm at fresh seed 46.

Advance to seeds 47--48 only if fixed-reference gain AUC is strictly larger in
at least two systems. Otherwise close the target branch without tuning the
reference, archive statistic, scale factor or bounds.
