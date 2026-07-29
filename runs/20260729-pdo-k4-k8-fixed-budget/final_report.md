# PdO K4/K8 fixed-budget result

## Outcome

**Do not promote K4 as the PdO default.** K4 bought more completed macro steps by cutting direction-oracle cost, but the paired best-energy result split 1–1 and therefore failed the pre-registered gate.

| Seed | Arm | ΔE best (eV) | Trials | Minima | Duplicate | Direction FE | Proposal FE | Landing FE | Wall (s) | Quench cert. |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | K8 | 6.455627 | 82 | 66 | 0.205 | 2352 | 12277 | 5055 | 450.1 | 83/83 |
| 42 | K4 | 8.381104 | 108 | 72 | 0.339 | 1248 | 13429 | 4957 | 450.1 | 109/109 |
| 43 | K8 | 5.355347 | 73 | 64 | 0.135 | 2288 | 12761 | 4649 | 453.4 | 73/74 |
| 43 | K4 | 4.835571 | 105 | 79 | 0.255 | 1256 | 12739 | 5644 | 448.7 | 105/106 |

Every arm used exactly 20,000 force evaluations; bootstrap cost was 37 FE, starter-quench cost was 0, and unattributed cost was 0.

## Paired interpretation

- Seed 42: K4−K8 = +1.925476 eV energy drop, +26 trials, +6 minima, -1104 direction FE.
- Seed 43: K4−K8 = -0.519775 eV energy drop, +32 trials, +15 minima, -1032 direction FE.

The clean mechanism is therefore supported: reducing K from 8 to 4 approximately halves candidate/HVP work and converts that budget into more macro attempts. It does not yet preserve final energy reliably; seed 43 lost 0.519775 eV despite 32 extra trials.

## Certificate boundary

Seed 42 had complete true-quench force certificates in both arms. Seed 43 had one non-certified true quench in each arm (K8 73/74; K4 105/106). The current walker records such a returned state in the archive, so archive counts are search outputs, not a guarantee that every stored state satisfies `fmax=0.03 eV/Å`.

The best seed-43 energies were reached before the final reported states (K8 trial 18; K4 trial 102), so this qualification does not change the paired best-energy decision. Per-quench termination identity was not persisted, so no stronger claim about which archive entry lacked the certificate is made.

## Decision

- Retain K8 in the current PdO production profile; do not promote K4.
- Keep K4 as a throughput-oriented experimental arm.
- Next direction experiment should improve candidate quality at K4-scale cost, rather than restoring candidate count blindly.
- Add an explicit true-quench certificate gate/log before using archive-minima counts as scientifically certified minima.

This is a two-seed mechanism ablation, not a statistical performance proof.
