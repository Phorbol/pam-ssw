# Branch preservation and cleanup manifest

Remote heads were fetched and compared with GitHub on 2026-09-08. No remote
branches were deleted or pushed by this implementation. Retain the production
reference experiment/c60-k4-profile-200 (fb02469), research history
feature/direction-continuation-ablation (2aef3f8), and current main.

The following refs are fully contained in the retained production or research
reference. Archive their exact SHA with a tag before removing a remote ref;
re-check ancestry and remote SHA immediately before any deletion.

| Redundant ref | Full SHA | Retained descendant |
|---|---|---|
| `autoresearch/20260427-lj-ssw` | `860af125d6b27708c8227b70116f400cb71fc9e1` | `experiment/c60-k4-profile-200` |
| `experiment/pdo-k4-k8-fixed-budget` | `91217d3d2a3d4628a6ae742c3f04225ea6894a3e` | `feature/direction-continuation-ablation` |
| `experiment/proposal-energy-traces` | `a26fd506921f9c3d307456a326d12e9e46a51bd3` | `experiment/c60-k4-profile-200` |
| `experiment/safe-history-capacity-ablation` | `dba2b363071e4c86810bc5233704b81ba14e615c` | `experiment/c60-k4-profile-200` |
| `experiment/safe-lbfgs-history-depth-ablation` | `b3332487f674987406d95b095d2914af6ca5d76d` | `experiment/c60-k4-profile-200` |
| `experiment/safe-lbfgs-scale-decomposition` | `31fd25720f05ae854a38d6842da1dcc00be4d531` | `experiment/c60-k4-profile-200` |
| `feature/bias-separated-relaxation` | `9c40809d34b3da46439d18bf6268fe92ce400de5` | `experiment/c60-k4-profile-200` |
| `feature/fixed-proposal-replay` | `afb3d04dd17a7a73275318575f5f4c55ed3f4885` | `experiment/c60-k4-profile-200` |
| `feature/posterior-parallel-pes` | `fbfe674aaf6380effee1738ca0dff7e02f518bff` | `experiment/c60-k4-profile-200` |
| `feature/posterior-terminal-outcome-validation` | `04c7bd011fd3b9beaa7dff775ffa5e203ef00a92` | `experiment/c60-k4-profile-200` |
| `feature/posterior-terminal-outcome-validation-base` | `4057d166f39b9cb5611b131ee085046b2c8df281` | `experiment/c60-k4-profile-200` |
| `feature/recoverable-budgeted-posterior-runner` | `2d922c1a1da6973f67fcf1def5c227aacc44e314` | `experiment/c60-k4-profile-200` |
| `feature/ssw-attempt-adapter` | `6b2a28958debfd43fc4352d6b3e344caaf0ca871` | `experiment/c60-k4-profile-200` |
| `feature/uphill-propagation-ablation` | `8c5dd1ca5f27d130fafb5af4447b4d58dcd56e90` | `feature/direction-continuation-ablation` |
| `feature/user-facing-validated-production` | `3f4e0d26d3fe9a94dc7d33f7b7323d74a49714f3` | `experiment/c60-k4-profile-200` |

Keep the following independent remote tips until their code/evidence has been archived or integrated:

- `experiment/fixed-direction-family-terminal-labels`: `1fa723020be344ff66f74df5c0086a3f236e2a56`
- `feature/direct-qp-ssw`: `4f385264f4e998705fe96c0bf520e1c36bb97f34`
- `feature/feature-atlas-pr14`: `6a63a020a22ced346d5e1b6d57e2820d4cbe3aba`
- `feature/vc-ssw-general-coordinate`: `a8fa259cef7cd855c434e94d68c20e689d709c6e`

Keep the local `experiment/productive-escape-benchmark-v1` worktree: it contains 51 additional commits and untracked complete LJ results. Deleting a redundant Git ref does not delete large tracked artifacts from surviving history.
