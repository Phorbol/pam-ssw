# 200-step direction-transport seed42 screen

- Decision: `seed42_screen_rejects`
- Conclusion: Pure transport is not promoted: its direction-oracle savings do not preserve search quality and reliability across both systems.
- Execution commit: `4ce3288cbfb9ca68ce74bb95137ee3a3b280d241`
- Model SHA-256: `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`

## Endpoint metrics

| system | arm | best eV | drop eV | mean best gain eV | minima | duplicate | FE | direction FE | proposal FE | quench FE | wall s | quench failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c60 | fixed_intent_ritz | -506.839081 | 32.169312 | 23.998228 | 123 | 0.3881 | 90315 | 29040 | 50470 | 9233 | 1713.132 | 0 |
| c60 | transported_direction | -505.636719 | 30.967102 | n/a | 185 | 0.0750 | 86722 | 7226 | 55644 | 22105 | 1687.249 | 5 |
| pdo | fixed_intent_ritz | -579.855957 | 11.458130 | 9.692992 | 195 | 0.0299 | 91368 | 21456 | 42694 | 25960 | 2211.330 | 0 |
| pdo | transported_direction | -574.980530 | 6.582703 | 5.659343 | 195 | 0.0299 | 77470 | 5584 | 40227 | 30726 | 1984.375 | 0 |

## Vector decision

| system | decision | FE saved | direction FE saved | wall s saved | final not worse | trajectory not worse | reliability not worse |
|---|---|---:|---:|---:|---|---|---|
| c60 | no_long_campaign_support | 3593 | 21814 | 25.883 | False | False | False |
| pdo | cost_search_tradeoff | 13898 | 15872 | 226.954 | False | False | True |

The comparison is at equal 200-step endpoints. Per-trial cumulative force counts were not recorded, so this artifact does not claim an equal-force-budget trajectory comparison.

Claim ceiling: one 200-step campaign seed per system and arm; a survivor gate for multi-seed production, not a significance claim.
