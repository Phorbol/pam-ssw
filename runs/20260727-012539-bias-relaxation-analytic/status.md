# Status

Execution completed once with exit code 0. Output hash:
`24b4db7b9170d006155ea0d29509d4d8391491e8896f9bef47cf2cc605d677bc`.

All four backends converged on both cases. Both custom objective-call ledgers
closed exactly and neither custom mode had a rejected secant or MIC reset.

The analytic cases do not show a call-count benefit from bias separation:

- one hill: total-secant 14 calls; bias-separated 18 calls;
- two hills: total-secant 16 calls; bias-separated 18 calls.

This is a correctness gate, not evidence that bias separation improves SSW.
The complete custom pair may advance to G1 because both are correct; neither
arm may be advanced or tuned alone.
