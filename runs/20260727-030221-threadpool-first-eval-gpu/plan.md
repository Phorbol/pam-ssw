# ThreadPool first-evaluation race verification

Re-run the exact arm that previously produced one first-evaluation
`worker_error`:

- system: PdO
- backend: safe total-gradient L-BFGS
- seed: 44
- policy: uniform
- ThreadPool workers: 2
- action budget: 1000
- total campaign budget: 3000
- common certified bootstrap state

The gate is strict: no `worker_error`, every attempt posterior-observed,
`benchmark_eligible=true`, and purpose-resolved accounting closed.
