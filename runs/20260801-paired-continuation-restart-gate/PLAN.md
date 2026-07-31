# S-CR1 paired continuation/restart gate

For seed 45, run C60, fixed-bottom PdO and packaged CuO with one exact shared
bootstrap minimum per system.  Compare four 20,000-FE starter policies while
freezing every inner SSW mechanism:

- uniform archive;
- current node-level UCB-like selector;
- classic Metropolis chain;
- snapshot-paired lowest-energy continuation and uniform restart.

The new arm is admitted to seeds 46--47 only if its gain AUC is strictly above
both UCB-like and Metropolis in at least two of the three systems.  No selector
weight, tie tolerance or posterior is fitted after observing the result.
