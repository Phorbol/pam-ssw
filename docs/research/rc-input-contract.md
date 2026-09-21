# LASP rigidbody/blist input bridge

`pamssw.standalone.rc_topology.read_rigid_topology` implements RC-SSW 2025 SI
section7 text format: body count, per-body membership count/list, and explicit
one-based bond endpoints in blist. Comments after # are accepted. No distance
cutoff, automatic chemical perception or invented rotatable-bond criterion is
introduced. Indices become zero-based. Two-atom body intersections define shared
joint axes and must be present in blist. Single-atom intersections between
nonadjacent bodies are retained through connected-membership validation, not
mistaken for extra rotatable joints. Loops/inconsistent overlaps fail explicitly.

Each graph component uses the lowest source body index as its root, BFS order,
and increasing atom order for the joint sign. This is a reproducible independent
coordinate gauge, not recovery of LASP's central-body choice. The returned
components dictionaries feed RigidForestChart/run_rc_forest_ssw. Input coordinates
must still be supplied and appropriately unwrapped for a molecule crossing PBC;
parsing topology does not imply isolated/periodic geometry support interchangeably.

The actual uploaded TYPE2-XXXII example parses into four 43-atom molecules,
32 overlapping rigid bodies and 28 joints, with 180 explicit chemical bonds.
The three-body SI phenylacetaldehyde example also passes. These are topology
checks and make no PES/efficiency claim. Tests: test_rc_topology.py (2 passed).
