import numpy as np

from research.ga_ssw.probe_native_lbfgs_vc import native_workspace_bytes, vc_norm


def test_flat_native_workspace_allocates_full_246_dimension_without_elf_run():
    # The LBFGS workspace is 246*801 doubles, beyond the old 1 MiB DATA map.
    assert native_workspace_bytes(246) == (246 * 801 + 800) * 8
    assert native_workspace_bytes(246) > 1 << 20


def test_vc_norm_keeps_atomic_max_and_six_cell_block_distinct_from_flat_l2():
    gradient = np.zeros(246)
    gradient[0] = .4
    gradient[-6] = .3
    assert vc_norm(gradient, 80) == .4
    assert np.linalg.norm(gradient) == .5


def test_original_instructions_accept_descending_flat_246_quadratic():
    import pytest
    pytest.importorskip('unicorn')
    from research.ga_ssw.probe_native_lbfgs_vc import FlatOracle, load_elf, ELF
    from pathlib import Path
    if not Path(ELF).exists():
        pytest.skip('uploaded ELF required for isolated ABI test')
    _, segments = load_elf(ELF)
    initial = np.linspace(-.1, .1, 246)
    oracle = FlatOracle(segments, initial, accepted_step_cap=3, joint_gtol=1e-10)
    first = .5 * initial @ initial
    for request in range(20):
        q = oracle.vector()
        row = dict(q=q.copy(), objective=.5*q@q, vc_norm=vc_norm(q, 80))
        flag = oracle.advance(row['objective'], q, row)
        if oracle.stop_reason or flag != 1:
            break
    assert oracle.accepted
    assert oracle.accepted[-1]['objective'] < first
    assert all(a['accepted_coordinate_error'] == 0 for a in oracle.accepted)
    assert len(oracle.accepted) <= 3


def test_failed_trial_counts_request_and_returns_initial_without_accepted(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import research.ga_ssw.probe_native_lbfgs_vc as probe
    q0 = np.zeros(246)
    record = dict(chart_reference={}, frozen_softening={},
                  frozen_gaussians=[dict(center=q0.tolist(), direction=q0.tolist(), width=1.)])
    climb = dict(q=q0.tolist(), status='maxiter', relaxation=dict(steps=300))
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(joint_config=dict(strain_length=1., pressure=0.))))
    monkeypatch.setattr(probe, '_pick_failed', lambda s: (record, climb))
    monkeypatch.setattr(probe, '_atoms', lambda s: None)
    monkeypatch.setattr(probe, '_softening', lambda s: None)
    monkeypatch.setattr(probe, 'SymmetricLogStrainChart', lambda *a, **k: SimpleNamespace(natoms=80))
    surfaces = []
    def surface_factory(calc):
        surface = SimpleNamespace(requests=0, label=len(surfaces))
        surfaces.append(surface)
        return surface
    monkeypatch.setattr(probe, 'ASEStressSurface', surface_factory)
    def objective(chart, surface, soft, q, terms, pressure):
        surface.requests += 1
        if surface.label == 0 and surface.requests == 2:
            raise RuntimeError('injected charged backend failure')
        return 1., np.ones(246), SimpleNamespace(atoms=None)
    monkeypatch.setattr(probe, 'frozen_objective', objective)
    class FakeOracle:
        accepted = []
        stop_reason = None
        calls = {}
        ptr = [0]*12
        def __init__(self, *a, **k): self.q=q0.copy()
        def vector(self): return self.q.copy()
        def advance(self, *a): self.q += 1.; return 1
        def integer(self, *a): return 1
    monkeypatch.setattr(probe, 'FlatOracle', FakeOracle)
    result = probe.run_case(source, source, lambda: None, [], request_cap=3,
                            accepted_step_cap=3, joint_gtol=.001, deadline=float('inf'))
    assert result['status'] == 'oracle_error'
    assert result['optimizer_requests'] == 2
    assert result['fresh_requests'] == 1
    assert result['total_requests'] == 3
    assert result['endpoint_source'] == 'evaluated_initial_no_accepted'
    np.testing.assert_array_equal(result['final_fresh']['source_q'], q0)
