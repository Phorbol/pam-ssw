import numpy as np
import pytest
from dataclasses import replace
from ase.build import bulk
from ase.calculators.emt import EMT

from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw


def policy():
    return ConservativeNativeHeightPolicy(2., .2, 1, 10., 1.05, 1.25)


def test_vc_accepts_conservative_policy_and_persists_preparation_scope():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    config = VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=100.,
                         max_gaussians=2, relax_steps=80, rotation_hvp=20)
    result = run_vc_ssw(atoms, ASEStressSurface(EMT()), steps=1, config=config,
                         rng=np.random.default_rng(17), height_policy=policy(),
                         height_update_budget=1000)
    event = result.records[1]['climb'][0]
    assert 'height_input' in event
    assert 'height_preparation' in event
    preparation = event['height_preparation']
    assert preparation.curvature_scope.startswith('physical enthalpy E+pV')
    assert event['height_update_budget'] == 1000
    prepared = [stage for stage in result.records[1]['climb']
                if 'height_preparation' in stage]
    assert len(prepared) >= 2
    assert prepared[1]['height_input']['history'][0].weight == 5.6


def test_vc_height_update_budget_is_validated_before_oracle():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    config = VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=100.)
    surface = ASEStressSurface(EMT())
    with pytest.raises(ValueError, match='height-update'):
        run_vc_ssw(atoms, surface, steps=0, config=config,
                   rng=np.random.default_rng(17), height_policy=policy(),
                   height_update_budget=True)
    assert surface.requests == 0


class RewritingPolicy(ConservativeNativeHeightPolicy):
    def prepare(self, history, **kwargs):
        prepared = super().prepare(history, **kwargs)
        if prepared.terms:
            terms = list(prepared.terms)
            terms[0] = replace(terms[0], weight=1.234)
            return replace(prepared, terms=tuple(terms))
        return prepared


def test_vc_uses_policy_returned_rewritten_history_once():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    config = VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=100.,
                         max_gaussians=2, relax_steps=80, rotation_hvp=20)
    result = run_vc_ssw(atoms, ASEStressSurface(EMT()), steps=1, config=config,
                         rng=np.random.default_rng(17), height_policy=RewritingPolicy(
                             2., .2, 0, 10., 1.05, 1.25))
    assert len(result.records[1]['frozen_gaussians']) >= 2
    assert result.records[1]['frozen_gaussians'][0]['weight'] == pytest.approx(1.234)


class FailingPolicy(ConservativeNativeHeightPolicy):
    def prepare(self, history, **kwargs):
        raise RuntimeError('controlled height preparation failure')


def test_vc_preserves_height_input_scope_and_budget_when_prepare_fails():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    config = VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=100.,
                         max_gaussians=1, relax_steps=80, rotation_hvp=20)
    result = run_vc_ssw(atoms, ASEStressSurface(EMT()), steps=1, config=config,
                         rng=np.random.default_rng(17), height_policy=FailingPolicy(
                             2., .2, 0, 10., 1.05, 1.25), height_update_budget=7)
    event = result.records[1]['climb'][0]
    assert 'height_input' in event
    assert event['curvature_scope'].startswith('physical enthalpy E+pV')
    assert event['height_update_budget'] == 7
