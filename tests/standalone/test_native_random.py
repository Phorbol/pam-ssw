"""Archived instruction outputs, no calculator or search-quality claim."""
import numpy as np
from pamssw.standalone.native_random import native_vmb2, native_ran3


def test_native_ran3_stream():
    rng = native_ran3(-11)
    expected = [.9412623013792834, .6362228000063899, .35219238703931877,
                .025734529272177786, .9734426564691607, .7571679055858286]
    assert [next(rng) for _ in expected] == expected


def test_native_vmb2_frozen_output():
    got = native_vmb2(np.arange(9).reshape(3, 3)*.01, np.ones((3, 3), bool), .17, temperature=1.)
    expected = [[-.0004763457464567895, .00025615085782416426, -.0010138272737654655],
                [.0011331739228916385, .0016096430882117096, .002038428316310897],
                [-.000656828176434849, -.0018657939460358746, -.0010246010425454314]]
    np.testing.assert_allclose(got, expected, atol=2e-18, rtol=0)


def test_seed_quantization_is_explicit_reference_behavior():
    initial = np.zeros((8, 3))
    mask = np.ones_like(initial, bool)
    np.testing.assert_array_equal(native_vmb2(initial, mask, .11), native_vmb2(initial, mask, .19))
