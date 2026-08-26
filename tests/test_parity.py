"""Numerical and behavioral parity with opensimplex 0.4.5.1."""

from __future__ import annotations

import inspect
from ctypes import addressof, c_double

import numpy as np
import pytest

import mojo_opensimplex as mos
from mojo_opensimplex._lib import lib
import opensimplex as reference


SEEDS = [3, 0, -1, 2**63 - 1, -(2**63), 10**40 + 17]


@pytest.mark.parametrize(
    ("name", "coordinates", "expected"),
    [
        ("noise2", (0.5, 0.5), -0.43906247097569345),
        ("noise3", (0.5, 0.5, 0.5), 0.39504955501618155),
        ("noise4", (0.5, 0.5, 0.5, 0.5), 0.04520359600370195),
    ],
)
def test_published_default_vectors(name, coordinates, expected):
    mos.seed()
    assert getattr(mos, name)(*coordinates) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("seed_value", SEEDS)
def test_scalar_parity_all_dimensions(seed_value):
    rng = np.random.default_rng(104)
    ours = mos.OpenSimplex(seed_value)
    theirs = reference.OpenSimplex(seed_value)
    for dimension in (2, 3, 4):
        coordinates = rng.uniform(-1_000_000_000, 1_000_000_000, (150, dimension))
        ours_fn = getattr(ours, f"noise{dimension}")
        reference_fn = getattr(theirs, f"noise{dimension}")
        actual = np.array([ours_fn(*point) for point in coordinates])
        expected = np.array([reference_fn(*point) for point in coordinates])
        assert np.allclose(actual, expected, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("seed_value", [-991, 0, 42])
def test_noise2array_parity(seed_value):
    x = np.linspace(-20.0, 30.0, 81)
    y = np.linspace(7.0, -11.0, 75)
    actual = mos.OpenSimplex(seed_value).noise2array(x, y)
    expected = reference.OpenSimplex(seed_value).noise2array(x, y)
    assert actual.shape == (y.size, x.size)
    assert actual.dtype == np.float64
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-14)


def test_noise3array_parity_and_axis_order():
    x = np.linspace(-2.0, 3.0, 19)
    y = np.linspace(1.0, 4.0, 18)
    z = np.linspace(-3.0, -1.0, 17)
    actual = mos.OpenSimplex(1234).noise3array(x, y, z)
    expected = reference.OpenSimplex(1234).noise3array(x, y, z)
    assert actual.shape == (z.size, y.size, x.size)
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-14)
    assert actual[4, 7, 11] == pytest.approx(
        mos.OpenSimplex(1234).noise3(x[11], y[7], z[4]), abs=2e-15
    )


def test_noise4array_parity_and_axis_order():
    x = np.linspace(-2.0, 2.0, 10)
    y = np.linspace(-1.0, 1.0, 9)
    z = np.linspace(0.0, 2.0, 9)
    w = np.linspace(3.0, 4.0, 9)
    actual = mos.OpenSimplex(-78).noise4array(x, y, z, w)
    expected = reference.OpenSimplex(-78).noise4array(x, y, z, w)
    assert actual.shape == (w.size, z.size, y.size, x.size)
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-14)
    assert actual[5, 4, 3, 2] == pytest.approx(
        mos.OpenSimplex(-78).noise4(x[2], y[3], z[4], w[5]), abs=2e-15
    )


def test_global_seed_and_get_seed():
    mos.seed(8675309)
    reference.seed(8675309)
    assert mos.get_seed() == reference.get_seed() == 8675309
    assert mos.noise2(-1.25, 9.5) == pytest.approx(
        reference.noise2(-1.25, 9.5), abs=2e-15
    )
    mos.seed()
    reference.seed()


def test_global_functions_all_dimensions():
    axes = [np.linspace(-1.0, 1.0, 4)] * 4
    mos.seed(91)
    reference.seed(91)
    for dimension in (2, 3, 4):
        point = tuple(axis[index] for index, axis in enumerate(axes[:dimension]))
        assert getattr(mos, f"noise{dimension}")(*point) == pytest.approx(
            getattr(reference, f"noise{dimension}")(*point), abs=2e-15
        )
        actual = getattr(mos, f"noise{dimension}array")(*axes[:dimension])
        expected = getattr(reference, f"noise{dimension}array")(*axes[:dimension])
        assert np.allclose(actual, expected, rtol=0.0, atol=2e-14)
    mos.seed()
    reference.seed()


def test_random_seed_uses_nanosecond_clock(monkeypatch):
    monkeypatch.setattr(mos.api.time, "time_ns", lambda: 123456789012345)
    mos.random_seed()
    assert mos.get_seed() == 123456789012345
    mos.seed()


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_noncontiguous_float32_axes(dimension):
    axes = [
        np.arange(20, dtype=np.float32)[::3],
        np.arange(18, dtype=np.int32)[::4],
        np.linspace(-2, 2, 14)[::2],
        np.linspace(1, 3, 16)[::3],
    ][:dimension]
    actual = getattr(mos.OpenSimplex(12), f"noise{dimension}array")(*axes)
    expected = getattr(reference.OpenSimplex(12), f"noise{dimension}array")(
        *(np.asarray(axis) for axis in axes)
    )
    # Upstream retains float32 intermediates for float32 axes. The Mojo API
    # normalizes coordinates to float64 before crossing the ABI.
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_empty_axis(dimension):
    axes = [np.array([])] + [np.arange(3.0)] * (dimension - 1)
    actual = getattr(mos.OpenSimplex(4), f"noise{dimension}array")(*axes)
    expected = getattr(reference.OpenSimplex(4), f"noise{dimension}array")(*axes)
    assert actual.shape == expected.shape
    assert actual.size == 0


def test_axes_are_validated_before_crossing_ffi():
    instance = mos.OpenSimplex(4)
    with pytest.raises(ValueError, match="one-dimensional"):
        instance.noise2array(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(TypeError, match="real numbers"):
        instance.noise2array(np.array([1 + 2j]), np.zeros(2))
    with pytest.raises(ValueError):
        instance.noise2array(np.array(["not a number"]), np.zeros(2))


def test_native_abi_rejects_invalid_addresses_lengths_and_overflow():
    native = lib()
    output = c_double()
    assert native.mos_noise2(0.0, 0.0, 0, 0, addressof(output)) == -1
    assert native.mos_noise2_array(0, -1, 0, 1, 0, 0, 0) == -2
    assert native.mos_noise2_array(0, 2**62, 0, 4, 0, 0, 0) == -3
    # Empty Cartesian products do not dereference any buffers.
    assert native.mos_noise4_array(0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0) == 0


def test_scalar_direct_return_abi_matches_checked_abi():
    native = lib()
    instance = mos.OpenSimplex(73)
    output = c_double()
    status = native.mos_noise2(
        -12.5,
        7.25,
        instance._perm_addr,
        mos.api._GRADIENTS2_ADDR,
        addressof(output),
    )
    direct = native.mos_noise2_value(
        -12.5, 7.25, instance._perm_addr, mos.api._GRADIENTS2_ADDR
    )
    assert status == 0
    assert direct == output.value


def test_public_api_names_and_signatures():
    names = {
        "DEFAULT_SEED", "OpenSimplex", "get_seed", "noise2", "noise2array",
        "noise3", "noise3array", "noise4", "noise4array", "random_seed", "seed",
    }
    assert names <= set(mos.__all__)
    for name in names - {"DEFAULT_SEED", "OpenSimplex"}:
        assert tuple(inspect.signature(getattr(mos, name)).parameters) == tuple(
            inspect.signature(getattr(reference, name)).parameters
        )


def test_values_stay_in_documented_range():
    rng = np.random.default_rng(55)
    instance = mos.OpenSimplex(44)
    for dimension in (2, 3, 4):
        points = rng.normal(size=(1000, dimension)) * 100
        values = [getattr(instance, f"noise{dimension}")(*point) for point in points]
        assert np.max(values) <= 1.0
        assert np.min(values) >= -1.0
