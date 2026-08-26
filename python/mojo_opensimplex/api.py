"""Python API compatible with opensimplex 0.4.5.1."""

from __future__ import annotations

from ctypes import c_int64
import time

import numpy as np

from ._lib import addr, check_status, lib


DEFAULT_SEED = 3

GRADIENTS2 = np.array(
    [5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5],
    dtype=np.int64,
)
GRADIENTS3 = np.array(
    [
        -11, 4, 4, -4, 11, 4, -4, 4, 11,
        11, 4, 4, 4, 11, 4, 4, 4, 11,
        -11, -4, 4, -4, -11, 4, -4, -4, 11,
        11, -4, 4, 4, -11, 4, 4, -4, 11,
        -11, 4, -4, -4, 11, -4, -4, 4, -11,
        11, 4, -4, 4, 11, -4, 4, 4, -11,
        -11, -4, -4, -4, -11, -4, -4, -4, -11,
        11, -4, -4, 4, -11, -4, 4, -4, -11,
    ],
    dtype=np.int64,
)
GRADIENTS4 = np.array(
    [
        3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3,
        -3, 1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 1, -1, 1, 1, 3,
        3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3, 1, 1, -1, 1, 3,
        -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3,
        3, 1, -1, 1, 1, 3, -1, 1, 1, 1, -3, 1, 1, 1, -1, 3,
        -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3, 1, -1, 1, -1, 3,
        3, -1, -1, 1, 1, -3, -1, 1, 1, -1, -3, 1, 1, -1, -1, 3,
        -3, -1, -1, 1, -1, -3, -1, 1, -1, -1, -3, 1, -1, -1, -1, 3,
        3, 1, 1, -1, 1, 3, 1, -1, 1, 1, 3, -1, 1, 1, 1, -3,
        -3, 1, 1, -1, -1, 3, 1, -1, -1, 1, 3, -1, -1, 1, 1, -3,
        3, -1, 1, -1, 1, -3, 1, -1, 1, -1, 3, -1, 1, -1, 1, -3,
        -3, -1, 1, -1, -1, -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3,
        3, 1, -1, -1, 1, 3, -1, -1, 1, 1, -3, -1, 1, 1, -1, -3,
        -3, 1, -1, -1, -1, 3, -1, -1, -1, 1, -3, -1, -1, 1, -1, -3,
        3, -1, -1, -1, 1, -3, -1, -1, 1, -1, -3, -1, 1, -1, -1, -3,
        -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3, -1, -1, -1, -1, -3,
    ],
    dtype=np.int64,
)

_GRADIENTS2_ADDR = addr(GRADIENTS2)
_GRADIENTS3_ADDR = addr(GRADIENTS3)
_GRADIENTS4_ADDR = addr(GRADIENTS4)


def _overflow(value: int) -> int:
    return c_int64(value).value


def _init(seed_value: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.zeros(256, dtype=np.int64)
    perm_grad_index3 = np.zeros(256, dtype=np.int64)
    source = np.arange(256, dtype=np.int64)
    state = int(seed_value)
    for _ in range(3):
        state = _overflow(state * 6364136223846793005 + 1442695040888963407)
    for i in range(255, -1, -1):
        state = _overflow(state * 6364136223846793005 + 1442695040888963407)
        r = int((state + 31) % (i + 1))
        if r < 0:
            r += i + 1
        perm[i] = source[r]
        perm_grad_index3[i] = int((perm[i] % (GRADIENTS3.size / 3)) * 3)
        source[r] = source[i]
    return perm, perm_grad_index3


def _axis(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("coordinate axes must be one-dimensional")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("coordinate axes must contain real numbers")
    return np.ascontiguousarray(array, dtype=np.float64)


class OpenSimplex:
    def __init__(self, seed: int) -> None:
        self._perm, self._perm_grad_index3 = _init(seed)
        self._perm_addr = addr(self._perm)
        self._perm_grad_index3_addr = addr(self._perm_grad_index3)
        self._seed = seed

    def get_seed(self) -> int:
        return self._seed

    def noise2(self, x: float, y: float) -> float:
        return lib().mos_noise2_value(
            x, y, self._perm_addr, _GRADIENTS2_ADDR
        )

    def noise3(self, x: float, y: float, z: float) -> float:
        return lib().mos_noise3_value(
            x,
            y,
            z,
            self._perm_addr,
            self._perm_grad_index3_addr,
            _GRADIENTS3_ADDR,
        )

    def noise4(self, x: float, y: float, z: float, w: float) -> float:
        return lib().mos_noise4_value(
            x, y, z, w, self._perm_addr, _GRADIENTS4_ADDR
        )

    def noise2array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xa, ya = _axis(x), _axis(y)
        result = np.empty((ya.size, xa.size), dtype=np.float64)
        check_status(
            lib().mos_noise2_array(
                addr(xa), xa.size, addr(ya), ya.size,
                self._perm_addr, _GRADIENTS2_ADDR, addr(result),
            )
        )
        return result

    def noise3array(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        xa, ya, za = _axis(x), _axis(y), _axis(z)
        result = np.empty((za.size, ya.size, xa.size), dtype=np.float64)
        check_status(
            lib().mos_noise3_array(
                addr(xa), xa.size, addr(ya), ya.size, addr(za), za.size,
                self._perm_addr, self._perm_grad_index3_addr,
                _GRADIENTS3_ADDR, addr(result),
            )
        )
        return result

    def noise4array(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        xa, ya, za, wa = _axis(x), _axis(y), _axis(z), _axis(w)
        result = np.empty(
            (wa.size, za.size, ya.size, xa.size), dtype=np.float64
        )
        check_status(
            lib().mos_noise4_array(
                addr(xa), xa.size, addr(ya), ya.size, addr(za), za.size,
                addr(wa), wa.size, self._perm_addr, _GRADIENTS4_ADDR,
                addr(result),
            )
        )
        return result


_default = OpenSimplex(DEFAULT_SEED)


def seed(seed: int = DEFAULT_SEED) -> None:
    global _default
    _default = OpenSimplex(seed)


def random_seed() -> None:
    seed(time.time_ns())


def get_seed() -> int:
    return _default.get_seed()


def noise2(x: float, y: float) -> float:
    return _default.noise2(x, y)


def noise2array(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _default.noise2array(x, y)


def noise3(x: float, y: float, z: float) -> float:
    return _default.noise3(x, y, z)


def noise3array(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return _default.noise3array(x, y, z)


def noise4(x: float, y: float, z: float, w: float) -> float:
    return _default.noise4(x, y, z, w)


def noise4array(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> np.ndarray:
    return _default.noise4array(x, y, z, w)
