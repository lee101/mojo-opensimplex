"""Load the compiled Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_OPENSIMPLEX_LIB") or os.path.join(
    ROOT, "dist", "libmojo-opensimplex.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mos_noise2_value": ([F, F, I, I], F),
    "mos_noise3_value": ([F, F, F, I, I, I], F),
    "mos_noise4_value": ([F, F, F, F, I, I], F),
    "mos_noise2": ([F, F, I, I, I], I),
    "mos_noise3": ([F, F, F, I, I, I, I], I),
    "mos_noise4": ([F, F, F, F, I, I, I], I),
    "mos_noise2_array": ([I, I, I, I, I, I, I], I),
    "mos_noise3_array": ([I] * 10, I),
    "mos_noise4_array": ([I] * 11, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_OPENSIMPLEX_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_OPENSIMPLEX_LIB does not exist: {LIB}")
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
            return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array) -> int:
    return int(array.ctypes.data)


def check_status(status: int) -> None:
    messages = {
        -1: "null buffer address",
        -2: "negative array length",
        -3: "array size exceeds the native index range",
    }
    if status:
        detail = messages.get(status, f"unknown status {status}")
        raise RuntimeError(f"Mojo kernel rejected the call: {detail}")
