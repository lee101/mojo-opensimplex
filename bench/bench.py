"""Benchmark Mojo kernels against opensimplex 0.4.5.1."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_opensimplex as mos  # noqa: E402
import opensimplex as reference  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def main() -> None:
    ours = mos.OpenSimplex(2026)
    theirs = reference.OpenSimplex(2026)
    rng = np.random.default_rng(8)
    points = rng.uniform(-1000.0, 1000.0, (100_000, 2))
    x2 = np.linspace(-100.0, 100.0, 400)
    y2 = np.linspace(-80.0, 80.0, 400)
    x3 = np.linspace(-20.0, 20.0, 40)
    y3 = np.linspace(-15.0, 15.0, 40)
    z3 = np.linspace(-10.0, 10.0, 40)
    axis4 = np.linspace(-4.0, 4.0, 16)

    cases = [
        (
            "noise2 scalar x 100,000",
            lambda: [ours.noise2(x, y) for x, y in points],
            lambda: [theirs.noise2(x, y) for x, y in points],
        ),
        (
            "noise2array 400 x 400",
            lambda: ours.noise2array(x2, y2),
            lambda: theirs.noise2array(x2, y2),
        ),
        (
            "noise3array 40 x 40 x 40",
            lambda: ours.noise3array(x3, y3, z3),
            lambda: theirs.noise3array(x3, y3, z3),
        ),
        (
            "noise4array 16^4",
            lambda: ours.noise4array(axis4, axis4, axis4, axis4),
            lambda: theirs.noise4array(axis4, axis4, axis4, axis4),
        ),
    ]

    print(f"Machine: {cpu_name()} ({os.cpu_count()} logical CPUs), {platform.system()}")
    print()
    print("| case | mojo-opensimplex | opensimplex 0.4.5.1 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, reference_fn in cases:
        actual = mojo_fn()
        expected = reference_fn()
        if not np.allclose(actual, expected, rtol=0.0, atol=2e-13):
            raise RuntimeError(f"parity failed before timing {name}")
        mojo_time = timeit(mojo_fn)
        reference_time = timeit(reference_fn)
        print(
            f"| {name} | {format_time(mojo_time)} | "
            f"{format_time(reference_time)} | {reference_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
