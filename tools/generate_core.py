"""Mechanically translate the upstream scalar evaluators into Mojo syntax."""

from __future__ import annotations

import inspect
from pathlib import Path

import opensimplex.internals


ROOT = Path(__file__).resolve().parents[1]
source = inspect.getsource(opensimplex.internals)
source = source[source.index("def _noise2(x, y, perm):") :]

lines: list[str] = []
for line in source.splitlines():
    if line.lstrip().startswith("@njit"):
        continue
    if "#" in line:
        line = line[: line.index("#")].rstrip()
    if line.strip():
        lines.append(line)
    elif lines and lines[-1]:
        lines.append("")

body = "\n".join(lines).rstrip() + "\n"
body = body.replace(
    "def _noise2(x, y, perm):",
    "def noise2(x: Float64, y: Float64, perm: IPtr, gradients: IPtr) -> Float64:",
)
body = body.replace(
    "def _noise3(x, y, z, perm, perm_grad_index3):",
    "def noise3(x: Float64, y: Float64, z: Float64, perm: IPtr, "
    "perm_grad_index3: IPtr, gradients: IPtr) -> Float64:",
)
body = body.replace(
    "def _noise4(x, y, z, w, perm):",
    "def noise4(x: Float64, y: Float64, z: Float64, w: Float64, "
    "perm: IPtr, gradients: IPtr) -> Float64:",
)
body = body.replace("_extrapolate2(perm,", "extrapolate2(perm, gradients,")
body = body.replace(
    "_extrapolate3(perm, perm_grad_index3,",
    "extrapolate3(perm, perm_grad_index3, gradients,",
)
body = body.replace("_extrapolate4(perm,", "extrapolate4(perm, gradients,")
for name in ("xs", "ys", "zs", "ws"):
    axis = name[0]
    body = body.replace(f"{axis}sb = floor({name})", f"{axis}sb = Int(floor({name}))")
    body = body.replace(f"{name} - {axis}sb", f"{name} - Float64({axis}sb)")
for axes in ("xsb + ysb", "xsb + ysb + zsb", "xsb + ysb + zsb + wsb"):
    body = body.replace(f"({axes}) * SQUISH", f"Float64({axes}) * SQUISH")
for axis in ("x", "y", "z", "w"):
    body = body.replace(f"{axis}b = {axis}sb + squish_offset", f"{axis}b = Float64({axis}sb) + squish_offset")
body = body.replace("value = 0\n", "value = 0.0\n")

header = '''"""Exact OpenSimplex 2014 scalar evaluators.

The region decisions are retained from opensimplex 0.4.5.1. Evaluating every
positive-radius lattice point differs near simplex boundaries.
"""

from std.math import floor

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]

comptime STRETCH_CONSTANT2 = -0.211324865405187
comptime SQUISH_CONSTANT2 = 0.366025403784439
comptime STRETCH_CONSTANT3 = -1.0 / 6.0
comptime SQUISH_CONSTANT3 = 1.0 / 3.0
comptime STRETCH_CONSTANT4 = -0.138196601125011
comptime SQUISH_CONSTANT4 = 0.309016994374947
comptime NORM_CONSTANT2 = 47.0
comptime NORM_CONSTANT3 = 103.0
comptime NORM_CONSTANT4 = 30.0


def extrapolate2(
    perm: IPtr, gradients: IPtr, xsb: Int, ysb: Int, dx: Float64, dy: Float64
) -> Float64:
    var index = Int(perm[(Int(perm[xsb & 255]) + ysb) & 255] & 14)
    return Float64(gradients[index]) * dx + Float64(gradients[index + 1]) * dy


def extrapolate3(
    perm: IPtr,
    perm_grad_index3: IPtr,
    gradients: IPtr,
    xsb: Int,
    ysb: Int,
    zsb: Int,
    dx: Float64,
    dy: Float64,
    dz: Float64,
) -> Float64:
    var pxy = Int(perm[(Int(perm[xsb & 255]) + ysb) & 255])
    var index = Int(perm_grad_index3[(pxy + zsb) & 255])
    return (
        Float64(gradients[index]) * dx
        + Float64(gradients[index + 1]) * dy
        + Float64(gradients[index + 2]) * dz
    )


def extrapolate4(
    perm: IPtr,
    gradients: IPtr,
    xsb: Int,
    ysb: Int,
    zsb: Int,
    wsb: Int,
    dx: Float64,
    dy: Float64,
    dz: Float64,
    dw: Float64,
) -> Float64:
    var pxy = Int(perm[(Int(perm[xsb & 255]) + ysb) & 255])
    var pxyz = Int(perm[(pxy + zsb) & 255])
    var index = Int(perm[(pxyz + wsb) & 255] & 252)
    return (
        Float64(gradients[index]) * dx
        + Float64(gradients[index + 1]) * dy
        + Float64(gradients[index + 2]) * dz
        + Float64(gradients[index + 3]) * dw
    )


'''

(ROOT / "src").mkdir(exist_ok=True)
(ROOT / "src" / "core.mojo").write_text(header + body)
