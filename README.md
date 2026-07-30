# mojo-opensimplex

OpenSimplex gradient noise implemented in Mojo and callable from Python.
`mojo_opensimplex` mirrors the public API of Python
`opensimplex==0.4.5.1`, while moving every scalar and Cartesian-grid noise
evaluation into a compiled shared library.

```python
import numpy as np
import mojo_opensimplex as opensimplex

opensimplex.seed(1234)
print(opensimplex.noise2(10.0, 10.0))

x = np.linspace(0.0, 8.0, 512)
y = np.linspace(0.0, 8.0, 512)
heightmap = opensimplex.noise2array(x, y)  # shape: (512, 512)
```

The module name is different so this project and the reference package can be
installed together. Importing it as `opensimplex`, as above, makes the covered
API a drop-in replacement.

## Coverage

The complete public computational API of `opensimplex` 0.4.5.1 is covered:

| API | support |
| --- | --- |
| Global state | `seed`, `random_seed`, `get_seed`, `DEFAULT_SEED` |
| Scalar noise | `noise2`, `noise3`, `noise4` |
| Grid noise | `noise2array`, `noise3array`, `noise4array` |
| Independent generators | `OpenSimplex` with all methods above |

The implementation is the original 2014 OpenSimplex algorithm used by that
package. It does not implement OpenSimplex2, OpenSimplex2S, analytic
derivatives, octave/fractal composition, or GPU kernels.

Array axes must be one-dimensional, real, NumPy-compatible inputs. They are
normalized to contiguous `float64`. This gives full double-precision
evaluation; it also means a `float32` input does not retain upstream's
float32 intermediate rounding. Results for those inputs can differ by roughly
`1e-6`. Float64 scalar and array results agree with upstream to floating-point
roundoff, including negative, extreme, and wider-than-64-bit seeds.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles the pinned Mojo nightly into
`dist/libmojo-opensimplex.so`. Imports rebuild a missing or stale library when
the compiler is available. A prebuilt library can instead be selected with
`MOJO_OPENSIMPLEX_LIB=/absolute/path/to/library.so`.

Run the example directly in the managed environment:

```bash
pixi run python -c "import mojo_opensimplex as o; print(o.noise2(10.0, 10.0))"
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(72 logical CPUs, Linux). These are the best of three runs, with identical
inputs and an equality check before timing. The conda-forge build of
`opensimplex` 0.4.5.1 used here does not install its optional Numba accelerator,
so its documented pure-Python fallback is the reference.

| case | mojo-opensimplex | opensimplex 0.4.5.1 | speedup |
| --- | ---: | ---: | ---: |
| `noise2` scalar x 100,000 | 513.58 ms | 1.88 s | 3.66x |
| `noise2array` 400 x 400 | 7.30 ms | 2.34 s | 320.77x |
| `noise3array` 40 x 40 x 40 | 14.30 ms | 2.00 s | 139.88x |
| `noise4array` 16^4 | 8.88 ms | 4.19 s | 472.10x |

Scalar calls still pay one ctypes transition each, but reuse cached addresses
for their stable NumPy-owned permutation and gradient buffers. Grid calls
cross the ABI once and spread independent points across up to 16 worker tasks,
which is where the large gains come from.

No GPU path is provided. Each lattice contribution performs several dependent
permutation and gradient table reads for a small amount of arithmetic, leaving
the effective arithmetic intensity below the roughly 2-flop-per-byte threshold;
point-dependent region branches would also cause warp divergence. Run
`pixi run bench` to reproduce the table on the current machine.

## How it works

`src/core.mojo` retains the exact simplex-region decisions, gradient tables,
normalization, and 64-bit permutation behavior of upstream. `src/capi.mojo`
exports non-parametric C-ABI functions. Python owns the permutation tables,
coordinate axes, and output arrays; buffers cross the ABI as integer addresses
and are reconstructed as `UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo.
Calls are synchronous, so NumPy retains every buffer for the full native call.
The ABI checks non-null addresses, non-negative dimensions, and multiplication
overflow before dereferencing memory, and Python raises on a nonzero status.
The Mojo side performs no heap allocation.

Grid outputs are contiguous row-major `float64` arrays with upstream's layout:

- 2D: `(y, x)`
- 3D: `(z, y, x)`
- 4D: `(w, z, y, x)`

Tests compare all dimensions, seed behavior, public signatures, published
examples, array layouts, and thousands of values against the installed real
`opensimplex` package.

## License

MIT. The translated evaluator derives from the MIT-licensed Python package;
its attribution is preserved in `THIRD_PARTY_NOTICES`.
