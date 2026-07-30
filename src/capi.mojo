"""C ABI and Cartesian-grid kernels for the Python bindings."""

from std.algorithm import parallelize

from core import IPtr, noise2, noise3, noise4

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def worker_count(n: Int) -> Int:
    if n < 4096:
        return 1
    return min(16, n)


@export("mos_noise2")
def mos_noise2(
    x: Float64,
    y: Float64,
    perm_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if perm_addr == 0 or gradients_addr == 0 or dst_addr == 0:
        return -1
    FPtr(unsafe_from_address=dst_addr)[0] = noise2(
        x,
        y,
        IPtr(unsafe_from_address=perm_addr),
        IPtr(unsafe_from_address=gradients_addr),
    )
    return 0


@export("mos_noise3")
def mos_noise3(
    x: Float64,
    y: Float64,
    z: Float64,
    perm_addr: Int,
    perm_grad_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        perm_addr == 0
        or perm_grad_addr == 0
        or gradients_addr == 0
        or dst_addr == 0
    ):
        return -1
    FPtr(unsafe_from_address=dst_addr)[0] = noise3(
        x,
        y,
        z,
        IPtr(unsafe_from_address=perm_addr),
        IPtr(unsafe_from_address=perm_grad_addr),
        IPtr(unsafe_from_address=gradients_addr),
    )
    return 0


@export("mos_noise4")
def mos_noise4(
    x: Float64,
    y: Float64,
    z: Float64,
    w: Float64,
    perm_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if perm_addr == 0 or gradients_addr == 0 or dst_addr == 0:
        return -1
    FPtr(unsafe_from_address=dst_addr)[0] = noise4(
        x,
        y,
        z,
        w,
        IPtr(unsafe_from_address=perm_addr),
        IPtr(unsafe_from_address=gradients_addr),
    )
    return 0


@export("mos_noise2_array")
def mos_noise2_array(
    x_addr: Int,
    nx: Int,
    y_addr: Int,
    ny: Int,
    perm_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if nx < 0 or ny < 0:
        return -2
    if nx != 0 and ny > 9223372036854775807 // nx:
        return -3
    if nx == 0 or ny == 0:
        return 0
    if (
        x_addr == 0
        or y_addr == 0
        or perm_addr == 0
        or gradients_addr == 0
        or dst_addr == 0
    ):
        return -1
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var gradients = IPtr(unsafe_from_address=gradients_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    var total = nx * ny
    var workers = worker_count(total)

    @parameter
    def process(worker: Int):
        var start = worker * total // workers
        var stop = (worker + 1) * total // workers
        for i in range(start, stop):
            var xi = i % nx
            var yi = i // nx
            dst[i] = noise2(x[xi], y[yi], perm, gradients)

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    return 0


@export("mos_noise3_array")
def mos_noise3_array(
    x_addr: Int,
    nx: Int,
    y_addr: Int,
    ny: Int,
    z_addr: Int,
    nz: Int,
    perm_addr: Int,
    perm_grad_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if nx < 0 or ny < 0 or nz < 0:
        return -2
    if nx != 0 and ny > 9223372036854775807 // nx:
        return -3
    var plane_checked = nx * ny
    if plane_checked != 0 and nz > 9223372036854775807 // plane_checked:
        return -3
    if plane_checked == 0 or nz == 0:
        return 0
    if (
        x_addr == 0
        or y_addr == 0
        or z_addr == 0
        or perm_addr == 0
        or perm_grad_addr == 0
        or gradients_addr == 0
        or dst_addr == 0
    ):
        return -1
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var z = FPtr(unsafe_from_address=z_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var perm_grad = IPtr(unsafe_from_address=perm_grad_addr)
    var gradients = IPtr(unsafe_from_address=gradients_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    var plane = nx * ny
    var total = plane * nz
    var workers = worker_count(total)

    @parameter
    def process(worker: Int):
        var start = worker * total // workers
        var stop = (worker + 1) * total // workers
        for i in range(start, stop):
            var xi = i % nx
            var yi = (i // nx) % ny
            var zi = i // plane
            dst[i] = noise3(x[xi], y[yi], z[zi], perm, perm_grad, gradients)

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    return 0


@export("mos_noise4_array")
def mos_noise4_array(
    x_addr: Int,
    nx: Int,
    y_addr: Int,
    ny: Int,
    z_addr: Int,
    nz: Int,
    w_addr: Int,
    nw: Int,
    perm_addr: Int,
    gradients_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if nx < 0 or ny < 0 or nz < 0 or nw < 0:
        return -2
    if nx != 0 and ny > 9223372036854775807 // nx:
        return -3
    var plane_checked = nx * ny
    if plane_checked != 0 and nz > 9223372036854775807 // plane_checked:
        return -3
    var volume_checked = plane_checked * nz
    if volume_checked != 0 and nw > 9223372036854775807 // volume_checked:
        return -3
    if volume_checked == 0 or nw == 0:
        return 0
    if (
        x_addr == 0
        or y_addr == 0
        or z_addr == 0
        or w_addr == 0
        or perm_addr == 0
        or gradients_addr == 0
        or dst_addr == 0
    ):
        return -1
    var x = FPtr(unsafe_from_address=x_addr)
    var y = FPtr(unsafe_from_address=y_addr)
    var z = FPtr(unsafe_from_address=z_addr)
    var w = FPtr(unsafe_from_address=w_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var gradients = IPtr(unsafe_from_address=gradients_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    var plane = nx * ny
    var volume = plane * nz
    var total = volume * nw
    var workers = worker_count(total)

    @parameter
    def process(worker: Int):
        var start = worker * total // workers
        var stop = (worker + 1) * total // workers
        for i in range(start, stop):
            var xi = i % nx
            var yi = (i // nx) % ny
            var zi = (i // plane) % nz
            var wi = i // volume
            dst[i] = noise4(x[xi], y[yi], z[zi], w[wi], perm, gradients)

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    return 0
