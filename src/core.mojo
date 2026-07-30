"""Exact OpenSimplex 2014 scalar evaluators.

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


def noise2(x: Float64, y: Float64, perm: IPtr, gradients: IPtr) -> Float64:

    stretch_offset = (x + y) * STRETCH_CONSTANT2
    xs = x + stretch_offset
    ys = y + stretch_offset

    xsb = Int(floor(xs))
    ysb = Int(floor(ys))

    squish_offset = Float64(xsb + ysb) * SQUISH_CONSTANT2
    xb = Float64(xsb) + squish_offset
    yb = Float64(ysb) + squish_offset

    xins = xs - Float64(xsb)
    yins = ys - Float64(ysb)

    in_sum = xins + yins

    dx0 = x - xb
    dy0 = y - yb

    value = 0.0

    dx1 = dx0 - 1 - SQUISH_CONSTANT2
    dy1 = dy0 - 0 - SQUISH_CONSTANT2
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        value += attn1 * attn1 * extrapolate2(perm, gradients, xsb + 1, ysb + 0, dx1, dy1)

    dx2 = dx0 - 0 - SQUISH_CONSTANT2
    dy2 = dy0 - 1 - SQUISH_CONSTANT2
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * extrapolate2(perm, gradients, xsb + 0, ysb + 1, dx2, dy2)

    if in_sum <= 1:
        zins = 1 - in_sum
        if zins > xins or zins > yins:
            if xins > yins:
                xsv_ext = xsb + 1
                ysv_ext = ysb - 1
                dx_ext = dx0 - 1
                dy_ext = dy0 + 1
            else:
                xsv_ext = xsb - 1
                ysv_ext = ysb + 1
                dx_ext = dx0 + 1
                dy_ext = dy0 - 1
        else:
            xsv_ext = xsb + 1
            ysv_ext = ysb + 1
            dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT2
            dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT2
    else:
        zins = 2 - in_sum
        if zins < xins or zins < yins:
            if xins > yins:
                xsv_ext = xsb + 2
                ysv_ext = ysb + 0
                dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT2
                dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT2
            else:
                xsv_ext = xsb + 0
                ysv_ext = ysb + 2
                dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT2
                dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT2
        else:
            dx_ext = dx0
            dy_ext = dy0
            xsv_ext = xsb
            ysv_ext = ysb
        xsb += 1
        ysb += 1
        dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT2
        dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT2

    attn0 = 2 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0:
        attn0 *= attn0
        value += attn0 * attn0 * extrapolate2(perm, gradients, xsb, ysb, dx0, dy0)

    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0:
        attn_ext *= attn_ext
        value += attn_ext * attn_ext * extrapolate2(perm, gradients, xsv_ext, ysv_ext, dx_ext, dy_ext)

    return value / NORM_CONSTANT2

def noise3(x: Float64, y: Float64, z: Float64, perm: IPtr, perm_grad_index3: IPtr, gradients: IPtr) -> Float64:

    stretch_offset = (x + y + z) * STRETCH_CONSTANT3
    xs = x + stretch_offset
    ys = y + stretch_offset
    zs = z + stretch_offset

    xsb = Int(floor(xs))
    ysb = Int(floor(ys))
    zsb = Int(floor(zs))

    squish_offset = Float64(xsb + ysb + zsb) * SQUISH_CONSTANT3
    xb = Float64(xsb) + squish_offset
    yb = Float64(ysb) + squish_offset
    zb = Float64(zsb) + squish_offset

    xins = xs - Float64(xsb)
    yins = ys - Float64(ysb)
    zins = zs - Float64(zsb)

    in_sum = xins + yins + zins

    dx0 = x - xb
    dy0 = y - yb
    dz0 = z - zb

    value = 0.0
    if in_sum <= 1:

        a_point = 0x01
        a_score = xins
        b_point = 0x02
        b_score = yins
        if a_score >= b_score and zins > b_score:
            b_score = zins
            b_point = 0x04
        elif a_score < b_score and zins > a_score:
            a_score = zins
            a_point = 0x04

        wins = 1 - in_sum
        if wins > a_score or wins > b_score:
            c = b_point if (b_score > a_score) else a_point

            if (c & 0x01) == 0:
                xsv_ext0 = xsb - 1
                xsv_ext1 = xsb
                dx_ext0 = dx0 + 1
                dx_ext1 = dx0
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx_ext1 = dx0 - 1

            if (c & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0
                if (c & 0x01) == 0:
                    ysv_ext1 -= 1
                    dy_ext1 += 1
                else:
                    ysv_ext0 -= 1
                    dy_ext0 += 1
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1

            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0
                dz_ext1 = dz0 + 1
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz_ext1 = dz0 - 1
        else:
            c = a_point | b_point

            if (c & 0x01) == 0:
                xsv_ext0 = xsb
                xsv_ext1 = xsb - 1
                dx_ext0 = dx0 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3

            if (c & 0x02) == 0:
                ysv_ext0 = ysb
                ysv_ext1 = ysb - 1
                dy_ext0 = dy0 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3

            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3

        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 0, zsb + 0, dx0, dy0, dz0)

        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)
    elif in_sum >= 2:

        a_point = 0x06
        a_score = xins
        b_point = 0x05
        b_score = yins
        if a_score <= b_score and zins < b_score:
            b_score = zins
            b_point = 0x03
        elif a_score > b_score and zins < a_score:
            a_score = zins
            a_point = 0x03

        wins = 3 - in_sum
        if wins < a_score or wins < b_score:
            c = b_point if (b_score < a_score) else a_point

            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 2
                xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT3

            if (c & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                if (c & 0x01) != 0:
                    ysv_ext1 += 1
                    dy_ext1 -= 1
                else:
                    ysv_ext0 += 1
                    dy_ext0 -= 1
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT3

            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 3 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT3
        else:
            c = a_point & b_point

            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 1
                xsv_ext1 = xsb + 2
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx0 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3

            if (c & 0x02) != 0:
                ysv_ext0 = ysb + 1
                ysv_ext1 = ysb + 2
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy0 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3

            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz0 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3

        dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 1, zsb + 0, dx3, dy3, dz3)

        dx2 = dx3
        dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 0, zsb + 1, dx2, dy2, dz2)

        dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy1 = dy3
        dz1 = dz2
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 1, zsb + 1, dx1, dy1, dz1)

        dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
        dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
        dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0)
    else:

        p1 = xins + yins
        if p1 > 1:
            a_score = p1 - 1
            a_point = 0x03
            a_is_further_side = True
        else:
            a_score = 1 - p1
            a_point = 0x04
            a_is_further_side = False

        p2 = xins + zins
        if p2 > 1:
            b_score = p2 - 1
            b_point = 0x05
            b_is_further_side = True
        else:
            b_score = 1 - p2
            b_point = 0x02
            b_is_further_side = False

        p3 = yins + zins
        if p3 > 1:
            score = p3 - 1
            if a_score <= b_score and a_score < score:
                a_point = 0x06
                a_is_further_side = True
            elif a_score > b_score and b_score < score:
                b_point = 0x06
                b_is_further_side = True
        else:
            score = 1 - p3
            if a_score <= b_score and a_score < score:
                a_point = 0x01
                a_is_further_side = False
            elif a_score > b_score and b_score < score:
                b_point = 0x01
                b_is_further_side = False

        if a_is_further_side == b_is_further_side:
            if a_is_further_side:

                dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1

                c = a_point & b_point
                if (c & 0x01) != 0:
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 2
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb
                elif (c & 0x02) != 0:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb + 2
                    zsv_ext1 = zsb
                else:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb + 2
            else:

                dx_ext0 = dx0
                dy_ext0 = dy0
                dz_ext0 = dz0
                xsv_ext0 = xsb
                ysv_ext0 = ysb
                zsv_ext0 = zsb

                c = a_point | b_point
                if (c & 0x01) == 0:
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb - 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb + 1
                elif (c & 0x02) == 0:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb - 1
                    zsv_ext1 = zsb + 1
                else:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb - 1
        else:
            if a_is_further_side:
                c1 = a_point
                c2 = b_point
            else:
                c1 = b_point
                c2 = a_point

            if (c1 & 0x01) == 0:
                dx_ext0 = dx0 + 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb - 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1
            elif (c1 & 0x02) == 0:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 + 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb - 1
                zsv_ext0 = zsb + 1
            else:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 + 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb - 1

            dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
            dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
            dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
            xsv_ext1 = xsb
            ysv_ext1 = ysb
            zsv_ext1 = zsb
            if (c2 & 0x01) != 0:
                dx_ext1 -= 2
                xsv_ext1 += 2
            elif (c2 & 0x02) != 0:
                dy_ext1 -= 2
                ysv_ext1 += 2
            else:
                dz_ext1 -= 2
                zsv_ext1 += 2

        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)

        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)

        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)

        dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz4 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 1, zsb + 0, dx4, dy4, dz4)

        dx5 = dx4
        dy5 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
        if attn5 > 0:
            attn5 *= attn5
            value += attn5 * attn5 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 1, ysb + 0, zsb + 1, dx5, dy5, dz5)

        dx6 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy6 = dy4
        dz6 = dz5
        attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
        if attn6 > 0:
            attn6 *= attn6
            value += attn6 * attn6 * extrapolate3(perm, perm_grad_index3, gradients, xsb + 0, ysb + 1, zsb + 1, dx6, dy6, dz6)

    attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
    if attn_ext0 > 0:
        attn_ext0 *= attn_ext0
        value += (
            attn_ext0
            * attn_ext0
            * extrapolate3(perm, perm_grad_index3, gradients, xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0)
        )

    attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
    if attn_ext1 > 0:
        attn_ext1 *= attn_ext1
        value += (
            attn_ext1
            * attn_ext1
            * extrapolate3(perm, perm_grad_index3, gradients, xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1)
        )

    return value / NORM_CONSTANT3

def noise4(x: Float64, y: Float64, z: Float64, w: Float64, perm: IPtr, gradients: IPtr) -> Float64:

    stretch_offset = (x + y + z + w) * STRETCH_CONSTANT4
    xs = x + stretch_offset
    ys = y + stretch_offset
    zs = z + stretch_offset
    ws = w + stretch_offset

    xsb = Int(floor(xs))
    ysb = Int(floor(ys))
    zsb = Int(floor(zs))
    wsb = Int(floor(ws))

    squish_offset = Float64(xsb + ysb + zsb + wsb) * SQUISH_CONSTANT4
    xb = Float64(xsb) + squish_offset
    yb = Float64(ysb) + squish_offset
    zb = Float64(zsb) + squish_offset
    wb = Float64(wsb) + squish_offset

    xins = xs - Float64(xsb)
    yins = ys - Float64(ysb)
    zins = zs - Float64(zsb)
    wins = ws - Float64(wsb)

    in_sum = xins + yins + zins + wins

    dx0 = x - xb
    dy0 = y - yb
    dz0 = z - zb
    dw0 = w - wb

    value = 0.0
    if in_sum <= 1:

        a_po = 0x01
        a_score = xins
        b_po = 0x02
        b_score = yins
        if a_score >= b_score and zins > b_score:
            b_score = zins
            b_po = 0x04
        elif a_score < b_score and zins > a_score:
            a_score = zins
            a_po = 0x04

        if a_score >= b_score and wins > b_score:
            b_score = wins
            b_po = 0x08
        elif a_score < b_score and wins > a_score:
            a_score = wins
            a_po = 0x08

        uins = 1 - in_sum
        if uins > a_score or uins > b_score:
            c = b_po if (b_score > a_score) else a_po
            if (c & 0x01) == 0:
                xsv_ext0 = xsb - 1
                xsv_ext1 = xsv_ext2 = xsb
                dx_ext0 = dx0 + 1
                dx_ext1 = dx_ext2 = dx0
            else:
                xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb + 1
                dx_ext0 = dx_ext1 = dx_ext2 = dx0 - 1

            if (c & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                dy_ext0 = dy_ext1 = dy_ext2 = dy0
                if (c & 0x01) == 0x01:
                    ysv_ext0 -= 1
                    dy_ext0 += 1
                else:
                    ysv_ext1 -= 1
                    dy_ext1 += 1

            else:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 1

            if (c & 0x04) == 0:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                dz_ext0 = dz_ext1 = dz_ext2 = dz0
                if (c & 0x03) != 0:
                    if (c & 0x03) == 0x03:
                        zsv_ext0 -= 1
                        dz_ext0 += 1
                    else:
                        zsv_ext1 -= 1
                        dz_ext1 += 1

                else:
                    zsv_ext2 -= 1
                    dz_ext2 += 1

            else:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 1

            if (c & 0x08) == 0:
                wsv_ext0 = wsv_ext1 = wsb
                wsv_ext2 = wsb - 1
                dw_ext0 = dw_ext1 = dw0
                dw_ext2 = dw0 + 1
            else:
                wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb + 1
                dw_ext0 = dw_ext1 = dw_ext2 = dw0 - 1

        else:
            c = a_po | b_po

            if (c & 0x01) == 0:
                xsv_ext0 = xsv_ext2 = xsb
                xsv_ext1 = xsb - 1
                dx_ext0 = dx0 - 2 * SQUISH_CONSTANT4
                dx_ext1 = dx0 + 1 - SQUISH_CONSTANT4
                dx_ext2 = dx0 - SQUISH_CONSTANT4
            else:
                xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb + 1
                dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT4
                dx_ext1 = dx_ext2 = dx0 - 1 - SQUISH_CONSTANT4

            if (c & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                dy_ext0 = dy0 - 2 * SQUISH_CONSTANT4
                dy_ext1 = dy_ext2 = dy0 - SQUISH_CONSTANT4
                if (c & 0x01) == 0x01:
                    ysv_ext1 -= 1
                    dy_ext1 += 1
                else:
                    ysv_ext2 -= 1
                    dy_ext2 += 1

            else:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT4
                dy_ext1 = dy_ext2 = dy0 - 1 - SQUISH_CONSTANT4

            if (c & 0x04) == 0:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                dz_ext0 = dz0 - 2 * SQUISH_CONSTANT4
                dz_ext1 = dz_ext2 = dz0 - SQUISH_CONSTANT4
                if (c & 0x03) == 0x03:
                    zsv_ext1 -= 1
                    dz_ext1 += 1
                else:
                    zsv_ext2 -= 1
                    dz_ext2 += 1

            else:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT4
                dz_ext1 = dz_ext2 = dz0 - 1 - SQUISH_CONSTANT4

            if (c & 0x08) == 0:
                wsv_ext0 = wsv_ext1 = wsb
                wsv_ext2 = wsb - 1
                dw_ext0 = dw0 - 2 * SQUISH_CONSTANT4
                dw_ext1 = dw0 - SQUISH_CONSTANT4
                dw_ext2 = dw0 + 1 - SQUISH_CONSTANT4
            else:
                wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb + 1
                dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT4
                dw_ext1 = dw_ext2 = dw0 - 1 - SQUISH_CONSTANT4

        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 0, wsb + 0, dx0, dy0, dz0, dw0)

        dx1 = dx0 - 1 - SQUISH_CONSTANT4
        dy1 = dy0 - 0 - SQUISH_CONSTANT4
        dz1 = dz0 - 0 - SQUISH_CONSTANT4
        dw1 = dw0 - 0 - SQUISH_CONSTANT4
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 0, wsb + 0, dx1, dy1, dz1, dw1)

        dx2 = dx0 - 0 - SQUISH_CONSTANT4
        dy2 = dy0 - 1 - SQUISH_CONSTANT4
        dz2 = dz1
        dw2 = dw1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 0, wsb + 0, dx2, dy2, dz2, dw2)

        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT4
        dw3 = dw1
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 1, wsb + 0, dx3, dy3, dz3, dw3)

        dx4 = dx2
        dy4 = dy1
        dz4 = dz1
        dw4 = dw0 - 1 - SQUISH_CONSTANT4
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 0, wsb + 1, dx4, dy4, dz4, dw4)

    elif in_sum >= 3:

        a_po = 0x0E
        a_score = xins
        b_po = 0x0D
        b_score = yins
        if a_score <= b_score and zins < b_score:
            b_score = zins
            b_po = 0x0B
        elif a_score > b_score and zins < a_score:
            a_score = zins
            a_po = 0x0B

        if a_score <= b_score and wins < b_score:
            b_score = wins
            b_po = 0x07
        elif a_score > b_score and wins < a_score:
            a_score = wins
            a_po = 0x07

        uins = 4 - in_sum
        if uins < a_score or uins < b_score:
            c = b_po if (b_score < a_score) else a_po

            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 2
                xsv_ext1 = xsv_ext2 = xsb + 1
                dx_ext0 = dx0 - 2 - 4 * SQUISH_CONSTANT4
                dx_ext1 = dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT4
            else:
                xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb
                dx_ext0 = dx_ext1 = dx_ext2 = dx0 - 4 * SQUISH_CONSTANT4

            if (c & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT4
                if (c & 0x01) != 0:
                    ysv_ext1 += 1
                    dy_ext1 -= 1
                else:
                    ysv_ext0 += 1
                    dy_ext0 -= 1

            else:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                dy_ext0 = dy_ext1 = dy_ext2 = dy0 - 4 * SQUISH_CONSTANT4

            if (c & 0x04) != 0:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT4
                if (c & 0x03) != 0x03:
                    if (c & 0x03) == 0:
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                    else:
                        zsv_ext1 += 1
                        dz_ext1 -= 1

                else:
                    zsv_ext2 += 1
                    dz_ext2 -= 1

            else:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                dz_ext0 = dz_ext1 = dz_ext2 = dz0 - 4 * SQUISH_CONSTANT4

            if (c & 0x08) != 0:
                wsv_ext0 = wsv_ext1 = wsb + 1
                wsv_ext2 = wsb + 2
                dw_ext0 = dw_ext1 = dw0 - 1 - 4 * SQUISH_CONSTANT4
                dw_ext2 = dw0 - 2 - 4 * SQUISH_CONSTANT4
            else:
                wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb
                dw_ext0 = dw_ext1 = dw_ext2 = dw0 - 4 * SQUISH_CONSTANT4

        else:
            c = a_po & b_po

            if (c & 0x01) != 0:
                xsv_ext0 = xsv_ext2 = xsb + 1
                xsv_ext1 = xsb + 2
                dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT4
                dx_ext1 = dx0 - 2 - 3 * SQUISH_CONSTANT4
                dx_ext2 = dx0 - 1 - 3 * SQUISH_CONSTANT4
            else:
                xsv_ext0 = xsv_ext1 = xsv_ext2 = xsb
                dx_ext0 = dx0 - 2 * SQUISH_CONSTANT4
                dx_ext1 = dx_ext2 = dx0 - 3 * SQUISH_CONSTANT4

            if (c & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb + 1
                dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT4
                dy_ext1 = dy_ext2 = dy0 - 1 - 3 * SQUISH_CONSTANT4
                if (c & 0x01) != 0:
                    ysv_ext2 += 1
                    dy_ext2 -= 1
                else:
                    ysv_ext1 += 1
                    dy_ext1 -= 1

            else:
                ysv_ext0 = ysv_ext1 = ysv_ext2 = ysb
                dy_ext0 = dy0 - 2 * SQUISH_CONSTANT4
                dy_ext1 = dy_ext2 = dy0 - 3 * SQUISH_CONSTANT4

            if (c & 0x04) != 0:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb + 1
                dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT4
                dz_ext1 = dz_ext2 = dz0 - 1 - 3 * SQUISH_CONSTANT4
                if (c & 0x03) != 0:
                    zsv_ext2 += 1
                    dz_ext2 -= 1
                else:
                    zsv_ext1 += 1
                    dz_ext1 -= 1

            else:
                zsv_ext0 = zsv_ext1 = zsv_ext2 = zsb
                dz_ext0 = dz0 - 2 * SQUISH_CONSTANT4
                dz_ext1 = dz_ext2 = dz0 - 3 * SQUISH_CONSTANT4

            if (c & 0x08) != 0:
                wsv_ext0 = wsv_ext1 = wsb + 1
                wsv_ext2 = wsb + 2
                dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT4
                dw_ext1 = dw0 - 1 - 3 * SQUISH_CONSTANT4
                dw_ext2 = dw0 - 2 - 3 * SQUISH_CONSTANT4
            else:
                wsv_ext0 = wsv_ext1 = wsv_ext2 = wsb
                dw_ext0 = dw0 - 2 * SQUISH_CONSTANT4
                dw_ext1 = dw_ext2 = dw0 - 3 * SQUISH_CONSTANT4

        dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT4
        dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT4
        dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT4
        dw4 = dw0 - 3 * SQUISH_CONSTANT4
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 1, wsb + 0, dx4, dy4, dz4, dw4)

        dx3 = dx4
        dy3 = dy4
        dz3 = dz0 - 3 * SQUISH_CONSTANT4
        dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT4
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 0, wsb + 1, dx3, dy3, dz3, dw3)

        dx2 = dx4
        dy2 = dy0 - 3 * SQUISH_CONSTANT4
        dz2 = dz4
        dw2 = dw3
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 1, wsb + 1, dx2, dy2, dz2, dw2)

        dx1 = dx0 - 3 * SQUISH_CONSTANT4
        dz1 = dz4
        dy1 = dy4
        dw1 = dw3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 1, wsb + 1, dx1, dy1, dz1, dw1)

        dx0 = dx0 - 1 - 4 * SQUISH_CONSTANT4
        dy0 = dy0 - 1 - 4 * SQUISH_CONSTANT4
        dz0 = dz0 - 1 - 4 * SQUISH_CONSTANT4
        dw0 = dw0 - 1 - 4 * SQUISH_CONSTANT4
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 1, wsb + 1, dx0, dy0, dz0, dw0)

    elif in_sum <= 2:
        a_is_bigger_side = True
        b_is_bigger_side = True

        if xins + yins > zins + wins:
            a_score = xins + yins
            a_po = 0x03
        else:
            a_score = zins + wins
            a_po = 0x0C

        if xins + zins > yins + wins:
            b_score = xins + zins
            b_po = 0x05
        else:
            b_score = yins + wins
            b_po = 0x0A

        if xins + wins > yins + zins:
            score = xins + wins
            if a_score >= b_score and score > b_score:
                b_score = score
                b_po = 0x09
            elif a_score < b_score and score > a_score:
                a_score = score
                a_po = 0x09

        else:
            score = yins + zins
            if a_score >= b_score and score > b_score:
                b_score = score
                b_po = 0x06
            elif a_score < b_score and score > a_score:
                a_score = score
                a_po = 0x06

        p1 = 2 - in_sum + xins
        if a_score >= b_score and p1 > b_score:
            b_score = p1
            b_po = 0x01
            b_is_bigger_side = False
        elif a_score < b_score and p1 > a_score:
            a_score = p1
            a_po = 0x01
            a_is_bigger_side = False

        p2 = 2 - in_sum + yins
        if a_score >= b_score and p2 > b_score:
            b_score = p2
            b_po = 0x02
            b_is_bigger_side = False
        elif a_score < b_score and p2 > a_score:
            a_score = p2
            a_po = 0x02
            a_is_bigger_side = False

        p3 = 2 - in_sum + zins
        if a_score >= b_score and p3 > b_score:
            b_score = p3
            b_po = 0x04
            b_is_bigger_side = False
        elif a_score < b_score and p3 > a_score:
            a_score = p3
            a_po = 0x04
            a_is_bigger_side = False

        p4 = 2 - in_sum + wins
        if a_score >= b_score and p4 > b_score:
            b_po = 0x08
            b_is_bigger_side = False
        elif a_score < b_score and p4 > a_score:
            a_po = 0x08
            a_is_bigger_side = False

        if a_is_bigger_side == b_is_bigger_side:
            if a_is_bigger_side:
                c1 = a_po | b_po
                c2 = a_po & b_po
                if (c1 & 0x01) == 0:
                    xsv_ext0 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 3 * SQUISH_CONSTANT4
                    dx_ext1 = dx0 + 1 - 2 * SQUISH_CONSTANT4
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT4
                    dx_ext1 = dx0 - 1 - 2 * SQUISH_CONSTANT4

                if (c1 & 0x02) == 0:
                    ysv_ext0 = ysb
                    ysv_ext1 = ysb - 1
                    dy_ext0 = dy0 - 3 * SQUISH_CONSTANT4
                    dy_ext1 = dy0 + 1 - 2 * SQUISH_CONSTANT4
                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT4
                    dy_ext1 = dy0 - 1 - 2 * SQUISH_CONSTANT4

                if (c1 & 0x04) == 0:
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0 - 3 * SQUISH_CONSTANT4
                    dz_ext1 = dz0 + 1 - 2 * SQUISH_CONSTANT4
                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT4
                    dz_ext1 = dz0 - 1 - 2 * SQUISH_CONSTANT4

                if (c1 & 0x08) == 0:
                    wsv_ext0 = wsb
                    wsv_ext1 = wsb - 1
                    dw_ext0 = dw0 - 3 * SQUISH_CONSTANT4
                    dw_ext1 = dw0 + 1 - 2 * SQUISH_CONSTANT4
                else:
                    wsv_ext0 = wsv_ext1 = wsb + 1
                    dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT4
                    dw_ext1 = dw0 - 1 - 2 * SQUISH_CONSTANT4

                xsv_ext2 = xsb
                ysv_ext2 = ysb
                zsv_ext2 = zsb
                wsv_ext2 = wsb
                dx_ext2 = dx0 - 2 * SQUISH_CONSTANT4
                dy_ext2 = dy0 - 2 * SQUISH_CONSTANT4
                dz_ext2 = dz0 - 2 * SQUISH_CONSTANT4
                dw_ext2 = dw0 - 2 * SQUISH_CONSTANT4
                if (c2 & 0x01) != 0:
                    xsv_ext2 += 2
                    dx_ext2 -= 2
                elif (c2 & 0x02) != 0:
                    ysv_ext2 += 2
                    dy_ext2 -= 2
                elif (c2 & 0x04) != 0:
                    zsv_ext2 += 2
                    dz_ext2 -= 2
                else:
                    wsv_ext2 += 2
                    dw_ext2 -= 2

            else:

                xsv_ext2 = xsb
                ysv_ext2 = ysb
                zsv_ext2 = zsb
                wsv_ext2 = wsb
                dx_ext2 = dx0
                dy_ext2 = dy0
                dz_ext2 = dz0
                dw_ext2 = dw0

                c = a_po | b_po

                if (c & 0x01) == 0:
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1 - SQUISH_CONSTANT4
                    dx_ext1 = dx0 - SQUISH_CONSTANT4
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx_ext1 = dx0 - 1 - SQUISH_CONSTANT4

                if (c & 0x02) == 0:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0 - SQUISH_CONSTANT4
                    if (c & 0x01) == 0x01:
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    else:
                        ysv_ext1 -= 1
                        dy_ext1 += 1

                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1 - SQUISH_CONSTANT4

                if (c & 0x04) == 0:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz_ext1 = dz0 - SQUISH_CONSTANT4
                    if (c & 0x03) == 0x03:
                        zsv_ext0 -= 1
                        dz_ext0 += 1
                    else:
                        zsv_ext1 -= 1
                        dz_ext1 += 1

                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1 - SQUISH_CONSTANT4

                if (c & 0x08) == 0:
                    wsv_ext0 = wsb
                    wsv_ext1 = wsb - 1
                    dw_ext0 = dw0 - SQUISH_CONSTANT4
                    dw_ext1 = dw0 + 1 - SQUISH_CONSTANT4
                else:
                    wsv_ext0 = wsv_ext1 = wsb + 1
                    dw_ext0 = dw_ext1 = dw0 - 1 - SQUISH_CONSTANT4

        else:
            if a_is_bigger_side:
                c1 = a_po
                c2 = b_po
            else:
                c1 = b_po
                c2 = a_po

            if (c1 & 0x01) == 0:
                xsv_ext0 = xsb - 1
                xsv_ext1 = xsb
                dx_ext0 = dx0 + 1 - SQUISH_CONSTANT4
                dx_ext1 = dx0 - SQUISH_CONSTANT4
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx_ext1 = dx0 - 1 - SQUISH_CONSTANT4

            if (c1 & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0 - SQUISH_CONSTANT4
                if (c1 & 0x01) == 0x01:
                    ysv_ext0 -= 1
                    dy_ext0 += 1
                else:
                    ysv_ext1 -= 1
                    dy_ext1 += 1

            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1 - SQUISH_CONSTANT4

            if (c1 & 0x04) == 0:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz_ext1 = dz0 - SQUISH_CONSTANT4
                if (c1 & 0x03) == 0x03:
                    zsv_ext0 -= 1
                    dz_ext0 += 1
                else:
                    zsv_ext1 -= 1
                    dz_ext1 += 1

            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz_ext1 = dz0 - 1 - SQUISH_CONSTANT4

            if (c1 & 0x08) == 0:
                wsv_ext0 = wsb
                wsv_ext1 = wsb - 1
                dw_ext0 = dw0 - SQUISH_CONSTANT4
                dw_ext1 = dw0 + 1 - SQUISH_CONSTANT4
            else:
                wsv_ext0 = wsv_ext1 = wsb + 1
                dw_ext0 = dw_ext1 = dw0 - 1 - SQUISH_CONSTANT4

            xsv_ext2 = xsb
            ysv_ext2 = ysb
            zsv_ext2 = zsb
            wsv_ext2 = wsb
            dx_ext2 = dx0 - 2 * SQUISH_CONSTANT4
            dy_ext2 = dy0 - 2 * SQUISH_CONSTANT4
            dz_ext2 = dz0 - 2 * SQUISH_CONSTANT4
            dw_ext2 = dw0 - 2 * SQUISH_CONSTANT4
            if (c2 & 0x01) != 0:
                xsv_ext2 += 2
                dx_ext2 -= 2
            elif (c2 & 0x02) != 0:
                ysv_ext2 += 2
                dy_ext2 -= 2
            elif (c2 & 0x04) != 0:
                zsv_ext2 += 2
                dz_ext2 -= 2
            else:
                wsv_ext2 += 2
                dw_ext2 -= 2

        dx1 = dx0 - 1 - SQUISH_CONSTANT4
        dy1 = dy0 - 0 - SQUISH_CONSTANT4
        dz1 = dz0 - 0 - SQUISH_CONSTANT4
        dw1 = dw0 - 0 - SQUISH_CONSTANT4
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 0, wsb + 0, dx1, dy1, dz1, dw1)

        dx2 = dx0 - 0 - SQUISH_CONSTANT4
        dy2 = dy0 - 1 - SQUISH_CONSTANT4
        dz2 = dz1
        dw2 = dw1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 0, wsb + 0, dx2, dy2, dz2, dw2)

        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT4
        dw3 = dw1
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 1, wsb + 0, dx3, dy3, dz3, dw3)

        dx4 = dx2
        dy4 = dy1
        dz4 = dz1
        dw4 = dw0 - 1 - SQUISH_CONSTANT4
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 0, wsb + 1, dx4, dy4, dz4, dw4)

        dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
        if attn5 > 0:
            attn5 *= attn5
            value += attn5 * attn5 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 0, wsb + 0, dx5, dy5, dz5, dw5)

        dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
        if attn6 > 0:
            attn6 *= attn6
            value += attn6 * attn6 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 1, wsb + 0, dx6, dy6, dz6, dw6)

        dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
        if attn7 > 0:
            attn7 *= attn7
            value += attn7 * attn7 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 0, wsb + 1, dx7, dy7, dz7, dw7)

        dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
        if attn8 > 0:
            attn8 *= attn8
            value += attn8 * attn8 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 1, wsb + 0, dx8, dy8, dz8, dw8)

        dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
        if attn9 > 0:
            attn9 *= attn9
            value += attn9 * attn9 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 0, wsb + 1, dx9, dy9, dz9, dw9)

        dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
        if attn10 > 0:
            attn10 *= attn10
            value += attn10 * attn10 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 1, wsb + 1, dx10, dy10, dz10, dw10)

    else:
        a_is_bigger_side = True
        b_is_bigger_side = True

        if xins + yins < zins + wins:
            a_score = xins + yins
            a_po = 0x0C
        else:
            a_score = zins + wins
            a_po = 0x03

        if xins + zins < yins + wins:
            b_score = xins + zins
            b_po = 0x0A
        else:
            b_score = yins + wins
            b_po = 0x05

        if xins + wins < yins + zins:
            score = xins + wins
            if a_score <= b_score and score < b_score:
                b_score = score
                b_po = 0x06
            elif a_score > b_score and score < a_score:
                a_score = score
                a_po = 0x06

        else:
            score = yins + zins
            if a_score <= b_score and score < b_score:
                b_score = score
                b_po = 0x09
            elif a_score > b_score and score < a_score:
                a_score = score
                a_po = 0x09

        p1 = 3 - in_sum + xins
        if a_score <= b_score and p1 < b_score:
            b_score = p1
            b_po = 0x0E
            b_is_bigger_side = False
        elif a_score > b_score and p1 < a_score:
            a_score = p1
            a_po = 0x0E
            a_is_bigger_side = False

        p2 = 3 - in_sum + yins
        if a_score <= b_score and p2 < b_score:
            b_score = p2
            b_po = 0x0D
            b_is_bigger_side = False
        elif a_score > b_score and p2 < a_score:
            a_score = p2
            a_po = 0x0D
            a_is_bigger_side = False

        p3 = 3 - in_sum + zins
        if a_score <= b_score and p3 < b_score:
            b_score = p3
            b_po = 0x0B
            b_is_bigger_side = False
        elif a_score > b_score and p3 < a_score:
            a_score = p3
            a_po = 0x0B
            a_is_bigger_side = False

        p4 = 3 - in_sum + wins
        if a_score <= b_score and p4 < b_score:
            b_po = 0x07
            b_is_bigger_side = False
        elif a_score > b_score and p4 < a_score:
            a_po = 0x07
            a_is_bigger_side = False

        if a_is_bigger_side == b_is_bigger_side:
            if a_is_bigger_side:
                c1 = a_po & b_po
                c2 = a_po | b_po

                xsv_ext0 = xsv_ext1 = xsb
                ysv_ext0 = ysv_ext1 = ysb
                zsv_ext0 = zsv_ext1 = zsb
                wsv_ext0 = wsv_ext1 = wsb
                dx_ext0 = dx0 - SQUISH_CONSTANT4
                dy_ext0 = dy0 - SQUISH_CONSTANT4
                dz_ext0 = dz0 - SQUISH_CONSTANT4
                dw_ext0 = dw0 - SQUISH_CONSTANT4
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT4
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT4
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT4
                dw_ext1 = dw0 - 2 * SQUISH_CONSTANT4
                if (c1 & 0x01) != 0:
                    xsv_ext0 += 1
                    dx_ext0 -= 1
                    xsv_ext1 += 2
                    dx_ext1 -= 2
                elif (c1 & 0x02) != 0:
                    ysv_ext0 += 1
                    dy_ext0 -= 1
                    ysv_ext1 += 2
                    dy_ext1 -= 2
                elif (c1 & 0x04) != 0:
                    zsv_ext0 += 1
                    dz_ext0 -= 1
                    zsv_ext1 += 2
                    dz_ext1 -= 2
                else:
                    wsv_ext0 += 1
                    dw_ext0 -= 1
                    wsv_ext1 += 2
                    dw_ext1 -= 2

                xsv_ext2 = xsb + 1
                ysv_ext2 = ysb + 1
                zsv_ext2 = zsb + 1
                wsv_ext2 = wsb + 1
                dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT4
                dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT4
                dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT4
                dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT4
                if (c2 & 0x01) == 0:
                    xsv_ext2 -= 2
                    dx_ext2 += 2
                elif (c2 & 0x02) == 0:
                    ysv_ext2 -= 2
                    dy_ext2 += 2
                elif (c2 & 0x04) == 0:
                    zsv_ext2 -= 2
                    dz_ext2 += 2
                else:
                    wsv_ext2 -= 2
                    dw_ext2 += 2

            else:

                xsv_ext2 = xsb + 1
                ysv_ext2 = ysb + 1
                zsv_ext2 = zsb + 1
                wsv_ext2 = wsb + 1
                dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT4
                dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT4
                dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT4
                dw_ext2 = dw0 - 1 - 4 * SQUISH_CONSTANT4

                c = a_po & b_po
                if (c & 0x01) != 0:
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT4
                    dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT4
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT4

                if (c & 0x02) != 0:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT4
                    if (c & 0x01) == 0:
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                    else:
                        ysv_ext1 += 1
                        dy_ext1 -= 1

                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT4

                if (c & 0x04) != 0:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT4
                    if (c & 0x03) == 0:
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                    else:
                        zsv_ext1 += 1
                        dz_ext1 -= 1

                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT4

                if (c & 0x08) != 0:
                    wsv_ext0 = wsb + 1
                    wsv_ext1 = wsb + 2
                    dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT4
                    dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT4
                else:
                    wsv_ext0 = wsv_ext1 = wsb
                    dw_ext0 = dw_ext1 = dw0 - 3 * SQUISH_CONSTANT4

        else:
            if a_is_bigger_side:
                c1 = a_po
                c2 = b_po
            else:
                c1 = b_po
                c2 = a_po

            if (c1 & 0x01) != 0:
                xsv_ext0 = xsb + 2
                xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT4
                dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT4
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT4

            if (c1 & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT4
                if (c1 & 0x01) == 0:
                    ysv_ext0 += 1
                    dy_ext0 -= 1
                else:
                    ysv_ext1 += 1
                    dy_ext1 -= 1

            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT4

            if (c1 & 0x04) != 0:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT4
                if (c1 & 0x03) == 0:
                    zsv_ext0 += 1
                    dz_ext0 -= 1
                else:
                    zsv_ext1 += 1
                    dz_ext1 -= 1

            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT4

            if (c1 & 0x08) != 0:
                wsv_ext0 = wsb + 1
                wsv_ext1 = wsb + 2
                dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT4
                dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT4
            else:
                wsv_ext0 = wsv_ext1 = wsb
                dw_ext0 = dw_ext1 = dw0 - 3 * SQUISH_CONSTANT4

            xsv_ext2 = xsb + 1
            ysv_ext2 = ysb + 1
            zsv_ext2 = zsb + 1
            wsv_ext2 = wsb + 1
            dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT4
            dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT4
            dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT4
            dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT4
            if (c2 & 0x01) == 0:
                xsv_ext2 -= 2
                dx_ext2 += 2
            elif (c2 & 0x02) == 0:
                ysv_ext2 -= 2
                dy_ext2 += 2
            elif (c2 & 0x04) == 0:
                zsv_ext2 -= 2
                dz_ext2 += 2
            else:
                wsv_ext2 -= 2
                dw_ext2 += 2

        dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT4
        dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT4
        dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT4
        dw4 = dw0 - 3 * SQUISH_CONSTANT4
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 1, wsb + 0, dx4, dy4, dz4, dw4)

        dx3 = dx4
        dy3 = dy4
        dz3 = dz0 - 3 * SQUISH_CONSTANT4
        dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT4
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 0, wsb + 1, dx3, dy3, dz3, dw3)

        dx2 = dx4
        dy2 = dy0 - 3 * SQUISH_CONSTANT4
        dz2 = dz4
        dw2 = dw3
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 1, wsb + 1, dx2, dy2, dz2, dw2)

        dx1 = dx0 - 3 * SQUISH_CONSTANT4
        dz1 = dz4
        dy1 = dy4
        dw1 = dw3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 1, wsb + 1, dx1, dy1, dz1, dw1)

        dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
        if attn5 > 0:
            attn5 *= attn5
            value += attn5 * attn5 * extrapolate4(perm, gradients, xsb + 1, ysb + 1, zsb + 0, wsb + 0, dx5, dy5, dz5, dw5)

        dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
        if attn6 > 0:
            attn6 *= attn6
            value += attn6 * attn6 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 1, wsb + 0, dx6, dy6, dz6, dw6)

        dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT4
        dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
        if attn7 > 0:
            attn7 *= attn7
            value += attn7 * attn7 * extrapolate4(perm, gradients, xsb + 1, ysb + 0, zsb + 0, wsb + 1, dx7, dy7, dz7, dw7)

        dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT4
        attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
        if attn8 > 0:
            attn8 *= attn8
            value += attn8 * attn8 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 1, wsb + 0, dx8, dy8, dz8, dw8)

        dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT4
        dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT4
        dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
        if attn9 > 0:
            attn9 *= attn9
            value += attn9 * attn9 * extrapolate4(perm, gradients, xsb + 0, ysb + 1, zsb + 0, wsb + 1, dx9, dy9, dz9, dw9)

        dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT4
        dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT4
        dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT4
        dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT4
        attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
        if attn10 > 0:
            attn10 *= attn10
            value += attn10 * attn10 * extrapolate4(perm, gradients, xsb + 0, ysb + 0, zsb + 1, wsb + 1, dx10, dy10, dz10, dw10)

    attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0 - dw_ext0 * dw_ext0
    if attn_ext0 > 0:
        attn_ext0 *= attn_ext0
        value += (
            attn_ext0
            * attn_ext0
            * extrapolate4(perm, gradients, xsv_ext0, ysv_ext0, zsv_ext0, wsv_ext0, dx_ext0, dy_ext0, dz_ext0, dw_ext0)
        )

    attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1 - dw_ext1 * dw_ext1
    if attn_ext1 > 0:
        attn_ext1 *= attn_ext1
        value += (
            attn_ext1
            * attn_ext1
            * extrapolate4(perm, gradients, xsv_ext1, ysv_ext1, zsv_ext1, wsv_ext1, dx_ext1, dy_ext1, dz_ext1, dw_ext1)
        )

    attn_ext2 = 2 - dx_ext2 * dx_ext2 - dy_ext2 * dy_ext2 - dz_ext2 * dz_ext2 - dw_ext2 * dw_ext2
    if attn_ext2 > 0:
        attn_ext2 *= attn_ext2
        value += (
            attn_ext2
            * attn_ext2
            * extrapolate4(perm, gradients, xsv_ext2, ysv_ext2, zsv_ext2, wsv_ext2, dx_ext2, dy_ext2, dz_ext2, dw_ext2)
        )

    return value / NORM_CONSTANT4
