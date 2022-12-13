# Taken from https://github.com/orangeduck/Motion-Matching (modified)

import numpy as np


def _fast_cross(a, b):
    """ 三维向量叉乘 """
    o = np.empty(np.broadcast(a, b).shape)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o


def eye(shape):
    """ 生成指定shape的四元数矩阵 """
    return np.ones(list(shape) + [4], dtype=np.float32) * np.asarray([1, 0, 0, 0], dtype=np.float32)


def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))


def normalize(x, eps=1e-8):
    return x / (length(x)[..., np.newaxis] + eps)


def abs(x):
    return np.where(x[..., 0:1] > 0.0, x, -x)


def from_angle_axis(angle, axis):
    """ 从轴和旋转角度生成四元数 """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


def to_xform(x):
    """ 转成 3x3 旋转矩阵 """
    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., np.newaxis, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., np.newaxis, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., np.newaxis, :],
    ], axis=-2)


def from_euler(e, order='zyx'):
    """ 从欧拉角生成四元数 """
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))


def from_xform(ts, eps=1e-10):
    """ 从旋转矩阵生成四元数 """
    qs = np.empty_like(ts[..., :1, 0].repeat(4, axis=-1))

    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[..., np.newaxis],
        (s * (ts[..., 2, 1] - ts[..., 1, 2]))[..., np.newaxis],
        (s * (ts[..., 0, 2] - ts[..., 2, 0]))[..., np.newaxis],
        (s * (ts[..., 1, 0] - ts[..., 0, 1]))[..., np.newaxis]
    ], axis=-1), qs)

    c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c0)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)[..., np.newaxis],
        (s0 * 0.25)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)[..., np.newaxis]
    ], axis=-1), qs)

    c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    qs = np.where(((t <= 0) & c1)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)[..., np.newaxis],
        ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)[..., np.newaxis],
        (s1 * 0.25)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)[..., np.newaxis]
    ], axis=-1), qs)

    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    qs = np.where(((t <= 0) & c2)[..., np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)[..., np.newaxis],
        ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)[..., np.newaxis],
        ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)[..., np.newaxis],
        (s2 * 0.25)[..., np.newaxis]
    ], axis=-1), qs)

    return qs


def inv(q):
    """ 四元数的共轭 """
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q


def mul(x, y):
    """ 四元数乘法 """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)


def inv_mul(x, y):
    return mul(inv(x), y)


def mul_inv(x, y):
    return mul(x, inv(y))


def mul_vec(q, x):
    """ 四元数乘三维向量 """
    # 下面的式子可以手推一下，很有意思
    t = 2.0 * _fast_cross(q[..., 1:], x)
    # return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)
    return x + q[..., :1] * t + _fast_cross(q[..., 1:], t)


def inv_mul_vec(q, x):
    """ 四元数的共轭乘三维向量 """
    return mul_vec(inv(q), x)


def unroll(x):
    """ 让相邻两帧之间的旋转角小于180 """
    # 看原作者的文章，好像是绘制曲线的时候，如果不做unroll会有数据跳变
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum(y[i] * y[i - 1], axis=-1)
        d1 = np.sum(-y[i] * y[i - 1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y


def between(x, y):
    """ 计算两个三维向量的夹角四元数（结果需归一化） """
    return np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis] +
        np.sum(x * y, axis=-1)[..., np.newaxis],
        _fast_cross(x, y)], axis=-1)


def log(x, eps=1e-5):
    """ 四元数的指数表示, log """
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))[..., np.newaxis]
    halfangle = np.divide(np.arctan2(length, x[..., 0:1]), length, out=np.ones_like(length), where=length > eps)
    return halfangle * x[..., 1:]


def exp(x, eps=1e-5):
    """ 四元数的指数表示, exp """
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[..., np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)


def to_scaled_angle_axis(x, eps=1e-5):
    """ 转化成旋转向量（方向为轴，长度为旋转角） """
    return 2.0 * log(x, eps)


def from_scaled_angle_axis(x, eps=1e-5):
    """ 旋转向量转化成四元数 """
    return exp(x / 2.0, eps)


def fk(lrot, lpos, parents):
    """ 局部坐标转全局坐标 """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def ik(grot, gpos, parents):
    """ 全局坐标转局部坐标 """
    return (
        np.concatenate([
            grot[..., :1, :],
            mul(inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            mul_vec(
                inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2))


def fk_vel(lrot, lpos, lvel, lang, parents):
    """ 局部坐标转全局坐标，包括速度和角速度 """
    gp, gr, gv, ga = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))
        gv.append(mul_vec(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[..., i:i + 1, :])) +
                  gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[..., i:i + 1, :]) + ga[parents[i]])

    return (
        np.concatenate(gr, axis=-2),
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))


def nlerp(q: np.ndarray, p: np.ndarray, alpha, axis=0):
    """ 线性插值，axis 指示 alpha 对齐到 q 的哪一轴，负数值是指在左侧新加一轴而不是倒数的轴数 """

    alpha = np.asarray(alpha)

    assert q.shape == p.shape
    assert q.ndim > axis + alpha.ndim

    q = q.copy()
    d = np.sum(q * p, axis=-1)
    q[d < 0] = -q[d < 0]

    alpha = np.moveaxis(np.array(alpha, ndmin=q.ndim - axis),
                        np.arange(-alpha.ndim, 0),
                        np.arange(0, alpha.ndim))

    return normalize(q * (1 - alpha) + p * alpha)


def slerp(q: np.ndarray, p: np.ndarray, alpha, eps=1e-5, axis=0):
    """ 球面插值，axis 指示 alpha 对齐到 q 的哪一轴，负数值是指在左侧新加一轴而不是倒数的轴数 """
    alpha = np.asarray(alpha)

    assert q.shape == p.shape
    assert q.ndim > axis + alpha.ndim

    q = q.copy()
    d = np.sum(q * p, axis=-1)
    q[d < 0] = -q[d < 0]

    alpha = np.moveaxis(np.array(alpha, ndmin=q.ndim - axis),
                        np.arange(-alpha.ndim, 0),
                        np.arange(0, alpha.ndim))

    dot = np.sum(q * p, axis=-1)[..., np.newaxis]
    theta = np.arccos(np.minimum(np.maximum(dot, -1), 1))

    return np.where(theta < eps,
                    normalize(q * (1 - alpha) + p * alpha),
                    q * np.cos(theta * alpha) + normalize(p - q * dot) * np.sin(theta * alpha))


# https://zeux.io/2015/07/23/approximating-slerp/
def slerp_approx(q: np.ndarray, p: np.ndarray, alpha, axis=0):
    """ 球面插值近似，axis 指示 alpha 对齐到 q 的哪一轴，负数值是指在左侧新加一轴而不是倒数的轴数 """
    alpha = np.asarray(alpha)

    assert q.shape == p.shape
    assert q.ndim > axis + alpha.ndim

    q = q.copy()
    d = np.sum(q * p, axis=-1)
    q[d < 0] = -q[d < 0]

    alpha = np.moveaxis(np.array(alpha, ndmin=q.ndim - axis),
                        np.arange(-alpha.ndim, 0),
                        np.arange(0, alpha.ndim))

    ca = np.sum(q * p, axis=-1)[..., np.newaxis]
    # t = q.copy()
    # t[ca < 0] = -t[ca < 0]

    d = np.abs(ca)
    a = 1.0904 + d * (-3.2452 + d * (3.55645 - d * 1.43519))
    b = 0.848013 + d * (-1.06021 + d * 0.215638)
    k = a * (alpha - 0.5) * (alpha - 0.5) + b
    oalpha = alpha + alpha * (alpha - 0.5) * (alpha - 1) * k

    return normalize(q * (1 - oalpha) + p * oalpha)


def vec_lerp(x: np.ndarray, y: np.ndarray, alpha, axis=0):
    """ 线性插值，axis 指示 alpha 对齐到 x 的哪一轴，负数值是指在左侧新加一轴而不是倒数的轴数 """

    alpha = np.asarray(alpha)

    assert x.shape == y.shape
    assert x.ndim > axis + alpha.ndim

    alpha = np.moveaxis(np.array(alpha, ndmin=x.ndim - axis),
                        np.arange(-alpha.ndim, 0),
                        np.arange(0, alpha.ndim))

    return x * (1 - alpha) + y * alpha


def ninterpolate(q, p, num):
    return nlerp(q, p, np.linspace(0, 1, num), axis=-1)


def sinterpolate(q, p, num):
    return slerp(q, p, np.linspace(0, 1, num), axis=-1)


def sinterpolate_approx(q, p, num):
    return slerp_approx(q, p, np.linspace(0, 1, num), axis=-1)
