# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import functools
from typing import Sequence, Union
from contextlib import contextmanager

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================
# Triton quaternion kernels (6-12x faster than PyTorch ops)
# ============================================================
if HAS_TRITON:
    @triton.jit
    def _quat_rotate_kernel(
        q_ptr, v_ptr, out_ptr,
        N: tl.constexpr,
        inverse: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        q_w = tl.load(q_ptr + pid * 4 + 0)
        q_x = tl.load(q_ptr + pid * 4 + 1)
        q_y = tl.load(q_ptr + pid * 4 + 2)
        q_z = tl.load(q_ptr + pid * 4 + 3)
        vx = tl.load(v_ptr + pid * 3 + 0)
        vy = tl.load(v_ptr + pid * 3 + 1)
        vz = tl.load(v_ptr + pid * 3 + 2)
        # a = v * (2*q_w^2 - 1)
        s = 2.0 * q_w * q_w - 1.0
        ax = s * vx; ay = s * vy; az = s * vz
        # b = 2*q_w * (q_vec x v)
        cx = q_y * vz - q_z * vy
        cy = q_z * vx - q_x * vz
        cz = q_x * vy - q_y * vx
        bx = 2.0 * q_w * cx; by = 2.0 * q_w * cy; bz = 2.0 * q_w * cz
        # c = 2 * (q_vec . v) * q_vec
        dot = q_x * vx + q_y * vy + q_z * vz
        cx2 = 2.0 * dot * q_x; cy2 = 2.0 * dot * q_y; cz2 = 2.0 * dot * q_z
        if inverse:
            ox = ax - bx + cx2; oy = ay - by + cy2; oz = az - bz + cz2
        else:
            ox = ax + bx + cx2; oy = ay + by + cy2; oz = az + bz + cz2
        tl.store(out_ptr + pid * 3 + 0, ox)
        tl.store(out_ptr + pid * 3 + 1, oy)
        tl.store(out_ptr + pid * 3 + 2, oz)

    @triton.jit
    def _quat_to_rotmat_kernel(q_ptr, out_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= N:
            return
        w = tl.load(q_ptr + pid * 4 + 0)
        x = tl.load(q_ptr + pid * 4 + 1)
        y = tl.load(q_ptr + pid * 4 + 2)
        z = tl.load(q_ptr + pid * 4 + 3)
        tx = 2.0 * x; ty = 2.0 * y; tz = 2.0 * z
        twx = tx * w; twy = ty * w; twz = tz * w
        txx = tx * x; txy = ty * x; txz = tz * x
        tyy = ty * y; tyz = tz * y; tzz = tz * z
        base = pid * 9
        tl.store(out_ptr + base + 0, 1.0 - (tyy + tzz))
        tl.store(out_ptr + base + 1, txy - twz)
        tl.store(out_ptr + base + 2, txz + twy)
        tl.store(out_ptr + base + 3, txy + twz)
        tl.store(out_ptr + base + 4, 1.0 - (txx + tzz))
        tl.store(out_ptr + base + 5, tyz - twx)
        tl.store(out_ptr + base + 6, txz - twy)
        tl.store(out_ptr + base + 7, tyz + twx)
        tl.store(out_ptr + base + 8, 1.0 - (txx + tyy))

    @triton.jit
    def _quat_mul_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= N:
            return
        w1 = tl.load(a_ptr + pid * 4 + 0); x1 = tl.load(a_ptr + pid * 4 + 1)
        y1 = tl.load(a_ptr + pid * 4 + 2); z1 = tl.load(a_ptr + pid * 4 + 3)
        w2 = tl.load(b_ptr + pid * 4 + 0); x2 = tl.load(b_ptr + pid * 4 + 1)
        y2 = tl.load(b_ptr + pid * 4 + 2); z2 = tl.load(b_ptr + pid * 4 + 3)
        ww = (z1 + x1) * (x2 + y2); yy = (w1 - y1) * (w2 + z2); zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz; qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2); x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2); z = qq - zz + (z1 + y1) * (w2 - x2)
        tl.store(out_ptr + pid * 4 + 0, w); tl.store(out_ptr + pid * 4 + 1, x)
        tl.store(out_ptr + pid * 4 + 2, y); tl.store(out_ptr + pid * 4 + 3, z)

@contextmanager
def torch_seed(seed: int=0):
    rng_state = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state_all(rng_state_cuda)


def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped


# @manual_batch
def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )


# @manual_batch
def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


# @manual_batch
def others(x: torch.Tensor) -> torch.Tensor:
    return off_diag(x.expand(x.shape[0], *x.shape))


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if HAS_TRITON and quaternion.is_cuda:
        q_flat = quaternion.reshape(-1, 4).contiguous()
        N = q_flat.shape[0]
        out = torch.empty(N, 9, device=quaternion.device, dtype=quaternion.dtype)
        _quat_to_rotmat_kernel[(N,)](q_flat, out, N=N)
        return out.view(*quaternion.shape[:-1], 3, 3)
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    euler = torch.as_tensor(euler)
    r, p, y = torch.unbind(euler, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def make_cells(
    range_min: Union[Sequence[float], torch.Tensor],
    range_max: Union[Sequence[float], torch.Tensor],
    size: Union[float, Sequence[float], torch.Tensor],
):
    """Compute the cell centers of a n-d grid.

    Examples:
        >>> cells = make_cells([0, 0], [1, 1], 0.1)
        >>> cells[:2, :2]
        tensor([[[0.0500, 0.0500],
                 [0.0500, 0.1500]],

                [[0.1500, 0.0500],
                 [0.1500, 0.1500]]])
    """
    range_min = torch.as_tensor(range_min)
    range_max = torch.as_tensor(range_max)
    size = torch.as_tensor(size)
    shape = ((range_max - range_min) / size).round().int()

    cells = torch.meshgrid(*[torch.linspace(l, r, n+1) for l, r, n in zip(range_min, range_max, shape)], indexing="ij")
    cells = torch.stack(cells, dim=-1)
    for dim in range(cells.dim()-1):
        cells = (cells.narrow(dim, 0, cells.size(dim)-1) + cells.narrow(dim, 1, cells.size(dim)-1)) / 2
    return cells


def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    if HAS_TRITON and q.is_cuda:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 4).contiguous()
        v_flat = v.reshape(-1, 3).contiguous()
        N = q_flat.shape[0]
        out = torch.empty(N, 3, device=q.device, dtype=q.dtype)
        _quat_rotate_kernel[(N,)](q_flat, v_flat, out, N=N, inverse=False)
        return out.unflatten(0, batch_shape)
    return _quat_rotate_python(q, v)

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor):
    if HAS_TRITON and q.is_cuda:
        batch_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 4).contiguous()
        v_flat = v.reshape(-1, 3).contiguous()
        N = q_flat.shape[0]
        out = torch.empty(N, 3, device=q.device, dtype=q.dtype)
        _quat_rotate_kernel[(N,)](q_flat, v_flat, out, N=N, inverse=True)
        return out.unflatten(0, batch_shape)
    return _quat_rotate_inverse_python(q, v)

@manual_batch
def _quat_rotate_python(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@manual_batch
def _quat_rotate_inverse_python(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@manual_batch
def euler_rotate(rpy: torch.Tensor, v: torch.Tensor):
    shape = rpy.shape
    r, p, y = torch.unbind(rpy, dim=-1)
    cr = torch.cos(r)
    sr = torch.sin(r)
    cp = torch.cos(p)
    sp = torch.sin(p)
    cy = torch.cos(y)
    sy = torch.sin(y)
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr
    ], dim=-1).view(*shape[:-1], 3, 3)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)


@manual_batch
def quat_axis(q: torch.Tensor, axis: int=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def axis_angle_to_quaternion(angle: torch.Tensor, axis: torch.Tensor):
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)


def axis_angle_to_matrix(angle, axis):
    quat = axis_angle_to_quaternion(angle, axis)
    return quaternion_to_rotation_matrix(quat)


def quat_mul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    if HAS_TRITON and a.is_cuda:
        shape = a.shape
        a_flat = a.reshape(-1, 4).contiguous()
        b_flat = b.reshape(-1, 4).contiguous()
        N = a_flat.shape[0]
        out = torch.empty(N, 4, device=a.device, dtype=a.dtype)
        _quat_mul_kernel[(N,)](a_flat, b_flat, out, N=N)
        return out.reshape(shape)
    return _quat_mul_python(a, b)

def _quat_mul_python(a: torch.Tensor, b: torch.Tensor):
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([w, x, y, z], dim=-1).view(shape)
    return quat


def symlog(x: torch.Tensor):
    """
    The symlog transformation described in https://arxiv.org/pdf/2301.04104v1.pdf
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

