import numpy as np


def rmsd_kabsch(p, q):
    A = np.dot(q.T, p)

    V, S, W = np.linalg.svd(A)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0.:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)

    rotated = np.dot(p, U.T)

    return np.sqrt(np.sum((rotated - q) ** 2) / len(p))


def rmsd_qcp(src, dst):
    src = np.array(src)

    n = src.shape[0]

    rmsd = np.zeros(len(dst), dtype=np.float64)
    rmsd[:] = np.inf

    valid = [len(segment) == n for segment in dst]

    dst = np.array([segment for i, segment in enumerate(dst) if valid[i]])
    N = dst.shape[0] #max(src.shape[0], dst.shape[0])

    if len(dst) == 0:
        return rmsd

    M = np.matmul(np.swapaxes(dst, 1, 2), src)

    xx = M[:, 0, 0]
    xy = M[:, 0, 1]
    yx = M[:, 1, 0]
    yy = M[:, 1, 1]

    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx ** 2

    xx_yy_u = xx_yy + np.sqrt(xy_yx_2 + xx_yy ** 2)
    xx_yy_u_2 = xx_yy_u ** 2

    denom = xx_yy_u_2 + xy_yx_2

    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom

    U = np.zeros((N, 2, 2))
    U[:, 0, 0] = Uxx
    U[:, 1, 1] = Uxx
    U[:, 0, 1] = -Uxy
    U[:, 1, 0] = Uxy

    rmsd[valid] = np.sqrt(np.sum((np.matmul(src, U) - dst) ** 2, axis=(1, 2)) / src.shape[1])
    return rmsd


def batch_rmsd_qcp(src, dst):
    M = np.einsum('aji,bjk', src, dst)

    xx = M[..., 0, 0]
    xy = M[..., 0, 1]
    yx = M[..., 1, 0]
    yy = M[..., 1, 1]

    xx_yy = xx + yy
    xy_yx = xy - yx
    xy_yx_2 = xy_yx ** 2

    xx_yy_u = xx_yy + np.sqrt(xy_yx_2 + xx_yy ** 2)
    xx_yy_u_2 = xx_yy_u ** 2

    denom = xx_yy_u_2 + xy_yx_2

    Uxx = (xx_yy_u_2 - xy_yx_2) / denom
    Uxy = 2 * xy_yx * xx_yy_u / denom

    U = np.zeros(M.shape[:-2] + (2, 2))
    U[..., 0, 0] = Uxx
    U[..., 1, 1] = Uxx
    U[..., 1, 0] = -Uxy
    U[..., 0, 1] = Uxy

    rmsd = np.sqrt(np.sum((np.matmul(src[:, None], U) - dst[None]) ** 2, axis=(-1, -2)) / src.shape[-1])
    return rmsd
