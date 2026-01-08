import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

# Hyperparameters
c = 1
d = 2

##############################
# Non Editable Region Starting #
##############################
def my_kernel(X1, Z1, X2, Z2):
    """
    Computes pairwise kernel values using:
    K~(x1,z1; x2,z2) = x1*x2 * (z1^T*z2 + 1)^2 + 1
    """

    # Convert to fast dtype and ensure correct shapes
    X1 = X1.astype(np.float32, copy=False).reshape(-1, 1)
    X2 = X2.astype(np.float32, copy=False).reshape(-1, 1)
    Z1 = Z1.astype(np.float32, copy=False)
    Z2 = Z2.astype(np.float32, copy=False)

    # Z dot products (n1 × n2)
    Gz = Z1 @ Z2.T
    Gz = (Gz + c) * (Gz + c)    

    # X outer product (n1 × n2)
    Gx = X1 * X2.T

    # final kernel 
    G = Gx * Gz + 1.0

    return G



################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################

    k = 32
    dim = k + 1

    # reshape to 33x33 rank-1 model
    W = w.reshape(dim, dim)

    # --------- RANK-1 FACTORIZATION ----------
    v = np.ones(dim)
    for _ in range(3):        # 3 iterations are enough for rank-1 matrices
        u = W @ v
        v = W.T @ u
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    sigma = u @ W @ v
    u = np.sqrt(sigma) * u
    v = np.sqrt(sigma) * v

    # sign correction
    if u.sum() < 0:
        u = -u
        v = -v

    # --------- VECTORIZED α, β RECOVERY ----------
    # u[0] = α0 , u[32] = β31
    # for 1 ≤ i ≤ 31: αi = (u[i] + u[i−1])/2 , β(i−1) = (u[i] − u[i−1])/2
    alpha = np.empty(k)
    beta = np.empty(k)
    alpha[0] = u[0]
    beta[k-1] = u[k]
    diff = u[1:k] - u[0:k-1]
    summ = u[1:k] + u[0:k-1]
    alpha[1:k] = 0.5 * summ
    beta[0:k-1] = 0.5 * diff

    # delays for first PUF (clipped to non-negative)
    a = np.maximum(alpha + beta, 0)
    b = np.maximum(-alpha - beta, 0)
    c = np.maximum(alpha - beta, 0)
    d = np.maximum(-alpha + beta, 0)

    # --------- SECOND PUF (same vectorized formula on v) ----------
    alpha2 = np.empty(k)
    beta2 = np.empty(k)
    alpha2[0] = v[0]
    beta2[k-1] = v[k]
    diff2 = v[1:k] - v[0:k-1]
    summ2 = v[1:k] + v[0:k-1]
    alpha2[1:k] = 0.5 * summ2
    beta2[0:k-1] = 0.5 * diff2

    p = np.maximum(alpha2 + beta2, 0)
    q = np.maximum(-alpha2 - beta2, 0)
    r = np.maximum(alpha2 - beta2, 0)
    s = np.maximum(-alpha2 + beta2, 0)

    return a, b, c, d, p, q, r, s