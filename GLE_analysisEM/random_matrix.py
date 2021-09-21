"""
Somes utilities function
"""
import numpy as np


def generateRandomDefPosMat(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """Generate a random value of the A matrix"""
    # A = rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / dim_x + dim_h  # Eigenvalues of A mainly into the unit circle
    # mat = max_ev * scipy.linalg.expm(0.25 * scipy.linalg.logm(A)) + min_re_ev * np.identity(dim_x + dim_h)  # map the unit circle into the quarter disk
    # print(min_re_ev, max_ev)
    return sub_matrix_gen(dim_x=dim_x, dim_h=dim_h, rng=rng, max_ev=max_ev, min_re_ev=min_re_ev)


def sub_matrix_gen(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Build specific matrix
    """
    A = np.zeros((dim_x + dim_h, dim_x + dim_h))
    Ahh = random_clever_gen(dim_x=0, dim_h=dim_h, rng=rng, max_ev=max_ev, min_re_ev=min_re_ev)
    A[dim_x:, dim_x:] = Ahh
    A[:dim_x, :dim_x] = random_clever_gen(dim_x=dim_x, dim_h=0, rng=rng, max_ev=max_ev * 1e-2, min_re_ev=min_re_ev * 1e-2)
    min_dim = min(dim_x, dim_h)
    A[dim_x : dim_x + min_dim, :min_dim] = -np.eye(min_dim)
    A[:min_dim, dim_x : dim_x + min_dim] = np.eye(min_dim)
    return A


def random_clever_gen(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Generate random matrix with positive real part eigenvalues
    """
    notFound = True
    n = 0
    # We still had to avoid too small eigenvalues
    while notFound:
        A = max_ev * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / (dim_x + dim_h)
        # try:  # In case we are unlucky and have non diagonalisable matrix
        lamb, v = np.linalg.eig(A)
        lamb_p = 0.5 * np.abs(np.real(lamb)) + 1.0j * np.imag(lamb)
        Ap = np.real(np.matmul(v, np.matmul(np.diag(lamb_p), np.linalg.inv(v))))
        n += 1
        if np.all(np.real(np.linalg.eigvals(Ap)) > min_re_ev) and np.all(np.linalg.eigvals(Ap + Ap.T) > min_re_ev):
            notFound = False
        # except:
        #     continue
    # print(n)
    return Ap


def random_gen_bruteforce(dim_x=1, dim_h=1, rng=np.random.default_rng(), max_ev=1.0, min_re_ev=0.005):
    """
    Brute force generation of correct matrix
    """
    notFound = True
    n = 0
    while notFound:
        A = max_ev * rng.standard_normal(size=(dim_x + dim_h, dim_x + dim_h)) / (dim_x + dim_h)
        A[:dim_x, dim_x:] = 1
        n += 1
        if np.all(np.real(np.linalg.eigvals(A)) > min_re_ev) and np.all(np.linalg.eigvals(A + A.T) > min_re_ev):
            notFound = False
    # print("Brute force", n, np.linalg.eigvals(A))
    return A
