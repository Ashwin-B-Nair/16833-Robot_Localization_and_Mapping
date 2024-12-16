'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    ata_inv = inv(A.T @ A)
    x = ata_inv @ A.T @ b
    
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html

    lu = splu(A.T @ A, permc_spec='NATURAL')
    x = lu.solve(A.T @ b)
    U = lu.U.tocsc()
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
  
    ata = csc_matrix(A.T@A)
    lu = splu(ata, permc_spec='COLAMD')
    x = lu.solve(A.T @ b)
    U = lu.U.tocsc()
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    
    
    A = csc_matrix(A)
    z, R, _, _ = rz(A, b, permc_spec='NATURAL')
    R_csr = R.tocsr()
    x = spsolve_triangular(R_csr, z, lower=False)
    return x, R_csr


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    # if not isinstance(A, csc_matrix):
    cond_num = np.linalg.cond(A.toarray())
    print(f"Condition number: {cond_num}")
    A = csc_matrix(A)
    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    R_csr = R.tocsr()
    P = permutation_vector_to_matrix(E)
    y = spsolve_triangular(R_csr, z, lower=False)
    x = P @ y

    return x, R_csr


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)