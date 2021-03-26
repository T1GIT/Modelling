import numpy as np


def intensity(lam: int, m: int, _m: int, n: int) -> np.ndarray:
    size = m + n + 1
    P = np.zeros((size, size))
    for i in range(0, m + n):
        P[i][i + 1] = lam
        P[i + 1][i] = (i + 1) * _m if i < m else m * _m
    return P


def last_state(P: np.ndarray) -> np.ndarray:
    size = len(P)
    M = P.T - np.diag(P.sum(1))
    M[-1].fill(1)
    B = np.zeros((size, 1))
    B[-1][0] = 1
    return (np.linalg.inv(M) @ B)[:, 0]
