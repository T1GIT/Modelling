import numpy as np
import math
import random as rnd

P = np.array([
#   1       2       3       4       5       6       7       8       9       10      11      12      13      14      15
    [0.07,  0.44,   0,      0,      0.49,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0   ],  # 1
    [0,     0.14,   0,      0,      0.12,   0.74,   0,      0,      0,      0,      0,      0,      0,      0,      0   ],  # 2
    [0,     0.38,   0.16,   0,      0,      0.46,   0,      0,      0,      0,      0,      0,      0,      0,      0   ],  # 3
    [0,     0,      0.17,   0.17,   0.27,   0.23,   0.16,   0,      0,      0,      0,      0,      0,      0,      0   ],  # 4
    [0.19,  0,      0,      0.38,   0.13,   0.09,   0,      0,      0,      0,      0.21,   0,      0,      0,      0   ],  # 5
    [0,     0.23,   0.11,   0,      0,      0.21,   0.3,    0,      0,      0.03,   0.12,   0,      0,      0,      0   ],  # 6
    [0,     0.42,   0,      0.18,   0,      0.32,   0.08,   0,      0,      0,      0,      0,      0,      0,      0   ],  # 7
    [0,     0,      0,      0,      0.36,   0,      0,      0.33,   0,      0,      0,      0,      0,      0,      0.31],  # 8
    [0,     0,      0,      0,      0.09,   0.12,   0.29,   0.16,   0.19,   0.02,   0,      0,      0,      0.04,   0.09],  # 9
    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0.1,    0,      0.01,   0.38,   0.21,   0.3 ],  # 10
    [0,     0,      0,      0,      0.23,   0,      0.2,    0,      0,      0,      0.06,   0.23,   0,      0.09,   0.19],  # 11
    [0,     0,      0,      0,      0,      0,      0,      0.51,   0.42,   0,      0,      0.07,   0,      0,      0   ],  # 12
    [0,     0,      0,      0,      0,      0,      0,      0,      0.04,   0,      0.35,   0.21,   0.4,    0,      0   ],  # 13
    [0,     0,      0,      0,      0,      0,      0,      0.28,   0.32,   0.11,   0,      0,      0.25,   0.04,   0   ],  # 14
    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0.16,   0,      0,      0.75,   0.09],  # 15
])

A = np.array([0.12, 0.05, 0.11, 0.07, 0.01, 0.01, 0.11, 0.11, 0.03, 0.13, 0.08, 0.03, 0.01, 0.1, 0.03])


def print_mx(array: np.ndarray):
    NUM = 3
    for row in array:
        for cell in row:
            print(f"{str(round(cell, NUM)):<7}", end="")
        print()
    print()



SIZE = 5
TEST = np.array([[rnd.uniform(0, 1) for _ in range(SIZE)] for _ in range(SIZE)])
print_mx(TEST)


def f_1(p: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.matrix_power(p, k)


def f_2(p: np.ndarray, k: int, a: np.ndarray) -> np.ndarray:
    return a @ f_1(p, k)


def f_3(p: np.ndarray, k: int) -> np.ndarray:
    size = len(p)
    res = p.copy()
    for _ in range(k - 1):
        temp_arr = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                s = 0
                for m in range(size):
                    if m != j:
                        s += p[i][m] * res[m][j]
                temp_arr[i][j] = s
        res = temp_arr
    return res


def f_3_1(p: np.ndarray, k: int) -> np.ndarray:
    res = np.zeros(p.shape)
    for t in range(1, k + 1):
        res += f_3(p, t)
    return res


def f_3_2(p: np.ndarray, rounded: bool = True, accuracy: int = 10) -> np.ndarray:
    res = np.zeros(p.shape)
    for t in range(1, accuracy + 1):
        res += t * f_3(p, t)
    if rounded:
        size = len(p)
        for i in range(size):
            for j in range(size):
                res[i][j] = math.ceil(res[i][j])
    return res


def f_4(p: np.ndarray, k: int) -> np.ndarray:
    f = p
    if k > 1:
        p_inv_cache = np.linalg.inv(p)
        f_1_cache = p
        f_1_desc_cache = f_1(p, k)
        for m in range(1, k):
            f_1_cache = f_1_cache @ p
            f_1_desc_cache = f_1_desc_cache @ p_inv_cache
            f = f_1_cache - f * f_1_desc_cache
    return f


def f_4_1(p: np.ndarray, rounded: bool = True, accuracy: int = 40):
    res = np.zeros(p.shape)
    f = p
    p_inv_ = np.linalg.inv(p)
    p_ = p
    p_desc_ = f_1(p, accuracy)
    for t in range(1, accuracy + 1):
        res += t * f
        p_ = p_ @ p
        p_desc_ = p_desc_ @ p_inv_
        f = p_ - f * p_desc_
    if rounded:
        size = len(p)
        for i in range(size):
            for j in range(size):
                res[i][j] = math.ceil(res[i][j])
    return res


def f_4_2(p: np.ndarray, k: int) -> np.ndarray:
    res = np.zeros(p.shape)
    f = p
    p_inv_cache = np.linalg.inv(p)
    f_1_cache = p
    f_1_desc_cache = f_1(p, k)
    for t in range(1, k + 1):
        res += f
        f_1_cache = f_1_cache @ p
        f_1_desc_cache = f_1_desc_cache @ p_inv_cache
        f = f_1_cache - f * f_1_desc_cache
    return res


print_mx(f_4(P, 5))
print_mx(f_4_1(P))
# print_mx(f_4_2(P, 2))


