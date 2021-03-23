import numpy as np
import math
import random as rnd


def print_mx(array: np.ndarray):
    NUM = 3
    for row in array:
        for cell in row:
            print(f"{str(round(cell, NUM)):<7}", end="")
        print()
    print()


SIZE = 5
TEST = np.array([[rnd.uniform(0, .2) for _ in range(SIZE)] for _ in range(SIZE)])
TEST_VECTOR = np.array([[rnd.uniform(0, .2)] for _ in range(SIZE)])


def f_1(p: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.matrix_power(p, k)


def f_2(p: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    return a @ f_1(p, k)


def f_3(p: np.ndarray, k: int) -> np.ndarray:
    size = len(p)
    if k == 0: return np.eye(size)
    res = p.copy()
    for _ in range(k - 1):
        temp_arr = np.zeros(p.shape)
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


def f_3_2(p: np.ndarray, accuracy: int = 200) -> np.ndarray:
    res = np.zeros(p.shape)
    for t in range(1, accuracy + 1):
        res += t * f_3(p, t)
    return res


def f_4(p: np.ndarray, k: int) -> np.ndarray:
    if k > 1:
        f = p + f_1(p, k)
        for m in range(1, k):
            f -= f_4(p, m) * f_1(p, k - m)
        return f
    else:
        return p


def f_4_1(p: np.ndarray, k: int):
    res = np.zeros(p.shape)
    for t in range(1, k + 1):
        res += f_4(p, t)
    return res


def f_4_2(p: np.ndarray, accuracy: int = 20) -> np.ndarray:
    res = np.zeros(p.shape)
    for t in range(1, accuracy + 1):
        res += t * f_4(p, t)
    return res


def f_5(p: np.ndarray):
    size = len(p)
    m_ = p.T - np.eye(size)
    m_[-1].fill(1)
    b = np.array([[0]] * (size - 1) + [[1]])
    return np.linalg.inv(m_) @ b
