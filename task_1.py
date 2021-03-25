import sys
from datetime import time
from time import time_ns
from typing import Generator

import numpy as np
import math
import random as rnd


def print_mx(array: np.ndarray):
    NUM = 3
    for row in array:
        for cell in row:
            print(f"{str(round(cell, NUM)):<8}", end="")
        print()
    print()


SIZE = 100
TEST = np.array([[rnd.uniform(0, .2) for _ in range(SIZE)] for _ in range(SIZE)])
TEST_VECTOR = np.array([[rnd.uniform(0, .2)] for _ in range(SIZE)])


def transfer_gen(p: np.ndarray) -> Generator[np.ndarray, None, None]:
    res = p
    yield res
    while True:
        res = res @ p
        yield res


def transfer(p: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.matrix_power(p, k)


def state(p: np.ndarray, a: np.ndarray, k: int) -> np.ndarray:
    return a @ transfer(p, k)


def first_transfer_gen(p: np.ndarray) -> Generator[np.ndarray, None, None]:
    size = len(p)
    res = p.copy()
    yield res
    while True:
        p_cache = np.zeros(p.shape)
        for i in range(size):
            for j in range(size):
                s = 0
                for m in range(size):
                    if m != j:
                        s += p[i][m] * res[m][j]
                p_cache[i][j] = s
        res = p_cache
        yield res


def first_return_gen(p: np.ndarray) -> Generator[np.ndarray, None, None]:
    p_gen = transfer_gen(p)
    res = next(p_gen).copy()
    yield res
    eye = np.eye(len(res))
    p_list = [p]
    f_list = [p]
    while True:
        next_p = next(p_gen)
        res = (next_p - sum(map(lambda _f, _p: _f * _p, f_list, p_list))) * eye
        f_list.append(res)
        p_list.insert(0, next_p)
        yield res


def on_step(gen: Generator[np.ndarray, None, None], k: int):
    for _ in range(k - 1): next(gen)
    return next(gen)


def not_later(gen: Generator[np.ndarray, None, None], k: int):
    res = next(gen).copy()
    for t in range(1, k):
        res += next(gen)
    return res


def avg(gen: Generator[np.ndarray, None, None], accuracy: int = 1000):
    res = next(gen).copy()
    for t in range(2, accuracy + 1):
        res += t * next(gen)
    return res


def last_state(p: np.ndarray):
    size = len(p)
    m_ = p.T - np.eye(size)
    m_[-1].fill(1)
    b = np.array([[0]] * (size - 1) + [[1]])
    return np.linalg.inv(m_) @ b


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

tns = time_ns()
print_mx(avg(first_return_gen(P)))
print((time_ns() - tns) / 1e9)
