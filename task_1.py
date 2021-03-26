import sys
from datetime import time, datetime
from time import time_ns, time
from typing import Generator

import numpy as np
import math
import random as rnd


def profile_wrapper(func):
    def wrap(*args, **kwargs):
        t = time_ns()
        res = func(*args, **kwargs)
        length = time_ns() - t
        if length > 1e9:
            len_str = f"{length / 1e9:4.2f} s"
        elif length > 1e3:
            len_str = f"{length / 1e6:4.2f} ms"
        else:
            len_str = f"{length} ns"
        print(f"{func.__name__:>10}: {len_str}")
        return res
    return wrap


def print_mx(array: np.ndarray):
    NUM = 3
    if type(array[0]) is np.ndarray:
        for row in array:
            for cell in row:
                print(f"{str(round(cell, NUM)):<8}", end="")
            print()
    else:
        for cell in array:
            print(f"{str(round(cell, NUM)):<8}", end="")
    print()


SIZE = 15
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
    res = next(p_gen).diagonal()
    yield res
    p_list = [res]
    f_list = [res]
    while True:
        next_p = next(p_gen).diagonal()
        res = next_p - sum(map(lambda _f, _p: _f * _p, f_list, p_list))
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


@profile_wrapper
def avg(gen: Generator[np.ndarray, None, None], accuracy: int = 10000):
    res = next(gen).copy()
    for t in range(2, accuracy + 1):
        res += t * next(gen)
    return res


def last_state(p: np.ndarray):
    size = len(p)
    m_ = p.T - np.eye(size)
    m_[-1].fill(1)
    b = np.array([[0]] * (size - 1) + [[1]])
    return (np.linalg.inv(m_) @ b).T[0]
