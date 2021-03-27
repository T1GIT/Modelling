import random as rnd
from time import time_ns
from typing import Generator

import numpy as np


def profile_wrapper(func):
    def wrap(*args, **kwargs):
        t = time_ns()
        res = func(*args, **kwargs)
        length = time_ns() - t
        if length >= 1e9:
            len_str = f"{length / 1e9:4.2f} s"
        elif length >= 1e3:
            len_str = f"{length / 1e6:4.2f} ms"
        else:
            len_str = f"{length} ns"
        print(f"{func.__name__}: {len_str}")
        return res
    return wrap


def print_mx(array: np.ndarray, accuracy: int = 5):
    if type(array[0]) is np.ndarray:
        for row in array:
            for cell in row:
                print(f"{str(round(cell, accuracy)):<{5 + accuracy}}", end="")
            print()
    else:
        for cell in array:
            print(f"{str(round(cell, accuracy)):<{5 + accuracy}}", end="")
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
    res = p.copy()
    yield res
    while True:
        res = (p @ res) - (p * res.diagonal())
        yield res


def first_return_gen(p: np.ndarray) -> Generator[np.ndarray, None, None]:
    p_gen = transfer_gen(p)
    res = next(p_gen).diagonal()
    yield res
    p_list = [res]
    f_list = [res]
    while True:
        next_p = next(p_gen).diagonal()
        res = next_p - sum(map(lambda _f, _p: _f * _p, f_list, reversed(p_list)))
        p_list.append(next_p)
        f_list.append(res)
        yield res


def on_step(gen: Generator[np.ndarray, None, None], k: int):
    for _ in range(k - 1): next(gen)
    return next(gen)


def not_later(gen: Generator[np.ndarray, None, None], k: int):
    return sum(map(lambda t:
                   next(gen),
                   range(k)))


@profile_wrapper
def avg(gen: Generator[np.ndarray, None, None], accuracy: int = 10000):
    return sum(map(lambda t:
                   t * next(gen),
                   range(1, accuracy + 1)))


def last_state(p: np.ndarray):
    m_ = p.T - np.eye(len(p))
    m_[-1].fill(1)
    b = np.array([[0]] * (len(p) - 1) + [[1]])
    return (np.linalg.inv(m_) @ b)[:, 0]

