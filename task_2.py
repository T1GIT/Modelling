import numpy as np


lam = 6
m = 5
_m = 1
n = 14


def f_1(lam: int, m: int, _m: int, n: int):
    size = m + n + 1
    p = np.zeros((size, size))
    for i in range(0, m + n):
        p[i][i + 1] = lam
        p[i + 1][i] = (i + 1) * _m if i <= m else m * _m
    return p


def f_2(p: np.ndarray):
    size = len(p)
    m = p.T - p.sum(0)
    m[-1].fill(1)
    b = np.array([[0]] * (size - 1) + [[1]])
    return np.linalg.inv(m) @ b


# X = f_2(f_1(lam, m, _m, n))
#
# # B
# print("P отказа:", X[-1])
#
# # C
# print("Относительная проп. способность:", 1 - X[-1])
# print("Асболютная проп. способность:", (1 - X[-1]) * lam)
#
# # D
# s = 0
# for i in range(1, n + 1):
#     s += i * X[m + i]
# print("Средняя длина очереди:", s)
#
#
# # E
# s = 0
# for i in range(n + 1):
#     s += (i + 1) / (m * _m) * X[m + i]
# print("Среднее время в очереди:", s)
#
# # F
# s0 = 0
# for i in range(1, m + 1):
#     s0 += i * X[i]
# s1 = 0
# for i in range(m + 1, size):
#     s1 += m * X[i]
# print("Среднее количество занятых каналов:", s0 + s1)
#
# # G
# print("Сумма установившихся вероятностей от 0 до m-1", sum(X[0: m]))
#
# # H
# d = []
# for i in P:
#     d.append(round(1 / i.sum(), 3))
# print(d)
