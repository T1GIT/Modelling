import numpy as np


lam = 6
m = 5
_m = 1
n = 14

LENGTH = m + n + 1

# Init
P = np.zeros((LENGTH, LENGTH))
for i in range(0, m + n):
    P[i][i + 1] = lam
    P[i + 1][i] = (i + 1) * _m if i <= m else m * _m
print("Матрица интенсивностей")
print(P)

# A
d = []
for i in P:
    d.append(i.sum())
D = np.diag(d)
M = P.T - D
M_ = M.copy()
M_[-1] = np.array([1] * len(D[0]))
B = np.array([np.array([0])] * len(P))
B[-1][0] = 1
X = np.array(list(map(lambda x: x[0], np.linalg.inv(M_) @ B)))
print("Установившиеся вероятности")
print(X)
print("Сумма вероятностей: ", X.sum())

print(" Характеристики:")

# B
print("P отказа:", X[-1])

# C
print("Относительная проп. способность:", 1 - X[-1])
print("Асболютная проп. способность:", (1 - X[-1]) * lam)

# D
s = 0
for i in range(1, n + 1):
    s += i * X[m + i]
print("Средняя длина очереди:", s)


# E
s = 0
for i in range(n + 1):
    s += (i + 1) / (m * _m) * X[m + i]
print("Среднее время в очереди:", s)

# F
s0 = 0
for i in range(1, m + 1):
    s0 += i * X[i]
s1 = 0
for i in range(m + 1, LENGTH):
    s1 += m * X[i]
print("Среднее количество занятых каналов:", s0 + s1)

# G
print("Сумма установившихся вероятностей от 0 до m-1", sum(X[0: m]))

# H
d = []
for i in P:
    d.append(round(1 / i.sum(), 3))
print(d)
