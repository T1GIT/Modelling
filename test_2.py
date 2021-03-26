import numpy as np
import math
import task_1
import task_2

#       Задание 2
# Данные
lam = 6
m = 5
_m = 1
n = 14

# Решение
print("     ЗАДАЧА A")
A = task_2.intensity(lam, m, _m, n)
task_1.print_mx(A)
X = task_2.last_state(A)
task_1.print_mx(X)

print("     ЗАДАЧА B")
print(X[-1])

print("     ЗАДАЧА C")
print(1 - X[-1])
print((1 - X[-1]) * lam)

print("     ЗАДАЧА D")
print(sum(map(lambda i:
              i * X[m + i],
              range(1, n + 1))))

print("     ЗАДАЧА E")
print(sum(map(lambda i:
              (i + 1) / (m * _m) * X[m + i],
              range(1, n + 1))))

print("     ЗАДАЧА F")
print(sum(map(lambda i:
              i * X[i],
              range(1, m + 1)))
      +
      sum(map(lambda i:
              m * X[i],
              range(m + 1, m + n + 1))))

print("     ЗАДАЧА G")
print(sum(X[1: m + 1]))

print("     ЗАДАЧА H")
res = list(map(lambda x:
               1 / x,
               A.sum(1)))
print(res[0])
