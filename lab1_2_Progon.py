import numpy as np


def solve_tridiagonal(A, b):
    n = len(b)
    alpha = np.zeros(n-1)
    beta = np.zeros(n)
    x = np.zeros(n)

    # прямой ход
    alpha[0] = -A[0][1] / A[0][0]
    beta[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        alpha[i] = -A[i][i+1] / (A[i][i] + A[i][i-1]*alpha[i-1])
        beta[i] = (b[i] - A[i][i-1]*beta[i-1]) / (A[i][i] + A[i][i-1]*alpha[i-1])
    beta[n-1] = (b[n-1] - A[n-1][n-2]*beta[n-2]) / (A[n-1][n-1] + A[n-1][n-2]*alpha[n-2])

    # обратный ход
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = alpha[i]*x[i+1] + beta[i]

    return x


# Пример системы уравнений
A = np.array([[8.0, -4.0, 0.0, 0.0, 0.0],
              [-2.0, 12.0, -7.0, 0.0, 0.0],
              [0.0, 2.0, -9.0, 1.0, 0.0],
              [0.0, 0.0, -8.0, 17.0, -4.0],
              [0.0, 0.0, 0.0, -7.0, 13.0]])
b = np.array([32.0, 15.0, -10.0, 133.0, -76.0])

# решение системы уравнений методом прогонки
x = solve_tridiagonal(A, b)

print("Решение СЛАУ методом прогонки:")
print(x)