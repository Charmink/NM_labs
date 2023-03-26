import copy

import numpy as np


def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        x = R[j:, j]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0])
        u = x + e * np.linalg.norm(x)
        v = u / np.linalg.norm(u)
        Qj = np.eye(m)
        Qj[j:, j:] -= 2.0 * np.outer(v, v)
        R = np.dot(Qj, R)
        Q = np.dot(Q, Qj.T)

    return Q, R


def qr_eigenvalues(A, epsilon=0.01, max_iter=1000):
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(max_iter):
        Q_, R = qr_decomposition(A)
        A = np.dot(R, Q_)
        Q = np.dot(Q, Q_)
        if np.abs(np.triu(A, k=1)).max() < epsilon:
            break
    eigenvalues = np.diag(A)

    return eigenvalues


# Пример использования
A = np.array([[-5, -8, 4], [4, 2, 6], [-2, 5, -6]])
result = qr_eigenvalues(A)
eigenvalues = np.linalg.eigvals(A)
print(f"Собственные значения матрицы A:{result}")
print(eigenvalues)
