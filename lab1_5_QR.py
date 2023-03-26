import numpy as np
import sympy as sp


def Hausgolder(A):
    l = len(A)
    Ak = np.copy(A)
    Q = np.eye(l)
    E = np.eye(l)

    for i in range(l-1):
        v = np.zeros(l)

        norma = 0
        for j in range(l):
            norma += A[j][i]**2

        v[i] = Ak[i][i] + np.sign(Ak[i][i]) * np.sqrt(norma)
        for j in range(i+1, l):
            v[j] = Ak[j][i]
        v = [v]
        v_T = np.transpose(v)

        Hk = E - (2 / (np.dot(v, v_T))) * (np.dot(v_T, v))
        Q = np.matmul(Q, Hk)

        Ak = np.matmul(Hk, Ak)

    return Q, Ak


def search_SZ(A, eps):
    l = len(A)

    A_old = np.copy(A)
    iters = 0
    while True:
        norms = np.zeros(l-1)
        Q, R = Hausgolder(A_old)
        # print("R:\n{0}".format(R))
        A_new = np.dot(R, Q)
        # print("Ak:\n{0}\n".format(A_new))

        norma = 0
        for i in range(l - 1):
            for j in range(i + 1, l):
                norma += A_new[j][i] ** 2
                # print(A_new[j][i])
            norms[i] = np.sqrt(norma)

        if min(norms) < eps:
            break

        A_old = A_new
        iters += 1

    if max(norms) < eps:
        prop_val = np.diag(A_new)
        return prop_val, iters
    else:
        prop_val = np.zeros(l, dtype=complex)
        x = sp.Symbol('x')
        res = sp.solve((A_new[1][1] - x) * (A_new[2][2] - x) - A_new[1][2]*A_new[2][1], x)
        prop_val[0] = A_new[0][0]
        prop_val[1] = res[0]
        prop_val[2] = res[1]
        return prop_val, iters


eps = float(input("Введите точность вычислений: "))
n = 3
print("Введите матрицу 3х3:")
A = np.zeros((n, n))
for i in range(n):
    t = list(map(int, input().split()))
    for j in range(n):
        A[i][j] = t[j]

lamd, it = search_SZ(A, eps)
print("Количество итераций: {0}".format(it))
print("Собственные значения:")
for i in range(len(lamd)):
    print("lambda{0} = {1}".format(i+1, lamd[i]))
