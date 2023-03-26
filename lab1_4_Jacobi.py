import numpy as np


def maximum(matrix):
    max = abs(matrix[0][1])
    k = len(matrix)
    i_m, j_m = 0, 1
    for i in range(len(matrix)):
        for j in range(i+1, k):
            if i != j and max < abs(matrix[i][j]):
                max = abs(matrix[i][j])
                i_m, j_m = i, j
    return i_m, j_m


def t_check(m):
    t = 0
    l = len(m)
    for i in range(l):
        for j in range(i + 1, l):
            if i != j:
                t += m[i][j] ** 2
    return np.sqrt(t)


def method_Jacobi(matr, eps):
    l = len(matr)
    matrix_k = np.copy(matr)
    Vect = np.eye(l)
    iters = 0
    while True:
        i_m, j_m = maximum(matrix_k)

        T = t_check(matrix_k)
        if T <= eps or iters >= 1000:
            break

        if i_m != j_m:
            phi = 0.5 * np.arctan(2 * matrix_k[i_m][j_m] / (matrix_k[i_m][i_m] - matrix_k[j_m][j_m]))
        else:
            phi = np.pi / 4

        U = np.eye(l)
        U[i_m][i_m] = np.cos(phi)
        U[i_m][j_m] = - np.sin(phi)
        U[j_m][i_m] = np.sin(phi)
        U[j_m][j_m] = np.cos(phi)

        U_T = np.transpose(U)
        matrix_k = np.matmul(np.matmul(U_T, matrix_k), U)
        Vect = np.matmul(Vect, U)

        iters += 1

    proper_val = np.diag(matrix_k)

    return proper_val, Vect, iters


eps = float(input("Введите точность вычислений: "))
n = int(input("Введите размерность матрицы: "))
print("Введите матрицу:")
A = np.zeros((n, n))
for i in range(n):
    t = list(map(int, input().split()))
    for j in range(n):
        A[i][j] = t[j]

lamd, prop_vects, k = method_Jacobi(A, eps)
print("Количество итераций: {0}\n".format(k))

print("Собственные значения:")
for i in range(len(lamd)):
    print("l{0} = {1}".format(i+1, lamd[i]))
print()

print("Собственные векторы:")
prop_vects_T = np.transpose(prop_vects)
for i in range(len(prop_vects)):
    print("x{0}: {1}".format(i+1, prop_vects_T[i]))
print()

print("Проверка ортогональности:")
for i in range(len(prop_vects_T)):
    for j in range(i+1, len(prop_vects_T)):
        if i != j:
            print("(x{0}, x{1}) = {2}".format(i+1, j+1, np.dot(prop_vects_T[i], prop_vects_T[j])))
