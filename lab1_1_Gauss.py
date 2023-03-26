import numpy as np
import copy


def gaussian_elimination(A, b):
    n = len(b)
    # Прямой ход метода Гаусса
    for i in range(n-1):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i+1:n] -= factor * A[i, i+1:n]
            A[j, i] = factor
            b[j] -= factor * b[i]
    # Обратный ход метода Гаусса
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    return x


def inverse_matrix(A):
    n = len(A)
    B = np.concatenate([A, np.identity(n)], axis=1)

    for i in range(n):
        # Выбираем максимальный элемент в столбце i
        max_row = i
        for j in range(i+1, n):
            if abs(B[j, i]) > abs(B[max_row, i]):
                max_row = j
        # Меняем местами строки i и max_row
        B[[i, max_row]] = B[[max_row, i]]

        # Делаем элемент (i,i) равным 1 путем деления i-й строки на B[i,i]
        B[i] /= B[i, i]

        # Вычитаем i-ю строку умноженную на B[j,i] из всех строк j != i
        for j in range(n):
            if i != j:
                B[j] -= B[i]*B[j, i]

    # Извлекаем обратную матрицу из расширенной матрицы B
    inv_A = B[:, n:]

    return inv_A


def determinant(A):
    n = len(A)
    det = 1.0
    # Прямой ход метода Гаусса
    for i in range(n):
        # Находим главный элемент
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[max_index, i]):
                max_index = j
        if max_index != i:
            # Меняем местами строки
            A[[i, max_index], :] = A[[max_index, i], :]
            # Меняем знак определителя
            det *= -1
        # Вычитаем из остальных строк текущую строку, умноженную на множитель
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
        # Умножаем главный элемент на определитель
        det *= A[i, i]
    return det


A = np.array([[-7., -9., 1., -9.],
              [-6., -8., -5., 2.],
              [-3., 6., 5., -9.],
              [-2., 0., -5., -9.]])
b = np.array([29., 42., 11., 75.])

inv_A = inverse_matrix(copy.deepcopy(A))
print("Обратная матрица:\n", inv_A)

det_A = determinant(copy.deepcopy(A))
print("Определитель:", det_A)

e_matrix = inv_A @ A
print("Единичная матрица:\n", e_matrix)

x = gaussian_elimination(copy.deepcopy(A), b.copy())
print("Решение: x =", x)

