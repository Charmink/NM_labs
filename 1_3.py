import numpy as np
from numpy.linalg import norm, inv

from matrix import Matrix, Vector


def equivalent_form(A, B):
    sz = len(A)
    alpha = Matrix.zero(sz)
    beta = Vector(sz)

    # чтобы не делить на 0, добавим небольшую константу
    for i in range(sz):
        beta[i] = B[i] / (A[i][i] + 1e-3)
        for j in range(sz):
            if i != j:
                alpha[i][j] = -A[i][j] / (A[i][i] + 1e-3)

    return alpha, beta


def finish_iter_process(x, x_prev, norm_alpha, norm_C, eps, zeidel=False):
    # дикий костыль, потому что мне не хотелось писать велосипедный метод зейделя
    if zeidel:
        norm_vec = norm(x - x_prev, np.inf)
    else:
        vec = x - x_prev
        norm_vec = vec.norm()

    if norm_alpha == 1:
        return norm_vec <= eps
    else:
        coef = norm_C / (1 - norm_alpha)
        return coef * norm_vec <= eps


def iterative_method(alpha, beta, eps):
    x = Vector.copy(beta)
    norm_alpha = alpha.norm()
    while True:
        x_i = beta + alpha.multiply(x)
        if finish_iter_process(x_i, x, norm_alpha, norm_alpha, eps):
            break
        x = x_i
    return x_i


def seidel_method(alpha, beta, eps):
    sz = len(alpha)
    np_alpha = np.array(alpha.get_data())
    np_beta = np.array(beta.get_data())
    B = np.tril(np_alpha, -1)
    C = np_alpha - B
    tmp1 = inv(np.eye(sz, sz) - B) @ C
    tmp2 = inv(np.eye(sz, sz) - B) @ np_beta
    x = tmp2

    while True:
        x_i = tmp2 + tmp1 @ x
        if finish_iter_process(x_i, x, norm(tmp1), norm(C), eps, zeidel=True):
            break
        x = x_i
    return Vector.from_list(x_i.tolist())


if __name__ == '__main__':
    n = int(input("Введите размерность матрицы: "))
    inputMatrix = []
    for i in range(n):
        inputMatrix.append(list(map(int, input().split())))
    matrix = Matrix.from_list(inputMatrix)
    B = Vector.from_list(list(map(int, input().split())))
    eps = float(input("Введите эпсилон: "))

    alpha, beta = equivalent_form(matrix, B)
    x = iterative_method(alpha, beta, eps)
    x2 = seidel_method(alpha, beta, eps)
    print("{0}\n{1}".format(x, x2))
