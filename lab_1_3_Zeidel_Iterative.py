from copy import deepcopy


def norm_c(matrix):
    max = -100000000
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix[i])):
            sum += abs(matrix[i][j])
        if sum > max:
            max = sum

    return max


def invers(matr):
    mat_1 = []
    for i in range(len(matr)):
        row = []
        for j in range(len(matr)):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        mat_1.append(row)

    matr_inv = []
    mat = deepcopy(matr)
    for i in range(len(mat)):
        for j in mat_1[i]:
            mat[i].append(j)
        matr_inv.append(mat[i])

    for nrow, row in enumerate(matr_inv):
        divider = row[nrow]
        for i in range(len(row)):
            row[i] /= divider

        for lower_row in matr_inv[nrow + 1:]:
            factor = lower_row[nrow]
            for i in range(len(lower_row)):
                lower_row[i] -= factor*row[i]

    l = len(matr_inv) - 2
    for i in range(len(matr_inv) - 1, -1, -1):
        row = matr_inv[i]
        for k in range(l, -1, -1):
            lower_row = matr_inv[k]
            factor = lower_row[i]
            for j in range(len(lower_row)):
                lower_row[j] -= factor*row[j]
        l -= 1

    inv = []
    for i in range(len(matr_inv)):
        inv.append([matr_inv[i][j] for j in range(len(matr_inv), len(matr_inv[0]))])

    return inv


def multiply(X, Y):
    M = [[0 for x in range(len(Y[0]))]
         for y in range(len(X))]

    for i in range(len(X)):
        for j in range(len(Y[0])):
            M[i][j] = 0
            for k in range(len(X[0])):
                M[i][j] += X[i][k] * Y[k][j]
    return M


def equivalent_form(A, B):
    l = len(A)
    alpha = [[0 for x in range(l)]
                for y in range(l)]

    beta = [0 for x in range(l)]
    b = [0 for x in range(l)]

    for i in range(l):
        b[i] = B[i][0]

    for i in range(l):
        beta[i] = b[i] / A[i][i]
        for j in range(l):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]

    return alpha, beta


def iterative(A, B, eps):
    l = len(A)
    alpha, beta = equivalent_form(A, B)
    X = [[beta[x]] for x in range(len(beta))]
    norma_alpha = norm_c(alpha)

    x_i = [[0 for x in range(1)]
            for y in range(l)]
    dif_X = [[0 for x in range(1)]
            for y in range(l)]
    flag = False
    count = 0
    while True:
        count += 1
        alpha_X = multiply(alpha, X)
        for i in range(l):
            x_i[i][0] = beta[i] + alpha_X[i][0]

        for i in range(l):
            dif_X[i][0] = x_i[i][0] - X[i][0]

        norma_dif_X = norm_c(dif_X)

        if norma_alpha == 1:
            if norma_dif_X <= eps:
                flag = True
        else:
            c = norma_alpha / (1 - norma_alpha)
            if c * norma_dif_X <= eps:
                flag = True

        if flag:
            break
        X = deepcopy(x_i)
    print("Количество итераций итерационным методом: {}".format(count), end='\n')
    return x_i


def Zeidel(A, B, eps):
    l = len(A)
    alpha, beta = equivalent_form(A, B)
    beta_vec = [[beta[x]] for x in range(len(beta))]
    norma_alpha = norm_c(alpha)

    B = deepcopy(alpha)
    C = deepcopy(alpha)
    E = []
    for i in range(l):
        row = []
        for j in range(l):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        E.append(row)

    for i in range(l):
        for j in range(i, l):
            B[i][j] = 0

    for i in range(l):
        for j in range(l):
            C[i][j] = alpha[i][j] - B[i][j]

    E_B = deepcopy(E)
    for i in range(l):
        for j in range(l):
            E_B[i][j] = E[i][j] - B[i][j]

    E_B_inv = invers(E_B)
    mat1 = multiply(E_B_inv, C)
    mat2_b = multiply(E_B_inv, beta_vec)
    X = mat2_b

    dif_X = [[0 for x in range(1)]
             for y in range(l)]
    x_i = [[0 for x in range(1)]
           for y in range(l)]
    flag = False
    count = 0
    while True:
        count += 1
        mat1_x = multiply(mat1, X)
        for i in range(l):
            x_i[i][0] = mat2_b[i][0] + mat1_x[i][0]

        for i in range(l):
            dif_X[i][0] = x_i[i][0] - X[i][0]

        norma_dif_X = norm_c(dif_X)
        norma_C = norm_c(C)
        # print("norma_dif={}".format(norma_dif_X))
        if norma_alpha == 1:
            if norma_dif_X <= eps:
                flag = True
        else:
            c = norma_C / (1 - norma_alpha)
            if c * norma_dif_X <= eps:
                flag = True

        if flag:
            break
        X = deepcopy(x_i)
    print("Количество итераций методом Зейделя: {}".format(count), end='\n')
    return x_i


n = int(input("Введите размерность матрицы: "))
print("Введите матрицу коэффициентов:")
Q, Y = [], []
for i in range(n):
    Q.append(list(map(int, input().split())))
print("Введите вектор правой части:")
for i in range(n):
    Y.append(list(map(int, input().split())))

eps = float(input("Введите эпсилон: "))

print("Метод простых итераций:")
otvet_S = iterative(Q, Y, eps)
for i, x in enumerate(otvet_S):
    print("x{0} = {1}".format(i+1, x[0]))

print("\nМетод Зейделя:")
otvet_Z = Zeidel(Q, Y, eps)
for i, x in enumerate(otvet_Z):
    print("x{0} = {1}".format(i+1, x[0]))

#
# 12 -3 -1 3
# 5 20 9 1
# 6 -3 -21 -7
# 8 -7 3 -27
# #
# -31
# 90
# 119
# 71