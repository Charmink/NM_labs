from functools import reduce
from matrix import Matrix, Vector


def get_P(mat):
    p = Matrix.identity(len(mat))
    for i in range(len(mat)):
        row = max(range(i, len(mat)), key=lambda j: abs(mat[j][i]))
        if i != row:
            p[i], p[row] = p[row], p[i]
    return p


def get_new_B(B, P):
    perm = [row.index(1) for row in P]
    new_B = Vector(len(B))
    for i in range(len(B)):
        new_B[i] = B[perm[i]]
    return new_B


def LU_decomposition(mat):
    P = get_P(mat)
    PA = P.multiply(mat)
    LU = Matrix(PA)

    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            LU[j][i] /= LU[i][i]
            for k in range(i + 1, len(mat)):
                LU[j][k] -= LU[j][i] * LU[i][k]

    return LU, P


def LU_solve(LU, B):
    sz = len(LU)
    z = Vector(sz)
    x = Vector(sz)

    z[0] = B[0]
    for i in range(1, sz):
        s = sum([LU[i][j] * z[j] for j in range(i)])
        z[i] = B[i] - s

    x[-1] = z[-1] / LU[-1][-1]
    for i in reversed(range(sz - 1)):
        s = sum([LU[i][j] * x[j]
                      for j in range(i + 1, sz)])
        x[i] = (z[i] - s) / LU[i][i]

    return x


def LU_inverse(LU):
    E = Matrix.identity(len(LU))
    x = []
    for row in E:
        x.append(LU_solve(LU, row))
    res = Matrix.from_list(x)
    res.transpose()
    return res


def LU_determinant(LU):
    return reduce(lambda x, y: x * y, [LU[i][i] for i in range(len(LU))])


if __name__ == '__main__':
    n = int(input("Введите размерность матрицы:"))
    inputMatrix = []
    for i in range(n):
        inputMatrix.append(list(map(int, input().split())))
    matrix = Matrix.from_list(inputMatrix)
    B = Vector.from_list(list(map(int, input().split())))

    LU, P = LU_decomposition(matrix)
    new_B = get_new_B(B, P)
    x = LU_solve(LU, new_B)

    print("Answer: \n{}".format(x))

    inv = LU_inverse(LU)
    print("Inverse matrix: \n{}".format(inv))

    det = LU_determinant(LU)
    print("Determinant: \n{}".format(det))
