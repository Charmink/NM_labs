from matrix import TridiagonalMatrix, Vector


def tma(mat, D):
    sz = len(mat)
    x = Vector(sz)
    p, q = [], []
    p.append(-mat.c[0] / mat.b[0])
    q.append(D[0] / mat.b[0])

    for i in range(1, sz):
        p_i = 0 if i == sz - 1 else (-mat.c[i] / (mat.b[i] + mat.a[i] * p[i - 1]))
        q_i = (D[i] - mat.a[i] * q[i - 1]) / (mat.b[i] + mat.a[i] * p[i - 1])
        p.append(p_i)
        q.append(q_i)

    x[sz - 1] = q[sz - 1]
    for i in range(sz - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


if __name__ == '__main__':
    A = [0] + list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split())) + [0]
    D = Vector.from_list(list(map(int, input().split())))

    matrix = TridiagonalMatrix.from_lists(A, B, C)

    x = tma(matrix, D)
    print("Answer: \n{}".format(x))


# -2 2 -8 -7
# 8 12 -9 17 13
# -4 -7 1 -4
# 32 15 -10 133 -76
