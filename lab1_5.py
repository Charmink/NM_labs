import numpy as np
from numpy.linalg import norm
from numpy.linalg import eig


def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


def householder(a, sz, k):
    v = np.zeros(sz)
    a = np.copy(a)
    v[k] = a[k] + sign(a[k]) * norm(a[k:])
    for i in range(k + 1, sz):
        v[i] = a[i]
    # we make newaxis for v.T
    v = v[:, np.newaxis]
    H = np.eye(sz) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def get_QR(A):
    sz = len(A)
    Q = np.identity(sz)
    A_i = np.copy(A)

    for i in range(sz - 1):
        col = A_i[:, i]
        H = householder(col, len(A_i), i)
        Q = Q @ H
        A_i = H @ A_i

    return Q, A_i


# complex eigenvalues by solving the equation
def get_roots(A, i):
    size = len(A)
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < size else 0
    a21 = A[i + 1][i] if i + 1 < size else 0
    a22 = A[i + 1][i + 1] if i + 1 < size else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


# iteration termination criterion for blocks with complex eigenvalues
def finish_iter_for_complex(A, eps, i):
    Q, R = get_QR(A)
    A_next = R @ Q
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return True if abs(lambda1[0] - lambda2[0]) <= eps and \
                abs(lambda1[1] - lambda2[1]) <= eps else False


def get_eigenvalue(A, eps, i):
    A_i = np.copy(A)
    while True:
        Q, R = get_QR(A_i)
        A_i = R @ Q
        a = np.copy(A_i)
        if norm(a[i + 1:, i]) <= eps:
            res = (a[i][i], False, A_i)
            break
        elif norm(a[i + 2:, i]) <= eps and finish_iter_for_complex(A_i, eps, i):
            res = (get_roots(A_i, i), True, A_i)
            break
    return res


def QR_method(A, eps):
    res = np.array([])
    i = 0
    A_i = np.copy(A)
    while i < len(A):
        eigenval = get_eigenvalue(A_i, eps, i)
        # for real
        if eigenval[1]:
            res = np.append(res, eigenval[0])
            i += 2
        # for complex
        else:
            res = np.append(res, eigenval[0])
            i += 1
        A_i = eigenval[2]
    return res, i


if __name__ == '__main__':
    A = np.array([[-5., -8., 4.],
                  [4., 2., 6.],
                  [-2., 5., -6.]])
    print('Введите точность вычислений:')
    eps = float(input())
    tmp, count_iter = QR_method(A, eps)
    print("Number of iterations:")
    print(count_iter)
    print("My eigenvalues:")
    print(tmp)
    eig_np = eig(A)
    print("Numpy eigenvalues:")
    print(eig_np[0].round(3))
