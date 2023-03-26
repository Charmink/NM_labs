import math

import sympy as sp

x, y = sp.symbols('x, y')


def Newton(f1, f2, intX, intY, eps):
    df1_x = sp.diff(f1, x)
    df1_y = sp.diff(f1, y)
    df2_x = sp.diff(f2, x)
    df2_y = sp.diff(f2, y)
    x_old = (intX[0] + intX[1]) / 2
    y_old = (intY[0] + intY[1]) / 2

    iters = 0
    while True:
        f1_k = f1.evalf(subs={x: x_old, y: y_old})
        f2_k = f2.evalf(subs={x: x_old, y: y_old})
        df1_x_k = df1_x.evalf(subs={x: x_old, y: y_old})
        df1_y_k = df1_y.evalf(subs={x: x_old, y: y_old})
        df2_x_k = df2_x.evalf(subs={x: x_old, y: y_old})
        df2_y_k = df2_y.evalf(subs={x: x_old, y: y_old})
        A1 = sp.Matrix([[f1_k, df1_y_k],
                        [f2_k, df2_y_k]])
        A2 = sp.Matrix([[df1_x_k, f1_k],
                        [df2_x_k, f2_k]])
        J = sp.Matrix([[df1_x_k, df1_y_k],
                       [df2_x_k, df2_y_k]])

        x_new = x_old - A1.det() / J.det()
        y_new = y_old - A2.det() / J.det()

        check_x = abs(x_new - x_old)
        check_y = abs(y_new - y_old)
        check = max(check_x, check_y)
        if check <= eps:
            break

        x_old, y_old = x_new, y_new
        iters += 1

    return [x_new, y_new], iters


def simple_iter(phi1, phi2, intX, intY, q, eps):
    x_old = (intX[0] + intX[1]) / 2
    y_old = (intY[0] + intY[1]) / 2
    iters = 0
    while True:
        x_new = phi1.evalf(subs={x: x_old, y: y_old})
        y_new = phi2.evalf(subs={x: x_old, y: y_old})
        check_x = abs(x_new - x_old)
        check_y = abs(y_new - y_old)
        check = q * max(check_x, check_y) / (1 - q)
        if check <= eps:
            break

        x_old, y_old = x_new, y_new
        iters += 1

    return [x_new, y_new], iters


f1 = (x**2 + y**2 - 16)
f2 = (x - sp.exp(y) + 4)

epsilon = float(input("Введите эпсилон: "))
print()

p1 = sp.plot_implicit(sp.Eq(f1, 0),  line_color='r', show=False)
p2 = sp.plot_implicit(sp.Eq(f2, 0), line_color='r', show=False)
p1.extend(p2)
p1.show()

segm_X = [3, 4]
segm_Y = [1.5, 2.5]

print("Метод Ньютона:")
X_newt, k = Newton(f1, f2, segm_X, segm_Y, epsilon)

print("Ответ: {0}\nКоличество итераций: {1}".format(X_newt, k))
print("Проверка 1 уравнения: {0}".format(sp.Subs(f1, (x, y), (X_newt[0], X_newt[1])).doit()))
print("Проверка 2 уравнения: {0}\n".format(sp.Subs(f2, (x, y), (X_newt[0], X_newt[1])).doit()))

print("Метод простой итерации:")
phi1 = sp.solve(f1, x)[1]
phi2 = sp.solve(f2, y)[0]

dphi1 = abs(sp.diff(phi1, x)) + abs(sp.diff(phi1, y))
dphi2 = abs(sp.diff(phi2, x)) + abs(sp.diff(phi2, y))

print("phi1' = {0}".format(dphi1.evalf(subs={y: 4.5})))
print("phi2' = {0}".format(dphi2.evalf(subs={x: 3})))

q = max(dphi1.evalf(subs={y: 3}), dphi2.evalf(subs={x: 3}))

X_simple, k_s = simple_iter(phi1, phi2, segm_X, segm_Y, q, epsilon)

print("Ответ: {0}\nКоличество итераций: {1}".format(X_simple, k))
print("Проверка 1 уравнения: {0}".format(sp.Subs(f1, (x, y), (X_simple[0], X_simple[1])).doit()))
print("Проверка 2 уравнения: {0}".format(sp.Subs(f2, (x, y), (X_simple[0], X_simple[1])).doit()))
