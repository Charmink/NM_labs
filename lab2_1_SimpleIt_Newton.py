import sympy as sp


def Newton(expr, interval, eps):
    a = interval[0]
    b = interval[1]
    f_a = f.evalf(subs={x: a})
    f_b = f.evalf(subs={x: b})
    x_old = b

    df = sp.diff(expr, x)
    ddf = sp.diff(expr, x, 2)
    ddf_subs = ddf.evalf(subs={x: x_old})

    if f_a * f_b >= 0 >= f_b * ddf_subs:
        return "Метод расходится, выберите другой интервал приближения"

    k = 0
    while True:
        f_xk = f.evalf(subs={x: x_old})
        df_xk = df.evalf(subs={x: x_old})
        x_new = x_old - f_xk / df_xk

        if abs(x_new - x_old) < eps:
            return x_new, k

        x_old = x_new
        k += 1


def simple_iter(phi, interval, eps):
    dphi = sp.diff(phi, x)
    q_1 = abs(dphi.evalf(subs={x: interval[0]}))
    q_2 = abs(dphi.evalf(subs={x: interval[1]}))
    q = max(q_1, q_2)
    x_old = (interval[1] + interval[0]) / 2
    x_new = None

    iter = 0
    for i in range(10):
        x_new = phi.evalf(subs={x: x_old})

        check = q * abs(x_new - x_old) / (1 - q)
        if check <= eps:
            break

        iter += 1
        x_old = x_new

    return x_new, iter


x = sp.Symbol('x')
t = sp.Symbol('t')
expr1 = x**3 + x**2 - 2*x - 1

sp.plot(expr1, (x, -5, 5))

epsilon = float(input("Введите эпсилон: "))
print()

interval1 = [1, 2]  # интервал для первой точки

f = sp.sympify(expr1)

root1_N, iters1_N = Newton(f, interval1, epsilon)

print("Метод Ньютона:")

print("Первая точка:\nКоличество итераций: {0}\nx1* = {1}\n".format(iters1_N, root1_N))
print()

print("Метод простой итерации:")

df = sp.diff(f, x)
df_a = df.evalf(subs={x: interval1[0]})
df_b = df.evalf(subs={x: interval1[1]})
max_df = max(df_a, df_b)
c = 1 / max_df

phi1 = x - expr1 * c
# print("Функция phi для точки на отрезке [0, 1]:\nphi(x) = {0}".format(phi1))
sp.plot(phi1, (x, 1, 2), title='phi1(x)')
dphi1 = sp.diff(phi1, x)
print("phi'(x) = {0}\n".format(dphi1))

print("Проверка условий сходимости:")
print("phi(1) = {0}".format(phi1.evalf(subs={x: interval1[1]})))
abs_dphi_1 = dphi1.evalf(subs={x: interval1[1]}, n=3) + 0.001
print("|phi'(x)| <= q = {0}\n".format(abs_dphi_1.evalf(3)))

root1_S, iters1_S = simple_iter(phi1, interval1, epsilon)
print("Количество итераций: {0}\nx1* = {1}\n".format(iters1_S, root1_S))
