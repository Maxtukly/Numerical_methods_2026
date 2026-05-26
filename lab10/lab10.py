import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return -2 * y + x

def exact(x):

    return (5/4) * np.exp(-2 * x) + x / 2 - 1/4

X0 = 0.0
X1 = 1.0
Y0 = 1.0
H  = 0.01
EPS = 1e-4

def runge_kutta4(f, x0, y0, x_end, h):
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        h = min(h, x_end - x)
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def runge_error_rk4(f, x0, y0, x_end, h):
    _, y_h  = runge_kutta4(f, x0, y0, x_end, h)
    _, y_h2 = runge_kutta4(f, x0, y0, x_end, h / 2)
    y_h2_coarse = y_h2[::2]
    n = min(len(y_h), len(y_h2_coarse))
    runge_err = np.abs(y_h[:n] - y_h2_coarse[:n]) / (2**4 - 1)
    return runge_err


def adaptive_rk4(f, x0, y0, x_end, eps):
    xs = [x0]
    ys = [y0]
    hs = []
    x, y = x0, y0
    h = (x_end - x0) / 10
    while x < x_end - 1e-12:
        h = min(h, x_end - x)
        k1 = f(x, y);       k2 = f(x+h/2, y+h/2*k1)
        k3 = f(x+h/2, y+h/2*k2); k4 = f(x+h, y+h*k3)
        y1 = y + h/6*(k1+2*k2+2*k3+k4)
        h2 = h / 2
        k1 = f(x, y);       k2 = f(x+h2/2, y+h2/2*k1)
        k3 = f(x+h2/2, y+h2/2*k2); k4 = f(x+h2, y+h2*k3)
        ym = y + h2/6*(k1+2*k2+2*k3+k4)
        k1 = f(x+h2, ym);   k2 = f(x+h2+h2/2, ym+h2/2*k1)
        k3 = f(x+h2+h2/2, ym+h2/2*k2); k4 = f(x+h, ym+h2*k3)
        y2 = ym + h2/6*(k1+2*k2+2*k3+k4)
        err = abs(y2 - y1) / (2**4 - 1)
        if err > eps:
            h /= 2
            continue
        x += h
        y = y2
        xs.append(x)
        ys.append(y)
        hs.append(h)
        if err < eps / 32:
            h *= 2
    return np.array(xs), np.array(ys), np.array(hs)

def adams2(f, x0, y0, x_end, h):
    xs_rk, ys_rk = runge_kutta4(f, x0, y0, x0 + h, h)
    xs = list(xs_rk)
    ys = list(ys_rk)
    errl = []

    x, y_prev = xs[-2], ys[-2]
    y_curr    = ys[-1]

    errl.append(0)
    errl.append(y_curr - y_prev)

    while xs[-1] < x_end - 1e-12:
        xn   = xs[-1]
        h_   = min(h, x_end - xn)

        fn_1 = f(xs[-2], ys[-2])   # f_{n-1}
        fn   = f(xn,     y_curr)   # f_n

        y_pred = y_curr + h_ / 2 * (3 * fn - fn_1)

        f_pred = f(xn + h_, y_pred)
        y_corr = y_curr + h_ / 2 * (fn + f_pred)

        errl.append(y_corr - y_pred)
        xs.append(xn + h_)
        ys.append(y_corr)
        y_curr = y_corr

    return np.array(xs), np.array(ys), np.array(errl)

def adams2_runge_error(f, x0, y0, x_end, h):
    xs_rk, ys_rk = runge_kutta4(f, x0, y0, x0 + h, h)
    xs = list(xs_rk); ys = list(ys_rk); errs = [0.0, 0.0]
    while xs[-1] < x_end - 1e-12:
        xn = xs[-1]; h_ = min(h, x_end - xn)
        fn_1 = f(xs[-2], ys[-2])
        fn   = f(xn,     ys[-1])
        y_pred = ys[-1] + h_ / 2 * (3*fn - fn_1)
        y_corr = ys[-1] + h_ / 2 * (fn + f(xn + h_, y_pred))
        errs.append(abs(y_corr - y_pred) / 3)
        xs.append(xn + h_); ys.append(y_corr)
    return np.array(xs), np.array(errs)

def adams2_adaptive(f, x0, y0, x_end, eps):
    h = (x_end - x0) / 10
    xs_rk, ys_rk = runge_kutta4(f, x0, y0, x0 + h, h)
    xs = list(xs_rk)
    ys = list(ys_rk)
    hs = [h]

    while xs[-1] < x_end - 1e-12:
        xn  = xs[-1]
        h   = min(h, x_end - xn)
        fn_1 = f(xs[-2], ys[-2])
        fn   = f(xn,     ys[-1])

        y_pred = ys[-1] + h / 2 * (3 * fn - fn_1)
        f_pred = f(xn + h, y_pred)
        y_corr = ys[-1] + h / 2 * (fn + f_pred)

        err = abs(y_corr - y_pred) / 3

        if err > eps:
            h /= 2
            xs_rk, ys_rk = runge_kutta4(f, x0, y0, xs[-1], h)
            xs = list(xs_rk)
            ys = list(ys_rk)
            hs = [h] * len(xs)
            continue

        xs.append(xn + h)
        ys.append(y_corr)
        hs.append(h)

        if err < eps / 8:
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)

def plot_all():
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    x_fine = np.linspace(X0, X1, 500)
    y_exact_fine = exact(x_fine)

    xs_rk, ys_rk = runge_kutta4(f, X0, Y0, X1, H)
    y_exact_rk   = exact(xs_rk)

    xs_a, ys_a, errors = adams2(f, X0, Y0, X1, H)
    y_exact_a = exact(xs_a)
    ax = axes[2, 0]
    ax.semilogy(xs_a, np.abs(ys_a - y_exact_a) + 1e-16, 'b-o', ms=4)
    ax.set_title('Адамс 2-го порядку: |y_числ - y_точн|')
    ax.set_xlabel('x');
    ax.set_ylabel('Похибка')
    ax.grid(True)

    xs_aa, ys_aa, hs_aa = adams2_adaptive(f, X0, Y0, X1, EPS)
    ax = axes[2, 1]
    ax.plot(xs_aa[1:], hs_aa[1:], 'c-o', ms=3)
    ax.set_title(f'Адамс адаптивний: крок h(x), ε={EPS}')
    ax.set_xlabel('x');
    ax.set_ylabel('h')
    ax.grid(True)

    ax = axes[0, 0]
    xs_ar, errs_ar = adams2_runge_error(f, X0, Y0, X1, H)
    ax.semilogy(xs_ar[2:], errs_ar[2:] + 1e-16, 'c-o', ms=4)
    ax.set_title('Адамс 2: оцінка похибки (прогноз − корекція)/3')
    ax.set_xlabel('x');
    ax.set_ylabel('Похибка (оцінка)');
    ax.grid(True)

    ax = axes[0, 1]
    ax.semilogy(xs_rk, np.abs(ys_rk - y_exact_rk) + 1e-16, 'r-o', ms=4)
    ax.set_title('РК4: похибка |y_числ - y_точн|')
    ax.set_xlabel('x'); ax.set_ylabel('Похибка')
    ax.grid(True)

    runge_err = runge_error_rk4(f, X0, Y0, X1, H)
    xs_short  = xs_rk[:len(runge_err)]
    ax = axes[1, 0]
    ax.semilogy(xs_short, runge_err + 1e-16, 'g-o', ms=4)
    ax.set_title('РК4: оцінка похибки методом Рунге')
    ax.set_xlabel('x'); ax.set_ylabel('Похибка (оцінка)')
    ax.grid(True)

    xs_ad, ys_ad, hs_ad = adaptive_rk4(f, X0, Y0, X1, EPS)
    ax = axes[1, 1]
    ax.plot(xs_ad[1:], hs_ad, 'm-o', ms=3)
    ax.set_title(f'РК4 адаптивний: крок h(x), ε={EPS}')
    ax.set_xlabel('x'); ax.set_ylabel('h')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


print("=" * 55)
print("Рівняння: y' = -2y + x,  y(0) = 1,  x -> [0, 1]")
print(f"Крок h = {H},  Точність eps = {EPS}")
print("=" * 55)

xs_a, ys_a, _ = adams2(f, X0, Y0, X1, H)
print("\n--- Адамс 2-го порядку (перші 5 вузлів) ---")
print(f"{'x':>8} {'y_числ':>12} {'y_точн':>12} {'|похибка|':>12}")
for i in range(min(5, len(xs_a))):
    ye = exact(xs_a[i])
    print(f"{xs_a[i]:8.4f} {ys_a[i]:12.8f} {ye:12.8f} {abs(ys_a[i]-ye):12.2e}")

xs_rk, ys_rk = runge_kutta4(f, X0, Y0, X1, H)
print("\n--- РК4 (перші 5 вузлів) ---")
print(f"{'x':>8} {'y_числ':>12} {'y_точн':>12} {'|похибка|':>12}")
for i in range(min(5, len(xs_rk))):
    ye = exact(xs_rk[i])
    print(f"{xs_rk[i]:8.4f} {ys_rk[i]:12.8f} {ye:12.8f} {abs(ys_rk[i]-ye):12.2e}")

plot_all()