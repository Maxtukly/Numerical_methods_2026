import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-10
MAX_ITER = 1000

def f(x):
    return x * np.sin(x) - np.cos(x)

def df(x):
    return 2 * np.sin(x) + x * np.cos(x)

def ddf(x):
    return 2 * np.cos(x) - x * np.sin(x)


def tabulate_and_find_roots(a=-1.0, b=10.0, h=0.1, filename="tabulation.txt"):
    xs = np.arange(a, b + h, h)
    fxs = [f(x) for x in xs]

    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(f"{'x':>10}  {'f(x)':>15}\n")
        fp.write("-" * 28 + "\n")
        for x, fx in zip(xs, fxs):
            fp.write(f"{x:10.4f}  {fx:15.8f}\n")

    brackets = []
    for i in range(len(fxs) - 1):
        if fxs[i] * fxs[i + 1] < 0:
            brackets.append((xs[i], xs[i + 1]))

    print("=" * 60)
    print("[1] Tabulation")
    print(f"  Tabulated on [{a}, {b}] with step h={h}")
    print(f"  Results saved to '{filename}'")
    print(f"  Sign-change brackets found: {brackets}")
    return brackets


def plot_function(a=-1.0, b=10.0, h=0.05):
    xs = np.arange(a, b + h, h)
    ys = [f(x) for x in xs]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, label=r"$f(x) = x\sin x - \cos x$")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Transcendental function tabulation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def simple_iteration(x0, g, eps=EPS, max_iter=MAX_ITER):
    x = x0
    for n in range(1, max_iter + 1):
        x_new = g(x)
        if abs(x_new - x) < eps and abs(f(x_new)) < eps:
            return x_new, n
        x = x_new
    return x, max_iter


def newton(x0, func=f, dfunc=df, eps=EPS, max_iter=MAX_ITER):
    x = x0
    for n in range(1, max_iter + 1):
        fx = func(x)
        dfx = dfunc(x)
        if dfx == 0:
            raise ZeroDivisionError("f'(x) == 0 during Newton iteration")
        x_new = x - fx / dfx
        if abs(x_new - x) < eps and abs(func(x_new)) < eps:
            return x_new, n
        x = x_new
    return x, max_iter


def chebyshev(x0, func=f, dfunc=df, d2func=ddf, eps=EPS, max_iter=MAX_ITER):
    x = x0
    for n in range(1, max_iter + 1):
        fx = func(x)
        d1 = dfunc(x)
        d2 = d2func(x)
        if d1 == 0:
            raise ZeroDivisionError("f'(x) == 0 during Chebyshev iteration")
        ratio = fx / d1
        x_new = x - ratio - (ratio ** 2) * d2 / (2 * d1)
        if abs(x_new - x) < eps and abs(func(x_new)) < eps:
            return x_new, n
        x = x_new
    return x, max_iter


def chord(x0, x1, func=f, eps=EPS, max_iter=MAX_ITER):
    for n in range(1, max_iter + 1):
        f0, f1 = func(x0), func(x1)
        if f1 - f0 == 0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < eps and abs(func(x2)) < eps:
            return x2, n
        x0, x1 = x1, x2
    return x1, max_iter


def parabola(x0, x1, x2, func=f, eps=EPS, max_iter=MAX_ITER):
    for n in range(1, max_iter + 1):
        f0, f1, f2 = func(x0), func(x1), func(x2)
        d01 = (f1 - f0) / (x1 - x0) if x1 != x0 else 0
        d12 = (f2 - f1) / (x2 - x1) if x2 != x1 else 0
        d012 = (d12 - d01) / (x2 - x0) if x2 != x0 else 0
        w = d12 + d012 * (x2 - x1)
        disc = w * w - 4 * f2 * d012
        if disc < 0:
            disc = 0
        sq = np.sqrt(disc)
        denom1 = w + sq
        denom2 = w - sq
        denom = denom1 if abs(denom1) >= abs(denom2) else denom2
        if denom == 0:
            break
        x3 = x2 - 2 * f2 / denom
        if abs(x3 - x2) < eps and abs(func(x3)) < eps:
            return x3, n
        x0, x1, x2 = x1, x2, x3
    return x2, max_iter


def inverse_interpolation(x0, x1, x2, func=f, eps=EPS, max_iter=MAX_ITER):
    for n in range(1, max_iter + 1):
        f0, f1, f2 = func(x0), func(x1), func(x2)
        try:
            x3 = (x0 * f1 * f2 / ((f0 - f1) * (f0 - f2)) +
                  x1 * f0 * f2 / ((f1 - f0) * (f1 - f2)) +
                  x2 * f0 * f1 / ((f2 - f0) * (f2 - f1)))
        except ZeroDivisionError:
            break
        if abs(x3 - x2) < eps and abs(func(x3)) < eps:
            return x3, n
        x0, x1, x2 = x1, x2, x3
    return x2, max_iter


def aitken(x0, g, eps=EPS, max_iter=MAX_ITER):
    x = x0
    for n in range(1, max_iter + 1):
        x1 = g(x)
        x2 = g(x1)
        denom = x2 - 2 * x1 + x
        if abs(denom) < 1e-15:
            return x2, n
        x_new = x - (x1 - x) ** 2 / denom
        if abs(x_new - x) < eps and abs(f(x_new)) < eps:
            return x_new, n
        x = x_new
    return x, max_iter


def solve_transcendental(brackets):
    root1_bracket = None
    root2_bracket = None
    for a, b in brackets:
        mid = (a + b) / 2
        if df(mid) > 0 and root1_bracket is None and mid > 0:
            root1_bracket = (a, b)
        elif df(mid) < 0 and root2_bracket is None and mid > 0:
            root2_bracket = (a, b)

    if root1_bracket is None or root2_bracket is None:
        print("  Could not find two roots with different behaviour; using first two brackets.")
        root1_bracket = brackets[0]
        root2_bracket = brackets[1] if len(brackets) > 1 else brackets[0]

    cases = [
        ("Root (f increasing)", root1_bracket),
        ("Root (f decreasing)", root2_bracket),
    ]

    print("\n" + "=" * 60)
    print("[2-4] Root-finding methods comparison")
    print(f"  Accuracy ε = {EPS}")

    for label, (a, b) in cases:
        x0 = (a + b) / 2
        x_prev = a

        alpha = 1.0 / df(x0)
        g = lambda x, a=alpha: x - a * f(x)

        results = {}
        results["Simple Iteration"]      = simple_iteration(x0, g)
        results["Aitken"]                = aitken(x0, g)
        results["Newton"]                = newton(x0)
        results["Chebyshev"]             = chebyshev(x0)
        results["Chord (Secant)"]        = chord(x_prev, x0)
        results["Parabola (Muller)"]     = parabola(x_prev - 0.05, x_prev, x0)
        results["Inverse Interpolation"] = inverse_interpolation(x_prev - 0.05, x_prev, x0)

        print(f"\n  {label}  (starting x0 ≈ {x0:.4f})")
        print(f"  {'Method':<25} {'Root':>14}  {'f(root)':>14}  {'Iters':>6}")
        print("  " + "-" * 65)
        for name, (root, iters) in results.items():
            print(f"  {name:<25} {root:14.8f}  {f(root):14.2e}  {iters:6d}")


def horner(coeffs, x):
    n = len(coeffs)
    b = coeffs[0]
    db = 0.0
    for i in range(1, n):
        db = b + db * x
        b = coeffs[i] + b * x
    return b, db


def poly_func(coeffs):
    def p(x):
        val, _ = horner(coeffs, x)
        return val
    return p

def poly_dfunc(coeffs):
    def dp(x):
        _, d = horner(coeffs, x)
        return d
    return dp

POLY_COEFFS = [1, -1, 4, -6]

def save_coeffs(coeffs, filename="polynomial.txt"):
    with open(filename, "w", encoding="utf-8") as fp:
        fp.write("# Polynomial coefficients (highest degree first)\n")
        fp.write("# p(x) = a_n*x^n + ... + a_1*x + a_0\n")
        fp.write(" ".join(map(str, coeffs)) + "\n")
    print(f"\n  Coefficients saved to '{filename}'")

def load_coeffs(filename="polynomial.txt"):
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("#"):
                return list(map(float, line.split()))
    raise ValueError("No coefficient data found in file.")


def newton_horner(x0, coeffs, eps=EPS, max_iter=MAX_ITER):
    pf  = poly_func(coeffs)
    pdf = poly_dfunc(coeffs)
    return newton(x0, func=pf, dfunc=pdf, eps=eps, max_iter=max_iter)


def plot_polynomial(coeffs, a=-2.0, b=4.0, h=0.05):
    xs = np.arange(a, b + h, h)
    ys = [poly_func(coeffs)(x) for x in xs]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="p(x)")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    degree = len(coeffs) - 1
    title = "p(x) = " + " + ".join(
        f"{c}·x^{degree-i}" if degree-i > 0 else str(c)
        for i, c in enumerate(coeffs)
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve_algebraic_real(coeffs):
    print("\n" + "=" * 60)
    print("[5-8] Algebraic equation")
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = degree - i
        if c == 0:
            continue
        if p == 0:
            terms.append(str(int(c)))
        elif p == 1:
            terms.append(f"{int(c)}x")
        else:
            terms.append(f"{int(c)}x^{p}")
    print("  Equation: " + " + ".join(terms).replace("+ -", "- ") + " = 0")

    np_roots = np.roots(coeffs)
    print(f"  NumPy reference roots: {np_roots}")

    pf = poly_func(coeffs)
    xs = np.arange(-5.0, 5.0, 0.1)
    x0_real = None
    for i in range(len(xs) - 1):
        if pf(xs[i]) * pf(xs[i+1]) < 0:
            x0_real = (xs[i] + xs[i+1]) / 2
            break

    if x0_real is None:
        x0_real = 1.0
        print("  Warning: no sign change found; using x0=1.0")

    real_root, iters = newton_horner(x0_real, coeffs)
    print(f"\n  Newton–Horner result:")
    print(f"    Real root ≈ {real_root:.8f}  (f={pf(real_root):.2e}, {iters} iters)")

    return real_root

def lin_method(coeffs, p0=None, q0=None, eps=EPS, max_iter=MAX_ITER):
    n = len(coeffs) - 1
    a = [c / coeffs[0] for c in coeffs]

    if p0 is None:
        p0 = -a[1]
    if q0 is None:
        q0 = a[2] if n >= 2 else 1.0

    p, q = p0, q0

    for iteration in range(1, max_iter + 1):
        b = [0.0] * (n + 1)
        b[0] = a[0]
        b[1] = a[1] + p * b[0]
        for i in range(2, n + 1):
            b[i] = a[i] + p * b[i-1] - q * b[i-2]

        c = [0.0] * (n + 1)
        c[0] = b[0]
        c[1] = b[1] + p * c[0]
        for i in range(2, n):
            c[i] = b[i] + p * c[i-1] - q * c[i-2]

        det = c[n-2] * (-c[n-2]) - (-c[n-3]) * c[n-1]
        if abs(det) < 1e-15:
            break
        dp = (b[n-1] * (-c[n-2]) - (-c[n-3]) * b[n]) / det
        dq = (c[n-2] * b[n] - b[n-1] * c[n-1]) / det

        p += dp
        q += dq

        if abs(dp) < eps and abs(dq) < eps:
            return p, q, iteration

    return p, q, max_iter


def solve_complex_roots(coeffs, real_root):
    print("\n" + "=" * 60)
    print("[9] Complex roots via Lin's method")

    n = len(coeffs)
    deflated = [coeffs[0]]
    for i in range(1, n - 1):
        deflated.append(coeffs[i] + deflated[-1] * real_root)

    print(f"  Deflated polynomial coefficients: {[round(c, 6) for c in deflated]}")

    p, q, iters = lin_method(deflated)
    print(f"  Quadratic factor: x² - ({p:.6f})x + ({q:.6f})")
    print(f"  Lin's method converged in {iters} iterations")

    discriminant = p**2 - 4*q
    if discriminant < 0:
        re = p / 2
        im = np.sqrt(-discriminant) / 2
        r1 = complex(re,  im)
        r2 = complex(re, -im)
    else:
        sq = np.sqrt(discriminant)
        r1 = complex((p + sq) / 2, 0)
        r2 = complex((p - sq) / 2, 0)

    print(f"  Complex roots:")
    print(f"    z1 = {r1:.8f}")
    print(f"    z2 = {r2:.8f}")


    pf_c = lambda z: sum(c * z**(len(coeffs)-1-i) for i, c in enumerate(coeffs))
    print(f"  Verification  p(z1) = {pf_c(r1):.2e}")
    print(f"  Verification  p(z2) = {pf_c(r2):.2e}")



brackets = tabulate_and_find_roots(a=-1.0, b=10.0, h=0.1)
plot_function(a=-1.0, b=10.0)

if len(brackets) < 2:
    print("  Warning: fewer than 2 brackets found; results may be limited.")

solve_transcendental(brackets)

save_coeffs(POLY_COEFFS)
loaded = load_coeffs("polynomial.txt")
print(f"\n  Loaded coefficients from file: {loaded}")

plot_polynomial(loaded)
real_root = solve_algebraic_real(loaded)
solve_complex_roots(loaded, real_root)
