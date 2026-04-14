import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi as math_pi

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

def exact_integral(a, b):
    def F(x):
        t1 = 50 * x
        t2 = -(240 / math_pi) * np.cos(math_pi * x / 12)
        coeff = 5 * sqrt(math_pi / 0.2) / 2   # = 5·√(5π)/2
        t3 = coeff * erf(sqrt(0.2) * (x - 12))
        return t1 + t2 + t3
    return F(b) - F(a)

I0 = exact_integral(a, b)
print(f"Точне значення інтегралу I0 = {I0:.10f}")


def simpson(f, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    fx = f(x)
    result = fx[0] + fx[-1]
    result += 4 * np.sum(fx[1:-1:2])
    result += 2 * np.sum(fx[2:-2:2])
    return result * h / 3

N_values = range(10, 1001, 2)
errors = [abs(simpson(f, a, b, N) - I0) for N in N_values]

plt.figure(figsize=(10, 5))
plt.semilogy(list(N_values), errors)
plt.xlabel('N (кількість розбиттів)')
plt.ylabel('ε(N) = |I(N) - I₀|')
plt.title('Залежність похибки формули Сімпсона від N')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()


target_eps = 1e-12
N_opt = None
for N in range(10, 10001, 2):
    if abs(simpson(f, a, b, N) - I0) <= target_eps:
        N_opt = N
        break

if N_opt is None:
    N_opt = 1000
    print("Увага: точність 1e-12 не досягнута при N<=10000, використовуємо N=1000")

eps_opt = abs(simpson(f, a, b, N_opt) - I0)
print(f"\nN_opt = {N_opt}  (перше N при якому ε ≤ 1e-12)")
print(f"epsopt = |I(N_opt) - I0| = {eps_opt:.2e}")

N0_raw = N_opt // 10
N0 = max(8, (N0_raw // 8) * 8)
if N0 < 8:
    N0 = 8
if N0 < 16:
    N0 = 16

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"\nN0 = {N0}")
print(f"I(N0) = {I_N0:.10f}")
print(f"eps0 = |I(N0) - I0| = {eps0:.2e}")

I_N0_half = simpson(f, a, b, N0 // 2)

p = 4
I_R = I_N0 + (I_N0 - I_N0_half) / (2**p - 1)
epsR = abs(I_R - I0)
print(f"\n--- Метод Рунге-Ромберга ---")
print(f"I(N0/2)  = {I_N0_half:.10f}")
print(f"I(N0)    = {I_N0:.10f}")
print(f"I_R      = {I_R:.10f}")
print(f"epsR = |I_R - I0| = {epsR:.2e}")

h1 = (b - a) / N0
h2 = h1 / 2
h3 = h1 / 4

N1 = N0
N2 = N0 * 2
N3 = N0 * 4

I1 = simpson(f, a, b, N1)
I2 = simpson(f, a, b, N2)
I3 = simpson(f, a, b, N3)

log_ratio = np.log((I1 - I2) / (I2 - I3)) / np.log(2)
p_aitken = log_ratio
print(f"\n--- Метод Ейткена ---")
print(f"I(N0)    = {I1:.10f}")
print(f"I(2*N0)  = {I2:.10f}")
print(f"I(4*N0)  = {I3:.10f}")
print(f"Оцінка порядку p = {p_aitken:.4f}")

q = 2**p_aitken
I_A = I3 + (I3 - I2) / (q - 1)
epsA = abs(I_A - I0)
print(f"I_Aitken = {I_A:.10f}")
print(f"epsA = |I_A - I0| = {epsA:.2e}")

def adaptive_simpson(f, a, b, tol, depth=0, max_depth=50):
    c = (a + b) / 2
    h = b - a

    fa, fc, fb = f(a), f(c), f(b)
    S1 = h / 6 * (fa + 4*fc + fb)

    d1, d2 = (a + c) / 2, (c + b) / 2
    fd1, fd2 = f(d1), f(d2)
    S2 = h / 12 * (fa + 4*fd1 + fc) + h / 12 * (fc + 4*fd2 + fb)

    if depth >= max_depth or abs(S2 - S1) < 15 * tol:
        return S2 + (S2 - S1) / 15
    else:
        left  = adaptive_simpson(f, a, c, tol/2, depth+1, max_depth)
        right = adaptive_simpson(f, c, b, tol/2, depth+1, max_depth)
        return left + right

print(f"\n--- Адаптивний алгоритм ---")
call_counter = [0]
def f_counted(x):
    call_counter[0] += 1
    return f(x)

for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
    call_counter[0] = 0
    I_adapt = adaptive_simpson(f_counted, a, b, tol)
    eps_adapt = abs(I_adapt - I0)
    print(f"tol={tol:.0e}  I={I_adapt:.8f}  похибка={eps_adapt:.2e}  викликів f={call_counter[0]}")