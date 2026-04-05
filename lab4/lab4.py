import numpy as np

def M(t):
    """Функція вологості ґрунту."""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_exact_derivative(t):
    """Аналітична перша похідна M'(t) = -5*e^(-0.1*t) + 5*cos(t)."""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(f, t, h):
    """Центральна різницева формула: (f(t+h) - f(t-h)) / (2h)."""
    return (f(t + h) - f(t - h)) / (2 * h)


print("1. Аналітичне розв'язання")

t0 = 1.0
exact = M_exact_derivative(t0)
print(f"  M(t)  = 50·e^(-0.1t) + 5·sin(t)")
print(f"  M'(t) = -5·e^(-0.1t) + 5·cos(t)")
print(f"\n  Точне значення M'({t0}) = {exact:.6f}")


print("2. Залежність похибки від кроку h")

print(f"\n  {'h':>12}  {'D(h)':>12}  {'Похибка':>12}")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}")

best_h = None
best_error = float('inf')
best_D = None


for exp in range(1, 11):
    h = 10 ** (-exp)
    D = central_diff(M, t0, h)
    error = abs(D - exact)
    marker = ""
    if error < best_error:
        best_error = error
        best_h = h
        best_D = D
        marker = "  <- оптимальний" if exp >= 5 else ""
    print(f"  {h:>12.0e}  {D:>12.6f}  {error:>12.2e}{marker}")

print(f"\n  Оптимальний крок: h_opt = {best_h:.0e}")
print(f"  Найкраща точність: {best_error:.2e}")


print("3–5. Фіксований крок h = 0.01")

h1 = 0.01
h2 = h1 / 2

D_h1 = central_diff(M, t0, h1)
D_h2 = central_diff(M, t0, h2)

err_h1 = abs(D_h1 - exact)
err_h2 = abs(D_h2 - exact)

print(f"\n  h  = {h1}   ->  D(h)   = {D_h1:.7f}   похибка = {err_h1:.2e}")
print(f"  h/2 = {h2}  ->  D(h/2) = {D_h2:.7f}   похибка = {err_h2:.2e}")

print("6. Метод Рунге-Ромберга")

p = 2
q = 2

D_RR = D_h2 + (D_h2 - D_h1) / (q**p - 1)
err_RR = abs(D_RR - exact)

print(f"\n  Формула: D_RR = D(h/2) + (D(h/2) - D(h)) / (2^p - 1)")
print(f"  p = {p}  ->  2^p - 1 = {q**p - 1}")
print(f"\n  D_RR    = {D_RR:.7f}")
print(f"  Точне   = {exact:.7f}")
print(f"  Похибка = {err_RR:.2e}  (було {err_h2:.2e})")
print(f"  Покращення у {err_h2/err_RR:.1f} разів")

print("7. Метод Ейткена")

h_a  = 0.01
h_b  = 0.005
h_c  = 0.0025

D_a = central_diff(M, t0, h_a)
D_b = central_diff(M, t0, h_b)
D_c = central_diff(M, t0, h_c)

ratio = (D_b - D_a) / (D_c - D_b) if (D_c - D_b) != 0 else float('nan')
p_est = np.log(abs(ratio)) / np.log(h_a / h_b) if ratio > 0 else float('nan')

denom_aitken = D_c - 2 * D_b + D_a
D_aitken = D_a - (D_b - D_a)**2 / denom_aitken if denom_aitken != 0 else float('nan')
err_aitken = abs(D_aitken - exact)

print(f"\n  Три кроки: h1={h_a}, h2={h_b}, h3={h_c}")
print(f"  D(h1) = {D_a:.8f}")
print(f"  D(h2) = {D_b:.8f}")
print(f"  D(h3) = {D_c:.8f}")
print(f"\n  Оцінка порядку точності p ≈ {p_est:.2f}")
print(f"\n  D_Ейткен = {D_aitken:.8f}")
print(f"  Точне    = {exact:.8f}")
print(f"  Похибка  = {err_aitken:.2e}  (було {abs(D_a - exact):.2e})")
print(f"  Покращення у {abs(D_a - exact)/err_aitken:.1f} разів")

print("ПІДСУМОК")

print(f"\n  {'Метод':<30}  {'Значення':>12}  {'Похибка':>12}")
print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
print(f"  {'Точне (аналітичне)':<30}  {exact:>12.7f}  {'—':>12}")
print(f"  {'Центр. різниця h=0.01':<30}  {D_h1:>12.7f}  {err_h1:>12.2e}")
print(f"  {'Центр. різниця h=0.005':<30}  {D_h2:>12.7f}  {err_h2:>12.2e}")
print(f"  {'Рунге-Ромберг':<30}  {D_RR:>12.7f}  {err_RR:>12.2e}")
print(f"  {'Ейткен':<30}  {D_aitken:>12.7f}  {err_aitken:>12.2e}")
