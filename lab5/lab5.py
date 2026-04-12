import numpy as np
import matplotlib.pyplot as plt

def M(t):
    """Вологість ґрунту: M(t) = 50*e^(-0.1t) + 5*sin(t)"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    """Точна похідна: M'(t) = -5*e^(-0.1t) + 5*cos(t)"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0
exact = dM_exact(t0)
print("=" * 55)
print("1. АНАЛІТИЧНЕ ДИФЕРЕНЦІЮВАННЯ")
print("=" * 55)
print(f"   M(t)  = 50·e^(-0.1t) + 5·sin(t)")
print(f"   M'(t) = -5·e^(-0.1t) + 5·cos(t)")
print(f"   M'({t0}) = {exact:.6f}")


def central_diff(t, h):
    """Центральна різниця: (M(t+h) - M(t-h)) / (2h)"""
    return (M(t + h) - M(t - h)) / (2 * h)

print("\n" + "=" * 55)
print("2. ЧИСЕЛЬНЕ ДИФЕРЕНЦІЮВАННЯ — ЗАЛЕЖНІСТЬ ВІД КРОКУ")
print("=" * 55)
print(f"   {'h':>12}  {'D(h)':>12}  {'|похибка|':>12}")
print("   " + "-" * 40)

steps = [10**(-k) for k in range(1, 9)]
errors = []
for h in steps:
    d = central_diff(t0, h)
    err = abs(d - exact)
    errors.append(err)
    print(f"   {h:>12.1e}  {d:>12.7f}  {err:>12.2e}")

opt_idx = np.argmin(errors)
h_opt = steps[opt_idx]
D_opt = central_diff(t0, h_opt)
print(f"\n   Оптимальний крок: h_opt = {h_opt:.1e}")
print(f"   D(h_opt) = {D_opt:.7f},  похибка = {errors[opt_idx]:.2e}")


print("\n" + "=" * 55)
print("3-5. ОБЧИСЛЕННЯ З ДВОМА КРОКАМИ")
print("=" * 55)

h1 = 0.01
h2 = h1 / 2

D1 = central_diff(t0, h1)
D2 = central_diff(t0, h2)
err1 = abs(D1 - exact)
err2 = abs(D2 - exact)

print(f"   h  = {h1}  →  D1 = {D1:.7f},  похибка = {err1:.2e}")
print(f"   h/2= {h2}  →  D2 = {D2:.7f},  похибка = {err2:.2e}")


print("\n" + "=" * 55)
print("6. МЕТОД РУНГЕ-РОМБЕРГА")
print("=" * 55)

p = 2
D_RR = D2 + (D2 - D1) / (2**p - 1)
err_RR = abs(D_RR - exact)

print(f"   Формула: D* = D2 + (D2 - D1) / (2^p - 1), p = {p}")
print(f"   D_RR = {D_RR:.7f}")
print(f"   Похибка RR = {err_RR:.2e}  (зменшилась у {err1/err_RR:.1f} разів)")


print("\n" + "=" * 55)
print("7. МЕТОД ЕЙТКЕНА")
print("=" * 55)

h3 = h2 / 2
D3 = central_diff(t0, h3)
err3 = abs(D3 - exact)
print(f"   h/4 = {h3}  →  D3 = {D3:.7f},  похибка = {err3:.2e}")


numerator   = np.log(abs((D3 - D2) / (D2 - D1)))
denominator = np.log(h3 / h2)
p_est = numerator / denominator
print(f"\n   Оцінка порядку точності: p ≈ {p_est:.2f}")

denom_aitken = D3 - 2*D2 + D1
if abs(denom_aitken) > 1e-15:
    D_Aitken = D1 - (D2 - D1)**2 / denom_aitken
else:
    D_Aitken = D_RR
err_Aitken = abs(D_Aitken - exact)

print(f"\n   D_Aitken = {D_Aitken:.7f}")
print(f"   Похибка Ейткена = {err_Aitken:.2e}  (зменшилась у {err1/err_Aitken:.1f} разів)")

print("\n" + "=" * 55)
print("ПІДСУМКОВА ТАБЛИЦЯ")
print("=" * 55)
print(f"   {'Метод':<28} {'Значення':>10} {'Похибка':>10}")
print("   " + "-" * 52)
print(f"   {'Точне значення':<28} {exact:>10.7f} {'—':>10}")
print(f"   {'Центральна різниця h=0.01':<28} {D1:>10.7f} {err1:>10.2e}")
print(f"   {'Центральна різниця h=0.005':<28} {D2:>10.7f} {err2:>10.2e}")
print(f"   {'Рунге-Ромберг':<28} {D_RR:>10.7f} {err_RR:>10.2e}")
print(f"   {'Ейткен':<28} {D_Aitken:>10.7f} {err_Aitken:>10.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].loglog(steps, errors, 'o-', color='#1D9E75', linewidth=2, markersize=6)
axes[0].axvline(h_opt, color='#D85A30', linestyle='--', label=f'h_opt={h_opt:.0e}')
axes[0].set_xlabel('Крок h')
axes[0].set_ylabel('|Похибка|')
axes[0].set_title('Похибка чисельного диференціювання від кроку h')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

methods = ['Центр. різн.\nh=0.01', 'Центр. різн.\nh=0.005', 'Рунге-\nРомберг', 'Ейткен']
errs    = [err1, err2, err_RR, err_Aitken]
colors  = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD']
bars = axes[1].bar(methods, errs, color=colors, edgecolor='none')
axes[1].set_yscale('log')
axes[1].set_ylabel('|Похибка| (log шкала)')
axes[1].set_title('Порівняння похибок методів')
for bar, err in zip(bars, errs):
    axes[1].text(bar.get_x() + bar.get_width()/2, err * 1.3,
                 f'{err:.1e}', ha='center', va='bottom', fontsize=10)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
