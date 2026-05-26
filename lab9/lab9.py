import matplotlib.pyplot as plt
import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

'''def powerf(x):
    return ((1 - x[0])**2 + 10 * (x[1] - x[0]**2)**2)**4'''

def system_f1(x):
    return x[0]**2 + x[1]**2 - 4

def system_f2(x):
    return x[0] * x[1] - 1

def system_target(x):
    return system_f1(x)**2 + system_f2(x)**2

def hooke_jeeves(f, x0, h0=1.0, alpha=2.0, epsilon1=1e-6, epsilon2=1e-6):
    n = len(x0)
    x_base = list(x0)
    h = [h0] * n
    trajectory = [list(x_base)]
    steps = 0

    def exploratory_search(start, h_vec):
        x = list(start)
        for i in range(n):
            x_new = list(x)
            x_new[i] += h_vec[i]
            if f(x_new) < f(x):
                x = x_new
            else:
                x_new[i] = x[i] - h_vec[i]
                if f(x_new) < f(x):
                    x = x_new
        return x

    max_iter = 100_000
    iteration = 0

    while iteration < max_iter:
        iteration += 1

        x_new = exploratory_search(x_base, h)

        if f(x_new) < f(x_base):
            while True:
                steps += 1
                x_pattern = [x_new[i] + (x_new[i] - x_base[i]) for i in range(n)]
                x_base_old = list(x_base)
                x_base = list(x_new)

                x_exp = exploratory_search(x_pattern, h)

                if f(x_exp) < f(x_base):
                    x_new = list(x_exp)
                    trajectory.append(list(x_base))
                else:
                    x_new = exploratory_search(x_base, h)
                    trajectory.append(list(x_base))

                    if f(x_new) < f(x_base):
                        continue
                    else:
                        break

            delta_x = np.sqrt(sum((x_base[i] - x_base_old[i])**2 for i in range(n)))
            delta_f = abs(f(x_base) - f(x_base_old))

            if delta_x < epsilon1 and delta_f < epsilon2:
                break
        else:
            h = [hi / alpha for hi in h]
            if max(h) < epsilon1:
                break

    return x_base, f(x_base), trajectory, steps

def test_rosenbrock():
    x0 = [-1.2, 1.0]
    x_min, f_min, traj, steps = hooke_jeeves(rosenbrock, x0, h0=0.5)

    print(f"Початкова точка:  {x0}")
    print(f"Знайдений мінімум: ({x_min[0]:.6f}, {x_min[1]:.6f})")
    print(f"Значення функції:  {f_min:.2e}")
    print(f"Кількість кроків:  {steps}")
    print(f"Точок траєкторії:  {len(traj)}")


    X = np.linspace(-2, 2, 400)
    Y = np.linspace(-1, 3, 400)
    Z = np.array([[rosenbrock([xi, yi]) for xi in X] for yi in Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(X, Y, np.log1p(Z), levels=40, cmap='viridis')
    plt.colorbar(cp, ax=ax, label='log(1 + f)')

    tx = [p[0] for p in traj]
    ty = [p[1] for p in traj]
    ax.plot(tx, ty, 'w.-', linewidth=1, markersize=3, label='Траєкторія')
    ax.plot(x0[0], x0[1], 'co', markersize=8, label='Початок')
    ax.plot(x_min[0], x_min[1], 'r*', markersize=14, label='Мінімум')
    ax.plot(1, 1, 'y+', markersize=12, markeredgewidth=2, label='Істинний мінімум')

    ax.set_title('Функція Розенброка — метод Хука-Дживса')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return traj

def solve_system():
    print()
    print("=" * 55)
    print("СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ")
    print("  f1(x,y) = x² + y² - 4 = 0")
    print("  f2(x,y) = x·y - 1     = 0")
    print("=" * 55)

    results = []

    starts = [(1.5, 0.5), (-1.5, -0.5), (0.5, 1.5), (-0.5, -1.5)]

    for x0 in starts:
        x_min, f_min, traj, steps = hooke_jeeves(
            system_target, list(x0), h0=0.5, epsilon1=1e-8, epsilon2=1e-12
        )
        results.append((x0, x_min, f_min, traj, steps))
        print(f"\nПоч. наближення: {x0}")
        print(f"  Розв'язок: x={x_min[0]:+.6f},  y={x_min[1]:+.6f}")
        print(f"  F(розв.) = {f_min:.2e}   (кроків: {steps})")
        print(f"  Перевірка: f1={system_f1(x_min):.2e},  f2={system_f2(x_min):.2e}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 500)

    ax.plot(2 * np.cos(t), 2 * np.sin(t), 'b-', linewidth=2, label='$x^2+y^2=4$')

    xh = np.linspace(0.3, 3.5, 300)
    ax.plot(xh, 1 / xh, 'r-', linewidth=2, label='$xy=1$')
    ax.plot(-xh, -1 / xh, 'r-', linewidth=2)

    unique = []
    for _, x_min, f_min, traj, _ in results:
        if f_min < 1e-8:
            key = (round(x_min[0], 3), round(x_min[1], 3))
            if key not in unique:
                unique.append(key)
                ax.plot(x_min[0], x_min[1], 'k*', markersize=14,
                        label=f'({x_min[0]:.3f}, {x_min[1]:.3f})')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title('Графіки рівнянь системи')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    X = np.linspace(-2.5, 2.5, 300)
    Y = np.linspace(-2.5, 2.5, 300)
    Z = np.array([[system_target([xi, yi]) for xi in X] for yi in Y])
    cp = ax2.contourf(X, Y, np.log1p(Z), levels=50, cmap='plasma')
    plt.colorbar(cp, ax=ax2, label='log(1 + F)')

    for x0, x_min, f_min, traj, _ in results:
        tx = [p[0] for p in traj]
        ty = [p[1] for p in traj]
        ax2.plot(tx, ty, 'w-', linewidth=0.8, alpha=0.5)
        ax2.plot(x0[0], x0[1], 'co', markersize=6)
        if f_min < 1e-8:
            ax2.plot(x_min[0], x_min[1], 'y*', markersize=12)

    ax2.set_title('Цільова функція $F = f_1^2 + f_2^2$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    with open('trajectory.txt', 'w', encoding='utf-8') as f:
        f.write("Траєкторія спуску (перший розв'язок)\n")
        f.write(f"{'Крок':>5}  {'x':>12}  {'y':>12}  {'F(x,y)':>14}\n")
        f.write("-" * 50 + "\n")
        _, _, _, traj0, _ = results[0]
        for k, pt in enumerate(traj0):
            f.write(f"{k:>5}  {pt[0]:>12.6f}  {pt[1]:>12.6f}  {system_target(pt):>14.6e}\n")
    print("Траєкторію збережено: trajectory.txt")



test_rosenbrock()
solve_system()