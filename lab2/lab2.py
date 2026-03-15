import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def write_data(filename, x, y, header=('rps', 'cpu')):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])


def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['rps']))
            y.append(float(row['cpu']))
    return x, y

def divided_differences(x, y):
    n = len(x)
    dd = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (x[i + j] - x[i])
    return dd


def print_divided_diff_table(x, y, dd):
    n = len(x)
    print("  Divided Diff table")
    for i in range(n):
        row = f"  {x[i]:>8.1f}  {y[i]:>10.4f}"
        for j in range(1, n - i):
            row += f"  {dd[i][j]:>16.8f}"
        print(row)

    print("\n  Coefs:")
    for k in range(n):
        print(f"    c[{k}] = dd[0][{k}] = {dd[0][k]:.8f}")
    print()


def newton_interpolate(x_nodes, y_nodes, t):
    dd = divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)

    result = dd[0][0]
    product = 1.0

    for k in range(1, n):
        product *= (t - x_nodes[k - 1])
        result += dd[0][k] * product

    return result


def newton_interpolate_vec(x_nodes, y_nodes, t_vals):
    return np.array([newton_interpolate(x_nodes, y_nodes, t) for t in t_vals])


def factorial_poly(m, s):
    result = 1.0
    for k in range(m):
        result *= (s - k)
    return result


def finite_differences(y_nodes):
    delta = list(y_nodes)
    result = [delta[0]]            # Δ⁰f(x₀) = f(x₀)
    for k in range(1, len(y_nodes)):
        delta = [delta[i + 1] - delta[i] for i in range(len(delta) - 1)]
        result.append(delta[0])    # Δᵏf(x₀)
    return result


def factorial_interpolate(x_nodes, y_nodes, t):
    n = len(x_nodes)
    h_eff = (x_nodes[-1] - x_nodes[0]) / (n - 1)
    s = (t - x_nodes[0]) / h_eff

    deltas = finite_differences(y_nodes)

    result = 0.0
    for k in range(n):
        fp = factorial_poly(k, s)           # s^(k)
        factorial_k = 1
        for j in range(1, k + 1):
            factorial_k *= j                # k!
        result += deltas[k] / factorial_k * fp

    return result


def factorial_interpolate_vec(x_nodes, y_nodes, t_vals):
    return np.array([factorial_interpolate(x_nodes, y_nodes, t) for t in t_vals])


COLORS = {
    'newton':    '#2563EB',
    'factorial': '#D97706',
    'nodes':     '#DC2626',
    'target':    '#16A34A',
    'err1':      '#7C3AED',
    'err2':      '#0891B2',   }


def plot_main(x_data, y_data, target_rps=600):
    t_dense = np.linspace(x_data[0], x_data[-1], 600)
    y_newton  = newton_interpolate_vec(x_data, y_data, t_dense)
    y_factor  = factorial_interpolate_vec(x_data, y_data, t_dense)
    pred_n    = newton_interpolate(x_data, y_data, target_rps)
    pred_f    = factorial_interpolate(x_data, y_data, target_rps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    configs = [
        (axes[0], y_newton, pred_n, COLORS['newton'],    'Newton'),
        (axes[1], y_factor, pred_f, COLORS['factorial'], 'Factorial'),
    ]

    for ax, y_interp, pred, color, title in configs:
        ax.plot(t_dense, y_interp, color=color, linewidth=2.5, label='Interpolation')
        ax.scatter(x_data, y_data, color=COLORS['nodes'], zorder=6, s=90, label='Nodes', edgecolors='white', linewidths=0.8)
        ax.axvline(target_rps, color=COLORS['target'], linestyle='--', alpha=0.8, linewidth=1.5)
        ax.scatter([target_rps], [pred], color=COLORS['target'], zorder=7, s=120, label=f'CPU({target_rps}) ≈ {pred:.2f}%', edgecolors='white', linewidths=1, marker='*')
        ax.annotate(f'  {pred:.2f}%', (target_rps, pred), fontsize=10, color=COLORS['target'], fontweight='bold', va='bottom')
        ax.set_xlabel('RPS', fontsize=11)
        ax.set_ylabel('CPU (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_data[0] - 20, x_data[-1] + 50)

    plt.tight_layout()
    plt.show()


def plot_node_study(x_data, y_data):
    t_dense = np.linspace(x_data[0], x_data[-1], 600)

    subsets = [
        ([0, 2, 4],        '3 nodes',     '#F59E0B'),
        ([0, 1, 2, 3],     '4 nodes',     '#3B82F6'),
        (list(range(5)),   '5 nodes', '#10B981'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = [
        (axes[0], newton_interpolate_vec,    'Newton'),
        (axes[1], factorial_interpolate_vec, 'Factorial'),
    ]

    for ax, fn, title in methods:
        for idx, label, color in subsets:
            xn = [x_data[i] for i in idx]
            yn = [y_data[i] for i in idx]
            y_interp = fn(xn, yn, t_dense)
            ax.plot(t_dense, y_interp, color=color, linewidth=2,
                    label=label, alpha=0.9)

        ax.scatter(x_data, y_data, color=COLORS['nodes'], zorder=6, s=90,
                   label='Nodes', edgecolors='white', linewidths=0.8)
        ax.set_xlabel('RPS', fontsize=11)
        ax.set_ylabel('CPU (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_data[0] - 20, x_data[-1] + 50)

    plt.tight_layout()
    plt.show()


def plot_error_analysis(x_data, y_data):
    t_dense = np.linspace(x_data[0], x_data[-1], 600)

    full_n = newton_interpolate_vec(x_data, y_data, t_dense)
    full_f = factorial_interpolate_vec(x_data, y_data, t_dense)

    subsets = [
        ([0, 2, 4],    '3 nodes',  COLORS['err1']),
        ([0, 1, 2, 3], '4 nodes',  COLORS['err2']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    configs = [
        (axes[0], newton_interpolate_vec,    full_n, 'Newton'),
        (axes[1], factorial_interpolate_vec, full_f, 'Factorial'),
    ]

    for ax, fn, full_ref, title in configs:
        for idx, label, color in subsets:
            xn = [x_data[i] for i in idx]
            yn = [y_data[i] for i in idx]
            y_sub = fn(xn, yn, t_dense)
            error = np.abs(y_sub - full_ref)
            ax.plot(t_dense, error, color=color, linewidth=2, label=label)
            ax.fill_between(t_dense, 0, error, color=color, alpha=0.08)

        ax.set_xlabel('RPS', fontsize=11)
        ax.set_ylabel('Error (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_step_study(x_data, y_data):
    t_dense = np.linspace(x_data[0], x_data[-1], 600)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    configs = [
        (axes[0], newton_interpolate_vec,    'Newton'),
        (axes[1], factorial_interpolate_vec, 'Factorial'),
    ]

    subsets_step = [
        ([0, 4],           '2 nodes', '#EF4444'),
        ([0, 2, 4],        '3 nodes', '#F59E0B'),
        ([0, 1, 2, 3, 4],  '5 nodes', '#10B981'),
    ]

    for ax, fn, title in configs:
        for idx, label, color in subsets_step:
            xn = [x_data[i] for i in idx]
            yn = [y_data[i] for i in idx]
            y_interp = fn(xn, yn, t_dense)
            ax.plot(t_dense, y_interp, color=color, linewidth=2, label=label)

        ax.scatter(x_data, y_data, color=COLORS['nodes'], zorder=6, s=90,
                   edgecolors='white', linewidths=0.8, label='Data')
        ax.set_xlabel('RPS', fontsize=11)
        ax.set_ylabel('CPU (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def generate_extended_data(x_base, y_base, n_total):
    x_new = list(np.linspace(x_base[0], x_base[-1], n_total))
    y_new = [newton_interpolate(x_base, y_base, xi) for xi in x_new]
    return x_new, y_new


def plot_runge_effect(x_base, y_base):
    t_dense = np.linspace(x_base[0], x_base[-1], 800)

    y_ref = newton_interpolate_vec(x_base, y_base, t_dense)

    configs = [
        (5,  '#10B981', '5 nodes '),
        (10, '#3B82F6', '10 nodes '),
        (20, '#EF4444', '20 nodes '),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for n, color, label in configs:
        xn, yn = generate_extended_data(x_base, y_base, n)
        y_interp = newton_interpolate_vec(xn, yn, t_dense)

        axes[0].plot(t_dense, y_interp, color=color, linewidth=1.8, label=label, alpha=0.85)
        axes[1].plot(t_dense, np.abs(y_interp - y_ref), color=color,
                     linewidth=1.8, label=label, alpha=0.85)

    axes[0].scatter(x_base, y_base, color=COLORS['nodes'], zorder=6, s=90,
                    edgecolors='white', linewidths=0.8, label='base 5 nodes')
    axes[0].set_title('Interpolation', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('RPS', fontsize=11)
    axes[0].set_ylabel('CPU (%)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title('Deviation from 5', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('RPS', fontsize=11)
    axes[1].set_ylabel('Error (%)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def write_extended_csv(x_base, y_base, out_dir):
    for n in [10, 20]:
        xn, yn = generate_extended_data(x_base, y_base, n)
        path = f'{out_dir}/data_{n}_nodes.csv'
        write_data(path, xn, yn)


def main():

    OUT = '/mnt/user-data/outputs'
    os.makedirs(OUT, exist_ok=True)

    rps_vals = [50.0, 100.0, 200.0, 400.0, 800.0]
    cpu_vals = [20.0,  35.0,  60.0, 110.0, 210.0]

    csv_path = f'{OUT}/data.csv'
    write_data(csv_path, rps_vals, cpu_vals)
    print(f"[1] Path to data {csv_path}")

    x_data, y_data = read_data(csv_path)
    print(f"    RPS : {x_data}")
    print(f"    CPU : {y_data}\n")

    print("Divided differences table")
    dd = divided_differences(x_data, y_data)
    print_divided_diff_table(x_data, y_data, dd)

    target = 600.0
    pred_newton = newton_interpolate(x_data, y_data, target)
    pred_factor = factorial_interpolate(x_data, y_data, target)

    print(f"   Prediction RPS = {int(target)}:                    ")
    print(f"   Newton             : {pred_newton:>10.4f} %             ")
    print(f"   Factorial      : {pred_factor:>10.4f} %             ")
    print(f"   Method diff    : {abs(pred_newton-pred_factor):>10.4f} %             \n")

    print("Accuracy assertion")
    print(f"\n  {'Method':<22} {'Nodes':<20} {'CPU(600)':<12} {'|Diff from 5|'}")
    print("  " + "─" * 68)

    subsets_study = [
        ([0, 4],        '2 nodes [50,800]'),
        ([0, 2, 4],     '3 nodes [50,200,800]'),
        ([0, 1, 2, 3],  '4 nodes [50,100,200,400]'),
        (list(range(5)),'5 nodes'),
    ]

    for idx, label in subsets_study:
        xn = [x_data[i] for i in idx]
        yn = [y_data[i] for i in idx]
        pn = newton_interpolate(xn, yn, target)
        pf = factorial_interpolate(xn, yn, target)
        print(f"  {'Newton':<22} {label:<20} {pn:<12.4f} {abs(pn - pred_newton):.4f}")
        print(f"  {'Factorial':<22} {label:<20} {pf:<12.4f} {abs(pf - pred_factor):.4f}")

    print("\nGraphs:")
    plot_main(x_data, y_data, target_rps=600)
    plot_step_study(x_data, y_data)
    plot_node_study(x_data, y_data)
    plot_error_analysis(x_data, y_data)

    print("\nRunge:")
    write_extended_csv(x_data, y_data, OUT)
    plot_runge_effect(x_data, y_data)

    print(f"\n  CPU prediction:")
    print(f"  {'Nodes':<10} {'CPU(600)':<14} {'Diff from 5|'}")
    ref = newton_interpolate(x_data, y_data, 600.0)
    for n in [5, 10, 20]:
        xn, yn = generate_extended_data(x_data, y_data, n)
        p = newton_interpolate(xn, yn, 600.0)
        print(f"  {n:<10} {p:<14.4f} {abs(p - ref):.4f}")


if __name__ == "__main__":
    main()