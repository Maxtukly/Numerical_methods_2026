import csv
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    x, y = [], []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return x, y


def form_matrix(x, m):
    n = m + 1
    A = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(xk**(i+j) for xk in x)
    return A

def form_vector(x, y, m):
    n = m + 1
    b = [0.0]*n
    for i in range(n):
        b[i] = sum(y[k] * x[k]**i for k in range(len(x)))
    return b

def gauss_solve(A_in, b_in):
    n = len(b_in)

    A = [row[:] for row in A_in]
    b = b_in[:]

    for k in range(n):

        max_row = k
        for i in range(k+1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        if abs(A[k][k]) < 1e-12:
            raise ValueError(f"Unable to solve on step {k}")


        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]


    x_sol = [0.0]*n
    for i in range(n-1, -1, -1):
        s = sum(A[i][j] * x_sol[j] for j in range(i+1, n))
        x_sol[i] = (b[i] - s) / A[i][i]

    return x_sol

def polynomial(x_vals, coef):
    result = []
    for xv in x_vals:
        val = sum(coef[i] * xv**i for i in range(len(coef)))
        result.append(val)
    return result

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i])**2 for i in range(n)) / n



def find_optimal_degree(x, y, max_degree):
    variances = []
    for m in range(1, max_degree+1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append((m, var, coef))
    return variances

csv_path = "data.csv"
x, y = read_data(csv_path)

print(f"\nN: {len(x)}")

print("\n=== Data ===")
print("Month: ")
print(x)
print("Temps: ")
print(y)

max_deg = 6
variances_data = find_optimal_degree(x, y, max_degree=max_deg)

print("Variances: ")
print(variances_data)

optimal = min(variances_data, key=lambda t: t[1])
opt_m, opt_var, opt_coef = optimal
print(f"\nOptimal m: m = {opt_m}  (dispr = {opt_var:.6f})")

y_approx = polynomial(x, opt_coef)
errors   = [y[i] - y_approx[i] for i in range(len(x))]


x_future = [25.0, 26.0, 27.0]
y_future = polynomial(x_future, opt_coef)
print("Prediction for 25, 26, 27:")
print(y_future)

x_np   = np.array(x)
y_np   = np.array(y)
x_fine = np.linspace(min(x)-0.5, 27.5, 500)
y_fine = np.array(polynomial(x_fine.tolist(), opt_coef))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


ax1 = axes[0]
ax1.plot(x_fine, y_fine, 'b-', linewidth=2, label=f'Poly m={opt_m}')
ax1.scatter(x_np, y_np, color='red', zorder=5, s=60, label='Data')
ax1.scatter(x_future, y_future, color='green', marker='*', s=150, zorder=5,
            label='Prediction (25-27)')
for xf, yf in zip(x_future, y_future):
    ax1.annotate(f'{yf:.1f}', (xf, yf), textcoords="offset points",
                 xytext=(5, 5), fontsize=8, color='green')

ax1.set_xlabel('Month')
ax1.set_ylabel('Temp (°C)')
ax1.set_title('Data and Prediction')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.bar(x_np, errors, color='orange', alpha=0.7, label='Error')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xlabel('Month')
ax2.set_ylabel('Error (°C)')
ax2.set_title('Approximation error')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
