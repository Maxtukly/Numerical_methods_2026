import numpy as np

def generate_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.uniform(-10, 10, (n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) + rng.uniform(1, 5)
    return A


def compute_b(A: np.ndarray, x_exact: np.ndarray) -> np.ndarray:
    return A @ x_exact

def save_matrix(A: np.ndarray, filename: str) -> None:
    np.savetxt(filename, A, fmt="%.10f")
    print(f"  Матрицю збережено у '{filename}'")


def save_vector(v: np.ndarray, filename: str) -> None:
    np.savetxt(filename, v, fmt="%.10f")
    print(f"  Вектор збережено у '{filename}'")


def load_matrix(filename: str) -> np.ndarray:
    return np.loadtxt(filename)


def load_vector(filename: str) -> np.ndarray:
    return np.loadtxt(filename)


def matrix_vector_product(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x


def vector_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def matrix_norm(A: np.ndarray) -> float:
    return float(np.max(np.sum(np.abs(A), axis=1)))


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return vector_norm(matrix_vector_product(A, x) - b)


def simple_iteration(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-14,
    max_iter: int = 100_000,) -> tuple[np.ndarray, int]:
    eigenvalues = np.linalg.eigvalsh(A)
    lam_min, lam_max = eigenvalues.min(), eigenvalues.max()
    tau = 2.0 / (lam_min + lam_max)

    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = x - tau * (matrix_vector_product(A, x) - b)
        if vector_norm(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-14,
    max_iter: int = 100_000,) -> tuple[np.ndarray, int]:
    d = np.diag(A)
    R = A - np.diag(d)

    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = (b - matrix_vector_product(R, x)) / d
        if vector_norm(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-14,
    max_iter: int = 100_000,) -> tuple[np.ndarray, int]:
    n = len(b)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = x.copy()
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x_new[j]
            x_new[i] = s / A[i, i]
        if vector_norm(x_new - x) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


N = 100
EPS = 1e-14


A = generate_matrix(N)
x_exact = np.full(N, 2.5)
b = compute_b(A, x_exact)

save_matrix(A, "matrix_A.txt")
save_vector(b, "vector_b.txt")

A = load_matrix("matrix_A.txt")
b = load_vector("vector_b.txt")

x0 = np.array([1.0 * (1 + i) for i in range(N)])

results = {}

x_si, iters_si = simple_iteration(A, b, x0, eps=EPS)
err_si = vector_norm(x_si - x_exact)
res_si = residual_norm(A, x_si, b)
results["Проста ітерація"] = (x_si, iters_si, err_si, res_si)

x_jac, iters_jac = jacobi(A, b, x0, eps=EPS)
err_jac = vector_norm(x_jac - x_exact)
res_jac = residual_norm(A, x_jac, b)
results["Якобі"] = (x_jac, iters_jac, err_jac, res_jac)

x_sei, iters_sei = seidel(A, b, x0, eps=EPS)
err_sei = vector_norm(x_sei - x_exact)
res_sei = residual_norm(A, x_sei, b)
results["Зейдель"] = (x_sei, iters_sei, err_sei, res_sei)

print(f"{'Метод':<20} {'Ітерацій':>10} {'‖x−x*‖':>14} {'‖Ax−b‖':>14}")
print("-" * 72)
for name, (_, iters, err, res) in results.items():
    print(f"{name:<20} {iters:>10} {err:>14.4e} {res:>14.4e} ")

print("\nПерші 5 компонент розв'язку (точний = 2.5):")
print(f"{'Метод':<20}", end="")
for j in range(5):
    print(f"  x[{j}]", end="")
print()
for name, (x, *_) in results.items():
    print(f"{name:<20}", end="")
    for j in range(5):
        print(f"  {x[j]:.14f}", end="")
    print()