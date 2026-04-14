import numpy as np

N = 100
X_TRUE = 2.5
EPS_REFINE = 1e-14
MAX_ITER = 50

def generate_and_save(n, x_val, file_a="matrix_A.txt", file_b="vector_B.txt"):
    rng = np.random.default_rng(42)
    A = rng.uniform(-1.0, 1.0, (n, n))
    np.fill_diagonal(A, n)

    x = np.full(n, x_val)
    b = A @ x

    np.savetxt(file_a, A, fmt="%.10f")
    np.savetxt(file_b, b, fmt="%.10f")
    print(f"[GEN] Матриця A ({n}x{n})  ->  {file_a}")
    print(f"[GEN] Вектор  B ({n})      ->  {file_b}")
    return A, b

def read_matrix(path):
    return np.loadtxt(path)

def read_vector(path):
    return np.loadtxt(path)

def lu_decompose(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k:, k] = A[k:, k] - L[k:, :k] @ U[:k, k]

        U[k, k] = 1.0
        if k + 1 < n:
            U[k, k+1:] = (A[k, k+1:] - L[k, :k] @ U[:k, k+1:]) / L[k, k]

    return L, U

def save_lu(L, U, file_l="matrix_L.txt", file_u="matrix_U.txt"):
    np.savetxt(file_l, L, fmt="%.10f")
    np.savetxt(file_u, U, fmt="%.10f")
    print(f"[LU]  L  ->  {file_l}")
    print(f"[LU]  U  ->  {file_u}")

def forward_substitution(L, b):
    n = len(b)
    z = np.zeros(n)
    z[0] = b[0] / L[0, 0]
    for k in range(1, n):
        z[k] = (b[k] - L[k, :k] @ z[:k]) / L[k, k]
    return z

def back_substitution(U, z):
    n = len(z)
    x = np.zeros(n)
    x[n-1] = z[n-1]          # u_{n-1,n-1} = 1
    for k in range(n-2, -1, -1):
        x[k] = z[k] - U[k, k+1:] @ x[k+1:]
    return x

def solve_lu(L, U, b):
    z = forward_substitution(L, b)
    x = back_substitution(U, z)
    return x

def mat_vec(A, x):
    return A @ x

def vec_norm(v):
    return float(np.max(np.abs(v)))

def iterative_refinement(A, L, U, b, x0, eps=EPS_REFINE, max_iter=MAX_ITER):
    x = x0.copy()
    prev_err_ax = vec_norm(b - mat_vec(A, x))

    for it in range(1, max_iter + 1):
        r  = b - mat_vec(A, x)
        dx = solve_lu(L, U, r)
        x  = x + dx

        err_x  = vec_norm(dx)
        err_ax = vec_norm(b - mat_vec(A, x))

        print(f"  Ітерація {it:3d}:  ||dX|| = {err_x:.3e}   ||AX-B|| = {err_ax:.3e}")

        if err_x <= eps:
            print(f"\n[ITER] Збіжність досягнута за {it} ітерацій.")
            return x, it

        if err_ax >= prev_err_ax * 0.99 and it > 1:
            print(f"\n[ITER] Нев'язка стабілізована (машинна точність). Ітерацій: {it}.")
            return x, it

        prev_err_ax = err_ax

    print(f"\n[ITER] Досягнуто максимум ітерацій ({max_iter}).")
    return x, max_iter

if __name__ == "__main__":
    sep = "=" * 62
    print(sep)
    print("  Лабораторна робота №7: LU-розклад та ітераційне уточнення")
    print(sep)

    A, b = generate_and_save(N, X_TRUE)

    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_B.txt")

    print("\n[LU] Виконання LU-розкладу (метод Дулітла)...")
    L, U = lu_decompose(A)
    save_lu(L, U)

    err_lu = vec_norm(A - L @ U)
    print(f"[LU] Верифікація  ||A - L*U|| = {err_lu:.3e}")

    print("\n[SOLVE] Розв'язок системи AX = B через LU-розклад...")
    x0 = solve_lu(L, U, b)

    eps_init  = vec_norm(b - mat_vec(A, x0))
    err_exact = vec_norm(x0 - X_TRUE)
    print(f"[SOLVE] Нев'язка  ||AX-B||  = {eps_init:.6e}")
    print(f"[SOLVE] Похибка   ||X-X*||  = {err_exact:.6e}")

    print(f"\n[ITER] Ітераційне уточнення (eps = {EPS_REFINE:.0e}):")
    x_ref, iters = iterative_refinement(A, L, U, b, x0,
                                         eps=EPS_REFINE, max_iter=MAX_ITER)

    eps_final = vec_norm(b - mat_vec(A, x_ref))
    err_final = vec_norm(x_ref - X_TRUE)

    print("\nПерші 5 компонент уточненого розв'язку:")
    for i in range(5):
        print(f"  x[{i+1}] = {x_ref[i]:.15f}  (еталон {X_TRUE})")