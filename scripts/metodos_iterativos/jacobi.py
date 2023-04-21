import numpy as np

def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)
    x_prev = np.zeros(n)
    iter = 0
    while iter < max_iter and np.linalg.norm(x - x_prev) > tol:
        x_prev = np.copy(x)
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :n], x_prev) + A[i, i] * x_prev[i]) / A[i, i]
        iter += 1
    return x

# Ejemplo de uso
A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
b = np.array([7, -8, 6])
x0 = np.array([0, 0, 0])
tol = 1e-8
max_iter = 1000

sol = jacobi(A, b, x0, tol, max_iter)

print("La solución es:", sol)
