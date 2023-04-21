import numpy as np

def factorizacionLU(A):
    """
    Función que realiza la factorización LU de una matriz A.
    Devuelve dos matrices L y U tales que A = LU.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[0, j] = A[0, j]
        L[j, 0] = A[j, 0] / U[0, 0]

    for i in range(1, n):
        for j in range(i, n):
            s1 = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = A[i, j] - s1

        for j in range(i, n):
            s2 = sum(U[k, i] * L[j, k] for k in range(i))
            L[j, i] = (A[j, i] - s2) / U[i, i]

    np.fill_diagonal(L, 1)
    return L, U


A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 4]])

b = np.array([12, 1, 2])

L, U = factorizacionLU(A)

y = np.zeros_like(b)
for i in range(len(b)):
    y[i] = b[i] - np.dot(L[i,:i], y[:i])

x = np.zeros_like(b)
for i in reversed(range(len(b))):
    x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]

print("La solución del sistema Ax = b es:")
print(x)

