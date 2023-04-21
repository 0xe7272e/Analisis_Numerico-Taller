import numpy as np

# Definir función de eliminación gaussiana
def eliminacion_gaussiana(A, b):
    n = len(b)
    for k in range(n-1):
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i,k+1:n] = A[i,k+1:n] - factor*A[k,k+1:n]
            b[i] = b[i] - factor*b[k]
    return A, b

# Resolver un sistema de ecuaciones utilizando eliminación gaussiana
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=np.float64)
b = np.array([8, -11, -3], dtype=np.float64)

A_triang, b_triang = eliminacion_gaussiana(A, b)

n = len(b)
x = np.zeros(n, dtype=np.float64)
x[n-1] = b_triang[n-1]/A_triang[n-1,n-1]
for i in range(n-2, -1, -1):
    x[i] = (b_triang[i] - np.dot(A_triang[i,i+1:n], x[i+1:n]))/A_triang[i,i]

# Imprimir la solución por pantalla
print("La solución del sistema Ax=b es:")
print(x)
