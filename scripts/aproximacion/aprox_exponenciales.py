import numpy as np

def exponential_approximation(x, y, n, alpha, c):
    A = np.zeros((len(x), n+1))
    for i in range(len(x)):
        A[i,0] = 1
        for j in range(n):
            A[i,j+1] = np.exp(alpha[j]*x[i])
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c

# Datos de ejemplo
x = np.array([0, 1, 2])
y = np.array([1, 2, 5])
n = 2
alpha = np.array([-1, 1])
c = np.zeros(n+1)

# Calcular los coeficientes de la aproximación exponencial
c = exponential_approximation(x, y, n, alpha, c)

# Imprimir los coeficientes
print("Coeficientes de la aproximación exponencial:", c)
