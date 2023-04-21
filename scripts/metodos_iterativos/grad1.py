import numpy as np

# Definir la función objetivo
def f(x):
    return x[0] + x[1]**3 + 3*(x[0] + x[1])**4

# Definir el gradiente de la función
def grad_f(x):
    return np.array([1 + 12*(x[0] + x[1])**3, 3*x[1]**2 + 12*(x[0] + x[1])**3])

# Definir el punto inicial
x0 = np.array([0, 0])

# Definir el tamaño de paso
alpha = 0.001

# Definir el criterio de parada
epsilon = 0.0001

# Definir el número máximo de iteraciones
max_iter = 10000

# Algoritmo de gradiente descendente
x = x0
for i in range(max_iter):
    grad = grad_f(x)
    x = x - alpha * grad
    if np.linalg.norm(grad) < epsilon:
        break

# Imprimir la solución
print("El mínimo se encuentra en:", x)
print("El valor mínimo es:", f(x))
