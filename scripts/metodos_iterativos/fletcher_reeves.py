import numpy as np
from scipy.optimize import line_search

def f(x):
    return 2*(x[0]**2) + 3*(x[1]**2) + 2*x[0]*x[1] + x[0] - 6*x[1]

def grad_f(x):
    return np.array([4*x[0] + 2*x[1] + 1, 6*x[1] + 2*x[0] - 6])

def fr(x_k, g_k, H_k):
    """
    Implementación de la regla de Fletcher-Reeves para obtener la dirección de descenso.
    """
    if np.all(np.abs(g_k) < 1e-8):
        # Si el gradiente es cercano a cero, se devuelve cero.
        return np.zeros_like(x_k)
    if H_k is None:
        # Si no se tiene una matriz Hessiana, se utiliza el método del gradiente conjugado sin matriz.
        return -g_k
    beta_k = (g_k @ H_k @ g_k) / (np.linalg.norm(g_k) ** 2)
    return -g_k + beta_k * fr(None, g_k, None)

x0 = np.array([0, 0])  # Punto de inicio
max_iter = 1000  # Número máximo de iteraciones
tolerance = 1e-8  # Tolerancia para la convergencia

x_k = x0
g_k = grad_f(x_k)
H_k = None  # No se utiliza matriz Hessiana en este caso.
for i in range(max_iter):
    d_k = fr(x_k, g_k, H_k)
    alpha_k = line_search(f, grad_f, x_k, d_k)[0]
    x_k_1 = x_k + alpha_k * d_k
    g_k_1 = grad_f(x_k_1)
    if np.linalg.norm(g_k_1) < tolerance:
        break
    beta_k_1 = (np.linalg.norm(g_k_1) ** 2) / (np.linalg.norm(g_k) ** 2)
    d_k_1 = -g_k_1 + beta_k_1 * d_k
    x_k = x_k_1
    g_k = g_k_1
    H_k = None  # No se utiliza matriz Hessiana en este caso.

print(f"Solución encontrada: x = {x_k}, f(x) = {f(x_k)}, iteraciones = {i+1}")
