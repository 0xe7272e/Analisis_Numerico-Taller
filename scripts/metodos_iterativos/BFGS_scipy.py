import numpy as np
from scipy.optimize import minimize

# definir la función objetivo y su gradiente
def f(x):
    return (x[0] - 1)**2 + 100*(x[1] - x[0]**2)**2

def grad_f(x):
    return np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1-x[0]), 200*(x[1] - x[0]**2)])

# punto inicial
x0 = np.array([-2, 2])

# usar BFGS para minimizar la función objetivo
res = minimize(f, x0, method='BFGS', jac=grad_f, options={'disp': True})

# imprimir el resultado
print(res.x)
