import numpy as np
from scipy.optimize import minimize

def f(x):
    return (x[0] + x[1]**3) + 3*(x[0] + x[1])**4

def grad_f(x):
    dfdx0 = 1 + 12*(x[0] + x[1])**3
    dfdx1 = 3*x[1]**2 + 12*(x[0] + x[1])**3
    return np.array([dfdx0, dfdx1])

def hessian(x):
    h00 = 36*(x[0] + x[1])**2 + 1
    h01 = h10 = 36*(x[0] + x[1])**2
    h11 = 6*x[1] + 36*(x[0] + x[1])**2
    return np.array([[h00, h01], [h10, h11]])

x0 = np.array([1.0, 1.0])  # punto inicial
H0 = np.linalg.inv(hessian(x0))
res = minimize(f, x0, method='BFGS', jac=grad_f, hess=H0, options={'maxiter': 1000})

print(res)
