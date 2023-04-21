import numpy as np

# Definir la función objetivo
def f(x):
    return x[0] + x[1]**3 + 3*(x[0] + x[1])**4

# Definir la función gradiente de la función objetivo
def grad_f(x):
    return np.array([1 + 12*(x[0]+x[1])**3, 3*x[1]**2 + 12*(x[0]+x[1])**3])

# Definir la matriz identidad
I = np.eye(2)

# Definir la aproximación inicial
x0 = np.array([1.0, 1.0])

# Definir el parámetro de tolerancia
tol = 1e-8

# Definir el número máximo de iteraciones
max_iter = 1000

# Definir el parámetro de reducción del tamaño del paso
alpha = 0.1

# Definir el parámetro de ampliación del tamaño del paso
beta = 0.5

# Definir el vector de dirección inicial
d = -grad_f(x0)

# Definir la matriz aproximada H0
H0 = I

# Definir la aproximación inicial del Hessiano
Hk = H0

# Definir la aproximación inicial del gradiente
gk = grad_f(x0)

# Definir el número de iteraciones actual
k = 0

# Realizar el ciclo de iteración del método BFGS
while np.linalg.norm(gk) > tol and k < max_iter:
    # Calcular el tamaño del paso usando la regla de Armijo
    alpha_k = 1.0
    while f(x0 + alpha_k*d) > f(x0) + alpha*alpha_k*np.dot(grad_f(x0), d):
        alpha_k *= beta
        
    # Calcular la aproximación del punto xk+1
    xk = x0 + alpha_k*d
    
    # Calcular el gradiente en el punto xk+1
    gk1 = grad_f(xk)
    
    # Calcular la diferencia del gradiente entre dos iteraciones
    delta_gk = gk1 - gk
    
    # Calcular la diferencia de los puntos entre dos iteraciones
    delta_xk = xk - x0
    
    # Actualizar la aproximación del Hessiano
    Hk1 = Hk + np.outer(delta_gk, delta_gk)/(np.dot(delta_gk, delta_xk)) \
          - np.dot(np.dot(Hk, np.outer(delta_xk, delta_xk)), Hk)/(np.dot(np.dot(delta_xk, Hk), delta_xk))
    
    # Actualizar la aproximación del gradiente
    gk = gk1
    
    # Actualizar la aproximación del punto
    x0 = xk
    
    # Actualizar la aproximación del Hessiano
    Hk = Hk1
    
    # Actualizar el vector de dirección
    d = -np.dot(Hk, gk)
    
    # Incrementar el número de iteraciones
    k += 1

# Imprimir la solución
print("La solución es:", x0)
