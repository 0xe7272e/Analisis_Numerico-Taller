import numpy as np

# Definir la función objetivo y su gradiente
def f(x):
    return (x[0] + x[1]**3) + 3*(x[0] + x[1])**4

def grad_f(x):
    return np.array([1 + 12*(x[0] + x[1])**3 + 3*x[1]**2,
                     3*x[1]**2 + 12*(x[0] + x[1])**3 + 3*x[1]**2])

# Definir la función que realiza la optimización usando el método DFP
def optimize(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    f: función objetivo
    grad_f: gradiente de la función objetivo
    x0: punto inicial
    alpha: tamaño de paso
    max_iter: número máximo de iteraciones
    tol: tolerancia para detener el algoritmo
    """
    # Inicializar el punto actual, la matriz Hessiana y la iteración
    x = x0
    H = np.identity(x0.size)
    iter_count = 0
    
    # Realizar las iteraciones hasta que se alcance la tolerancia o se llegue al número máximo de iteraciones
    while iter_count < max_iter:
        # Calcular el gradiente en el punto actual
        grad = grad_f(x)
        
        # Calcular la dirección de búsqueda usando la matriz Hessiana y el gradiente
        d = -np.dot(H, grad)
        
        # Actualizar el punto usando el tamaño de paso y la dirección de búsqueda
        x_new = x + alpha*d
        
        # Calcular el nuevo gradiente
        grad_new = grad_f(x_new)
        
        # Calcular el cambio en el gradiente y en el punto
        delta_x = x_new - x
        delta_grad = grad_new - grad
        
        # Actualizar la matriz Hessiana usando la fórmula de DFP
        H = H + np.outer(delta_x, delta_x)/np.dot(delta_x, delta_grad) - np.dot(np.dot(H, np.outer(delta_grad, delta_grad)), H)/np.dot(delta_grad, np.dot(H, delta_grad))
        
        # Verificar si se ha alcanzado la tolerancia
        if np.linalg.norm(x_new - x) < tol:
            break
        
        # Actualizar el punto actual y la iteración
        x = x_new
        iter_count += 1
    
    # Retornar el punto óptimo y el valor de la función objetivo en ese punto
    return x, f(x)

# Definir el punto inicial
x0 = np.array([1, 1])

# Optimizar la función usando el método DFP
x_opt, f_opt = optimize(f, grad_f, x0)

# Imprimir el resultado
print("El mínimo de la función se encuentra en x =", x_opt, "con un valor de f(x) =", f_opt)

