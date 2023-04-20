# Implementación de coeficientes de grado n 
# Rodrigo Castillo Camargo


import numpy as np
def polynomial_regression(x, y, n):
    """
    Realiza una aproximación polinómica de grado n utilizando el método de mínimos cuadrados.
    
    Args:
        x: una lista de valores de la variable independiente
        y: una lista de valores de la variable dependiente
        n: el grado del polinomio
    
    Returns:
        Una tupla que contiene los coeficientes del polinomio
    """
    
    # Construir la matriz de diseño
    X = []
    for i in range(len(x)):
        row = [x[i]**j for j in range(n+1)]
        X.append(row)
    
    # Convertir a arrays
    X = np.array(X)
    y = np.array(y)
    
    # Calcular los coeficientes utilizando el método de mínimos cuadrados
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)
    coeffs = np.linalg.solve(X_transpose_X, X_transpose_y)
    
    return tuple(coeffs)


x = [0, 1, 2, 3, 4, 5]
y = [1, 3, 5, 4, 6, 8]

coeffs = polynomial_regression(x, y, 3)

print("Coeficientes del polinomio: ", coeffs)
