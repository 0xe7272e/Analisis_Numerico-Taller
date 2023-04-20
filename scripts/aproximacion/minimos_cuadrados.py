# Implementación minimos cuadrados
# Rodrigo Castillo Camargo

def linear_regression(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    xy_sum = 0
    x_squared_sum = 0
    
    for i in range(n):
        xy_sum += x[i] * y[i]
        x_squared_sum += x[i] ** 2
    
    # Calcular los coeficientes de la línea recta
    b = (xy_sum - n * x_mean * y_mean) / (x_squared_sum - n * x_mean ** 2)
    a = y_mean - b * x_mean
    
    return a, b
    
# Datos
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# Calcular los coeficientes de la línea recta
a, b = linear_regression(x, y)

# Imprimir los coeficientes
print("Intersección de la línea:", a)
print("Pendiente de la línea:", b)
