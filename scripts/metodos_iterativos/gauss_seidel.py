def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    k = 0
    err = tol + 1

    while err > tol and k < max_iter:
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i)) - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        err = max(abs(x[i] - x_old[i]) for i in range(n))
        k += 1

    return x, k

# Definir la matriz de coeficientes y el vector de términos independientes
A = [[4, -1, 0, 0],
     [-1, 4, -1, 0],
     [0, -1, 4, -1],
     [0, 0, -1, 3]]

b = [15, 10, 10, 10]

# Definir el vector de aproximación inicial, la tolerancia y el número máximo de iteraciones
x0 = [0, 0, 0, 0]
tol = 1e-6
max_iter = 1000

# Resolver el sistema de ecuaciones lineales mediante el método de Gauss-Seidel
x, k = gauss_seidel(A, b, x0, tol, max_iter)

# Mostrar la solución por pantalla
print("La solución del sistema es:")
for i in range(len(x)):
    print(f"x_{i} = {x[i]:.6f}")

print(f"\nSe realizaron {k} iteraciones.")

