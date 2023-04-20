import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn  # Función Bessel de primera especie

def trigonometric_approximation(x, y, degree):
    """
    Aproximación de una función mediante polinomios ortogonales trigonométricos de grado n.

    Args:
        x (list): Lista con los valores x de los puntos de la función.
        y (list): Lista con los valores y de los puntos de la función.
        degree (int): Grado del polinomio.

    Returns:
        tuple: Tupla con el polinomio de aproximación y los coeficientes del polinomio.

    """
    n = len(x)
    a = np.zeros(degree+1)
    b = np.zeros(degree+1)
    for k in range(degree+1):
        a[k] = (2/n)*np.sum(y*np.cos(k*np.array(x)))
        b[k] = (2/n)*np.sum(y*np.sin(k*np.array(x)))
    p = np.zeros(n)
    for k in range(degree+1):
        p += a[k]*np.cos(k*np.array(x)) + b[k]*np.sin(k*np.array(x))
    c = np.zeros(degree+1)
    for k in range(degree+1):
        c[k] = np.sqrt(a[k]**2 + b[k]**2)
    return p, c


# Definir la función seno en el intervalo [0, pi]
x = np.linspace(0, np.pi, 100)
y = np.sin(x)

# Aproximación mediante polinomios ortogonales trigonométricos de grado 5
p, c = trigonometric_approximation(x, y, degree=5)

# Graficar la función original y la aproximación
plt.plot(x, y, label='Seno original')
plt.plot(x, p, label='Aproximación')
plt.legend()
plt.show()
