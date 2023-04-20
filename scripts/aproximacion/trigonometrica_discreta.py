import numpy as np
import matplotlib.pyplot as plt

def DFT(f):
    N = len(f)
    c = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            c[k] += f[n] * np.exp(-2j*np.pi*k*n/N)
    return c

def IDFT(c):
    N = len(c)
    f = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            f[n] += c[k] * np.exp(2j*np.pi*k*n/N)
        f[n] /= N
    return f

def trig_approx(f, M):
    N = len(f)
    L = N/2
    x = np.linspace(-L, L, N+1)[:-1]
    c = DFT(f)
    cf = np.zeros(N, dtype=complex)
    cf[0] = c[0]/N
    for m in range(1,M+1):
        cf[m] = c[m]/N
        cf[-m] = c[-m]/N
    return lambda x: np.real(np.sum(cf[k] * np.exp(2j*np.pi*k*x/L) for k in range(-M,M+1)))

# Ejemplo de uso
f = lambda x: np.sin(x)
M = 2
N = 2*M+1
L = np.pi
x = np.linspace(-L,L,N)
f_vals = f(x)
f_approx = trig_approx(f_vals, M)

plt.plot(x, f_vals, label='Función original')
plt.plot(x, f_approx(x), label='Aproximación')
plt.legend()
plt.show()
