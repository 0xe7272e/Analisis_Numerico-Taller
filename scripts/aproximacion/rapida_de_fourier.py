import math
import matplotlib.pyplot as plt

def FFT(f):
    # Calcula la FFT de una señal f
    N = len(f)
    if N == 1:
        return f
    else:
        Feven = FFT([f[i] for i in range(0, N, 2)])
        Fodd = FFT([f[i] for i in range(1, N, 2)])
        combined = [0] * N
        for m in range(N//2):
            combined[m] = Feven[m] + math.e**(-2j*math.pi*m/N)*Fodd[m]
            combined[m + N//2] = Feven[m] - math.e**(-2j*math.pi*m/N)*Fodd[m]
        return combined

# Crear un arreglo de puntos de prueba
t = [i/100 for i in range(20)]
f = [math.sin(2*math.pi*5*i) + math.sin(2*math.pi*10*i) for i in t]

# Calcular la transformada rápida de fourier de los puntos
f_fft = FFT(f)

# Calcular la frecuencia de cada muestra en los puntos original
freq = [i/(t[-1]-t[0]) for i in range(len(t))]

# Graficar los puntos original y los puntos reconstruidos de la FFT
f_ifft = [i.real for i in FFT(f_fft[::-1])] # Calcula la transformada inversa de Fourier
plt.plot(t, f, label='Señal original')
plt.plot(t, f_ifft, label='FFT reconstruida')
plt.legend()
plt.show()

