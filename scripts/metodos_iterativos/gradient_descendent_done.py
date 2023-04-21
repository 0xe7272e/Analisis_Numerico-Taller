# https://realpython.com/gradient-descent-algorithm-python/ 
# Implementation of gradient descendent by Rodrigo Castillo Camargo
from time import sleep


def fun(x,y):
    return x**2 + y**2

def derivadas_parciales(x_0: float , y_0: float , f: type(fun)):
    """
    docstring
    """
    h = 0.00000001
    f_0 = f(x_0, y_0) # es el f(x)
    f_0x = f(x_0 +h , y_0) # corrimiento en x
    f_0y = f(x_0 , y_0 + h) # corrimento en y
    dx = (f_0x - f_0)/h
    dy = (f_0y - f_0)/h
    gradiente = (dx,dy)
    return gradiente

def gradiente_descendente(f, x_0, y_0 , tolerance):
    """
    Docstring
    """
    alpha = 10**-3 # valor estándar para definir el tamaño del paso
    while True:
        p_inicial = (x_0,y_0)
        primer_gradiente = derivadas_parciales(x_0, y_0, f)
        shift =   [i*alpha for i in primer_gradiente]# corrimiento
        x_0 = x_0 - shift[0]
        y_0 = y_0 - shift[1]
        magnitud_gradiente = sum([i**2 for i in primer_gradiente])
        print(magnitud_gradiente)
        print(x_0, y_0)
        if(magnitud_gradiente <= tolerance):
            minimo = [x_0,y_0]
            return minimo



gradiente_descendente(fun, -10, 10, 0.000000001)
