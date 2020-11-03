import numpy as np
import math

a = 0.1
b = 0.5
c = 0.09
d = 0.5

step = 0.1

def get_etalons(number):
    etalons = np.zeros(number)
    for i in range(number):
        x = i * step
        y = a * math.cos(b * x) + c * math.sin(d * x)
        etalons[i] = y
    return etalons
