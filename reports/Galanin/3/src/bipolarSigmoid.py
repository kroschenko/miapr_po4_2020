import math

def BiSigmFunc(S):
    y = 2 / (1 + math.exp(-S)) - 1
    return y

def dBiSigmFunc(y):
    dy = 2 * (1 - y) * y
    return dy
