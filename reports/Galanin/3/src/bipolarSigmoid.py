import math

def BiSigmFunc(S):
    y = 2 / (1 + math.exp(-S)) - 1
    return y

def dBiSigmFunc(y):
    dy = y * (1 - y)
    return dy
