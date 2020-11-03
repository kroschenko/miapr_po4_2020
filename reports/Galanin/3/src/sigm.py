import math

def SigmFunc(S):
    y = 1 / (1 + math.exp(-S))
    return y

def dSigmFunc(y):
    dy = y * (1 - y)
    return dy