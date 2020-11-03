import math

def sigm(S):
    y = 1 / (1 + math.exp(-S))
    return y