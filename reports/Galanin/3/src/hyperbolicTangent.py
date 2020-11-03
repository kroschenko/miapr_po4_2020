import math

def HyperbolicTangent(S):
    y = (math.exp(S) - math.exp(-S)) / (math.exp(S) + math.exp(-S))
    return y

def dHyperbolicTangent(y):
    y = 1 - y ** 2
    return y