import random

from modules.lab5 import lab5

random.seed(3)

obj = lab5(
    5,      # variant
    1000,   # maxIterations - max number of itererations
    1e-2,   # Ee - desired squared error
    0.001,  # alpha_ki - learning rate (inputs - hiddens)
    0.001   # alpha_ij - learning rate (hiddens - outputs)
)

obj.learning()
obj.test()
obj.print_graph()