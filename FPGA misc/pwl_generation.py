"""
on the FPGA i will be using a piecewise linear approximation of the sigmoid
and arctan functions. i need to generate the points used to create this
approximation
maybe one day i'll want to do something more fancy here, to try and minimize
error
"""
import numpy as np
import math
from math import atan

def sigmoid(x):
    return 1/(1+math.exp(-x))

f = sigmoid
name = 'sigmoid'
f = atan
name = 'arctan'
points = 128
x_range = [-8, 8]

x = np.expand_dims(np.linspace(x_range[0], x_range[1], points), axis=1)
y = np.expand_dims(np.array([f(x_) for x_ in x]), axis=1)

savarray = np.append(x, y, axis=1).tolist()

import csv

with open(name + '.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(savarray)