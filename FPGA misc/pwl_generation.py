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

f = atan
name = 'arctan'
x_range = [-20, 20]
# f = atan
# name = 'arctan'
# x_range = [-20, 20]
points = 128

x = np.expand_dims(np.linspace(x_range[0], x_range[1], points), axis=1)
y = np.expand_dims(np.array([f(x_) for x_ in x]), axis=1)

savarray = np.append(x, y, axis=1).tolist()

import csv

with open(name + '.csv', 'w', newline='') as file:
    write = csv.writer(file)
    write.writerows(savarray)
#%%plot
import matplotlib.pyplot as plt
x_true = np.linspace(x_range[0], x_range[1], 1000);
y_true = np.array([f(x_) for x_ in x_true])

plt.figure()
plt.title(name + " and piecewise linear approximation")
plt.plot(x_true, y_true, label=name)
plt.legend()
plt.plot(x, y, label='piecewise approximation')

