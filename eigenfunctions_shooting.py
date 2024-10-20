import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-6  # define a tolerance level 
col = ['r', 'g', 'm', 'b', 'k']  # eigenfunc colors
n0 = 10
L = 4
xspan = np.arange(-L, L+0.1, 0.1)
A1 = []
A2 = []


def shoot2(y, x, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

epsilon_start = 0
for modes in  range(1, 6):
    epsilon = epsilon_start
    delta_epsilon = 0.1
    
    for _ in range(1000):
        y0 = [1, np.sqrt(L**2 - epsilon)]
        y = odeint(shoot2, y0, xspan, args = (epsilon,))
        desired_conditions = np.sqrt(L**2 - epsilon)*y[-1,0] + y[-1,1]
        if(abs(desired_conditions) < tol): 
            break
        if (-1)**(modes+1) * desired_conditions > 0:
            epsilon += delta_epsilon
        else:
            epsilon = epsilon - delta_epsilon/2
            delta_epsilon /= 2
    norm = np.trapz(y[:,0]**2, xspan)
    eigenfunction_normalized = y[:,0] / np.sqrt(norm)
    A1.append(abs(eigenfunction_normalized))
    A2.append(epsilon)
    
    epsilon_start = epsilon + 0.1

A1 = np.array(A1).T
