import numpy as np

def f(x):
    return x * np.sin(3 * x) - np.exp(x)
def f_prime(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)


def newton_raphson(x0, tol=1e-6, max_iter=2000):
    x = x0
    x_list = [x]
    
    for iterations in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        x_new = x - fx / fpx
        x_list.append(x_new)  
        
        if abs(fx) < tol:
            break
        
        x = x_new
    
    return np.array(x_list), iterations + 1


def bisection(xleft, xright, tol = 1e-6, max_iter = 2000):
    if f(xleft) * f(xright) >= 0:
        raise ValueError("function values at endpoints require opposite signs")
    mid_points = []
    
    for iterations in range(max_iter):
        xmid = (xleft + xright) / 2
        mid_points.append(xmid)
        if(abs(f(xmid)) < tol):
            break
        elif f(xleft) * f(xmid) < 0:
            xright = xmid
        else:
            xleft = xmid
    return mid_points, iterations + 1

x_newton, iter_newton = newton_raphson(-1.6)

x_bisection, iter_bisection = bisection(-0.7, -0.4)


A1 = np.array(x_newton)
A2 = np.array(x_bisection)
A3 = np.array([iter_newton, iter_bisection])



A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])

x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

# operations
A4 = A + B
A5 = 3 * x - 4 * y
A6 = np.dot(A, x)
A7 = np.dot(B, x - y)
A8 = np.dot(D, x)
A9 = np.dot(D, y) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)

