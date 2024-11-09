import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
fig, ax1 = plt.subplots()

tol = 1e-4  # define a tolerance level
col = ['r', 'g', 'm', 'b', 'k']  # eigenfunc colors
n0 = 10
L = 4
xspan = np.linspace(-L, L, int(2*L/0.1) + 1)
xp = [-L,L]
A1 = []
A2 = []

def shoot2(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

epsilon_start = 0
for modes in  range(1, 6):
    epsilon = epsilon_start
    delta_epsilon = 0.1

    for _ in range(1000):
        y0 = [1, np.sqrt(L**2 - epsilon)]
        sol = solve_ivp(shoot2, xp, y0, args=(epsilon,), t_eval=xspan, method = 'RK45')
        y = sol.y
        xs = sol.t
        desired_conditions = np.sqrt(L**2 - epsilon)*y[0,-1] + y[1,-1]
        if(abs(desired_conditions) < tol):
            break
        if (-1)**(modes+1) * desired_conditions > 0:
            epsilon += delta_epsilon
        else:
            epsilon = epsilon - delta_epsilon
            delta_epsilon /= 2
    norm = np.trapz(y[0,:]**2, xs)
    eigenfunction_normalized = y[0,:] / np.sqrt(norm)
    A1.append(abs(eigenfunction_normalized))
    A2.append(epsilon)


    epsilon_start = epsilon + 0.1

A1 = np.array(A1).T


import scipy.linalg as la
L = 4

xspan = np.linspace(-L, L, int(2*L/0.1) + 1)
x = xspan[1:-1]
N = len(x)
dx = 0.1

fig, ax1 = plt.subplots()

M1 = np.zeros((N, N))
for i in range(N):
    M1[i, i] = -2 - (x[i]**2) * (dx**2)
    if i < N - 1:
        M1[i, i+1] = 1
        M1[i+1, i] = 1

M2 = np.zeros((N, N))
M2[0, 0] = 4/3
M2[0, 1] = -1/3
M2[-1, -1] = 4/3
M2[-1, -2] = -1/3

D = M1 + M2 

eigenvalues, eigenfunctions = la.eig(-D)

eigenvalues = np.real(eigenvalues)
eigenfunctions = np.real(eigenfunctions)

sorted_indices = np.argsort(eigenvalues)
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenfunctions_sorted = eigenfunctions[:, sorted_indices]

num_eigenfunctions = eigenfunctions_sorted.shape[1]

final_matrix = np.zeros((N+2, num_eigenfunctions))
final_matrix[1:-1, :] = eigenfunctions_sorted

final_matrix[0, :] = (4/3) * final_matrix[1, :] - (1/3) * final_matrix[2, :]
final_matrix[-1, :] = (4/3) * final_matrix[-2, :] - (1/3) * final_matrix[-3, :]

for i in range(num_eigenfunctions):
    norm = np.trapz(final_matrix[:, i]**2, xspan)
    final_matrix[:, i] = np.abs(final_matrix[:, i] / np.sqrt(norm))

A3 = final_matrix[:, :5]  # First five eigenfunctions
A4 = eigenvalues_sorted[:5] / dx**2  # Corresponding eigenvalues



tol = 1e-4  # define a tolerance level
col = ['r', 'g', 'm', 'b', 'k']  # eigenfunc colors
n0 = 10
L = 2
xspan = np.linspace(-L, L, int(2*L/0.1) + 1)
xp = [-L,L]
A5 = []
A6 = []
A7 = []
A8 = []


def shoot2(x, y, epsilon, gamma):
    return [y[1], (x**2 - epsilon + gamma * abs(y[0])**2) * y[0]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for gamma in [0.05, -0.05]:
    epsilon_start = 0.1
    for modes in range(1,3):
        A_start = 1e-6
        delta_A = 0.01
        A = A_start
        for _ in range(100): #A Loop
            epsilon = epsilon_start
            delta_epsilon = 0.2
            for _ in range(1000):
                y0 = [A, A* np.sqrt(L**2 - epsilon)]
                sol = solve_ivp(shoot2, xp, y0, args = (epsilon, gamma, ), t_eval = xspan)
                y = sol.y
                xs = sol.t
                desired_conditions = np.sqrt(L**2 - epsilon)*y[0,-1] + y[1,-1]
                if(abs(desired_conditions) < tol):
                    break
                if (-1)**(modes+1) * desired_conditions > 0:
                    epsilon += delta_epsilon
                else:
                    epsilon = epsilon - delta_epsilon/2
                    delta_epsilon /= 2


            area = np.trapz(y[0,:]**2, xs)
            if (abs(abs(area) - 1)) < tol:
                break
            if (area < 1):
                A += delta_A
            else:
                A -= delta_A/2
                delta_A /= 2

        norm = np.trapz(y[0,:]**2, xs)
        eigenfunction_normalized = abs(y[0,:] / np.sqrt(norm))
        if(gamma > 0):
            A5.append(eigenfunction_normalized)
            A6.append(epsilon)
        else:
            A7.append(eigenfunction_normalized)
            A8.append(epsilon)
        epsilon_start = epsilon + 2

A5 = np.array(A5).T
A7 = np.array(A7).T


L = 2
epsilon = 1
gamma = 0
x_span = (-L, L)
y0 = [1, np.sqrt(L**2 - epsilon)] 


def hw1_rhs_a(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tol_values = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
methods = ['RK45', 'RK23', 'Radau', 'BDF']

avg_step_sizes = {method: [] for method in methods}

for method in methods:
    print(f"Processing {method}...")
    for tol in tol_values:
        options = {'rtol': tol, 'atol': tol}

        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method, args=(epsilon,), **options)

        step_sizes = np.diff(sol.t)
        avg_step_size = np.mean(step_sizes)
        avg_step_sizes[method].append(avg_step_size)

avg_step_sizes = {method: np.array(steps) for method, steps in avg_step_sizes.items()}

plt.figure(figsize=(10, 6))
slopes = []

for method in methods:
    log_step_sizes = np.log10(avg_step_sizes[method])
    log_tols = np.log10(tol_values)

    coeffs = np.polyfit(log_step_sizes, log_tols, 1)
    slopes.append(coeffs[0])

    plt.scatter(log_step_sizes, log_tols, label=f'{method} (slope: {coeffs[0]:.3f})')
    fit_line = coeffs[0] * log_step_sizes + coeffs[1]
    plt.plot(log_step_sizes, fit_line, '--')

plt.xlabel('log10(Average Step Size)')
plt.ylabel('log10(Tolerance)')
plt.title('Convergence Study of Different ODE Solvers')
plt.legend()
plt.grid(True)

A9 = np.array(slopes)

print("\nSlopes for each method:")
for method, slope in zip(methods, slopes):
    print(f"{method}: {slope}")

plt.show()

A10 = []
A11 = []
A12 = []
A13 = []

# Define the domain
L = 4
xspan = np.linspace(-L, L, int(2*L/0.1) + 1)

exact_eigenvalues = [1, 3, 5, 7, 9]
exact_eigenvalues = np.array(exact_eigenvalues)


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# Calculate exact eigenfunctions and eigenvalues
def hermite_polynomial(x, n):
    """Calculate the nth Hermite polynomial"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 - 2
    elif n == 3:
        return 8*x**3 - 12*x
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    else:
        raise ValueError("Hermite polynomial not defined for this n")



def exact_eigenfunction(x, n):
    """Calculate the exact eigenfunction for quantum harmonic oscillator"""
    # Calculate normalization factor
    norm_factor = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    # Calculate Hermite polynomial
    h_n = hermite_polynomial(x, n)
    # Return normalized wavefunction
    return norm_factor * h_n * np.exp(-x**2/2)

exact_eigenfunc = []

for i in range (5):
  exact_eigenfunc.append(exact_eigenfunction(xspan, i))
exact_eigenfunc = np.array(exact_eigenfunc).T

for i in range(5):
  func_diff_a = abs(exact_eigenfunc[:,i]) - abs(A1[:,i])
  norm_func_a = np.trapz(func_diff_a**2, xspan)
  A10.append(norm_func_a)

  func_diff_b = abs(exact_eigenfunc[:,i]) - abs(A3[:,i])
  norm_func_b = np.trapz(func_diff_b**2, xspan)
  A12.append(norm_func_b)


A10 = np.array(A10)
A12 = np.array(A12)


error_func_A = abs(exact_eigenfunc) - abs(A1)
error_func_B = abs(exact_eigenfunc) - abs(A3)

error_eigenval_A = abs(A2 - exact_eigenvalues)
error_eigenval_B = abs(A4 - exact_eigenvalues)

error_A_percentage = (error_eigenval_A / exact_eigenvalues) * 100
error_B_percentage = (error_eigenval_B / exact_eigenvalues) * 100

A11 = error_A_percentage

A13 = error_B_percentage