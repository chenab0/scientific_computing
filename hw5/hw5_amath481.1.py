
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

L = 20
m = 64   # N value in x and y directions
n = m * m  # total size of matrix
dx = L/m


e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]


# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
             e2.flatten(), -4 * e1.flatten(), e3.flatten(),
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()/dx**2
diagonals = [e1.flatten(), -1 * e1.flatten(), e1.flatten(),-1 * e1.flatten()]
offsets = [-(n-m), -m, m, (n-m)]

matA[0, 0] = 2 / dx**2

matB = spdiags(diagonals, offsets, n, n)/(2*dx)


diagonals = [e5.flatten(), -1 * e2.flatten(), e3.flatten(), -1 * e4.flatten()]
offsets = [-(m-1), -1, 1, (m-1)]

matC = spdiags(diagonals, offsets, n, n)/(2*dx)


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
import scipy
import scipy.linalg
from scipy.sparse.linalg import spsolve
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import bicgstab


tspan = np.arange(0, 4.5, 0.5)
tp = (tspan[0], tspan[-1])

# Parameters
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

#LU 
P, L, U = lu(matA)

#GMRES
psi0 = np.zeros(nx * ny)

# Create grid
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
dx = x[1] - x[0]
print(dx)
X, Y = np.meshgrid(x, y)

w0 = 1 * np.exp(-X**2 - (1/20)*Y**2)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2



def compute_derivatives(psi_flat, w_flat):
  psi_x_flat = matB @ psi_flat
  psi_y_flat = matC @ psi_flat
  w_x_flat = matB @ w_flat
  w_y_flat = matC @ w_flat
  jacobian_flat = psi_x_flat * w_y_flat - psi_y_flat * w_x_flat
  laplacian_flat = matA @ w_flat
  return - jacobian_flat + nu * laplacian_flat


#FFT SOLVE
def spc_rhs(t, w_flat):
    w = w_flat.reshape((nx, ny))
    w_cap = fft2(w)
    psi_cap = -w_cap / K
    psi = ifft2(psi_cap).real
    psi_flat = psi.flatten()

    return compute_derivatives(psi_flat, w_flat)


#AX = B LINEAR SOLVE
def AB_rhs(t, w_flat):
    psi_flat = scipy.linalg.solve(matA, w_flat)
    return compute_derivatives(psi_flat, w_flat)


#LU DECOMPOSITION
def LU_rhs(t, w_flat):
    Pb = np.dot(P, w_flat)
    y = solve_triangular(L, Pb, lower=True)
    psi_flat = solve_triangular(U, y)
    return compute_derivatives(psi_flat, w_flat)

#GMRES
def gmres_rhs(t, w_flat, psi0):
    psi_flat = gmres(matA, w_flat, x0 = psi0, rtol = 1e-6)
    psi0 = psi_flat
    return compute_derivatives(psi_flat, w_flat)

#BICGSTAB
def bicgstab_rhs(t, w_flat):
    psi_flat = bicgstab(matA, w_flat)
    return compute_derivatives(psi_flat, w_flat)



w0_flat = w0.reshape(N)


#FFT
sol = solve_ivp(spc_rhs, tp, w0_flat, t_eval = tspan, args=(), method='RK45')
w_sol = sol.y
A1 = w_sol

#A/b
sol = solve_ivp(AB_rhs, tp, w0_flat, t_eval = tspan, args=(), method='RK45')
w_sol = sol.y
A2 = w_sol

#LU
sol = solve_ivp(LU_rhs, tp, w0_flat, t_eval = tspan, args=(), method='RK45')
w_sol = sol.y
A3 = w_sol


