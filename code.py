import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp


tspan = np.arange(0, 4.5, 0.5)
tp = (tspan[0], tspan[-1])

# Parameters
n = 64
L = 20
Lx, Ly = L, L
nx, ny = n, n
N = nx * ny

D1 = 0.1
D2 = 0.1
beta = 1.0

#Create Grid
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

#Initial conditions
m = 1  #number of spirals
r = np.sqrt(X**2 + Y**2)
theta = np.angle(X + 1j*Y)

u = np.tanh(r) * np.cos(m * theta - r)
v = np.tanh(r) * np.sin(m * theta - r)
u_cap = fft2(u)
v_cap = fft2(v)
y_flat = np.concatenate([u_cap.flatten(), v_cap.flatten()])

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2


#FFT SPECTRAL METHOD
def spc_rhs(t, y_flat):
  # Split the 1D array back into u and v components
  n_half = len(y_flat)// 2
  u_cap_flat = y_flat[:n_half]
  v_cap_flat = y_flat[n_half:]

  #Reshape to 2D
  u_cap = u_cap_flat.reshape(nx, ny)
  v_cap = v_cap_flat.reshape(nx, ny)

  u = np.real(ifft2(u_cap))
  v = np.real(ifft2(v_cap))

  A2 = u**2 + v**2
  lamda = 1 - A2
  omega = -1*beta*A2

  f = lamda * u - omega * v
  g = omega * u + lamda * v

  u_t = fft2(f) + D1 * K**2 * u_cap
  v_t = fft2(g) + D2 * K**2 * v_cap

  return np.concatenate([u_t.flatten(), v_t.flatten()])

#Solve ODE
sol_fft = solve_ivp(spc_rhs, tp, y_flat, t_eval = tspan, args=(), method='RK45')
A1 = sol_fft.y
print(A1.shape)
print(A1)

#CHEBYCHEV METHOD
def cheb(N):
  if N == 0:
    D = 0.; x = 1.
  else:
    n = np.arange(0, N+1)
    x = np.cos(np.pi*n/N).reshape(N+1, 1)
    c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**n).reshape(N+1, 1)
    X = np.tile(x, (1, N+1))
    dX = X - X.T
    D = np.dot(c, 1./c.T) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N+1)

n = 30
nx, ny = n, n
L = 20
Lx, Ly = L, L

D, x_cheb = cheb(n)

#No flux boundaries
D[0, :] = 0
D[-1, :] = 0

#Making Grid
x_cheb = (L/2) * x_cheb #Rescaling to [-10, 10]
y_cheb = (L/2) * x_cheb #Same for Y
X, Y = np.meshgrid(x_cheb, y_cheb)

#Computing Laplacian
D2 = D @ D
I = np.eye(len(D2))
Laplacian = ((2/L)**2)*(np.kron(I, D2) + np.kron(D2, I)) #2D Laplacian

#Initial conditions
m = 1  #number of spirals
r = np.sqrt(X**2 + Y**2)
theta = np.angle(X + 1j*Y)

u = np.tanh(r) * np.cos(m * theta - r)
v = np.tanh(r) * np.sin(m * theta - r)

u_flat = u.flatten()
v_flat = v.flatten()
y_flat = np.concatenate([u_flat, v_flat])

# Parameters
D1 = 0.1
D2 = 0.1
beta = 1.0


def cheb_rhs(t, y_flat):
  # Split the 1D array back into u and v components
  n_half = len(y_flat)// 2
  u_flat = y_flat[:n_half]
  v_flat = y_flat[n_half:]

  #Reshape to 2D
  u = u_flat.reshape(nx + 1, ny + 1)
  v = v_flat.reshape(nx + 1, ny + 1)

  A2 = u**2 + v**2
  lamda = 1 - A2
  omega = -1*beta*A2

  f = lamda * u - omega * v
  g = omega * u + lamda * v

  u_t = f.flatten() + D1 * Laplacian @ u_flat
  v_t = g.flatten() + D2 * Laplacian @ v_flat

  return np.concatenate([u_t, v_t])

#Solve ODE
sol_cheb = solve_ivp(cheb_rhs, tp, y_flat, t_eval = tspan, args=(), method='RK45')
A2 = sol_cheb.y