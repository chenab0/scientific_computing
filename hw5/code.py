
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
def gmres_rhs(t, w_flat):
    global psi0
    psi_flat, info = gmres(matA, w_flat, x0=psi0, rtol=1e-6)
    psi0 = psi_flat  # Update psi0 for the next iteration
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


start_time = time.time()
sol_fft = solve_ivp(spc_rhs, tp, w0_flat, t_eval=tspan, method='RK45', rtol=1e-6)
end_time = time.time()
elapsed_time_fft = end_time - start_time
print(f"FFT method elapsed time: {elapsed_time_fft:.2f} seconds")

# Timing and solving with Direct Solve (A\b)
start_time = time.time()
sol_direct = solve_ivp(AB_rhs, tp, w0_flat, t_eval=tspan, method='RK45', rtol=1e-6)
end_time = time.time()
elapsed_time_direct = end_time - start_time
print(f"Direct Solve method elapsed time: {elapsed_time_direct:.2f} seconds")

# Timing and solving with LU decomposition
start_time = time.time()
sol_lu = solve_ivp(LU_rhs, tp, w0_flat, t_eval=tspan, method='RK45', rtol=1e-6)
end_time = time.time()
elapsed_time_lu = end_time - start_time
print(f"LU decomposition method elapsed time: {elapsed_time_lu:.2f} seconds")

# Timing and solving with GMRES
start_time = time.time()
sol_gmres = solve_ivp(
    gmres_rhs, tp, w0_flat, t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-8
)
end_time = time.time()
elapsed_time_gmres = end_time - start_time
print(f"GMRES method elapsed time: {elapsed_time_gmres:.2f} seconds")

# Timing and solving with BICGSTAB
start_time = time.time()
sol_bicgstab = solve_ivp(
    bicgstab_rhs, tp, w0_flat, t_eval=tspan, method='RK45', rtol=1e-6, atol=1e-8
)
end_time = time.time()
elapsed_time_bicgstab = end_time - start_time
print(f"BICGSTAB method elapsed time: {elapsed_time_bicgstab:.2f} seconds")

def gaussian_vortex(X, Y, x0, y0, A, sigma_x, sigma_y):
    return A * np.exp(-((X - x0)**2) / (2 * sigma_x**2) - ((Y - y0)**2) / (2 * sigma_y**2))

# Parameters
A = 1.0  
sigma_x = 1.0  # Width in x
sigma_y = 1.0  # Width in y
d = 4.0  # Separation distance

# Positions of the vortices
x0, y0 = -d / 2, 0
x1, y1 = d / 2, 0

# Two Oppositely charged vortices
w0_opposite = (gaussian_vortex(X, Y, x0, y0, A, sigma_x, sigma_y) +
      gaussian_vortex(X, Y, x1, y1, -A, sigma_x, sigma_y))
w0_opposite_flat = w0_opposite.reshape(N)

# Two similarly charged vorticies
w0_samesign = (gaussian_vortex(X, Y, x0, y0, A, sigma_x, sigma_y) +
      gaussian_vortex(X, Y, x1, y1, A, sigma_x, sigma_y))
w0_samesign_flat = w0_samesign.reshape(N)

# Two pairs of oppositely “charged” vorticies made to collide with each other.
vortices = [
    {'x0': -d / 2, 'y0': -d / 2, 'A': -A},
    {'x0': d / 2, 'y0': -d / 2, 'A': -A},
    {'x0': -d / 2, 'y0': d / 2, 'A': A},
    {'x0': d / 2, 'y0': d / 2, 'A': A},
]
w_pair = np.zeros_like(X)
for vortex in vortices:
    w_pair += gaussian_vortex(X, Y, vortex['x0'], vortex['y0'],
                          vortex['A'], sigma_x, sigma_y)
w_pair_flat = w_pair.reshape(N)


# Assortment of random vortices
num_vortices = 275
w0_random_vorticities = np.zeros_like(X)

import random
for _ in range(num_vortices):
    x0 = random.uniform(-Lx / 2, Lx / 2)
    y0 = random.uniform(-Ly / 2, Ly / 2)
    A = random.uniform(-1, 1)  # Random amplitude between -1 and 1
    sigma_x = random.uniform(0.5, 1.5)
    sigma_y = random.uniform(0.5, 1.5)
    w0_random_vorticities += gaussian_vortex(X, Y, x0, y0, A, sigma_x, sigma_y)

w0_random_vorticities_flat = w0_random_vorticities.reshape(N)

import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def create_vorticity_animation(w0, tspan, nx, ny, Lx, Ly, output_filename='vorticity_animation.gif', 
                             cmap='magma', frames_dir='vorticity_frames', frame_duration=0.1):

    # Create directory for frames
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Set up spatial grid
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)
    
    # Solve the vorticity equation
    tp = (tspan[0], tspan[-1])
    w0_flat = w0.reshape(nx * ny)
    sol = solve_ivp(spc_rhs, tp, w0_flat, t_eval=tspan, method='RK45')
    w_sol = sol.y
    
    # Generate frames
    for j, t in enumerate(tspan):
        omega = w_sol[:, j].reshape((ny, nx))
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolor(x, y, omega, shading='auto', cmap=cmap)
        fig.colorbar(im, ax=ax, label='Vorticity')
        ax.set_title(f'Vorticity at t = {t:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        frame_filename = os.path.join(frames_dir, f'frame_{j:04d}.png')
        plt.savefig(frame_filename)
        plt.close(fig)
    
    # Create animation
    images = []
    for j in range(len(tspan)):
        frame_filename = os.path.join(frames_dir, f'frame_{j:04d}.png')
        images.append(imageio.imread(frame_filename))
    
    # Save animation
    imageio.mimsave(output_filename, images, duration=frame_duration)
    
    # Clean up frames
    for j in range(len(tspan)):
        os.remove(os.path.join(frames_dir, f'frame_{j:04d}.png'))
    os.rmdir(frames_dir)
    
    return output_filename


single_vorticity_file = create_vorticity_animation(w0, tspan, nx, ny, Lx, Ly, 
                                                   output_filename='single_vorticity.gif')

opposite_vorticities_file = create_vorticity_animation(w0_opposite, tspan, nx, ny, Lx, Ly, 
                                                      output_filename='opposite_vorticities.gif')

samesign_vorticies_file = create_vorticity_animation(w0_samesign, tspan, nx, ny, Lx, Ly, 
                                                    output_filename='samesign_vorticities.gif')

random_vorticites_file = create_vorticity_animation(w0_random_vorticities, tspan, nx, ny, Lx, Ly,
                                                   output_filename='random_vorticities.gif')

pair_vorticity_file = create_vorticity_animation(w_pair, tspan, nx, ny, Lx, Ly, 
                                                   output_filename='pair_vorticities.gif')

# Display all animations
from IPython.display import Image, display

print("Single Vortices Animation:")
display(Image(filename='single_vorticities.gif'))

print("Opposite Vortices Animation:")
display(Image(filename='opposite_vorticities.gif'))

print("\nSame-Sign Vortices Animation:")
display(Image(filename='samesign_vorticities.gif'))

print("Pair Vortices Animation:")
display(Image(filename='pair_vorticities.gif'))

print("\nRandom Vortices Animation:")
display(Image(filename='random_vorticities.gif'))
