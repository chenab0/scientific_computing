import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftn, ifftn
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

n = 16
L = 2 * np.pi

tspan = np.arange(0, 4.5, 0.5)
tp = (tspan[0], tspan[-1])

Lx, Ly, Lz = L, L, L
nx, ny, nz = n, n, n
N = nx * ny * nz

x2 = np.linspace(-Lx/2, Lx/2, n + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, n + 1)
y = y2[:ny]
z2 = np.linspace(-Lz/2, Lz/2, n + 1)
z = z2[:nz]


dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, y, z)


# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
kz = (2 * np.pi / Lz) * np.concatenate((np.arange(0, nz/2), np.arange(-nz/2, 0)))
kz[0] = 1e-6
KX, KY, KZ = np.meshgrid(kx, ky, kz)
K = KX**2 + KY**2 + KZ**2

#Parameters
A1 = A2 = A3 = -1
B1 = B2 = B3 = 1

#Initial Condition:
psi0 = np.cos(X) * np.cos(Y) * np.cos(Z)
psi0_2 = np.sin(X) * np.sin(Y) * np.sin(Z)

psi0_cap = fftn(psi0)

def spc_rhs(t, psi_hat_flat):
  psi_hat = psi_hat_flat.reshape((nx, ny, nz))
  psi = ifftn(psi_hat)
  term1 = -np.abs(psi)**2 * psi
  term2 = (A1 * np.sin(X)**2 + B1) * (A2 * np.sin(Y)**2 + B2) * (A3 * np.sin(Z)**2 + B3) * psi
  rhs = -1j * fftn(term1 + term2)
  return rhs.flatten()


sol_fft = solve_ivp(spc_rhs, tp, psi0.flatten(), t_eval = tspan, args=(), method='RK45')
As1 = sol_fft.y

for idx, t in enumerate(tspan):
    psi_hat_flat = sol_fft.y[:, idx]
    psi_hat = psi_hat_flat.reshape((nx, ny, nz))
    psi = ifftn(psi_hat)
    psi_abs = np.abs(psi)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=psi_abs.flatten().real,
        isomin=psi_abs.min(),
        isomax=psi_abs.max(),
        opacity=0.1,  # Adjust for better visibility
        surface_count=20,  # Number of isosurfaces
        colorscale='Viridis',
    ))
    fig.update_layout(
        title=f'Isosurface at Time = {t}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        )
    )
    fig.show()

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from mpl_toolkits.mplot3d import Axes3D

# Create a directory for temporary frames
if not os.path.exists('temp_frames'):
    os.makedirs('temp_frames')

# Reshape the solution back to 3D
sol_3d = []
for t_idx in range(len(tspan)):
    sol_t = sol_fft.y[:, t_idx].reshape((nx, ny, nz))
    sol_3d.append(np.abs(sol_t))

# Create frames for the animation
frames = []
for t_idx in range(len(tspan)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create slice plots for visualization
    slice_xy = sol_3d[t_idx][:, :, nz//2]
    slice_xz = sol_3d[t_idx][:, ny//2, :]
    slice_yz = sol_3d[t_idx][nx//2, :, :]

    # Plot the three orthogonal slices
    ax.plot_surface(X[:, :, nz//2], Y[:, :, nz//2], Z[:, :, nz//2],
                   facecolors=plt.cm.viridis(slice_xy/np.max(slice_xy)),
                   alpha=0.7)
    ax.plot_surface(X[:, ny//2, :], Y[:, ny//2, :], Z[:, ny//2, :],
                   facecolors=plt.cm.viridis(slice_xz/np.max(slice_xz)),
                   alpha=0.7)
    ax.plot_surface(X[nx//2, :, :], Y[nx//2, :, :], Z[nx//2, :, :],
                   facecolors=plt.cm.viridis(slice_yz/np.max(slice_yz)),
                   alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Time = {tspan[t_idx]:.2f}')

    # Save frame
    filename = f'temp_frames/frame_{t_idx:03d}.png'
    plt.savefig(filename)
    frames.append(imageio.imread(filename))
    plt.close()

# Create animated GIF
imageio.mimsave('quantum_evolution.gif', frames, duration=0.5)

# Clean up temporary files
for filename in os.listdir('temp_frames'):
    os.remove(os.path.join('temp_frames', filename))
os.rmdir('temp_frames')

print("Animation saved as 'quantum_evolution.gif'")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from skimage import measure

# Create a directory for temporary frames
if not os.path.exists('temp_frames'):
    os.makedirs('temp_frames')

# Reshape the solution back to 3D
sol_3d = []
for t_idx in range(len(tspan)):
    sol_t = sol_fft.y[:, t_idx].reshape((nx, ny, nz))
    sol_3d.append(np.abs(sol_t))

# Create frames for the animation
frames = []
for t_idx in range(len(tspan)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate isosurface
    data = sol_3d[t_idx]
    max_val = np.max(data)
    # Create three isosurfaces at different thresholds
    levels = [0.3 * max_val, 0.5 * max_val, 0.7 * max_val]
    colors = ['blue', 'green', 'red']
    alphas = [0.3, 0.4, 0.5]

    for level, color, alpha in zip(levels, colors, alphas):
        verts, faces, _, _ = measure.marching_cubes(data, level=level)

        # Scale vertices to match the actual coordinates
        verts = verts * np.array([Lx/nx, Ly/ny, Lz/nz]) - np.array([Lx/2, Ly/2, Lz/2])

        # Plot the isosurface
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                       triangles=faces,
                       color=color,
                       alpha=alpha,
                       shade=True)

    # Set the view
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Time = {tspan[t_idx]:.2f}')

    # Set axis limits
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_zlim([-Lz/2, Lz/2])

    # Save frame
    filename = f'temp_frames/frame_{t_idx:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    frames.append(imageio.imread(filename))
    plt.close()

# Create animated GIF
imageio.mimsave('quantum_isosurface.gif', frames, duration=0.5)

# Clean up temporary files
for filename in os.listdir('temp_frames'):
    os.remove(os.path.join('temp_frames', filename))
os.rmdir('temp_frames')

print("Animation saved as 'quantum_isosurface.gif'")

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftn, ifftn
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

n = 16
L = 2 * np.pi

tspan = np.arange(0, 4.5, 0.5)
tp = (tspan[0], tspan[-1])

Lx, Ly, Lz = L, L, L
nx, ny, nz = n, n, n
N = nx * ny * nz

x2 = np.linspace(-Lx/2, Lx/2, n + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, n + 1)
y = y2[:ny]
z2 = np.linspace(-Lz/2, Lz/2, n + 1)
z = z2[:nz]

dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, y, z)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
kz = (2 * np.pi / Lz) * np.concatenate((np.arange(0, nz/2), np.arange(-nz/2, 0)))
kz[0] = 1e-6
KX, KY, KZ = np.meshgrid(kx, ky, kz)
K = KX**2 + KY**2 + KZ**2

# Parameters
A1 = A2 = A3 = -1
B1 = B2 = B3 = 1

# Initial Condition:
psi0 = np.cos(X) * np.cos(Y) * np.cos(Z)
psi0_2 = np.sin(X) * np.sin(Y) * np.sin(Z)

psi0_cap = fftn(psi0)

def spc_rhs(t, psi_hat_flat):
    psi_hat = psi_hat_flat.reshape((nx, ny, nz))
    psi = ifftn(psi_hat)
    term1 = -np.abs(psi)**2 * psi
    term2 = (A1 * np.sin(X)**2 + B1) * (A2 * np.sin(Y)**2 + B2) * (A3 * np.sin(Z)**2 + B3) * psi
    rhs = -1j * fftn(term1 + term2)
    return rhs.flatten()

sol_fft = solve_ivp(spc_rhs, tp, psi0.flatten(), t_eval=tspan, args=(), method='RK45')
As1 = sol_fft.y.reshape(len(tspan), nx, ny, nz)

# Visualization using isosurface for a specific time step
time_step = 5  # Select the time step for visualization
data_to_visualize = np.abs(As1[time_step])

# Create the isosurface plot
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=data_to_visualize.flatten(),
    isomin=np.min(data_to_visualize),
    isomax=np.max(data_to_visualize),
    surface_count=10,  # Number of isosurfaces
    colorscale="Viridis",  # Colormap
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(
    title=f"Isosurface Visualization at Time Step {time_step}",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

fig.show()
