import pyquibbler as qb
qb.initialize_quibbler()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Globals
RADIUS = 1
NUM_POINTS = 20
GUIDELINE_WIDTH = 0.5

PHI = np.linspace(0, np.pi, NUM_POINTS)         # latitude (polar)
THETA = np.linspace(0, 2*np.pi, NUM_POINTS)     # longitude (azimuthal)
THETA_MESH, PHI_MESH = np.meshgrid(THETA, PHI)

def sphere_cartesian_coords(r=RADIUS, phi=PHI_MESH, theta=THETA_MESH):
    """Return x, y, z coordinates on the sphere."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def rotate_sphere(x, y, z, tilt=0, roll=0, pan=0):
    """Apply 3D rotation to given sphere coordinates."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(pan), -np.sin(pan), 0],
                   [np.sin(pan), np.cos(pan), 0],
                   [0, 0, 1]])
    coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    rotated = Rz @ Ry @ Rx @ coords
    return (rotated[0].reshape(x.shape), rotated[1].reshape(y.shape), rotated[2].reshape(z.shape))

def project_coords(x, y, z, r, projection):
    """Project sphere points to 2D using given projection."""
    match projection:
        case "Stereographic":
            denom = (1 - z/r)
            x_proj = x / denom
            y_proj = y / denom
        case "Azimuthal":
            phi = np.arcsin(z / r)
            lambd = np.arctan2(y, x)
            phi0, lambda0 = np.pi/2, 0
            rho = r * np.arccos(np.sin(phi0)*np.sin(phi) + np.cos(phi0)*np.cos(phi)*np.cos(lambd-lambda0))
            theta = np.arctan2(np.cos(phi)*np.sin(lambd-lambda0),
                               np.cos(phi0)*np.sin(phi)-np.sin(phi0)*np.cos(phi)*np.cos(lambd-lambda0))
            x_proj = -rho * np.cos(theta) / (0.5 * np.pi * r)
            y_proj = rho * np.sin(theta) / (0.5 * np.pi * r)
        case "Orthographic":
            x_proj, y_proj = x, y
        case _:
            x_proj, y_proj = x, y
    return x_proj, y_proj

def plot_guidelines(ax, r, num_divisions, tilt, roll, pan, projection):
    """Draw longitude and latitude guidelines on the projected sphere."""
    # Longitude, latitude divisions
    angles = np.linspace(0, 2*np.pi, num_divisions)

    # Plot longitude guidelines (red)
    for angle in angles:
        x_long = r * np.sin(PHI_MESH) * np.cos(angle)
        y_long = r * np.sin(PHI_MESH) * np.sin(angle)
        z_long = r * np.cos(PHI_MESH)
        x_r, y_r, z_r = rotate_sphere(x_long, y_long, z_long, tilt, roll, pan)
        x_proj, y_proj = project_coords(x_r, y_r, z_r, r, projection)
        ax.plot(x_proj, y_proj, color='r', linewidth=GUIDELINE_WIDTH)
    # Plot latitude guidelines (green)
    for angle in angles:
        x_lat = r * np.sin(PHI_MESH) * np.sin(angle)
        y_lat = r * np.cos(PHI_MESH)
        z_lat = r * np.sin(PHI_MESH) * np.cos(angle)
        x_r, y_r, z_r = rotate_sphere(x_lat, y_lat, z_lat, tilt, roll, pan)
        x_proj, y_proj = project_coords(x_r, y_r, z_r, r, projection)
        ax.plot(x_proj, y_proj, color='g', linewidth=GUIDELINE_WIDTH)
    # Plot "meridian" guidelines (blue)
    for angle in angles:
        x_mer = r * np.cos(PHI_MESH)
        y_mer = r * np.sin(PHI_MESH) * np.cos(angle)
        z_mer = r * np.sin(PHI_MESH) * np.sin(angle)
        x_r, y_r, z_r = rotate_sphere(x_mer, y_mer, z_mer, tilt, roll, pan)
        x_proj, y_proj = project_coords(x_r, y_r, z_r, r, projection)
        ax.plot(x_proj, y_proj, color='b', linewidth=GUIDELINE_WIDTH)
    # Draw sphere boundary
    ax.add_patch(plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2))

def setup_sliders(fig, tilt0, roll0, pan0):
    """Create matplotlib sliders for tilt, roll, pan."""
    axcolor = 'lightgoldenrodyellow'
    ax_tilt = fig.add_axes([0.75, 0.1, 0.2, 0.05], facecolor=axcolor)
    ax_roll = fig.add_axes([0.75, 0.2, 0.2, 0.05], facecolor=axcolor)
    ax_pan  = fig.add_axes([0.75, 0.3, 0.2, 0.05], facecolor=axcolor)
    tilt = qb.iquib(0.)
    roll = qb.iquib(0.)
    pan  = qb.iquib(0.)
    s_tilt = Slider(ax_tilt, 'Tilt', 0, np.pi / 2, valinit=tilt, valstep=np.pi / 16)
    s_roll = Slider(ax_roll, 'Roll', 0, np.pi / 2, valinit=roll, valstep=np.pi / 16)
    s_pan  = Slider(ax_pan,  'Pan',  0, np.pi / 2, valinit=pan,  valstep=np.pi / 16)
    return s_tilt, s_roll, s_pan

def plot_sphere_projection(divisions, projection, tilt=0, roll=0, pan=0):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.05, 0.1, 0.7, 0.7])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect("equal")
    plot_guidelines(ax, RADIUS, divisions, tilt, roll, pan, projection)
    setup_sliders(fig, tilt, roll, pan)
    plt.show()

# Run the plot with default values
plot_sphere_projection(divisions=16, projection="Azimuthal")
