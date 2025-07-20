import pyquibbler as qb
qb.initialize_quibbler()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

r = 1
N = 100
DIVISIONS = 16
GUIDELINE_WIDTH = 0.5


phi = np.linspace(0, np.pi, num=N)  # Polar angles
theta = np.linspace(0, 2*np.pi, num=N)  # Azimuthal angles
theta, phi = np.meshgrid(theta, phi)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)


def rotate_sphere(x, y, z, tilt, roll, pan):

    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(pan), -np.sin(pan), 0], [np.sin(pan), np.cos(pan), 0], [0, 0, 1]])

    # Apply rotations
    coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    rotated_coords = np.dot(Rz, np.dot(Ry, np.dot(Rx, coords)))
    return rotated_coords[0].reshape(x.shape), rotated_coords[1].reshape(y.shape), rotated_coords[2].reshape(z.shape)

def get_projection(x, y, z, r, projection):

    match projection:
        case "Stereographic":
            x_proj = x / (1 - z/r)
            y_proj = y / (1 - z/r)
            return x_proj, y_proj
        case "Azimuthal":
            phi = np.arcsin(z / r)  # Latitude
            lambda_ = np.arctan2(y, x)  # Longitude
            phi_0 = np.pi/2
            lambda_0 = 0
            rho = r * np.arccos(
                np.sin(phi_0) * np.sin(phi) +
                np.cos(phi_0) * np.cos(phi) * np.cos(lambda_ - lambda_0)
            )
            theta = np.arctan2(
                np.cos(phi) * np.sin(lambda_ - lambda_0),
                np.cos(phi_0) * np.sin(phi) - np.sin(phi_0) * np.cos(phi) * np.cos(lambda_ - lambda_0)
            )
            x_proj = -rho * np.cos(theta) / 0.5 / np.pi / r
            y_proj = rho * np.sin(theta) / 0.5 / np.pi / r
            return x_proj, y_proj
        case "Orthographic":
            x_proj = x
            y_proj = y
            return x_proj, y_proj
        case _: return x, y

def plot(num_divisions, projection):

    plot_longitudes(num_divisions, projection)
    ax = plt.gca()
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_aspect("equal")
    plt.show()

def plot_longitudes(num_divisions, projection):

    tilt = qb.iquib(0.)
    roll = qb.iquib(0.)
    pan = qb.iquib(0.)

    ax_tilt = plt.axes([0.75, 0.1, 0.2, 0.05])
    ax_roll = plt.axes([0.75, 0.2, 0.2, 0.05])
    ax_pan = plt.axes([0.75, 0.3, 0.2, 0.05])

    Slider(ax=ax_tilt, valmin=0, valmax=np.pi/2, valstep=np.pi/32, label="tilt", valinit=tilt)
    Slider(ax=ax_roll, valmin=0, valmax=np.pi/2, valstep=np.pi/32, label="roll", valinit=roll)
    Slider(ax=ax_pan, valmin=0, valmax=np.pi/2, valstep=np.pi/132, label="pan", valinit=pan)

    ax = plt.axes([0.05, 0.1, 0.7, 0.7]) # Axes for the main view

    z_longitudes = np.linspace(0, 2*np.pi, num_divisions)
    for angle in z_longitudes:
        # Parametric curve for longitude
        x_long = r * np.sin(phi) * np.cos(angle)
        y_long = r * np.sin(phi) * np.sin(angle)
        z_long = r * np.cos(phi)
        # Apply rotation
        x_rot_l, y_rot_l, z_rot_l = rotate_sphere(x_long, y_long, z_long, tilt, pan, roll)
        # Project the rotated longitude
        x_proj_l, y_proj_l = get_projection(x_rot_l, y_rot_l, z_rot_l, r, projection)
        plt.plot(x_proj_l, y_proj_l, color='r', linewidth=GUIDELINE_WIDTH)

    y_longitudes = np.linspace(0, 2*np.pi, num_divisions)
    for angle in y_longitudes:
        x_long = r * np.sin(phi) * np.sin(angle)
        y_long = r * np.cos(phi)
        z_long = r * np.sin(phi) * np.cos(angle)
        x_rot_l, y_rot_l, z_rot_l = rotate_sphere(x_long, y_long, z_long, tilt, pan, roll)
        x_proj_l, y_proj_l = get_projection(x_rot_l, y_rot_l, z_rot_l, r, projection)
        plt.plot(x_proj_l, y_proj_l, color='g', linewidth=GUIDELINE_WIDTH)

    x_longitudes = np.linspace(0, 2*np.pi, num_divisions)
    for angle in x_longitudes:
        x_long = r * np.cos(phi)
        y_long = r * np.sin(phi) * np.cos(angle)
        z_long = r * np.sin(phi) * np.sin(angle)
        x_rot_l, y_rot_l, z_rot_l = rotate_sphere(x_long, y_long, z_long, tilt, pan, roll)
        x_proj_l, y_proj_l = get_projection(x_rot_l, y_rot_l, z_rot_l, r, projection)
        plt.plot(x_proj_l, y_proj_l, color='b', linewidth=GUIDELINE_WIDTH)

    ax.add_patch(plt.Circle((0,0), 1.0, color='black', fill=False, linewidth=2))
