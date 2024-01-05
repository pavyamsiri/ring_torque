"""
This script tests the accurracy the Ring class' moment of inertia approximations by comparing it to an approximation obtained
by splitting the ring into point particles of uniform mass.
"""

# External libraries
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# Internal libraries
from ring import Ring


def test_calculation(num_samples: int, num_particles: int):
    # Use a fixed semi-major axis and mass, but vary the eccentricity
    semi_major_axis = 1
    mass = 1
    radius = 1
    eccentricities = np.linspace(0, 1, num_samples, endpoint=False)
    semi_minor_axes = semi_major_axis * np.sqrt(1 - np.power(eccentricities, 2))
    # Initialize arrays to store the results
    ixx_approx = np.full(eccentricities.shape, np.nan)
    iyy_approx = np.full(eccentricities.shape, np.nan)
    izz_approx = np.full(eccentricities.shape, np.nan)
    # Initialise arrays for the particle approximation
    ixx_particle = np.full(eccentricities.shape, np.nan)
    iyy_particle = np.full(eccentricities.shape, np.nan)
    izz_particle = np.full(eccentricities.shape, np.nan)

    for idx, b in tqdm(enumerate(semi_minor_axes), total=num_samples, leave=False):
        # Obtain the moment of inertia calculated by the Ring class
        ring = Ring(radius, semi_major_axis, b, (0, 0, 1), 0, mass)
        ixx_approx[idx] = ring.moment_of_inertia_x
        iyy_approx[idx] = ring.moment_of_inertia_y
        izz_approx[idx] = ring.moment_of_inertia_z

        # Approximate the moment of inertia by splitting the ring into point particles
        current_ixx = 0
        current_iyy = 0
        current_izz = 0
        points = ring.sample_points(num_particles)
        for point in points.T:
            current_x = point[0]
            current_y = point[1]
            current_z = point[2]
            current_ixx += current_y**2 + current_z**2
            current_iyy += current_x**2 + current_z**2
            current_izz += current_x**2 + current_y**2
        ixx_particle[idx] = mass / num_particles * current_ixx
        iyy_particle[idx] = mass / num_particles * current_iyy
        izz_particle[idx] = mass / num_particles * current_izz

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(
        eccentricities, ixx_approx, color="red", label=r"Elliptic integrals $I_{xx}$"
    )
    ax.plot(
        eccentricities,
        ixx_particle,
        color="tab:red",
        linestyle="--",
        label=f"{num_particles} particle approximation" + r" $I_{xx}$",
    )
    ax.plot(
        eccentricities,
        iyy_approx,
        color="blue",
        label=r"Elliptic integrals $I_{yy}$",
    )
    ax.plot(
        eccentricities,
        iyy_particle,
        color="tab:blue",
        linestyle="--",
        label=f"{num_particles} particle approximation" + r" $I_{yy}$",
    )
    ax.plot(
        eccentricities, izz_approx, color="green", label=r"Elliptic integrals $I_{zz}$"
    )
    ax.plot(
        eccentricities,
        izz_particle,
        color="tab:green",
        linestyle="--",
        label=f"{num_particles} particle approximation" + r" $I_{zz}$",
    )

    ax.set_title("Comparing moment of inertia calculations")
    ax.set_xlabel(r"Eccentricity $e$")
    ax.set_ylabel(r"Moment of inertia $I$")
    ax.set_ylim(0, 1.1 * max(izz_approx.max(), izz_particle.max()))
    ax.set_xlim(0, 1.01)
    ax.legend()

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    test_calculation(100, 300)
