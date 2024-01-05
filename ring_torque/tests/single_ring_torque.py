# Standard libraries
import logging
import os
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Internal libraries
from .common import NUM_FRAMES_PER_TEST, FRAME_RATE, init_plot, FORCE_GRID_POINTS
from ring_torque.common import PLOT_PADDING_FACTOR
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.ring import Ring, RingDiscretisation, RingDiscretisationType

log = logging.getLogger(__name__)


def test_single_ring_torque(
    num_eccentricities: int,
    num_points: int,
    softening_length: float,
    output_path: str,
    force: bool = False,
):
    """Test if the torque from a single ring is accurate.

    Parameters
    ----------
    num_eccentricities : int
        Number of eccentricity samples to use.
    num_points : int
        Number of sample points to use.
    softening_length : float
        The softening length to use.
    output_path : str
        Path to save the plot.
    force : bool
        Whether to overwrite the output file if it already exists, by default False.
    """
    if os.path.isfile(output_path) and not force:
        log.info("Skipping single ring torque test...")
        return
    # Calculate the total number of frames
    total_frames: int = num_eccentricities * NUM_FRAMES_PER_TEST

    # Log test info
    log.info("Setting up single ring torque test...")
    log.info(f"Testing softening length of {softening_length} Rd")
    log.info(f"Testing {num_points} sample points.")
    log.info(f"Testing {num_eccentricities} different eccentricities.")
    log.info(f"Generating {total_frames} frames at {FRAME_RATE} FPS.")

    # Generate scale heights from eccentricities
    eccentricities = np.linspace(0, 1, num_eccentricities, endpoint=False)
    # The axes limits
    max_plot_extent: float = PLOT_PADDING_FACTOR / np.sqrt(1 - eccentricities[-1] ** 2)

    # Setup the plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title(f"Test torque from a single ring sampled with {num_points} points")
    ax.set_aspect("equal")
    ax.set_xlabel(r"x $(R_{\mathrm{d}})$")
    ax.set_ylabel(r"y $(R_{\mathrm{d}})$")
    ax.set_xlim(-max_plot_extent, max_plot_extent)
    ax.set_ylim(-max_plot_extent, max_plot_extent)
    eccentricity_label = ax.annotate(
        f"Ring eccentricity: {eccentricities[0]:.2f}",
        (0.05, 0.95),
        xycoords="axes fraction",
    )
    eccentricity_label.set_bbox(dict(facecolor="white", alpha=0.95, edgecolor="white"))

    # Initialise plots
    discretisation = RingDiscretisation(
        num_points, RingDiscretisationType.UNIFORM_NUM_PARTICLES
    )
    ring = Ring(1.0, eccentricities[0], (0, 0, 1), 0, 1, 0, 1, discretisation)
    pos_x = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    pos_y = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    mesh_x, mesh_y = np.meshgrid(pos_x, pos_y)
    pos_x_serial = mesh_x.flatten()
    pos_y_serial = mesh_y.flatten()
    pos_z_serial = np.zeros_like(pos_x_serial)
    pos = np.vstack((pos_x_serial, pos_y_serial, pos_z_serial))
    _, torque_vector, _ = ring.calculate_particle_torque_approx_array(
        pos, 1.0, num_points, softening_length
    )
    torque_z = torque_vector[:, 2].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
    torque_sign = np.sign(torque_z)

    theta = np.arctan2(mesh_y, mesh_x)
    curl_x = -np.sin(theta)
    curl_y = np.cos(theta)

    # Plot the torque as a vector field
    quiver_map = ax.quiver(
        mesh_x,
        mesh_y,
        torque_sign * curl_x,
        torque_sign * curl_y,
        torque_sign,
        cmap="seismic",
        pivot="mid",
        norm=mplcolors.Normalize(vmin=-1, vmax=+1),
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(quiver_map, cax=cax, orientation="vertical")

    def update_plot(frame: int) -> Sequence[Artist]:
        """Update the plot.

        Parameters
        ----------
        frame : int
            the frame number.

        Returns
        -------
        Sequence[Artist]
            the artists to update.
        """
        nonlocal ring
        eccentricity_index = frame // NUM_FRAMES_PER_TEST
        eccentricity = eccentricities[eccentricity_index]
        # Update the eccentricity every NUM_FRAMES_PER_TEST frames
        if frame % NUM_FRAMES_PER_TEST == 0:
            ring = Ring(1.0, eccentricity, (0, 0, 1), 0, 1, 0, 1, discretisation)
        # Update the rotation every frame
        angle = 2 * np.pi * frame / NUM_FRAMES_PER_TEST
        new_rotation = Quaternion.from_axis_angle((0, 0, 1), angle)
        ring.set_rotation_quaternion(new_rotation)

        # Update the plots
        _, torque_vector, _ = ring.calculate_particle_torque_approx_array(
            pos, 1.0, num_points, softening_length
        )
        torque_z = torque_vector[:, 2].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
        torque_sign = np.sign(torque_z)
        quiver_map.set_UVC(
            torque_sign * curl_x,
            torque_sign * curl_y,
            torque_sign,
        )

        eccentricity_label.set_text(f"Ring eccentricity: {eccentricity:.2f}")
        return [quiver_map, eccentricity_label]

    # Create function animation
    animation = FuncAnimation(
        fig,
        update_plot,
        init_func=init_plot,
        frames=total_frames,
        blit=True,
    )

    # Track progress and save animation
    with CustomProgress() as progress:
        animation_task = progress.add_task(
            "Running single ring torque test...",
            total=total_frames,
            leave=False,
        )
        animation.save(
            output_path,
            fps=FRAME_RATE,
            progress_callback=lambda _i, _n: progress.advance(animation_task),
            dpi=120,
        )
    plt.close(fig)
    log.info("Finished single ring torque test.")
