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
from .common import (
    NUM_FRAMES_PER_TEST,
    FRAME_RATE,
    FORCE_GRID_POINTS,
    init_plot,
)
from ring_torque.common import PLOT_PADDING_FACTOR
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.system import RingSystem
from ring_torque.radii import calculate_uniform_radii
from ring_torque.ring import RingDiscretisation, RingDiscretisationType

log = logging.getLogger(__name__)


def test_multiple_ring_force(
    num_rings: int,
    ring_spacing: float,
    num_eccentricities: int,
    num_points: int,
    softening_length: float,
    output_path: str,
    force: bool = False,
):
    """Test if the force from multiple rings is accurate.

    Parameters
    ----------
    num_rings: int
        Number of rings to use.
    ring_spacing: float
        The spacing between rings.
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
        log.info("Skipping multiple ring force test...")
        return
    # Calculate the total number of frames
    total_frames: int = num_eccentricities * NUM_FRAMES_PER_TEST

    # Log test info
    log.info("Setting up multiple ring force test...")
    log.info(f"Testing {num_rings} rings")
    log.info(f"Testing softening length of {softening_length} Rd")
    log.info(f"Testing {num_points} sample points.")
    log.info(f"Testing {num_eccentricities} different eccentricities.")
    log.info(f"Generating {total_frames} frames at {FRAME_RATE} FPS.")

    # Generate radii
    radii = calculate_uniform_radii(num_rings, 1.0, ring_spacing)
    # Generate scale heights from eccentricities
    eccentricities = np.linspace(0, 1, num_eccentricities, endpoint=False)
    # The axes limits
    max_plot_extent: float = (
        PLOT_PADDING_FACTOR / np.sqrt(1 - eccentricities[-1] ** 2) * max(radii)
    )

    # Setup the plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title(f"Test force from multiple rings sampled with {num_points} points")
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
    ring_system = RingSystem(radii, eccentricities[0], 1, 1, 0, discretisation)
    pos_x = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    pos_y = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    mesh_x, mesh_y = np.meshgrid(pos_x, pos_y)
    pos_x_serial = mesh_x.flatten()
    pos_y_serial = mesh_y.flatten()
    pos_z_serial = np.zeros_like(pos_x_serial)
    pos = np.vstack((pos_x_serial, pos_y_serial, pos_z_serial))
    ring_force = ring_system.calculate_force_on_ring_array(
        pos, 1.0, softening_length, 0
    )
    force_x = ring_force[0, :].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
    force_y = ring_force[1, :].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
    force_mag = np.hypot(force_x, force_y)
    force_x[force_mag > 0] /= force_mag[force_mag > 0]
    force_y[force_mag > 0] /= force_mag[force_mag > 0]
    # Plot the force as a vector field
    quiver_map = ax.quiver(
        mesh_x,
        mesh_y,
        force_x,
        force_y,
        force_mag,
        cmap="viridis",
        pivot="mid",
        norm=mplcolors.Normalize(vmin=0, vmax=force_mag.max()),
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
        nonlocal ring_system
        eccentricity_index = frame // NUM_FRAMES_PER_TEST
        eccentricity = eccentricities[eccentricity_index]
        # Update the eccentricity every NUM_FRAMES_PER_TEST frames
        if frame % NUM_FRAMES_PER_TEST == 0:
            ring_system = RingSystem(radii, eccentricity, 1, 1, 0, discretisation)
        # Update the rotation every frame
        angle = 2 * np.pi * frame / NUM_FRAMES_PER_TEST
        new_rotation = Quaternion.from_axis_angle((0, 0, 1), angle)
        for ring in ring_system.rings:
            ring.set_rotation_quaternion(new_rotation)

        # Update the plots
        ring_force = ring_system.calculate_force_on_ring_array(
            pos, 1.0, softening_length, 0
        )
        force_x = ring_force[0, :].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
        force_y = ring_force[1, :].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
        force_mag = np.hypot(force_x, force_y)
        force_x[force_mag > 0] /= force_mag[force_mag > 0]
        force_y[force_mag > 0] /= force_mag[force_mag > 0]
        quiver_map.set_UVC(force_x, force_y, force_mag)
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
            "Running multiple ring force test...",
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
    log.info("Finished multiple ring force test")
