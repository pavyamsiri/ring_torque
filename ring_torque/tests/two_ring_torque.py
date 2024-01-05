# Standard libraries
import logging
import os
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mplcolors
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Internal libraries
from .common import NUM_FRAMES_PER_TEST, FRAME_RATE, init_plot, FORCE_GRID_POINTS
from ring_torque.common import PLOT_PADDING_FACTOR
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.ring import RingDiscretisation, RingDiscretisationType
from ring_torque.system import RingSystem

log = logging.getLogger(__name__)


def test_two_rings_torque(
    radius_ratio: float,
    num_eccentricities: int,
    num_points: int,
    softening_length: float,
    output_path: str,
    force: bool = False,
):
    """Test if the torque from two rings is accurate. The inner ring is fixed and the outer ring is rotated.

    Parameters
    ----------
    radius_ratio : float
        The ratio of the outer ring radius to the inner ring radius times its semi-major axis.
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
    crossing_flag = "crossing" if radius_ratio < 1 else "non-crossing"
    if os.path.isfile(output_path) and not force:
        log.info(f"Skipping two {crossing_flag} rings torque test...")
        return
    # Calculate the total number of frames
    total_frames: int = num_eccentricities * NUM_FRAMES_PER_TEST

    # Log test info
    log.info(f"Setting up two {crossing_flag} rings torque test...")
    log.info(f"Testing softening length of {softening_length} Rd")
    log.info(f"Testing {num_points} sample points.")
    log.info(f"Testing {num_eccentricities} different eccentricities.")
    log.info(f"Generating {total_frames} frames at {FRAME_RATE} FPS.")

    # Generate eccentricities
    eccentricities = np.linspace(0, 1, num_eccentricities, endpoint=False)
    scale_heights = 1 / np.sqrt(1 - eccentricities**2)
    # Generate non-crossing radii
    inner_radius = 1.0
    outer_radii = radius_ratio * scale_heights * inner_radius
    # The axes limits
    max_plot_extent: float = (
        PLOT_PADDING_FACTOR / np.sqrt(1 - eccentricities[-1] ** 2) * outer_radii.max()
    )

    # Setup the plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title(
        f"Test torque from two {crossing_flag} rings sampled with {num_points} points"
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"x $(R_{\mathrm{d}})$")
    ax.set_xlim(-max_plot_extent, max_plot_extent)
    ax.set_ylim(-max_plot_extent, max_plot_extent)
    eccentricity_label = ax.annotate(
        f"Ring eccentricity: {eccentricities[0]:.2f}",
        (0.05, 0.95),
        xycoords="axes fraction",
    )
    eccentricity_label.set_bbox(dict(facecolor="white", alpha=0.95, edgecolor="white"))

    # Initialise the plots
    discretisation = RingDiscretisation(
        num_points, RingDiscretisationType.UNIFORM_NUM_PARTICLES
    )
    ring_system = RingSystem(
        [inner_radius, outer_radii[0]], eccentricities[0], 1, 1, 0, discretisation
    )
    pos_x = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    pos_y = np.linspace(-max_plot_extent, max_plot_extent, FORCE_GRID_POINTS)
    mesh_x, mesh_y = np.meshgrid(pos_x, pos_y)
    pos_x_serial = mesh_x.flatten()
    pos_y_serial = mesh_y.flatten()
    pos_z_serial = np.zeros_like(pos_x_serial)
    pos = np.vstack((pos_x_serial, pos_y_serial, pos_z_serial))
    _, torque_vector, _ = ring_system.calculate_torque_on_ring_array(
        pos, 1.0, softening_length, 1
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
    # Plot the rings as ellipse patches
    ellipse_patches: list[patches.Ellipse] = []
    for ring in ring_system.rings:
        ellipse_patch = patches.Ellipse(
            (0, 0),
            ring.total_width,
            ring.total_height,
            angle=0,
            fill=False,
            edgecolor="black",
        )
        ax.add_patch(ellipse_patch)
        ellipse_patches.append(ellipse_patch)

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
            outer_radius = outer_radii[eccentricity_index]
            ring_system = RingSystem(
                [inner_radius, outer_radius], eccentricity, 1, 1, 0, discretisation
            )
            # Update the patches
            for ellipse_patch, ring in zip(ellipse_patches, ring_system.rings):
                ellipse_patch.set_width(ring.total_width)
                ellipse_patch.set_height(ring.total_height)
        # Update the rotation every frame
        angle = 2 * np.pi * frame / NUM_FRAMES_PER_TEST
        new_rotation = Quaternion.from_axis_angle((0, 0, 1), angle)
        ring_system.rings[1].set_rotation_quaternion(new_rotation)
        ellipse_patches[1].set_angle(np.rad2deg(angle))
        total_torque, _, _ = ring_system.calculate_torque_on_ring(
            1.0, softening_length, 1
        )
        ellipse_patches[1].set_edgecolor("red" if total_torque[2] > 0 else "blue")

        # Update the plots
        _, torque_vector, _ = ring_system.calculate_torque_on_ring_array(
            pos, 1.0, softening_length, 1
        )
        torque_z = torque_vector[:, 2].reshape((FORCE_GRID_POINTS, FORCE_GRID_POINTS))
        torque_sign = np.sign(torque_z)
        quiver_map.set_UVC(
            torque_sign * curl_x,
            torque_sign * curl_y,
            torque_sign,
        )
        eccentricity_label.set_text(f"Ring eccentricity: {eccentricity:.2f}")
        return [quiver_map, eccentricity_label] + ellipse_patches

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
            f"Running two {crossing_flag} rings torque test...",
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
    log.info(f"Finished two {crossing_flag} rings torque test")
