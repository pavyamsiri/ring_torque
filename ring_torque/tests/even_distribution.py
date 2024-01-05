# Standard libraries
import logging
import os
from typing import Sequence

# External libraries
import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np

# Internal libraries
from .common import (
    NUM_FRAMES_PER_TEST,
    FRAME_RATE,
    init_plot,
)
from ring_torque.common import PLOT_PADDING_FACTOR
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.ring import Ring, RingDiscretisation, RingDiscretisationType

log = logging.getLogger(__name__)


def test_even_distribution(
    num_eccentricities: int, num_points: int, output_path: str, force: bool = False
):
    """Test if the sample points are evenly distributed for all rotations.

    Parameters
    ----------
    num_eccentricities : int
        Number of eccentricity samples to use.
    num_points : int
        Number of sample points to use.
    output_path : str
        Path to save the plot.
    force : bool
        Whether to overwrite the output file if it already exists, by default False.
    """
    if os.path.isfile(output_path) and not force:
        log.info("Skipping sample point uniformity test...")
        return

    # Calculate the total number of frames
    total_frames: int = num_eccentricities * NUM_FRAMES_PER_TEST

    # Log test info
    log.info("Setting up sample point uniformity test...")
    log.info(f"Testing {num_points} sample points.")
    log.info(f"Testing {num_eccentricities} different eccentricities.")
    log.info(f"Generating {total_frames} frames at {FRAME_RATE} FPS.")

    # Generate scale heights from eccentricities
    eccentricities = np.linspace(0, 1, num_eccentricities, endpoint=False)
    # The axes limits
    max_plot_extent: float = PLOT_PADDING_FACTOR / np.sqrt(1 - eccentricities[-1] ** 2)
    # Generate colors for the sample points
    point_cmap = mpl.colormaps["tab10"]
    point_color = point_cmap(list(np.linspace(0, 1, num_points, dtype=float)))

    # Setup the plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title(f"Test uniformity of {num_points} sample points")
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
    points = ring.autosample_points()
    points_x = points[0, :]
    points_y = points[1, :]
    closed_x = np.append(points_x, points_x[0])
    closed_y = np.append(points_y, points_y[0])
    (ellipse_line,) = ax.plot(closed_x, closed_y, color="blue")
    ellipse_points = ax.scatter(
        points_x, points_y, facecolors=point_color, edgecolors=None, alpha=1.0, s=100
    )

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
        points = ring.autosample_points()
        points_x = points[0, :]
        points_y = points[1, :]
        closed_x = np.append(points_x, points_x[0])
        closed_y = np.append(points_y, points_y[0])
        ellipse_line.set_data(closed_x, closed_y)
        ellipse_points.set_offsets(np.column_stack([points_x, points_y]))
        eccentricity_label.set_text(f"Ring eccentricity: {eccentricity:.2f}")
        return [ellipse_line, ellipse_points, eccentricity_label]

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
            "Running sample point uniformity test...",
            total=total_frames,
            leave=False,
        )
        animation.save(
            output_path,
            fps=FRAME_RATE,
            progress_callback=lambda _, _n: progress.advance(animation_task),
            dpi=120,
        )
    plt.close(fig)
    log.info("Finished sample point uniformity test.")
