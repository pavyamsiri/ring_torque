# Standard libraries
import logging
import os
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np

# Internal libraries
from .common import NUM_FRAMES_PER_TEST, FRAME_RATE, init_plot
from ring_torque.common import NumpyFloatType
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.system import RingSystem
from ring_torque.ring import RingDiscretisation, RingDiscretisationType

log = logging.getLogger(__name__)


def test_two_ring_torque_response(
    radius_ratio: float,
    num_eccentricities: int,
    num_points: int,
    softening_length: float,
    output_path: str,
    force: bool = False,
):
    """Test the torque response of two rings. This outputs an animated plot of the torque along the z-axis for two gravitating
    elliptic rings.

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
        log.info("Skipping two ring torque response test...")
        return
    # Calculate the total number of frames
    total_frames: int = num_eccentricities * NUM_FRAMES_PER_TEST

    # Log test info
    log.info(f"Setting up two {crossing_flag} rings torque response test...")
    log.info(f"Testing softening length of {softening_length} Rd")
    log.info(f"Testing {num_points} sample points.")
    log.info(f"Testing {num_eccentricities} different eccentricities.")
    log.info(f"Generating {total_frames} frames at {FRAME_RATE} FPS.")

    # Generate eccentricities
    eccentricities = np.linspace(0, 1, num_eccentricities, endpoint=False)
    scale_heights = 1 / np.sqrt(1 - eccentricities**2)
    # Generate radii
    inner_radius = 1.0
    outer_radii = radius_ratio * scale_heights * inner_radius

    # Setup the plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title(
        f"Test torque response from two {crossing_flag} rings sampled with {num_points} points"
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"Normalised $\tau_z$")
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
    theta = np.linspace(0, 2 * np.pi, NUM_FRAMES_PER_TEST, endpoint=False)
    outer_total_torque, _, _ = ring_system.calculate_torque_on_ring(
        1.0, softening_length, 1
    )
    inner_total_torque, _, _ = ring_system.calculate_torque_on_ring(
        1.0, softening_length, 0
    )

    # Add guidelines
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(90, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(180, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(270, color="black", linestyle="--", linewidth=0.5)

    # Use natural angles for axis ticks
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

    # Calculate the torque response across all angles
    outer_torque_response = np.full(NUM_FRAMES_PER_TEST, np.nan, dtype=NumpyFloatType)
    inner_torque_response = np.full(NUM_FRAMES_PER_TEST, np.nan, dtype=NumpyFloatType)
    for idx in range(NUM_FRAMES_PER_TEST):
        angle = 2 * np.pi * idx / NUM_FRAMES_PER_TEST
        new_rotation = Quaternion.from_axis_angle((0, 0, 1), angle)
        ring_system.rings[1].set_rotation_quaternion(new_rotation)
        outer_total_torque, _, _ = ring_system.calculate_torque_on_ring(
            1.0, softening_length, 1
        )
        outer_torque_response[idx] = outer_total_torque[2]
        inner_total_torque, _, _ = ring_system.calculate_torque_on_ring(
            1.0, softening_length, 0
        )
        inner_torque_response[idx] = inner_total_torque[2]
    # Normalise the torques
    outer_torque_response /= outer_torque_response.max()
    inner_torque_response /= inner_torque_response.max()
    # Plot the torque response
    (outer_torque_line,) = ax.plot(
        np.rad2deg(theta), outer_torque_response, label="Outer ring"
    )
    (inner_torque_line,) = ax.plot(
        np.rad2deg(theta), inner_torque_response, label="Inner ring"
    )

    ax.legend(handles=[outer_torque_line, inner_torque_line], loc="upper right")

    def update_plot(frame: int) -> Sequence[Artist]:
        """Update the plot.

        Parameters
        ----------
        frame : int
            The frame number.

        Returns
        -------
        Sequence[Artist]
            The artists to update.
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
            # Calculate the torque response across all angles
            outer_torque_response = np.full(
                NUM_FRAMES_PER_TEST, np.nan, dtype=NumpyFloatType
            )
            inner_torque_response = np.full(
                NUM_FRAMES_PER_TEST, np.nan, dtype=NumpyFloatType
            )
            for idx in range(NUM_FRAMES_PER_TEST):
                angle = 2 * np.pi * idx / NUM_FRAMES_PER_TEST
                new_rotation = Quaternion.from_axis_angle((0, 0, 1), angle)
                ring_system.rings[1].set_rotation_quaternion(new_rotation)
                outer_total_torque, _, _ = ring_system.calculate_torque_on_ring(
                    1.0, softening_length, 1
                )
                outer_torque_response[idx] = outer_total_torque[2]
                inner_total_torque, _, _ = ring_system.calculate_torque_on_ring(
                    1.0, softening_length, 0
                )
                inner_torque_response[idx] = inner_total_torque[2]
            # Normalise the torques
            outer_torque_response /= outer_torque_response.max()
            inner_torque_response /= inner_torque_response.max()
            outer_torque_line.set_ydata(outer_torque_response)
            inner_torque_line.set_ydata(inner_torque_response)
            eccentricity_label.set_text(f"Ring eccentricity: {eccentricity:.2f}")

        return [outer_torque_line, inner_torque_line, eccentricity_label]

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
            f"Running two {crossing_flag} rings torque response test...",
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
    log.info(f"Finished two {crossing_flag} rings torque response test")
