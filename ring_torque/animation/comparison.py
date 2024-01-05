# Standard libraries
import time
import logging

# External libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import numpy as np

# Internal libraries
from .common import (
    init_plot_identity,
    read_hdf5_file,
)
from .panel import trace_density_wave
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.system import RingSystem
from ring_torque.ring import RingDiscretisation


log = logging.getLogger(__name__)


# TODO: Fix this up because its ugly
def animate_rings_comparison(
    first_path: str,
    second_path: str,
    output_path: str,
    discretisation: RingDiscretisation,
    max_time_step: int | None,
    fps: int,
    show_points: bool,
    trace: bool,
    display: bool = True,
) -> None:
    """Animates a comparison between two ring systems.

    Parameters
    ----------
    first_path : str
        The path to the HDF5 file containing the first ring system.
    second_path : str
        The path to the HDF5 file containing the second ring system.
    output_path : str
        The path to the output file.
    num_particles : int
        The number of particles to use when drawing the rings.
    max_time_step : int | None
        The maximum number of time steps to animate. If None, all time steps will be animated.
    fps : int
        The frames per second of the output animation.
    show_points : bool
        Whether to show the individual particles of the rings.
    trace : bool
        Whether to trace the density wave of the rings.
    display : bool, optional
        Whether to display the progress bar during animation creation.
        Default is True.
    """
    animation_data = read_hdf5_file(first_path)
    disc: RingSystem = animation_data.ring_system
    disc.set_discretisation(discretisation)
    other_animation_data = read_hdf5_file(second_path)
    other_disc: RingSystem = other_animation_data.ring_system
    other_disc.set_discretisation(discretisation)
    rotate_clockwise: bool = animation_data.tangential_velocity < 0

    log.debug("Starting ring simulation...")
    # Set up figure and axes
    fig = plt.figure(figsize=(11, 9))
    fig.suptitle(f"{animation_data.name} vs {other_animation_data.name}")

    ax_xy = fig.add_subplot(221)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel(r"$x$")
    ax_xy.set_ylabel(r"$y$")
    ax_xy.set_xlim(-animation_data.max_plot_extent, animation_data.max_plot_extent)
    ax_xy.set_ylim(-animation_data.max_plot_extent, animation_data.max_plot_extent)

    ax_yz = fig.add_subplot(223)
    ax_yz.set_aspect("equal")
    ax_yz.set_xlabel(r"$y$")
    ax_yz.set_ylabel(r"$z$")
    ax_yz.set_xlim(-animation_data.max_plot_extent, animation_data.max_plot_extent)
    ax_yz.set_ylim(-animation_data.max_plot_extent, animation_data.max_plot_extent)

    other_ax_xy = fig.add_subplot(222)
    other_ax_xy.set_aspect("equal")
    other_ax_xy.set_xlabel(r"$x$")
    other_ax_xy.set_ylabel(r"$y$")
    other_ax_xy.set_xlim(
        -other_animation_data.max_plot_extent, other_animation_data.max_plot_extent
    )
    other_ax_xy.set_ylim(
        -other_animation_data.max_plot_extent, other_animation_data.max_plot_extent
    )

    other_ax_yz = fig.add_subplot(224)
    other_ax_yz.set_aspect("equal")
    other_ax_yz.set_xlabel(r"$y$")
    other_ax_yz.set_ylabel(r"$z$")
    other_ax_yz.set_xlim(
        -other_animation_data.max_plot_extent, other_animation_data.max_plot_extent
    )
    other_ax_yz.set_ylim(
        -other_animation_data.max_plot_extent, other_animation_data.max_plot_extent
    )

    rings = animation_data.ring_system._rings
    other_rings = other_animation_data.ring_system._rings
    radii = list(animation_data.ring_system.radii) + list(
        other_animation_data.ring_system.radii
    )
    # Create a color map to help differentiate between the rings
    ellipse_cmap = mpl.colormaps["cool"]
    min_radius = min(radii)
    max_radius = max(radii)
    marker_style = "o" if show_points else ""

    # Store the rings' patches
    ellipse_xy: list[Line2D] = []
    ellipse_yz: list[Line2D] = []
    # Draw the rings
    for _, ring in reversed(list(enumerate(rings))):
        ring_color = ellipse_cmap(
            (ring.radius - min_radius) / (max_radius - min_radius)
        )
        points = ring.autosample_points()
        points_x = np.append(points[0, :], points[0, 0])
        points_y = np.append(points[1, :], points[1, 0])
        points_z = np.append(points[2, :], points[2, 0])
        (xy_line,) = ax_xy.plot(
            points_x, points_y, color=ring_color, marker=marker_style, markersize=3.5
        )
        ellipse_xy.append(xy_line)

        (yz_line,) = ax_yz.plot(
            points_y, points_z, color=ring_color, marker=marker_style, markersize=3.5
        )
        ellipse_yz.append(yz_line)
    ellipse_xy = list(reversed(ellipse_xy))
    ellipse_yz = list(reversed(ellipse_yz))

    # Store the rings' patches
    other_ellipse_xy: list[Line2D] = []
    other_ellipse_yz: list[Line2D] = []
    # Draw the rings
    for _, ring in reversed(list(enumerate(other_rings))):
        ring_color = ellipse_cmap(
            (ring.radius - min_radius) / (max_radius - min_radius)
        )
        points = ring.autosample_points()
        points_x = np.append(points[0, :], points[0, 0])
        points_y = np.append(points[1, :], points[1, 0])
        points_z = np.append(points[2, :], points[2, 0])
        (xy_line,) = other_ax_xy.plot(
            points_x, points_y, color=ring_color, marker=marker_style, markersize=3.5
        )
        other_ellipse_xy.append(xy_line)

        (yz_line,) = other_ax_yz.plot(
            points_y, points_z, color=ring_color, marker=marker_style, markersize=3.5
        )
        other_ellipse_yz.append(yz_line)
    other_ellipse_xy = list(reversed(other_ellipse_xy))
    other_ellipse_yz = list(reversed(other_ellipse_yz))

    wave_line: Line2D | None = None
    complement_wave_line: Line2D | None = None
    other_wave_line: Line2D | None = None
    other_complement_wave_line: Line2D | None = None
    if trace:
        wave_points = trace_density_wave(rings, rotate_clockwise)
        (wave_line,) = ax_xy.plot(
            wave_points[0, :], wave_points[1, :], color="red", linewidth=3.5
        )
        (complement_wave_line,) = ax_xy.plot(
            -wave_points[0, :], -wave_points[1, :], color="red", linewidth=3.5
        )

        other_wave_points = trace_density_wave(
            other_rings,
            rotate_clockwise,
        )
        (other_wave_line,) = other_ax_xy.plot(
            other_wave_points[0, :], other_wave_points[1, :], color="red", linewidth=3.5
        )
        (other_complement_wave_line,) = other_ax_xy.plot(
            -other_wave_points[0, :],
            -other_wave_points[1, :],
            color="red",
            linewidth=3.5,
        )

    def update_plot(frame: int) -> list[Line2D]:
        """Update the plot for the current frame.

        Parameters
        ----------
        frame : int
            The current frame number.
        """
        # Update the rings' rotation and patches
        for (
            ring_idx,
            (ring, xy_line, yz_line),
        ) in enumerate(zip(rings, ellipse_xy, ellipse_yz)):
            actual_frame = frame // animation_data.save_interval
            if actual_frame > animation_data.num_snapshots:
                actual_frame = animation_data.num_snapshots - 1
            w = animation_data.orientations[0, ring_idx, actual_frame]
            x = animation_data.orientations[1, ring_idx, actual_frame]
            y = animation_data.orientations[2, ring_idx, actual_frame]
            z = animation_data.orientations[3, ring_idx, actual_frame]
            next_rotation = Quaternion(w, x, y, z)
            # Update the ring's rotation
            ring.set_rotation_quaternion(next_rotation)
            # Draw new points
            points = ring.autosample_points()
            points_x = np.append(points[0, :], points[0, 0])
            points_y = np.append(points[1, :], points[1, 0])
            points_z = np.append(points[2, :], points[2, 0])
            xy_line.set_data(points_x, points_y)
            yz_line.set_data(points_y, points_z)
        # Timestamp the plot
        time_stamp = f"Time = {frame * animation_data.time_step:.3f}"
        ax_xy.set_title(time_stamp)
        ax_yz.set_title(time_stamp)
        # Update the other rings' rotation and patches
        for (
            ring_idx,
            (ring, xy_line, yz_line),
        ) in enumerate(zip(other_rings, other_ellipse_xy, other_ellipse_yz)):
            actual_frame = frame // other_animation_data.save_interval
            if actual_frame > other_animation_data.num_snapshots:
                actual_frame = other_animation_data.num_snapshots - 1
            w = other_animation_data.orientations[0, ring_idx, actual_frame]
            x = other_animation_data.orientations[1, ring_idx, actual_frame]
            y = other_animation_data.orientations[2, ring_idx, actual_frame]
            z = other_animation_data.orientations[3, ring_idx, actual_frame]
            next_rotation = Quaternion(w, x, y, z)
            # Update the ring's rotation
            ring.set_rotation_quaternion(next_rotation)
            # Draw new points
            points = ring.autosample_points()
            points_x = np.append(points[0, :], points[0, 0])
            points_y = np.append(points[1, :], points[1, 0])
            points_z = np.append(points[2, :], points[2, 0])
            xy_line.set_data(points_x, points_y)
            yz_line.set_data(points_y, points_z)
        # Timestamp the plot
        time_stamp = f"Time = {frame * other_animation_data.time_step:.3f}"
        other_ax_xy.set_title(time_stamp)
        other_ax_yz.set_title(time_stamp)

        wave_lines: list[Line2D] = []
        if (
            wave_line is not None
            and complement_wave_line is not None
            and other_wave_line is not None
            and other_complement_wave_line is not None
        ):
            wave_points = trace_density_wave(rings, rotate_clockwise)
            wave_line.set_data(wave_points[0, :], wave_points[1, :])
            complement_wave_line.set_data(-wave_points[0, :], -wave_points[1, :])
            other_wave_points = trace_density_wave(
                other_rings,
                rotate_clockwise,
            )
            other_wave_line.set_data(other_wave_points[0, :], other_wave_points[1, :])
            other_complement_wave_line.set_data(
                -other_wave_points[0, :], -other_wave_points[1, :]
            )
            wave_lines = [
                wave_line,
                complement_wave_line,
                other_wave_line,
                other_complement_wave_line,
            ]

        return (
            ellipse_xy + ellipse_yz + other_ellipse_xy + other_ellipse_yz + wave_lines
        )

    num_time_steps = max(
        animation_data.num_time_steps, other_animation_data.num_time_steps
    )

    num_frames = (
        min(num_time_steps, max_time_step)
        if max_time_step is not None
        else num_time_steps
    )

    animation = FuncAnimation(
        fig,
        update_plot,
        init_func=init_plot_identity,
        frames=num_frames,
        blit=True,
        repeat=False,
    )
    start_time = time.perf_counter()
    with CustomProgress(disable=not display) as progress:
        animation_task = progress.add_task(
            "Creating animation...", total=num_frames, leave=False
        )
        animation.save(
            output_path,
            fps=fps,
            progress_callback=lambda _i, _n: progress.advance(animation_task),
        )
    plt.close(fig)
    elapsed_time = time.perf_counter() - start_time
    elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    if elapsed_time > num_time_steps:
        iteration_rate = elapsed_time / num_time_steps
        rate_unit = "s/iteration"
    else:
        iteration_rate = num_time_steps / elapsed_time
        rate_unit = "iterations/s"
    log.debug(
        f"Finished ring simulation in {int(elapsed_hours)}:{int(elapsed_minutes):02}:{round(elapsed_seconds):02}."
    )
    log.debug(f"Iteration speed: {(iteration_rate):.2f} {rate_unit}")
    log.debug(f"Animation saved to {output_path}")
