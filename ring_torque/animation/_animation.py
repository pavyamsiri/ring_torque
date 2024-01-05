"""This module contains the functions for creating animations of ring systems."""

# Standard libraries
import logging
import time
from typing import Sequence

# External libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import text as mpltext
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation

from ring_torque.animation.panel import PanelInterface

# Internal libraries
from .common import (
    RingAnimationData,
    calculate_grid_layout,
    read_hdf5_file,
    init_plot_identity,
)
from .panel import RingAnimationPanelKind, RingAnimationTraceOptions, LengthUnit
from ring_torque.common import format_execution_time_s
from ring_torque.progress import CustomProgress
from ring_torque.quaternion import Quaternion
from ring_torque.ring import RingDiscretisation


log = logging.getLogger(__name__)


def animate_rings(
    input_path: str,
    output_path: str,
    discretisation: RingDiscretisation,
    panel_specs: Sequence[RingAnimationPanelKind],
    trace: RingAnimationTraceOptions,
    max_time_step: int | None,
    fps: int,
    show_axes: bool,
    show_points: bool,
    as_kpc: bool,
    skip_interpolation: bool,
    display: bool = True,
) -> None:
    """Create an animation of a system of rings from an HDF5 file.

    Parameters
    ----------
    input_path : str
        The path to the input HDF5 file.
    output_path : str
        The path to the output animation file.
    num_particles : int
        The number of particles to sample per ring.
    panel_specs : Sequence[RingAnimationPanel]
        The panels to draw.
    trace : RingAnimationTraceOptions
        The trace options for the animation.
    max_time_step : int | None
        The maximum number of time steps to include in the animation.
        If None, all time steps are included.
    fps : int
        The frames per second of the animation.
    show_axes : bool
        Whether to show the axes or not.
    show_points : bool
        Whether to show the sampled particles as points or not.
    as_kpc : bool
        Whether to plot the animation in kpc or in disk scale lengths.
    skip_interpolation : bool
        Skips grid interpolation. Grid interpolation is a visual effect.
    display : bool, optional
        Whether to display the progress bar during animation creation.
        Defaults to True.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    animation_data: RingAnimationData = read_hdf5_file(input_path)
    # Extract metadata
    disc = animation_data.ring_system
    length_unit = LengthUnit.KPC if as_kpc else LengthUnit.SCALE_LENGTH
    # Set the actual discretisation
    disc.set_discretisation(discretisation)
    animation_data.log_info()

    log.info("Starting ring animation...")
    # Set up figure and axes
    mpl.rcParams.update({"font.size": 16})
    num_rows, num_cols = calculate_grid_layout(len(panel_specs))
    width = 17
    height = 17
    title_fontsize: int = 18
    # Square grid
    if num_rows == num_cols:
        height = width
        mpl.rcParams.update({"font.size": 21})
        title_fontsize = 25
    # Special case for (1, 2) and (1, 3)
    elif num_rows == 1:
        height = width // num_cols
    # Special case for (2, 3)
    elif num_rows == 2 and num_cols == 3:
        height = width // 2
    else:
        log.error(f"Unsupported grid layout: {num_rows}x{num_cols}")
        return
    height += 1 if show_axes else 0
    fig = plt.figure(figsize=(width, height))
    title_label: mpltext.Text | None = None
    figure_title: str = animation_data.name
    if animation_data.name == "":
        if animation_data.with_gravity:
            figure_title = f"{animation_data.num_rings} rings with gravity"
        else:
            figure_title = f"{animation_data.num_rings} rings without gravity"
    else:
        figure_title = animation_data.name
    if show_axes:
        title_label = fig.suptitle(figure_title, fontsize=title_fontsize)

    conversion_factor = disc._scale_length if as_kpc else 1
    max_plot_extent = conversion_factor * animation_data.max_plot_extent

    # Create panels
    grid_spec = fig.add_gridspec(num_rows, num_cols)
    panels: Sequence[PanelInterface] = []
    with CustomProgress(disable=not display, transient=True) as progress:
        creation_task = progress.add_task("Creating panels...", total=len(panel_specs))
        for panel_idx, panel_spec in enumerate(panel_specs):
            current_panel = panel_spec.create(
                fig,
                grid_spec[panel_idx],
                animation_data,
                max_plot_extent,
                length_unit,
                show_points,
                show_axes,
                skip_interpolation,
                trace,
                mpl.colormaps["cool"],
            )
            current_panel.initialise(animation_data)
            if not show_axes:
                current_panel.turn_off_axes()
            panels.append(current_panel)
            progress.advance(creation_task)

    padding: float = 1.0 if show_axes else 0.0
    fig.tight_layout(pad=padding)

    def update_plot(frame: int) -> Sequence[Artist]:
        """Update the plot for the current frame.

        Parameters
        ----------
        frame : int
            The current frame number.
        """
        redrawn_artists: list[Artist] = []

        current_time = frame * animation_data.time_step

        # Update the rings' rotation and patches
        for ring_idx, ring in enumerate(disc.rings):
            actual_frame = frame // animation_data.save_interval
            w = animation_data.orientations[0, ring_idx, actual_frame]
            x = animation_data.orientations[1, ring_idx, actual_frame]
            y = animation_data.orientations[2, ring_idx, actual_frame]
            z = animation_data.orientations[3, ring_idx, actual_frame]
            next_rotation = Quaternion(w, x, y, z)
            # Update the ring's rotation
            ring.set_rotation_quaternion(next_rotation)
        for current_panel in panels:
            redrawn_artists.extend(current_panel.draw(frame, animation_data))

        # Timestamp the plot
        time_stamp = f"{current_time:.3f} Gyr"
        if title_label is not None:
            title_label.set_text(f"{figure_title}: {time_stamp}")
            # NOTE: For some reason, adding title label to redrawn artists makes it disappear.
            # redrawn_artists.append(title_label)
        return redrawn_artists

    num_time_steps = animation_data.num_time_steps
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
    with CustomProgress(disable=not display, transient=True) as progress:
        animation_task = progress.add_task("Creating animation...", total=num_frames)
        animation.save(
            output_path,
            fps=fps,
            progress_callback=lambda _i, _n: progress.advance(animation_task),
            dpi=60,
        )
    plt.close(fig)
    elapsed_time = time.perf_counter() - start_time
    elapsed_time_str, iteration_rate_str = format_execution_time_s(
        elapsed_time, num_time_steps
    )
    log.info(f"Finished ring simulation in {elapsed_time_str}.")
    log.info(f"Iteration speed: {iteration_rate_str}.")
    log.info(f"Animation saved to {output_path}")
