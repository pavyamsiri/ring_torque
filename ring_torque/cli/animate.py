"""This module contains the animate command for the ring_torque CLI."""

# Standard libraries
import argparse
from dataclasses import dataclass
from functools import partial
import glob
import logging
import os
from typing import Sequence

# Internal libraries
from .parser import _parse_animate
from .parallel import process_tasks_with_pool
from .file_helpers import (
    VIDEO_EXTENSIONS,
    check_both_directories,
    check_both_files,
    format_output_path,
)
from .system_helpers import parse_discretisation
from ring_torque.animation import (
    RingAnimationTraceOptions,
    RingAnimationPanelKind,
    animate_rings,
)
from ring_torque.common import setup_logging
from ring_torque.ring import RingDiscretisation

log = logging.getLogger(__name__)


@dataclass
class AnimationParameters:
    """Contains animation parameters.

    Attributes
    ----------
    discretisation : RingDiscretisation
        The discretisation to use for each ring.
    max_time_step : int | None
        The maximum time step to animate.
    fps : int
        The number of frames per second to use in the animation.
    show_points : bool
        Whether to show the ring particles.
    show_axes : bool
        Whether to show the axes.
    as_kpc : bool
        Whether to show the rings in kpc.
    skip_interpolation : bool
        Skips grid interpolation to increase performance.
    trace : RingAnimationTraceOptions
        The trace options to use.
    panel_specs : Sequence[RingAnimationPanel]
        The panels to use in the animation.
    """

    discretisation: RingDiscretisation
    max_time_step: int | None
    fps: int
    show_points: bool
    show_axes: bool
    as_kpc: bool
    skip_interpolation: bool
    trace: RingAnimationTraceOptions
    panel_specs: Sequence[RingAnimationPanelKind]


@dataclass
class AnimationJobTask:
    """Contains the input and output file paths for a single animation job.

    Attributes
    ----------
    data_file : str
        The path to the data file to animate.
    output_file : str
        The path to the output file.
    """

    data_file: str
    output_file: str


def main(args: list[str]):
    parsed_args = _parse_args(args)
    if check_both_files(parsed_args.input_path, parsed_args.output_path):
        _animate(parsed_args)
    elif check_both_directories(parsed_args.input_path, parsed_args.output_path):
        _batch_animate(parsed_args)
    else:
        raise ValueError(
            "The input and output paths must both be files or both be directories."
        )


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the `animate` command.

    Parameters
    ----------
    args : list[str]
        The command line arguments to parse.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="ring_torque animate",
        description="Animate a simulation or comparison.",
    )
    _parse_animate(parser)
    return parser.parse_args(args)


def _animate(args: argparse.Namespace):
    """The entry point for the `animate` subcommand. This function expects a single input file and a single output file.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed command line arguments.

    Raises
    ------
    ValueError
        If the given input and output paths are not both files or both directories.
    """
    # File paths
    input_path: str = args.input_path
    output_path: str = args.output_path
    # Animation parameters
    max_time_step: int | None = args.max_time_step
    fps: int = args.fps
    show_axes: bool = not args.remove_axes
    show_points: bool = args.show_points
    as_kpc: bool = args.as_kpc
    skip_interpolation: bool = args.skip_interpolation
    trace_options: RingAnimationTraceOptions = args.trace
    panel_specs: Sequence[RingAnimationPanelKind] = args.panel_specs
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)
    # Log out to a file in the same directory as the output file
    output_path = format_output_path(output_path, VIDEO_EXTENSIONS)
    # Animate the rings
    log.info(f"Animating {input_path} and outputting to {output_path}.")
    animate_rings(
        input_path,
        output_path,
        discretisation,
        panel_specs,
        trace_options,
        max_time_step,
        fps,
        show_axes,
        show_points,
        as_kpc,
        skip_interpolation,
    )


def _batch_animate(args: argparse.Namespace) -> None:
    """The entry point for the `batch_animate` subcommand.
    This function expects a single input directory and a single output directory.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    # File paths
    input_path: str = args.input_path
    output_path: str = args.output_path
    # Animation parameters
    max_time_step: int | None = args.max_time_step
    fps: int = args.fps
    show_axes: bool = not args.remove_axes
    show_points: bool = args.show_points
    as_kpc: bool = args.as_kpc
    skip_interpolation: bool = args.skip_interpolation
    trace_options: RingAnimationTraceOptions = args.trace
    panel_specs: Sequence[RingAnimationPanelKind] = args.panel_specs
    # Number of workers
    num_processes: int = args.num_processes
    # Force flag
    force: bool = args.force
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)

    # Animation parameters
    parameters = AnimationParameters(
        discretisation,
        max_time_step,
        fps,
        show_points,
        show_axes,
        as_kpc,
        skip_interpolation,
        trace_options,
        panel_specs,
    )

    # Glob the input directory
    data_template = os.path.join(input_path, "*.hdf5")
    data_files = glob.glob(data_template)
    tasks = __generate_animation_task(
        data_files, output_path, VIDEO_EXTENSIONS[0], force
    )

    log.info(f"Animating data files from {input_path} and saving to {output_path}")

    display = num_processes == 1
    process_tasks_with_pool(
        num_processes,
        partial(__animate_from_data_file, parameters, display=display),
        tasks,
        "Animating rings",
    )


def __animate_from_data_file(
    parameters: AnimationParameters, task: AnimationJobTask, display: bool = True
) -> None:
    """Animate the rings from the given data file.

    Parameters
    ----------
    parameters : AnimationParameters
        The animation parameters to use.
    task : AnimationJobTask
        The task to complete. This contains the input and output file paths.
    """
    # Turn off logs
    if not display:
        logging.getLogger("ring_torque.animation").setLevel(logging.WARNING)
        logging.getLogger("ring_torque.system").setLevel(logging.WARNING)
    animate_rings(
        task.data_file,
        task.output_file,
        parameters.discretisation,
        parameters.panel_specs,
        parameters.trace,
        parameters.max_time_step,
        parameters.fps,
        parameters.show_axes,
        parameters.show_points,
        parameters.as_kpc,
        parameters.skip_interpolation,
        display=display,
    )


def __generate_animation_task(
    data_files: Sequence[str], output_path: str, output_extension: str, force: bool
) -> Sequence[AnimationJobTask]:
    """Generate an animation task from the given data file.

    Parameters
    ----------
    data_files : Sequence[str]
        The paths to the data files to animate.
    output_path : str
        The output directory to save the animations to.
    output_extension : str
        The output extension to use. The extension must start with `.`.
    force : bool
        Whether to overwrite existing files.

    Returns
    -------
    Sequence[AnimationJobTask]
        The animation job tasks.
    """
    tasks: list[AnimationJobTask] = []
    for data_file in data_files:
        # Format output path file name
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        output_file = os.path.join(output_path, f"{base_name}{output_extension}")
        if not force and os.path.exists(output_file):
            continue
        tasks.append(AnimationJobTask(data_file, output_file))
    return tasks


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
