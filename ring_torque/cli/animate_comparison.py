"""This module contains the animate_comparison command for the ring_torque CLI."""

# Standard libraries
import argparse
import logging

# Internal libraries
from .parser import _parse_animate_comparison
from .file_helpers import (
    VIDEO_EXTENSIONS,
    check_all_files,
    format_output_path,
)
from .system_helpers import parse_discretisation
from ring_torque.animation import animate_rings_comparison
from ring_torque.common import setup_logging
from ring_torque.ring import RingDiscretisation


log = logging.getLogger(__name__)


def main(args: list[str]) -> None:
    """Parse command line arguments for the `animate_comparison` command.

    Parameters
    ----------
    args : list[str]
        The command line arguments to parse.
    """
    parsed_args = _parse_args(args)
    if check_all_files(
        [parsed_args.first_path, parsed_args.second_path], parsed_args.output_path
    ):
        _animate_comparison(parsed_args)
    else:
        raise ValueError("The input and output paths must both be files.")


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the `animate_comparison` command.

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
        prog="ring_torque animate_comparison",
        description="Animate a comparison between two simulations.",
    )
    _parse_animate_comparison(parser)
    return parser.parse_args(args)


def _animate_comparison(args: argparse.Namespace) -> None:
    """The entry point for the `animate_comparison` subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed command line arguments.

    Notes
    -----
    The `animate_comparison` subcommand is used to animate two ring simulations to compare them with each other.
    """
    # File paths
    first_path: str = args.first_path
    second_path: str = args.second_path
    output_path: str = args.output_path
    # Animation parameters
    max_time_step: int | None = args.max_time_step
    fps: int = args.fps
    show_points: bool = args.show_points
    trace: bool = args.trace
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)

    output_path = format_output_path(output_path, VIDEO_EXTENSIONS)
    log.info(f"Animation {first_path} vs {second_path} and outputting to {output_path}")
    animate_rings_comparison(
        first_path,
        second_path,
        output_path,
        discretisation,
        max_time_step,
        fps,
        show_points,
        trace,
    )


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
