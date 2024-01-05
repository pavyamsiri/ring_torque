"""This module contains the batch_animate_comparison command for the ring_torque CLI."""

# Standard libraries
import argparse
from dataclasses import dataclass
from functools import partial
import glob
import itertools
import logging
import os
import re

# Internal libraries
from .file_helpers import check_both_directories
from .parallel import process_tasks_with_pool
from .parser import _parse_batch_animate_comparison
from .simulate_regimes import Regime
from .system_helpers import parse_discretisation
from ring_torque.animation import animate_rings_comparison
from ring_torque.common import setup_logging
from ring_torque.ring import RingDiscretisation

log = logging.getLogger(__name__)


@dataclass
class AnimationComparisonParameters:
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
    trace : bool
        Whether to trace the ring particles.
    """

    discretisation: RingDiscretisation
    max_time_step: int | None
    fps: int
    show_points: bool
    trace: bool


@dataclass
class AnimationComparisonJobTask:
    """A dataclass to hold the information for a comparison task.

    Attributes
    ----------
    first_path : str
        The path to the first input file.
    second_path : str
        The path to the second input file.
    output_path : str
        The path to the output file.
    """

    first_path: str
    second_path: str
    output_path: str


def main(args: list[str]) -> None:
    """Parse command line arguments for the `batch_animate_comparison` command.

    Parameters
    ----------
    args : list[str]
        The command line arguments to parse.
    """
    parsed_args = _parse_args(args)
    if check_both_directories(parsed_args.input_path, parsed_args.output_path):
        _batch_animate_comparison(parsed_args)
    else:
        raise ValueError("The input and output paths must both be directories.")


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the `batch_animate_comparison` command.

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
        prog="ring_torque batch_animate_comparison",
        description="Batch animate a comparison between two simulations.",
    )
    _parse_batch_animate_comparison(parser)
    return parser.parse_args(args)


def _batch_animate_comparison(args: argparse.Namespace) -> None:
    """Animate all possible comparisons between two simulations given a folder of data files.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.
    """
    # File paths
    input_path: str = args.input_path
    output_path: str = args.output_path
    # Animation parameters
    max_time_step: int | None = args.max_time_step
    fps: int = args.fps
    show_points: bool = args.show_points
    trace: bool = args.trace
    # File handling parameters
    force: bool = args.force
    # Multiprocessing parameters
    num_processes: int = args.num_processes
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)

    parameters: AnimationComparisonParameters = AnimationComparisonParameters(
        discretisation, max_time_step, fps, show_points, trace
    )

    log.info(
        f"Animating all comparisons in {input_path} and outputting to the {output_path} directory."
    )
    # Generate tasks
    tasks = __generate_comparison_tasks(input_path, output_path, trace, force)

    display = num_processes == 1
    process_tasks_with_pool(
        num_processes,
        partial(__animate_comparison_from_task, parameters, display=display),
        tasks,
        "Animating ring comparisons",
    )


def __animate_comparison_from_task(
    parameters: AnimationComparisonParameters,
    task: AnimationComparisonJobTask,
    display: bool = True,
) -> None:
    """Animate the given task.

    Parameters
    ----------
    parameters : AnimationComparisonParameters
        The animation parameters.
    task : AnimationComparisonJobTask
        The task to animate.
    display : bool, optional
        Whether to display the animation. Defaults to True.
    """
    # Turn off logs
    if not display:
        logging.getLogger("ring_torque.system").setLevel(logging.WARNING)
        logging.getLogger("ring_torque.animation").setLevel(logging.WARNING)
    animate_rings_comparison(
        task.first_path,
        task.second_path,
        task.output_path,
        parameters.discretisation,
        parameters.max_time_step,
        parameters.fps,
        parameters.show_points,
        parameters.trace,
        display=display,
    )


def __generate_comparison_tasks(
    input_dir: str,
    output_dir: str,
    trace: bool,
    force: bool,
) -> list[AnimationComparisonJobTask]:
    """Generate comparison tasks for the given input path.

    Parameters
    ----------
    input_dir : str
        The input directory.
    output_dir : str
        The output directory.
    trace : bool
        A flag only used to tag in the output name to help distinguish between traced and untraced animations.
    force : bool
        Whether to force the animation to be generated even if the output file already exists.

    Returns
    -------
    tasks : list[ComparisonTask]
        A list of tasks to be completed by the pool.
    """

    def get_first_group(text: str, pattern: re.Pattern) -> str | None:
        """Get the first capture group of the given pattern from the text.

        Parameters
        ----------
        text : str
            The text to search.
        pattern : re.Pattern
            The pattern to search for.

        Returns
        -------
        group : str | None
            The first capture group of the given pattern from the text or None if no match was found.
        """
        groups = pattern.search(text)
        if groups is None:
            return None
        else:
            return groups.group(1)

    # Grab all hdf5 files in the input directory and assume the file names are formatted by the `simulate_regimes` command.
    input_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    # Generate all pairs of input files without repeats
    possible_tasks: list[tuple[str, str]] = list(itertools.combinations(input_files, 2))

    # Filter out tasks which are not useful comparison
    discriminant_pattern = re.compile("([0-9]+r[gt]*)")
    regime_pattern = re.compile("(m[di]i[di])_[0-9]+myr[0-9]+r")
    whol_pattern = re.compile("(whol)_[0-9]+myr[0-9]+r")
    tasks: list[AnimationComparisonJobTask] = []
    for first, second in possible_tasks:
        # Extract the discriminant (number of rings and settings) and regime (MIII, MDII, MDID)
        first_base_name = os.path.splitext(os.path.basename(first))[0]
        second_base_name = os.path.splitext(os.path.basename(second))[0]
        first_discriminant = get_first_group(first_base_name, discriminant_pattern)
        second_discriminant = get_first_group(second_base_name, discriminant_pattern)
        # Attempt to get the regime
        first_regime = get_first_group(first_base_name, regime_pattern)
        # Try to see if it is a whol file
        if first_regime is None:
            first_regime = get_first_group(first_base_name, whol_pattern)
        second_regime = get_first_group(second_base_name, regime_pattern)
        # Try to see if it is a whol file
        if second_regime is None:
            second_regime = get_first_group(second_base_name, whol_pattern)

        # Failed to match the file names
        if (
            first_discriminant is None
            or second_discriminant is None
            or first_regime is None
            or second_regime is None
        ):
            log.warning(f"First path: {first_base_name}")
            log.warning(f"Second path: {second_base_name}")
            log.warning("The file names are not formatted correctly! Skipping...")
            # Match did not succeed so the file names are not formatted correctly, skip.
            continue
        # Extract the number of rings
        first_num_rings, first_discriminant = first_discriminant.split("r")
        first_num_rings = int(first_num_rings)
        second_num_rings, second_discriminant = second_discriminant.split("r")
        second_num_rings = int(second_num_rings)
        # Add the r back to the discriminant
        first_discriminant = f"r{first_discriminant}"
        second_discriminant = f"r{second_discriminant}"

        # Conditions
        same_settings = first_discriminant == second_discriminant
        same_regime = first_regime == second_regime

        # Sanity check that we have not included repeats
        assert not (
            same_settings and same_regime and first_num_rings == second_num_rings
        )

        # Both regime and settings are different
        both_different = not same_settings and not same_regime
        # Settings differ more than once
        too_different = abs(len(first_discriminant) - len(second_discriminant)) > 1
        # Can't compare between planar with gravity and tilted with no gravity
        too_different |= {first_discriminant, second_discriminant} == {"rg", "rt"}
        # Filter out tasks which are too different to compare
        if both_different or too_different:
            continue

        # Swap first and second so that the regimes go from MIII to MDII to MDID
        first_regime = Regime.from_str(first_regime)
        assert first_regime is not None
        second_regime = Regime.from_str(second_regime)
        assert second_regime is not None

        # Swap if the first regime comes after the second or it has more settings active
        if first_regime > second_regime or len(first_discriminant) > len(
            second_discriminant
        ):
            (
                (first, first_regime, first_discriminant, first_num_rings),
                (
                    second,
                    second_regime,
                    second_discriminant,
                    second_num_rings,
                ),
            ) = (
                (second, second_regime, second_discriminant, second_num_rings),
                (first, first_regime, first_discriminant, first_num_rings),
            )

        # Case 1: Same settings but different regimes
        if not same_settings and same_regime:
            out_base = f"{first_regime}_{first_num_rings}{first_discriminant}_vs_{second_num_rings}{second_discriminant}"
        # Case 2: Same regime but different settings
        else:
            # Sanity check: We should have already filtered out the other cases
            assert same_settings and not same_regime
            out_base = f"{first_discriminant}_{first_num_rings}{first_regime}_vs_{second_num_rings}{second_regime}"

        # Add the trace tag if trace is True
        out_base += "_trace" if trace else ""
        output_path = os.path.join(output_dir, f"{out_base}.mp4")
        # Don't overwrite existing files unless
        if not force and os.path.exists(output_path):
            continue
        tasks.append(AnimationComparisonJobTask(first, second, output_path))
    return tasks


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
