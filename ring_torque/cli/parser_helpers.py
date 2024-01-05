"""
Contains helpful utilities to be use to generate parsers for the `ring_dynamics` command line interface.
"""

# Standard libraries
import argparse

# Internal libraries
from ring_torque.animation import RingAnimationTraceOptions
from ring_torque.animation.panel import RingAnimationPanelKind
from ring_torque.system import NumericalScheme, SofteningKernel


def ensure_positive_integer(value) -> int:
    try:
        integer = int(value)
        if integer > 0:
            return integer
        else:
            raise argparse.ArgumentTypeError(f"{value} must be a positive integer!")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")


def ensure_positive_float(value) -> float:
    try:
        number = float(value)
        if number > 0:
            return number
        else:
            raise argparse.ArgumentTypeError(f"{value} must be a positive float!")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")


def _add_radii_generation_options(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify how the radii of the rings should be generated.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    radii_options = parser.add_mutually_exclusive_group(required=True)
    radii_options.add_argument(
        "-range",
        "--range",
        nargs=2,
        type=float,
        dest="radius_range",
        help="Use uniformly distributed rings with respect to radius, that lie within [`base_radius` and `max_radius`]",
    )


def _add_simulation_parameters(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify simulation flags.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    DEFAULT_ECCENTRICITY: float = 0.5
    parser.add_argument(
        "-eccentricity",
        "--eccentricity",
        type=float,
        dest="eccentricity",
        default=DEFAULT_ECCENTRICITY,
        help=f"The eccentricity of the rings. Defaults to {DEFAULT_ECCENTRICITY}.",
    )
    parser.add_argument(
        "-softening_length",
        "--softening_length",
        type=float,
        dest="softening_length",
        default=0.0,
        help="The softening length to be used. Defaults to 0 or no softening.",
    )
    parser.add_argument(
        "-bend_amplitude",
        "--bend_amplitude",
        type=float,
        dest="bend_amplitude",
        default=0.0,
        help="The amplitude of the bending mode. Defaults to 0 or no bending.",
    )


def _add_simulation_flags(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify simulation flags.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "-g",
        "--g",
        "-gravity",
        "--gravity",
        action="store_true",
        dest="gravity",
        help="Turn on gravity between rings.",
    )
    parser.add_argument(
        "-t",
        "--t",
        "-tilt",
        "--tilt",
        action="store_true",
        dest="with_tilt",
        help="Perturb the ring's tilt slightly.",
    )
    parser.add_argument(
        "-scheme",
        "--scheme",
        type=NumericalScheme,
        choices=tuple(NumericalScheme),
        default=NumericalScheme.KICK_DRIFT_KICK,
        dest="numerical_scheme",
        help="Choose the numerical scheme to evolve the rings with.",
    )
    parser.add_argument(
        "-kernel",
        "--kernel",
        type=SofteningKernel,
        choices=tuple(SofteningKernel),
        default=SofteningKernel.NONE,
        dest="kernel_type",
        help="Choose the softening kernel.",
    )


def _add_time_discretisation(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify the discretisation of the simulation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "num_time_steps", type=int, help="Number of time steps to simulate."
    )
    parser.add_argument("time_step", type=float, help="The size of each time step.")


def _add_ring_discretisation(parser: argparse.ArgumentParser, required: bool):
    """Adds arguments to the parser that allows the user to specify the discretisation of the rings.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument(
        "-num_particles",
        "-num_points",
        "--num_particles",
        "--num_points",
        dest="num_particles",
        type=ensure_positive_integer,
        help="The number of particles to discretise each ring into",
    )
    parser.add_argument(
        "-arc_length",
        "--arc_length",
        type=ensure_positive_float,
        dest="arc_length",
        help="The desired arc length between points in units of the scale length.",
        default=0.1,
    )


def _add_animation_parameters(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify the parameters of the animation.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "-max_time_step",
        "--max_time_step",
        type=int,
        dest="max_time_step",
        help="The maximum time step to plot up to.",
        default=None,
    )
    parser.add_argument(
        "-fps", "--fps", type=int, default=30, help="Frames per second of output video."
    )
    parser.add_argument(
        "-remove_axes",
        "--remove_axes",
        action="store_true",
        dest="remove_axes",
        help="Removes plot axes.",
    )
    parser.add_argument(
        "-show_points",
        "--show_points",
        action="store_true",
        dest="show_points",
        help="Shows ellipse sample points.",
    )
    parser.add_argument(
        "-as_kpc",
        "--as_kpc",
        action="store_true",
        dest="as_kpc",
        help="Use kpc as units of length instead of disk scale lengths.",
    )
    parser.add_argument(
        "-skip_interpolation",
        "--skip_interpolation",
        action="store_true",
        dest="skip_interpolation",
        help="Skips grid interpolation. This is to increase animation speed.",
    )
    parser.add_argument(
        "-trace",
        "--trace",
        type=RingAnimationTraceOptions,
        choices=list(RingAnimationTraceOptions),
        default=RingAnimationTraceOptions.NO_TRACE,
        dest="trace",
        help="Traces the density wave.",
    )


def _add_single_animation_parameters(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-layout",
        "--layout",
        type=RingAnimationPanelKind,
        choices=list(RingAnimationPanelKind),
        default=[RingAnimationPanelKind.XY],
        nargs="*",
        dest="panel_specs",
        help="The panels to plot.",
    )


def _add_file_options(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify the file options.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "-force",
        "--force",
        action="store_true",
        dest="force",
        help="Overwrites existing files.",
    )


def _add_serialisation_parameters(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify the serialisation parameters.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "-save_interval",
        "--save_interval",
        type=int,
        default=1,
        help="Frames between each data snapshot.",
    )


def _add_multiprocessing_parameters(parser: argparse.ArgumentParser):
    """Adds arguments to the parser that allows the user to specify the multiprocessing parameters.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "-num_processes",
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use when simulating.",
    )
