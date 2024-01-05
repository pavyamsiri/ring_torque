"""
Contains parsers for the `ring_dynamics` command line interface.
"""
# Standard libraries
import argparse

# Internal libraries
from .parser_helpers import (
    _add_file_options,
    _add_serialisation_parameters,
    _add_simulation_flags,
    _add_simulation_parameters,
    _add_single_animation_parameters,
    _add_time_discretisation,
    _add_ring_discretisation,
    _add_radii_generation_options,
    _add_animation_parameters,
    _add_multiprocessing_parameters,
)


def _parse_commands(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `commands` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to commands file.")
    parser.add_argument("output_path", type=str, help="Path to output directory.")
    _add_multiprocessing_parameters(parser)
    _add_file_options(parser)


def _parse_simulate(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `simulate` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to config file.")
    parser.add_argument("output_path", type=str, help="Path to output file.")
    parser.add_argument("num_rings", type=int, help="Number of rings to simulate.")
    _add_time_discretisation(parser)
    parser.add_argument(
        "-simulation_name",
        "--simulation_name",
        type=str,
        default="",
        help="The name of the simulation.",
    )
    _add_radii_generation_options(parser)
    _add_simulation_flags(parser)
    _add_serialisation_parameters(parser)
    _add_ring_discretisation(parser, required=False)
    _add_simulation_parameters(parser)
    _add_multiprocessing_parameters(parser)
    _add_file_options(parser)


def _parse_animate(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `animate` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to input file/directory.")
    parser.add_argument("output_path", type=str, help="Path to output file/directory.")
    _add_ring_discretisation(parser, required=True)
    _add_animation_parameters(parser)
    _add_single_animation_parameters(parser)
    _add_file_options(parser)
    _add_multiprocessing_parameters(parser)


def _parse_animate_comparison(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `animate_comparison` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("first_path", type=str, help="Path to first input file.")
    parser.add_argument(
        "second_path",
        type=str,
        help="Path to second input file.",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to output file or directory."
    )
    _add_ring_discretisation(parser, required=True)
    _add_animation_parameters(parser)


def _parse_batch_animate_comparison(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `batch_animate_comparison` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to input directory.")
    parser.add_argument("output_path", type=str, help="Path to output directory.")
    _add_ring_discretisation(parser, required=True)
    _add_animation_parameters(parser)
    _add_file_options(parser)
    _add_multiprocessing_parameters(parser)


def _parse_simulate_regimes(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `simulate_regimes` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to input file.")
    parser.add_argument(
        "output_path", type=str, help="The directory to output the files to."
    )
    parser.add_argument(
        "ring_spacing",
        type=float,
        help="The radial spacing between the rings in each regime.",
    )
    _add_ring_discretisation(parser, required=False)
    _add_time_discretisation(parser)
    _add_serialisation_parameters(parser)
    _add_file_options(parser)
    _add_multiprocessing_parameters(parser)
    _add_simulation_parameters(parser)


def _parse_animate_grid(parser: argparse.ArgumentParser):
    """Parse command line arguments for the `animate_grid` subcommand.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument("input_path", type=str, help="Path to input file.")
    parser.add_argument("output_path", type=str, help="Path to output file.")
    _add_ring_discretisation(parser, required=True)
    _add_animation_parameters(parser)
    parser.add_argument(
        "-num_cells",
        "--num_cells",
        type=int,
        dest="num_cells",
        help="The number of cells to use in the grid.",
        default=128,
    )
    parser.add_argument(
        "-skip_interpolation",
        "--skip_interpolation",
        action="store_true",
        dest="skip_interpolation",
        help="Skip the interpolation step.",
    )
    _add_multiprocessing_parameters(parser)
    _add_file_options(parser)
