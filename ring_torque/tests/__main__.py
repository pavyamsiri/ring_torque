# Standard libraries
import argparse
import os


# Internal libraries
from ring_torque.common import setup_logging

# Test functions
from .even_distribution import test_even_distribution
from .single_ring_potential import test_single_ring_potential
from .single_ring_force import test_single_ring_force
from .single_ring_torque import test_single_ring_torque
from .multiple_ring_potential import test_multiple_ring_potential
from .multiple_ring_force import test_multiple_ring_force
from .multiple_ring_torque import test_multiple_ring_torque
from .two_ring_torque import test_two_rings_torque
from .two_ring_torque_response import test_two_ring_torque_response


def main(args: list[str]):
    """Run the tests.

    Parameters
    ----------
    args : list[str]
        The command line arguments.
    """
    parsed_args = _parse_args(args)
    test_dir: str = parsed_args.test_dir
    softening_length: float = parsed_args.softening_length
    force: bool = parsed_args.force

    # Check that the test directory exists
    if not os.path.isdir(test_dir):
        raise ValueError(f"Test directory {test_dir} does not exist.")

    # Set up logging
    _ = setup_logging()

    # Setup test parameters
    NUM_ECCENTRICITIES: int = 10
    NUM_POINTS: int = 128
    NUM_RINGS: int = 5
    RING_SPACING: float = 0.2
    # Test sample point uniformity
    test_even_distribution(
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        os.path.join(test_dir, "test_sample_point_uniformity.mp4"),
        force=force,
    )
    # Test single ring potential
    test_single_ring_potential(
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_single_ring_potential.mp4"),
        force=force,
    )
    # Test multiple ring potential
    test_multiple_ring_potential(
        NUM_RINGS,
        RING_SPACING,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_multiple_ring_potential.mp4"),
        force=force,
    )
    # Test single ring force
    test_single_ring_force(
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_single_ring_force.mp4"),
        force=force,
    )
    # Test multiple ring force
    test_multiple_ring_force(
        NUM_RINGS,
        RING_SPACING,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_multiple_ring_force.mp4"),
        force=force,
    )
    # Test single ring torque
    test_single_ring_torque(
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_single_ring_torque.mp4"),
        force=force,
    )
    # Test multiple ring torque
    test_multiple_ring_torque(
        NUM_RINGS,
        RING_SPACING,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_multiple_ring_torque.mp4"),
        force=force,
    )
    # Test two non-crossing rings torque
    NONCROSSING_RATIO: float = 1.1
    test_two_rings_torque(
        NONCROSSING_RATIO,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_two_noncrossing_rings_torque.mp4"),
        force=force,
    )
    # Test two crossing rings torque
    CROSSING_RATIO: float = 0.9
    test_two_rings_torque(
        CROSSING_RATIO,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_two_crossing_rings_torque.mp4"),
        force=force,
    )
    # Test two non-crossing rings torque response
    test_two_ring_torque_response(
        NONCROSSING_RATIO,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_two_noncrossing_rings_torque_response.mp4"),
        force=force,
    )
    # Test two crossing rings torque response
    test_two_ring_torque_response(
        CROSSING_RATIO,
        NUM_ECCENTRICITIES,
        NUM_POINTS,
        softening_length,
        os.path.join(test_dir, "test_two_crossing_rings_torque_response.mp4"),
        force=force,
    )


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse the command line arguments.

    Parameters
    ----------
    args : list[str]
        The command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test the ring_torque package.")
    parser.add_argument(
        "test_dir", type=str, help="The directory to save the test results to."
    )
    parser.add_argument(
        "-softening_length",
        "--softening_length",
        type=float,
        dest="softening_length",
        default=0.0,
        help="The softening length to use",
    )
    parser.add_argument(
        "-f",
        "--f",
        "-force",
        "--force",
        action="store_true",
        dest="force",
        help="Don't skip any tests.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
