"""This module contains common utilities and constants for the tests module"""
# Standard libraries
from typing import Sequence

# External libraries
from matplotlib.artist import Artist


FRAME_RATE: int = 30
NUM_SECONDS_PER_TEST: int = 3
NUM_FRAMES_PER_TEST: int = int(FRAME_RATE * NUM_SECONDS_PER_TEST)
FORCE_GRID_POINTS: int = 48


def init_plot() -> Sequence[Artist]:
    """Initialize the plot. This is a dummy function that does nothing.

    Returns
    -------
    Sequence[Artist]
        the artists to update.
    """
    return []
