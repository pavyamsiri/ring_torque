"""Common definitions for the ring torque package."""

# Standard libraries
import logging

# External libraries
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

NumpyFloatType: type = np.float64
"""The type of floating point numbers to use for calculations."""
EPSILON: float = np.sqrt(np.finfo(NumpyFloatType).eps)
"""The epsilon value to use for finite difference approximations."""
PLOT_PADDING_FACTOR: float = 1.1
"""The padding factor to use when plotting."""

CURRENT_MAJOR_VERSION = 1
"""The current major version of the hdf5 file format."""
CURRENT_MINOR_VERSION = 0
"""The current minor version of the hdf5 file format."""
CURRENT_PATCH_VERSION = 0
"""The current patch version of the hdf5 file format."""


def setup_logging() -> Console:
    """Setup rich logging to the console and optionally to a file.

    Returns
    -------
    Console
        the console object to use for rich printing.
    """
    FORMAT = "%(message)s"
    console = Console()
    console_handler = RichHandler(console=console, show_time=False)
    handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
        encoding="utf-8",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    return console


def format_num_bytes(num_bytes: int) -> str:
    """Format the given number of bytes in a human readable way.

    Parameters
    ----------
    num_bytes : int
        the number of bytes to format.

    Returns
    -------
    str
        the formatted number of bytes.
    """
    if num_bytes < 1000:
        return f"{num_bytes} B"
    elif num_bytes < 1000**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1000**3:
        return f"{num_bytes / 1024**2:.2f} MB"
    elif num_bytes < 1000**4:
        return f"{num_bytes / 1024**3:.2f} GB"
    else:
        return f"{num_bytes / 1024**4:.2f} TB"


def format_execution_time_s(
    elapsed_time_s: float, num_iterations: int
) -> tuple[str, str]:
    """Formats elapsed time in seconds into a more human friendly form.

    Parameters
    ----------
    elapsed_time_s : float
        The number of elapsed seconds.
    num_iterations : int
        The number of iterations executed.

    Returns
    -------
    elapsed_time_str : str
        The formatted elapsed time in hours, minutes and seconds.
    iteration_rate_str : str
        The formatted iteration rate either in iterations/s or s/iteration, whichever
        is larger than 1.
    """
    elapsed_minutes, elapsed_seconds = divmod(elapsed_time_s, 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    if elapsed_time_s > num_iterations:
        iteration_rate = elapsed_time_s / num_iterations
        rate_unit = "s/iteration"
    else:
        iteration_rate = num_iterations / elapsed_time_s
        rate_unit = "iterations/s"
    iteration_rate_str = f"{iteration_rate:.2f} {rate_unit}"
    elapsed_time_str = (
        f"{int(elapsed_hours)}:{int(elapsed_minutes):02}:{round(elapsed_seconds):02}"
    )
    return (elapsed_time_str, iteration_rate_str)
