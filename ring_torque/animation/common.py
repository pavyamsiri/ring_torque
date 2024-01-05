"""This module contains common functions and classes used by the ring animation modules."""

# Standard libraries
from dataclasses import dataclass
import logging
from typing import cast, Sequence

# External libraries
import h5py
from matplotlib.artist import Artist
import numpy as np
from numpy import typing as npt

# Internal libraries
from ring_torque.common import (
    CURRENT_MAJOR_VERSION,
    CURRENT_MINOR_VERSION,
    NumpyFloatType,
)
from ring_torque.ring import RingDiscretisation, RingDiscretisationType
from ring_torque.system import RingSystem


log = logging.getLogger(__name__)


@dataclass
class RingAnimationData:
    """Data class for storing parameters required for to create ring animations.

    Attributes:
    -----------
    name : str
        Name of the animation.
    orientations : npt.NDArray[np.floating]
        Array of ring orientations stored as quaternions.
    angular_velocities : npt.NDArray[np.floating]
        Array of ring angular velocities.
    tangential_velocity : float
        Tangential velocity of the ring.
    save_interval : int
        Interval at which to save snapshots of the animation.
    time_step : float
        Time step for the simulation.
    num_time_steps : int
        Number of time steps for the simulation.
    num_rings : int
        Number of rings in the system.
    num_snapshots : int
        Number of snapshots to save.
    with_gravity : bool
        Whether to include gravity in the simulation.
    max_plot_extent : float
        Maximum extent of the plot.
    ring_system : RingSystem
        Ring system object.
    """

    name: str
    orientations: npt.NDArray[np.floating]
    angular_velocities: npt.NDArray[np.floating]
    tangential_velocity: float
    save_interval: int
    time_step: float
    num_time_steps: int
    num_rings: int
    num_snapshots: int
    with_gravity: bool
    max_plot_extent: float
    ring_system: RingSystem

    def log_info(self) -> None:
        log.info(f"Number of rings: {self.num_rings}")
        log.info(f"Number of snapshots: {self.num_snapshots}")
        log.info(f"Save interval: {self.save_interval}")
        log.info(f"Time step: {self.time_step}")
        log.info(f"Shape: {self.orientations.shape}")


def read_hdf5_file(file_name: str) -> RingAnimationData:
    """Reads ring animation data from an HDF5 file. This will raise a KeyError if the file is invalid.

    Parameters
    ----------
    file_name : str
        The path to the HDF5 file.

    Returns
    -------
    RingAnimationData
        An object containing the ring animation data.
    """
    try:
        return _read_hdf5_file_unchecked(file_name)
    except KeyError as e:
        # Extract the name of the missing attribute
        start_idx = str(e).find(": '") + 3
        attribute_name = str(e)[start_idx:-3]
        error_message = f"Invalid HDF5 file: missing attribute: {attribute_name}. Please re-run the simulation."
        log.error(error_message)
        raise e


def _read_hdf5_file_unchecked(file_name: str) -> RingAnimationData:
    """Reads ring animation data from an HDF5 file. This is an unchecked operation that assumes the file is valid.

    Parameters
    ----------
    file_name : str
        The path to the HDF5 file.

    Returns
    -------
    RingAnimationData
        An object containing the ring animation data.
    """
    with h5py.File(file_name, "r") as input_file:
        # Read orientations (quaternions) into array
        orientations_data = input_file["orientations"]
        assert isinstance(orientations_data, h5py.Dataset)
        orientations: npt.NDArray[np.floating] = np.zeros(
            orientations_data.shape, dtype=NumpyFloatType
        )
        orientations_data.read_direct(orientations)
        # Read angular velocities into array
        angular_velocities_data = input_file["angular_velocity"]
        assert isinstance(angular_velocities_data, h5py.Dataset)
        angular_velocities: npt.NDArray[np.floating] = np.zeros(
            angular_velocities_data.shape, dtype=NumpyFloatType
        )
        angular_velocities_data.read_direct(angular_velocities)

        # Check the version
        major_version: int = cast(int, input_file.attrs["major_version"])
        minor_version: int = cast(int, input_file.attrs["minor_version"])
        patch_version: int = cast(int, input_file.attrs["patch_version"])
        if (
            major_version != CURRENT_MAJOR_VERSION
            and minor_version != CURRENT_MINOR_VERSION
            and patch_version
        ):
            raise ValueError(
                f"Invalid version: {major_version}.{minor_version}.{patch_version}."
            )
        else:
            log.info(
                f"Loaded Data Version: {major_version}.{minor_version}.{patch_version}"
            )

        # General/orientations attributes
        save_interval = cast(int, input_file.attrs["save_interval"])
        time_step = cast(float, input_file.attrs["time_step"])
        num_rings: int = orientations_data.shape[1]
        num_snapshots: int = orientations_data.shape[2]
        with_gravity: bool = cast(bool, input_file.attrs["with_gravity"])
        max_plot_extent: float = cast(float, input_file.attrs["max_plot_extent"])
        radii: list[float] = [
            float(radius) for radius in cast(np.ndarray, input_file.attrs["radii"])
        ]
        eccentricity: float = cast(float, input_file.attrs["eccentricity"])
        scale_length: float = cast(float, input_file.attrs["scale_length"])
        scale_mass: float = cast(float, input_file.attrs["scale_mass"])
        bend_amplitude: float = cast(float, input_file.attrs["bend_amplitude"])
        simulation_name = cast(str, input_file.attrs["name"])
        # Pre-compute the number of time steps
        num_time_step: int = (num_snapshots - 1) * save_interval + 1

        # Angular velocities attributes
        tangential_velocity: float = cast(
            float, input_file.attrs["tangential_velocity"]
        )
        # Create default discretisation
        discretisation: RingDiscretisation = RingDiscretisation(
            64, RingDiscretisationType.UNIFORM_NUM_PARTICLES
        )
        # Create the ring system
        ring_system = RingSystem(
            radii,
            eccentricity,
            scale_length,
            scale_mass,
            bend_amplitude,
            discretisation,
        )
        return RingAnimationData(
            name=simulation_name,
            orientations=orientations,
            angular_velocities=angular_velocities,
            tangential_velocity=tangential_velocity,
            save_interval=save_interval,
            time_step=time_step,
            num_time_steps=num_time_step,
            num_rings=num_rings,
            num_snapshots=num_snapshots,
            with_gravity=with_gravity,
            max_plot_extent=max_plot_extent,
            ring_system=ring_system,
        )


def calculate_grid_layout(num_cells: int) -> tuple[int, int]:
    """Calculates the grid layout for a given number of cells.

    Parameters
    ----------
    num_cells : int
        The number of cells.

    Returns
    -------
    num_rows : int
        The number of rows.
    num_cols : int
        The number of columns.
    """
    if num_cells < 4:
        # Special case for less than 4 cells
        return 1, num_cells
    sqrt = np.sqrt(num_cells)
    if sqrt.is_integer():
        # The number of cells is a perfect square
        num_rows = num_cols = int(sqrt)
    else:
        # The number of cells is not a perfect square
        num_rows = int(np.floor(sqrt))
        num_cols = int(np.ceil(sqrt))
        if num_rows * num_cols < num_cells:
            # The number of cells is not a perfect square and the number of rows and columns
            # is not enough to fit all the cells. We need to increase the number of rows.
            num_rows += 1
    return (num_rows, num_cols)


def init_plot_identity() -> Sequence[Artist]:
    """Initialises the plot for FuncAnimation. This function does nothing.

    Returns
    -------
    Sequence[Artist]
        A list of artists to be redrawn.
    """
    return []
