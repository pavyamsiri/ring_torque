"""
Represents a system of concentric elliptic rings that aims to model a galactic disk.
"""

# Standard libraries
from dataclasses import dataclass
import logging
import os
import tomllib

# External libraries
import numpy as np

# Internal libraries
from ring_torque.units import (
    convert_mass_to_simulation_units,
    convert_velocity_to_simulation_units,
)


log = logging.getLogger(__name__)


@dataclass
class RingModelConfig:
    """A dataclass to configure a RingSystem.

    Attributes
    ----------
    tangential_velocity : float
        The tangential velocity of the rings in Rd/Gyr.
    scale_length : float
        The scale length of the disk in kpc.
    scale_mass : float
        The scale mass of the disk in simulation units.
    """

    tangential_velocity: float
    scale_length: float
    scale_mass: float


def parse_config(config_path: str) -> RingModelConfig:
    """Parse the given config file into a `RingModelConfig`.

    Parameters
    ----------
    config_path : str
        The path to the config file.

    Returns
    -------
    RingModelConfig
        The parsed config.
    """
    with open(config_path, "rb") as config_file:
        config = tomllib.load(config_file)
        tangential_velocity: float = config["tangential_velocity"]
        scale_length: float = config["scale_length"]
        scale_mass: float = config["scale_mass"]

        # Convert tangential velocity from km/s to Rd/Gyr
        tangential_velocity = convert_velocity_to_simulation_units(
            tangential_velocity, scale_length
        )
        # Precompute pattern speed Omega - kappa/2
        tangential_velocity = (1 - np.sqrt(2) / 2) * tangential_velocity
        # Convert scale mass from giga solar masses to simulation units
        scale_mass = convert_mass_to_simulation_units(scale_mass * 1e9, scale_length)
        return RingModelConfig(
            tangential_velocity,
            scale_length,
            scale_mass,
        )


def validate_config_path(config_path: str) -> bool:
    """Checks that the given file path exists and has the correct file extension.

    Parameters
    ----------
    config_path : str
        The path to the config file.

    Returns
    -------
    bool
        `True` if the file path exists and has the correct file extension, `False` otherwise.
    """
    result = os.path.isfile(config_path)
    result &= os.path.splitext(config_path)[1] == ".toml"
    return result
