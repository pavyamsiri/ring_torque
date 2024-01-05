"""This module contains the simulate_regimes command for the ring_torque CLI."""
from __future__ import annotations

# Standard libraries
import argparse
from dataclasses import dataclass
from enum import Enum
from functools import partial
import itertools
import logging
import os
from typing import Sequence

# Internal libraries
from ring_torque.cli.file_helpers import (
    check_file_and_directory,
)
from ring_torque.cli.parallel import process_tasks_with_pool
from ring_torque.cli.parser import _parse_simulate_regimes
from ring_torque.common import setup_logging
from ring_torque.models.model import RingModelConfig, parse_config
from ring_torque.radii import calculate_uniform_radii_to_max_radius
from ring_torque.system import RingSystem

log = logging.getLogger(__name__)


# The minimum and maximum radius of the disk in units of disk scale length
MINIMUM_DISK_RADIUS: float = 1 / 6
MAXIMUM_DISK_RADIUS: float = 5


class Regime(Enum):
    """An enum representing the different regimes of the ring system. The regimes are:
    - MIII: Rings are confined to the region R < Rd.
        - Mass is increasing with radius.
        - The moments of inertia for all three axes is increasing with radius.
    - MDII: Rings are confined to the region Rd < R < 3Rd.
        - Mass is decreasing with radius.
        - The moments of inertia for all three axes is increasing with radius.
    - MDID: Rings are confined to the region 3Rd < R.
        - Mass is decreasing with radius.
        - The moments of inertia for all three axes is decreasing with radius.
    - WHOL: The entire disk is simulated.
    """

    MIII = 0
    MDII = 1
    MDID = 2
    WHOL = 3

    @staticmethod
    def from_str(name: str) -> Regime | None:
        """Get the regime from the given name (case insensitive).

        Parameters
        ----------
        name : str
            The name of the regime.

        Returns
        -------
        regime : Regime | None
            The regime corresponding to the given name or None if no match was found.
        """
        match name.lower():
            case "miii":
                return Regime.MIII
            case "mdii":
                return Regime.MDII
            case "mdid":
                return Regime.MDID
            case "whol":
                return Regime.WHOL
            case _:
                return None

    def get_minimum_radius(self) -> float:
        """Get the minimum radius for the given regime.

        Returns
        -------
        minimum_radius : float
            The minimum radius for the given regime.
        """
        match self:
            case Regime.MIII:
                return MINIMUM_DISK_RADIUS
            case Regime.MDII:
                return 1
            case Regime.MDID:
                return 3
            case Regime.WHOL:
                return MINIMUM_DISK_RADIUS

    def get_maximum_radius(self) -> float:
        """Get the maximum radius for the given regime.

        Returns
        -------
        maximum_radius : float
            The maximum radius for the given regime.
        """
        match self:
            case Regime.MIII:
                return 1
            case Regime.MDII:
                return 3
            case Regime.MDID:
                return MAXIMUM_DISK_RADIUS
            case Regime.WHOL:
                return MAXIMUM_DISK_RADIUS

    def get_range_str(self) -> str:
        """Get a string that describes the range of radii for the given regime.

        Returns
        -------
        range_str : str
            A string that describes the range of radii for the given regime.
        """
        match self:
            case Regime.MIII:
                return f"{MINIMUM_DISK_RADIUS:0.0f}Rd < R < Rd"
            case Regime.MDII:
                return "Rd < R < 3Rd"
            case Regime.MDID:
                return f"3Rd < R < {MAXIMUM_DISK_RADIUS:0.0f}Rd"
            case Regime.WHOL:
                return f"{MINIMUM_DISK_RADIUS:0.0f}R < R < {MAXIMUM_DISK_RADIUS:0.0f}Rd"

    def __lt__(self, other: Regime) -> bool:
        return self.value < other.value

    def __str__(self) -> str:
        match self:
            case Regime.MIII:
                return "miii"
            case Regime.MDII:
                return "mdii"
            case Regime.MDID:
                return "mdid"
            case Regime.WHOL:
                return "whol"


@dataclass
class SimulateRegimeParameters:
    """Contains simulation parameters.

    Attributes
    ----------
    eccentricity : float
        The eccentricity of the rings.
    bend_amplitude : float
        The amplitude of the bending waves.
    num_particles : int
        The number of particles to use when approximating the ring's shape.
    num_time_steps : int
        The number of time steps to simulate.
    time_step : float
        The size of each time step in Gyr.
    softening_length : float
        The softening length to use for gravity in units of the scale length.
    save_interval : int
        The interval between saving data to disk.
    """

    eccentricity: float
    bend_amplitude: float
    num_particles: int
    num_time_steps: int
    time_step: float
    softening_length: float
    save_interval: int


@dataclass
class SimulateRegimeJobTask:
    """A task to be completed by the pool for the `simulate_regimes` subcommand."""

    output_path: str
    regime: Regime
    use_gravity: bool
    with_tilt: bool
    model_config: RingModelConfig
    num_time_steps: int
    radii: list[float]
    eccentricity: float
    bend_amplitude: float


def main(args: list[str]) -> None:
    """Parse command line arguments for the `simulate_regimes` command.

    Parameters
    ----------
    args : list[str]
        The command line arguments to parse.
    """
    parsed_args = _parse_args(args)
    if check_file_and_directory(parsed_args.input_path, parsed_args.output_path):
        _simulate_regimes(parsed_args)
    else:
        raise ValueError("The input and output paths must both be directories.")


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the `simulate_regimes` command.

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
        prog="ring_torque simulate_regimes",
        description="Simulate a ring torque model for a range of regimes.",
    )
    _parse_simulate_regimes(parser)
    return parser.parse_args(args)


def _simulate_regimes(args: argparse.Namespace) -> None:
    """The entry point for the `simulate_regimes` subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed command line arguments.

    Notes
    -----
    The `simulate_regimes` subcommand is used to simulate rings in the three regimes, MIII, MDII and MDID. See `Regime` for more
    information.
    """
    # File paths
    input_path: str = args.input_path
    output_dir: str = args.output_path
    # Simulation parameters
    num_particles: int = args.num_particles
    ring_spacing: float = args.ring_spacing
    num_time_steps: int = args.num_time_steps
    time_step: int = args.time_step
    softening_length: float = args.softening_length
    eccentricity: float = args.eccentricity
    bend_amplitude: float = args.bend_amplitude
    # Serialisation parameters
    save_interval: int = args.save_interval
    # File handling parameters
    force: bool = args.force
    # Multiprocessing parameters
    num_processes: int = args.num_processes

    log.info(
        f"Simulating regimes from the model file at {input_path} and saving to {output_dir}."
    )

    parameters: SimulateRegimeParameters = SimulateRegimeParameters(
        eccentricity,
        bend_amplitude,
        num_particles,
        num_time_steps,
        time_step,
        softening_length,
        save_interval,
    )

    # Generating tasks
    tasks = __generate_simulate_regime_tasks(
        input_path,
        output_dir,
        ring_spacing,
        num_time_steps,
        time_step,
        eccentricity,
        bend_amplitude,
        force,
    )

    display = num_processes == 1
    process_tasks_with_pool(
        num_processes,
        partial(__simulate_regime, parameters, display=display),
        tasks,
        "Simulating regimes",
    )


def __simulate_regime(
    parameters: SimulateRegimeParameters,
    task: SimulateRegimeJobTask,
    display: bool = True,
) -> None:
    """Simulate the given regime.

    Parameters
    ----------
    parameters : SimulateRegimeParameters
        The simulation parameters.
    task : SimulateRegimeJobTask
        The task to be completed.
    display : bool, optional
        Whether to display the simulation, by default True
    """
    # Turn off logs
    if not display:
        logging.getLogger("ring_torque.models.model").setLevel(logging.WARNING)
        logging.getLogger("ring_torque.system").setLevel(logging.WARNING)
    # Extract data
    regime = task.regime
    use_gravity = task.use_gravity
    with_tilt = task.with_tilt
    radii = task.radii
    model_config = task.model_config
    output_file = task.output_path
    # Generate a name for the simulation
    regime_name = regime.get_range_str()
    gravity_display = "with" if use_gravity else "without"
    tilt_display = "random tilts" if with_tilt else "planar"
    simulation_name = f"{regime_name} {gravity_display} gravity ({tilt_display})"

    # Create rings and simulate
    rings = RingSystem(
        radii,
        parameters.eccentricity,
        model_config.scale_length,
        model_config.scale_mass,
        parameters.bend_amplitude,
    )
    rings.simulate_rings(
        output_file,
        parameters.num_particles,
        parameters.num_time_steps,
        parameters.time_step,
        model_config.tangential_velocity,
        parameters.softening_length,
        parameters.save_interval,
        use_gravity,
        with_tilt,
        simulation_name,
        display=display,
    )


def __generate_simulate_regime_tasks(
    model_path: str,
    output_dir: str,
    ring_spacing: float,
    num_time_steps: int,
    time_step: float,
    eccentricity: float,
    bend_amplitude: float,
    force: bool,
) -> Sequence[SimulateRegimeJobTask]:
    """Generate simulation tasks for the given output path. These tasks are for the `simulate_regimes` subcommand.

    Parameters
    ----------
    model_path : str
        The path to the model config file.
    output_dir : str
        The output directory.
    ring_spacing : float
        The spacing between rings.
    num_time_steps : int
        The number of time steps to simulate.
    time_step : float
        The time step to use.
    eccentricity : float
        The eccentricity of the orbits.
    bend_amplitude : float
        The amplitude of the bending mode.
    force : bool
        Whether to force the simulation to be generated even if the output file already exists.

    Returns
    -------
    tasks : Sequence[SimulateRegimeTask]
        A list of tasks to be completed by the pool.

    Raises
    ------
    ValueError
        If the given spacing is too large causing a regime to have no rings if it were constrained to the given spacing.
    """
    # Parse the model
    model_config = parse_config(model_path)

    # Generate all possible tasks
    regimes = list(Regime)
    gravity_options = [True, False]
    tilt_options = [True, False]
    task_parameters: list[tuple[Regime, bool, bool]] = list(
        itertools.product(regimes, gravity_options, tilt_options)
    )

    # Filter out tasks where the output file already exists
    tasks: list[SimulateRegimeJobTask] = []
    for regime, use_gravity, with_tilt in task_parameters:
        # Get the radius range for the current regime
        base_radius = regime.get_minimum_radius()
        max_radius = regime.get_maximum_radius()
        # Calculate the number of rings given the spacing
        num_rings: int = int((max_radius - base_radius) / ring_spacing)
        # The number of rings must be greater than 1 for all regimes
        if num_rings == 0:
            error_message = f"The given spacing {ring_spacing} > {(max_radius - base_radius):.3f} the smallest allowed value!"
            log.error(error_message)
            raise ValueError(error_message)
        # Generate the output file name
        num_myr = int(num_time_steps * time_step * 1000)
        output_path = f"{regime}_{num_myr}myr{num_rings}r"
        # Add the tags
        output_path += "g" if use_gravity else ""
        output_path += "t" if with_tilt else ""
        # Join the output directory and the output file name
        output_path = os.path.join(output_dir, f"{output_path}.hdf5")
        # Check that the output file does not already exist
        if not force and os.path.exists(output_path):
            continue
        radii = calculate_uniform_radii_to_max_radius(
            num_rings, base_radius, max_radius
        )
        tasks.append(
            SimulateRegimeJobTask(
                output_path,
                regime,
                use_gravity,
                with_tilt,
                model_config,
                num_time_steps,
                radii,
                eccentricity,
                bend_amplitude,
            )
        )

    return tasks


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
