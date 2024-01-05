"""This module contains the simulate command for the ring_torque CLI."""

# Standard libraries
import argparse
from dataclasses import dataclass
from functools import partial
import glob
import logging
import os
from typing import Sequence

# External libraries

# Internal libraries
from .parser import _parse_simulate
from .file_helpers import (
    DATA_EXTENSIONS,
    check_both_directories,
    check_both_files,
    format_output_path,
)
from .system_helpers import parse_discretisation
from .parallel import process_tasks_with_pool
from ring_torque.common import setup_logging
from ring_torque.models.model import parse_config, validate_config_path
from ring_torque.radii import calculate_uniform_radii_to_max_radius
from ring_torque.ring import RingDiscretisation
from ring_torque.system import RingSystem, NumericalScheme, SofteningKernel


log = logging.getLogger(__name__)


@dataclass
class SimulationParameters:
    """Contains simulation parameters.

    Attributes
    ----------
    num_rings : int
        The number of rings to simulate.
    discretisation: RingDiscretisation
        The discretisation strategy to use. It can either be a uniform number of particles per ring
        or a uniform arc length between every point on a ring.
    num_time_steps : int
        The number of time steps to simulate.
    time_step : float
        The size of each time step in Gyr.
    eccentricity : float
        The eccentricity of the rings.
    kernel_type : SofteningKernel
        The softening kernel to use.
    softening_length : float
        The softening length to use for gravity in units of the scale length.
    bend_amplitude : float
        The amplitude of the bending waves.
    base_radius : float
        The base radius of the rings.
    max_radius : float
        The maximum radius of the rings.
    save_interval : int
        The interval between saving data to disk.
    use_gravity : bool
        Whether to use gravity in the simulation.
    with_tilt : bool
        Whether to perturb the ring's tilt slightly.
    numerical_scheme : NumericalScheme
        Which numerical scheme to use to evolve the system.
    """

    num_rings: int
    discretisation: RingDiscretisation
    num_time_steps: int
    time_step: float
    eccentricity: float
    kernel_type: SofteningKernel
    softening_length: float
    bend_amplitude: float
    base_radius: float
    max_radius: float
    save_interval: int
    use_gravity: bool
    with_tilt: bool
    numerical_scheme: NumericalScheme


@dataclass
class SimulationJobTask:
    """Contains the parameters for a simulation job.

    Attributes
    ----------
    input_path : str
        The path to the config file.
    output_path : str
        The path to the output file.
    simulation_name : str
        The name of the simulation.
    """

    input_path: str
    output_path: str
    simulation_name: str


def main(args: list[str]):
    """The `simulate` command."""
    parsed_args = _parse_args(args)
    if check_both_files(parsed_args.input_path, parsed_args.output_path):
        _simulate(parsed_args)
    elif check_both_directories(parsed_args.input_path, parsed_args.output_path):
        _batch_simulate(parsed_args)
    else:
        raise ValueError(
            f"Invalid input and output paths: {parsed_args.input_path}, {parsed_args.output_path}"
        )


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the `simulate` CLI command."""
    parser = argparse.ArgumentParser(
        description="Simulate a system of concentric elliptic rings for a given set of parameters."
    )
    _parse_simulate(parser)
    return parser.parse_args(args)


def _simulate(args: argparse.Namespace):
    """The `simulate` command. This function expects a single input file and a single output file.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed command line arguments.

    Raises
    ------
    ValueError
        If the given config file path is invalid.
    """
    # File paths
    input_path: str = args.input_path
    output_path: str = args.output_path
    # Simulation parameters
    num_rings: int = args.num_rings
    num_time_steps: int = args.num_time_steps
    time_step: float = args.time_step
    eccentricity: float = args.eccentricity
    bend_amplitude: float = args.bend_amplitude
    # Simulation flags
    use_gravity: bool = args.gravity
    with_tilt: bool = args.with_tilt
    # Serialisation parameters
    simulation_name: str = args.simulation_name
    save_interval: int = args.save_interval
    # Radii generation selection
    base_radius, max_radius = args.radius_range
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)
    # Numerical scheme
    numerical_scheme: NumericalScheme = args.numerical_scheme
    # Softening kernel
    kernel_type: SofteningKernel = args.kernel_type
    softening_length: float = args.softening_length

    # Generate simulation name if not given
    if simulation_name == "":
        simulation_name = os.path.splitext(os.path.basename(input_path))[0]
        # Add simulation flags to name
        if use_gravity:
            simulation_name += "_gravity"
        if with_tilt:
            simulation_name += "_tilt"

    parameters = SimulationParameters(
        num_rings,
        discretisation,
        num_time_steps,
        time_step,
        eccentricity,
        kernel_type,
        softening_length,
        bend_amplitude,
        base_radius,
        max_radius,
        save_interval,
        use_gravity,
        with_tilt,
        numerical_scheme,
    )
    __simulate_from_config_file(
        parameters, SimulationJobTask(input_path, output_path, simulation_name)
    )


def _batch_simulate(args: argparse.Namespace) -> None:
    """The `batch_simulate` command. This function expects a single input directory and a single output directory.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed command line arguments.
    """
    # File paths
    input_path: str = args.input_path
    output_path: str = args.output_path
    # Simulation parameters
    num_rings: int = args.num_rings
    num_time_steps: int = args.num_time_steps
    time_step: float = args.time_step
    eccentricity: float = args.eccentricity
    bend_amplitude: float = args.bend_amplitude
    # Simulation flags
    use_gravity: bool = args.gravity
    with_tilt: bool = args.with_tilt
    # Serialisation parameters
    simulation_name: str = args.simulation_name
    save_interval: int = args.save_interval
    # Radii generation selection
    base_radius, max_radius = args.radius_range
    # Number of workers
    num_processes: int = args.num_processes
    # Force flag
    force: bool = args.force
    # Ring discretisation
    arc_length: float | None = args.arc_length
    num_particles: int | None = args.num_particles
    discretisation: RingDiscretisation = parse_discretisation(num_particles, arc_length)
    # Numerical scheme
    numerical_scheme: NumericalScheme = args.numerical_scheme
    # Softening kernel
    kernel_type: SofteningKernel = args.kernel_type
    softening_length: float = args.softening_length

    parameters = SimulationParameters(
        num_rings,
        discretisation,
        num_time_steps,
        time_step,
        eccentricity,
        kernel_type,
        softening_length,
        bend_amplitude,
        base_radius,
        max_radius,
        save_interval,
        use_gravity,
        with_tilt,
        numerical_scheme,
    )

    # Glob the input directory
    config_template = os.path.join(input_path, "*.toml")
    config_files = glob.glob(config_template)
    tasks = __generate_simulation_tasks(
        config_files, output_path, simulation_name, force
    )

    log.info(f"Simulating model files from {input_path} and saving to {output_path}")

    display = num_processes == 1
    process_tasks_with_pool(
        num_processes,
        partial(__simulate_from_config_file, parameters, display=display),
        tasks,
        "Simulating rings",
    )


def __simulate_from_config_file(
    parameters: SimulationParameters, task: SimulationJobTask, display: bool = True
) -> None:
    """Simulate a system of concentric elliptic rings for a given set of parameters.

    Parameters
    ----------
    parameters : SimulationParameters
        The simulation parameters.
    task : SimulationJobTask
        The simulation job task. This contains the input and output paths and the simulation name.
    """
    # Turn off logs
    if not display:
        logging.getLogger("ring_torque.models.model").setLevel(logging.WARNING)
        logging.getLogger("ring_torque.system").setLevel(logging.WARNING)

    # Unpack task
    input_path: str = task.input_path
    output_path: str = task.output_path
    simulation_name: str = task.simulation_name
    # Format the output path and setup logging
    output_path = format_output_path(output_path, DATA_EXTENSIONS)
    _ = setup_logging()

    # Validate file paths
    if not validate_config_path(input_path):
        raise ValueError(f"Invalid config file path: {input_path}")

    # Parse config
    model_config = parse_config(input_path)
    # Generate radii
    radii = calculate_uniform_radii_to_max_radius(
        parameters.num_rings, parameters.base_radius, parameters.max_radius
    )

    # Create and simulate the rings
    rings = RingSystem(
        radii,
        parameters.eccentricity,
        model_config.scale_length,
        model_config.scale_mass,
        parameters.bend_amplitude,
        parameters.discretisation,
    )
    rings.simulate_rings(
        output_path,
        parameters.num_time_steps,
        parameters.time_step,
        model_config.tangential_velocity,
        parameters.kernel_type,
        parameters.softening_length,
        parameters.save_interval,
        parameters.use_gravity,
        parameters.with_tilt,
        parameters.numerical_scheme,
        simulation_name,
        display=display,
    )


def __generate_simulation_tasks(
    config_files: Sequence[str], output_path: str, simulation_name: str, force: bool
) -> Sequence[SimulationJobTask]:
    """Generate simulation tasks.

    Parameters
    ----------
    config_files : Sequence[str]
        The paths to the config files to generate tasks for.
    output_path : str
        The path to the output directory.
    simulation_name : str
        The name of the simulation.
    force : bool
        Whether to overwrite existing files.
    """
    tasks: list[SimulationJobTask] = []
    for config_file in config_files:
        base_name = os.path.splitext(os.path.basename(config_file))[0]
        # Format output path file name
        output_file = os.path.join(output_path, f"{base_name}{DATA_EXTENSIONS[0]}")
        if not force and os.path.exists(output_file):
            continue

        # Add model name to simulation name
        if simulation_name != "":
            name = f"{simulation_name}: {base_name}"
        else:
            name = base_name
        tasks.append(SimulationJobTask(config_file, output_file, name))
    return tasks


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
