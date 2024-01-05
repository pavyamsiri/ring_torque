"""This module contains the commands command for the ring_torque CLI."""

from __future__ import annotations

# Standard libraries
import argparse
import copy
from dataclasses import dataclass
from functools import partial
import os
import logging
import tomllib
from typing import Sequence

# Internal libraries
from .parser import _parse_commands
from .parallel import process_tasks_with_pool
from .file_helpers import (
    DATA_EXTENSIONS,
    check_file_and_directory,
    format_output_path,
)
from .system_helpers import parse_discretisation
from ring_torque.common import setup_logging
from ring_torque.models.model import RingModelConfig, parse_config
from ring_torque.ring import RingDiscretisation, RingDiscretisationType
from ring_torque.radii import calculate_uniform_radii_to_max_radius
from ring_torque.system import RingSystem, NumericalScheme, SofteningKernel


log = logging.getLogger(__name__)


@dataclass
class SimulationJobTask:
    """Contains the parameters for a simulation job.

    Attributes
    ----------
    num_rings : int
        The number of rings.
    discretisation : RingDiscretisation
        The discretisation to use for each ring.
    num_time_steps : int
        The number of time steps to simulate for.
    time_step : float
        The time step in Gyr.
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
        The numerical scheme to use.
    model : RingModelConfig
        The galaxy model.
    output_path : str
        The path to the output file.
    simulation_name : str
        The name of the simulation.
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
    model: RingModelConfig
    output_path: str
    simulation_name: str


class SimulationJobTaskBuilder:
    """This class is used to partially construct a `SimulationJobTask`."""

    def __init__(self, model: RingModelConfig, output_dir: str):
        """Constructs a `SimulationJobTaskBuilder`.

        Parameters
        ----------
        model : RingModelConfig
            The galaxy model.
        output_dir : str
            The directory in which to output the file to.
        """
        # Default to None
        self._num_rings: int | None = None
        self._discretisation: RingDiscretisation | None = None
        self._num_time_steps: int | None = None
        self._time_step: float | None = None
        self._eccentricity: float | None = None
        self._kernel_type: SofteningKernel | None = None
        self._softening_length: float | None = None
        self._bend_amplitude: float | None = None
        self._base_radius: float | None = None
        self._max_radius: float | None = None
        self._save_interval: int | None = None
        self._use_gravity: bool | None = None
        self._with_tilt: bool | None = None
        self._numerical_scheme: NumericalScheme | None = None
        self._file_name: str | None = None
        self._name: str | None = None
        # Given parameters
        self._model: RingModelConfig = model
        self._output_dir: str = output_dir

    def num_rings(self, value: int | None) -> SimulationJobTaskBuilder:
        """Sets `num_rings` to `value` if it is not `None`.

        Parameters
        ----------
        value : int | None
            The value to set `num_rings` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._num_rings = value
        return self

    def discretisation(
        self, value: RingDiscretisation | None
    ) -> SimulationJobTaskBuilder:
        """Sets `discretisation` to `value` if it is not `None`.

        Parameters
        ----------
        value : RingDiscretisation | None
            The value to set `discretisation` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._discretisation = value
        return self

    def num_time_steps(self, value: int | None) -> SimulationJobTaskBuilder:
        """Sets `num_time_steps` to `value` if it is not `None`.

        Parameters
        ----------
        value : int | None
            The value to set `num_time_steps` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._num_time_steps = value
        return self

    def time_step(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `time_step` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set time_step` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._time_step = value
        return self

    def eccentricity(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `eccentricity` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set `eccentricity` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._eccentricity = value
        return self

    def kernel_type(self, value: SofteningKernel | None) -> SimulationJobTaskBuilder:
        """Sets `kernel_type` to `value` if it is not `None`.

        Parameters
        ----------
        value : SofteningKernel | None
            The value to set `kernel_type` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._kernel_type = value
        return self

    def softening_length(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `softening_length` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set `softening_length` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._softening_length = value
        return self

    def bend_amplitude(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `bend_amplitude` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set `bend_amplitude` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._bend_amplitude = value
        return self

    def base_radius(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `base_radius` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set `base_radius` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._base_radius = value
        return self

    def max_radius(self, value: float | None) -> SimulationJobTaskBuilder:
        """Sets `max_radius` to `value` if it is not `None`.

        Parameters
        ----------
        value : float | None
            The value to set `max_radius` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._max_radius = value
        return self

    def save_interval(self, value: int | None) -> SimulationJobTaskBuilder:
        """Sets `save_interval` to `value` if it is not `None`.

        Parameters
        ----------
        value : int | None
            The value to set `save_interval` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._save_interval = value
        return self

    def use_gravity(self, value: bool | None) -> SimulationJobTaskBuilder:
        """Sets `use_gravity` to `value` if it is not `None`.

        Parameters
        ----------
        value : bool | None
            The value to set `use_gravity` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._use_gravity = value
        return self

    def with_tilt(self, value: bool | None) -> SimulationJobTaskBuilder:
        """Sets `with_tilt` to `value` if it is not `None`.

        Parameters
        ----------
        value : bool | None
            The value to set with_tilt` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._with_tilt = value
        return self

    def numerical_scheme(
        self, value: NumericalScheme | None
    ) -> SimulationJobTaskBuilder:
        """Sets `with_tilt` to `value` if it is not `None`.

        Parameters
        ----------
        numerical_scheme : NumericalScheme | None
            The value to set `numerical_scheme` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._numerical_scheme = value
        return self

    def file_name(self, value: str | None) -> SimulationJobTaskBuilder:
        """Sets `file_name` to `value` if it is not `None`.

        Parameters
        ----------
        : str | None
            The value to set file_name` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._file_name = value
        return self

    def name(self, value: str | None) -> SimulationJobTaskBuilder:
        """Sets `name` to `value` if it is not `None`.

        Parameters
        ----------
        | None
            The value to set to ` to if it is not `None`.

        Returns
        -------
        SimulationJobTaskBuilder
            itself.
        """
        if value is not None:
            self._name = value
        return self

    def generate_file_name(self) -> str:
        """Generates a file name from the existing given parameters.

        Returns
        -------
        file_name : str
            The generated file name.
        """
        assert self._num_rings is not None, "Missing `num_rings`!"
        assert self._discretisation is not None, "Missing `discretisation`!"
        assert self._num_time_steps is not None, "Missing `num_time_steps`!"
        assert self._time_step is not None, "Missing `time_step`!"
        assert self._eccentricity is not None, "Missing `eccentricity`!"
        assert self._kernel_type is not None, "Missing `kernel_type`!"
        assert self._softening_length is not None, "Missing `softening_length`!"
        assert self._bend_amplitude is not None, "Missing `bend_amplitude`!"
        assert self._base_radius is not None, "Missing `base_radius`!"
        assert self._max_radius is not None, "Missing `max_radius`!"
        assert self._save_interval is not None, "Missing `save_interval`!"
        assert self._use_gravity is not None, "Missing `use_gravity`!"
        assert self._with_tilt is not None, "Missing `with_tilt`!"
        assert self._numerical_scheme is not None, "Missing `numerical_scheme`!"
        gravity_flag: str = "g" if self._use_gravity else ""
        eccentricity_string: str = f"ecc{str(self._eccentricity).replace('.', 'p')}"
        discretisation: RingDiscretisation = self._discretisation
        match discretisation.get_type():
            case RingDiscretisationType.UNIFORM_NUM_PARTICLES:
                discretisation_string = f"np{discretisation.get_num_particles()}"
            case RingDiscretisationType.UNIFORM_ARC_LENGTH:
                discretisation_string = (
                    f"al{str(discretisation.get_arc_length()).replace('.', 'p')}"
                )
            case _ as discretisation_type:
                raise ValueError(
                    f"Unexpected discretisation type {discretisation_type}"
                )
        total_time: float = 1000 * self._num_time_steps * self._time_step
        total_time_string: str = f"time{total_time:.0f}myr"
        scheme_string: str = f"ns{self._numerical_scheme.get_identifier()}"
        num_ring_string: str = f"{self._num_rings}r"
        file_identifiers: list[str] = []
        file_identifiers.append(gravity_flag + eccentricity_string)
        file_identifiers.append(discretisation_string)
        file_identifiers.append(total_time_string)
        file_identifiers.append(num_ring_string)
        file_identifiers.append(scheme_string)
        # Softening
        if self._kernel_type != SofteningKernel.NONE and self._softening_length > 0:
            kernel_type_string: str = f"sk{self._kernel_type.get_identifier()}"
            softening_length_string: str = (
                f"sl{str(self._softening_length).replace('.', 'p')}"
            )
            file_identifiers.append(kernel_type_string + softening_length_string)
        # Bend amplitude
        if self._bend_amplitude > 0:
            bend_amplitude_string: str = (
                f"ba{str(self._bend_amplitude).replace('.', 'p')}"
            )
            file_identifiers.append(bend_amplitude_string)

        file_name: str = f"{'_'.join(file_identifiers)}.hdf5"
        return file_name

    def generate_name(self) -> str:
        """Generates a name from the existing given parameters.

        Returns
        -------
        name: str
            The name.
        """
        assert self._num_rings is not None, "Missing `num_rings`!"
        assert self._discretisation is not None, "Missing `discretisation`!"
        assert self._num_time_steps is not None, "Missing `num_time_steps`!"
        assert self._time_step is not None, "Missing `time_step`!"
        assert self._eccentricity is not None, "Missing `eccentricity`!"
        assert self._kernel_type is not None, "Missing `kernel_type`!"
        assert self._softening_length is not None, "Missing `softening_length`!"
        assert self._bend_amplitude is not None, "Missing `bend_amplitude`!"
        assert self._base_radius is not None, "Missing `base_radius`!"
        assert self._max_radius is not None, "Missing `max_radius`!"
        assert self._save_interval is not None, "Missing `save_interval`!"
        assert self._use_gravity is not None, "Missing `use_gravity`!"
        assert self._with_tilt is not None, "Missing `with_tilt`!"
        assert self._numerical_scheme is not None, "Missing `numerical_scheme`!"
        gravity_flag: str = "g" if self._use_gravity else ""
        eccentricity_string: str = f"e = {self._eccentricity}"
        discretisation: RingDiscretisation = self._discretisation
        match discretisation.get_type():
            case RingDiscretisationType.UNIFORM_NUM_PARTICLES:
                discretisation_string = (
                    f"Np = {discretisation.get_num_particles()} particles"
                )
            case RingDiscretisationType.UNIFORM_ARC_LENGTH:
                discretisation_string = f"La = {discretisation.get_arc_length()} Rd"
            case _ as discretisation_type:
                raise ValueError(
                    f"Unexpected discretisation type {discretisation_type}"
                )
        name_identifiers: list[str] = []
        name_identifiers.append(eccentricity_string)
        name_identifiers.append(discretisation_string)
        # Softening
        if self._kernel_type != SofteningKernel.NONE and self._softening_length > 0:
            softening_length_string: str = f"Ls = {self._softening_length} Rd"
            name_identifiers.append(softening_length_string)
        # Bend amplitude
        if self._bend_amplitude > 0:
            bend_amplitude_string: str = f"Av = {self._bend_amplitude} Rd"
            name_identifiers.append(bend_amplitude_string)

        # Gravity
        if self._kernel_type != SofteningKernel.NONE and self._softening_length > 0:
            kernel_type_string: str = f" [{self._kernel_type.get_name()}] "
        else:
            kernel_type_string: str = " "
        if self._use_gravity:
            gravity_use_flag: str = "with"
        else:
            gravity_use_flag: str = "without"
        gravity_flag: str = f"({gravity_use_flag}{kernel_type_string}gravity)"

        name: str = f"{', '.join(name_identifiers)} {gravity_flag}"
        return name

    def build(self) -> SimulationJobTask:
        """Builds a SimulationJobTask.

        Returns
        -------
        SimulationJobTask
            The simulation job task.

        Raises
        ------
        ValueError
            If this method is called while one of its field is still `None`.
        """
        try:
            assert self._num_rings is not None, "Missing `num_rings`!"
            assert self._discretisation is not None, "Missing `discretisation`!"
            assert self._num_time_steps is not None, "Missing `num_time_steps`!"
            assert self._time_step is not None, "Missing `time_step`!"
            assert self._eccentricity is not None, "Missing `eccentricity`!"
            assert self._kernel_type is not None, "Missing `kernel_type`!"
            assert self._softening_length is not None, "Missing `softening_length`!"
            assert self._bend_amplitude is not None, "Missing `bend_amplitude`!"
            assert self._base_radius is not None, "Missing `base_radius`!"
            assert self._max_radius is not None, "Missing `max_radius`!"
            assert self._save_interval is not None, "Missing `save_interval`!"
            assert self._use_gravity is not None, "Missing `use_gravity`!"
            assert self._with_tilt is not None, "Missing `with_tilt`!"
            assert self._numerical_scheme is not None, "Missing `numerical_scheme`!"
        except AssertionError as err:
            raise ValueError from err
        # If no file name is given, generate it
        if self._file_name is None:
            file_name = self.generate_file_name()
        else:
            file_name = self._file_name

        # If no name is given, generate it
        if self._name is None:
            name = self.generate_name()
        else:
            name = self._name

        # Output to the output directory
        output_path = os.path.join(self._output_dir, file_name)
        return SimulationJobTask(
            num_rings=self._num_rings,
            discretisation=self._discretisation,
            num_time_steps=self._num_time_steps,
            time_step=self._time_step,
            eccentricity=self._eccentricity,
            kernel_type=self._kernel_type,
            softening_length=self._softening_length,
            bend_amplitude=self._bend_amplitude,
            base_radius=self._base_radius,
            max_radius=self._max_radius,
            save_interval=self._save_interval,
            use_gravity=self._use_gravity,
            with_tilt=self._with_tilt,
            numerical_scheme=self._numerical_scheme,
            model=self._model,
            output_path=output_path,
            simulation_name=name,
        )


def main(args: Sequence[str]) -> None:
    """The main function for the commands command.

    Parameters
    ----------
    args : Sequence[str]
        The command line arguments.
    """
    parsed_args = _parse_args(args)
    # The input path must be an existing file and the output must be a writable directory
    if check_file_and_directory(parsed_args.input_path, parsed_args.output_path):
        # Generate the tasks for each job
        tasks = __generate_tasks(parsed_args.input_path, parsed_args.output_path)
        # Turn off progress bars if multiprocessing is on
        display = parsed_args.num_processes == 1 and len(tasks) > 1
        # Schedule jobs with pool
        process_tasks_with_pool(
            parsed_args.num_processes,
            partial(__simulate, display=display),
            tasks,
            "Simulating rings",
        )
    else:
        log.error(
            "The input path must be a valid file and the output path must be a valid directory."
        )
        log.error(f"Input path: {parsed_args.input_path}")
        log.error(f"Output path: {parsed_args.output_path}")


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments for the `commands` CLI command."""
    parser = argparse.ArgumentParser(
        description="Reads a list of simulation parameters and runs all of them. Can be parallelised using multiprocessing."
    )
    _parse_commands(parser)
    return parser.parse_args(args)


def __simulate(
    task: SimulationJobTask,
    display: bool = True,
) -> None:
    """Simulate a system of concentric elliptic rings for a given set of parameters.

    Parameters
    ----------
    task : SimulationJobTask
        The simulation job task.
    """
    # Turn off logs
    if not display:
        logging.getLogger("ring_torque.models.model").setLevel(logging.WARNING)
        logging.getLogger("ring_torque.system").setLevel(logging.WARNING)

    # Unpack task
    output_path: str = task.output_path
    simulation_name: str = task.simulation_name
    # Format the output path and setup logging
    output_path = format_output_path(output_path, DATA_EXTENSIONS)
    _ = setup_logging()

    # Generate radii
    radii = calculate_uniform_radii_to_max_radius(
        task.num_rings, task.base_radius, task.max_radius
    )

    # Create and simulate the rings
    rings = RingSystem(
        radii,
        task.eccentricity,
        task.model.scale_length,
        task.model.scale_mass,
        task.bend_amplitude,
        task.discretisation,
    )
    rings.simulate_rings(
        output_path,
        task.num_time_steps,
        task.time_step,
        task.model.tangential_velocity,
        task.kernel_type,
        task.softening_length,
        task.save_interval,
        task.use_gravity,
        task.with_tilt,
        task.numerical_scheme,
        simulation_name,
        display=display,
    )


def __generate_tasks(input_path: str, output_path: str) -> Sequence[SimulationJobTask]:
    """Reads a `.toml` file to get a list of runs with different simulation parameters.
    Returns a sequence of `SimulationJobTask` containing all of the simulation parameters
    for each run.

    Parameters
    ----------
    input_path : str
        The path to the commands file.
    output_path : str
        The path to the output directory.

    Returns
    -------
    tasks : Sequence[SimulationJobTask]
        The generated tasks parsed from the commands toml.
    """
    with open(input_path, "rb") as input_file:
        config = tomllib.load(input_file)
        directory_path, _ = os.path.split(input_path)
        model_path = os.path.join(directory_path, config["model_path"])
        # Parse galaxy model
        model = parse_config(model_path)
        # Builder to hold default options
        default_builder = SimulationJobTaskBuilder(model, output_path)
        default_parameters = config["simulation"]
        default_builder.num_rings(default_parameters.get("num_rings"))
        default_builder.num_time_steps(default_parameters.get("num_time_steps"))
        default_builder.time_step(default_parameters.get("time_step"))
        default_builder.eccentricity(default_parameters.get("eccentricity"))
        default_kernel = default_parameters.get("kernel_type")
        if default_kernel is not None:
            default_builder.kernel_type(SofteningKernel(default_kernel))
        default_builder.softening_length(default_parameters.get("softening_length"))
        default_builder.bend_amplitude(default_parameters.get("bend_amplitude"))
        default_builder.base_radius(default_parameters.get("base_radius"))
        default_builder.max_radius(default_parameters.get("max_radius"))
        default_builder.save_interval(default_parameters.get("save_interval"))
        default_builder.use_gravity(default_parameters.get("use_gravity"))
        default_builder.with_tilt(default_parameters.get("with_tilt"))
        default_builder.file_name(default_parameters.get("file_name"))
        default_builder.name(default_parameters.get("name"))
        default_scheme = default_parameters.get("numerical_scheme")
        if default_scheme is not None:
            default_builder.numerical_scheme(NumericalScheme(default_scheme))
        default_num_particles: int | None = default_parameters.get("num_particles")
        default_arc_length: float | None = default_parameters.get("arc_length")
        default_discretisation: RingDiscretisation | None = (
            parse_discretisation(default_num_particles, default_arc_length)
            if (default_num_particles is not None or default_arc_length is not None)
            else None
        )
        default_builder.discretisation(default_discretisation)
        # Keys which describe runs
        run_keys = [key for key in config.keys() if key.startswith("run")]
        # Store the tasks
        tasks: list[SimulationJobTask] = []
        for run in run_keys:
            # Current run parameters
            current_run = config[run]
            # Copy the default builder
            current_builder = copy.copy(default_builder)
            # Override the default builder with the current run's parameters
            current_builder.num_rings(current_run.get("num_rings"))
            current_builder.num_time_steps(current_run.get("num_time_steps"))
            current_builder.time_step(current_run.get("time_step"))
            current_builder.eccentricity(current_run.get("eccentricity"))
            current_kernel = current_run.get("kernel_type")
            if current_kernel is not None:
                current_builder.kernel_type(SofteningKernel(current_kernel))
            current_builder.softening_length(current_run.get("softening_length"))
            current_builder.bend_amplitude(current_run.get("bend_amplitude"))
            current_builder.base_radius(current_run.get("base_radius"))
            current_builder.max_radius(current_run.get("max_radius"))
            current_builder.save_interval(current_run.get("save_interval"))
            current_builder.use_gravity(current_run.get("use_gravity"))
            current_builder.with_tilt(current_run.get("with_tilt"))
            current_builder.file_name(current_run.get("file_name"))
            current_builder.name(current_run.get("name"))
            current_scheme = current_run.get("numerical_scheme")
            if current_scheme is not None:
                current_builder.numerical_scheme(NumericalScheme(current_scheme))
            num_particles: int | None = current_run.get("num_particles")
            arc_length: float | None = current_run.get("arc_length")
            discretisation: RingDiscretisation | None = (
                parse_discretisation(num_particles, arc_length)
                if (num_particles is not None or arc_length is not None)
                else None
            )
            current_builder.discretisation(discretisation)
            # Build task and add to list
            tasks.append(current_builder.build())
        return tasks


if __name__ == "__main__":
    import sys

    _ = setup_logging()
    main(sys.argv[1:])
