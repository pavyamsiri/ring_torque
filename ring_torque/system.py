"""Contains the RingSystem class, which represents a system of concentric elliptic rings."""

from __future__ import annotations

# Standard libraries
from enum import Enum
import logging
import time
from typing import Callable, Sequence, TypeAlias

# External libraries
import h5py

import numpy.typing as npt
import numpy as np

import scipy.spatial as spatial
from ring_torque.progress import CustomProgress

# Internal libraries
from ring_torque.ring import Ring, RingDiscretisation
from ring_torque.common import (
    CURRENT_MAJOR_VERSION,
    CURRENT_MINOR_VERSION,
    CURRENT_PATCH_VERSION,
    PLOT_PADDING_FACTOR,
    NumpyFloatType,
    EPSILON,
    format_execution_time_s,
    format_num_bytes,
)
from ring_torque.quaternion import Quaternion
from ring_torque.units import G_IN_SIMULATION_UNITS

# Local logger
log = logging.getLogger(__name__)

# Default tilt parameters
DEFAULT_TILT_RADIUS: float = 0.1
DEFAULT_TILT_SEED: int = 567457895

# Batch size for calculating the potential.
CALCULATE_POTENTIAL_BATCH_SIZE = 256


class NumericalScheme(Enum):
    EULER = "EULER"
    KICK_DRIFT_KICK = "KICK_DRIFT_KICK"
    DRIFT_KICK_DRIFT = "DRIFT_KICK_DRIFT"

    def __str__(self) -> str:
        return self.value

    def get_identifier(self) -> str:
        match self:
            case NumericalScheme.EULER:
                return "e"
            case NumericalScheme.KICK_DRIFT_KICK:
                return "kdk"
            case NumericalScheme.DRIFT_KICK_DRIFT:
                return "dkd"


class SofteningKernel(Enum):
    NONE = "NONE"
    PLUMMER = "PLUMMER"
    KERNEL_32 = "KERNEL_32"
    CUT = "CUT"

    def __str__(self) -> str:
        return self.value

    def get_identifier(self) -> str:
        match self:
            case SofteningKernel.NONE:
                return "none"
            case SofteningKernel.PLUMMER:
                return "plum"
            case SofteningKernel.KERNEL_32:
                return "32"
            case SofteningKernel.CUT:
                return "cut"

    def get_name(self) -> str:
        match self:
            case SofteningKernel.NONE:
                return ""
            case SofteningKernel.PLUMMER:
                return "Plummer"
            case SofteningKernel.KERNEL_32:
                return "Kernel32"
            case SofteningKernel.CUT:
                return "Cut"


def calculate_kernel_none(
    pos: npt.NDArray[np.floating],
    other_pos: npt.NDArray[np.floating],
    softening_length: float,
) -> npt.NDArray[np.floating]:
    """Calculates the gravitational kernel for the given positions and other positions with no softening.

    Parameters
    ----------
    pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential at.
    other_pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential from.
    softening_length : float
        The softening length to use. This is ignored.

    Returns
    -------
    kernel : npt.NDArray[np.floating]
        The gravitational kernel. This is a 2D array of shape (N, M) where N is the number of points in `pos` and M is
        the number of points in `other_pos`.
    """
    _ = softening_length
    return -np.reciprocal(spatial.distance.cdist(pos, other_pos, metric="euclidean"))


def calculate_kernel_plummer(
    pos: npt.NDArray[np.floating],
    other_pos: npt.NDArray[np.floating],
    softening_length: float,
) -> npt.NDArray[np.floating]:
    """Calculates the gravitational kernel for the given positions and other positions with softening.

    Parameters
    ----------
    pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential at.
    other_pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential from.
    softening_length : float
        The softening length to use.

    Returns
    -------
    kernel : npt.NDArray[np.floating]
        The gravitational kernel. This is a 2D array of shape (N, M) where N is the number of points in `pos` and M is
        the number of points in `other_pos`.
    """
    distances = spatial.distance.cdist(pos, other_pos, metric="sqeuclidean")
    return -np.reciprocal(np.sqrt(distances + softening_length**2))


def calculate_kernel_32(
    pos: npt.NDArray[np.floating],
    other_pos: npt.NDArray[np.floating],
    softening_length: float,
) -> npt.NDArray[np.floating]:
    """Calculates the gravitational kernel for the given positions and other positions with softening.

    Parameters
    ----------
    pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential at.
    other_pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential from.
    softening_length : float
        The softening length to use.

    Returns
    -------
    kernel : npt.NDArray[np.floating]
        The gravitational kernel. This is a 2D array of shape (N, M) where N is the number of points in `pos` and M is
        the number of points in `other_pos`.
    """
    distances = spatial.distance.cdist(pos, other_pos, metric="sqeuclidean")
    square_epsilon = softening_length**2
    return -(distances + 1.5 * square_epsilon) / np.sqrt(
        (distances + square_epsilon) ** 3
    )


def calculate_kernel_cut(
    pos: npt.NDArray[np.floating],
    other_pos: npt.NDArray[np.floating],
    softening_length: float,
) -> npt.NDArray[np.floating]:
    """Calculates the gravitational kernel for the given positions and other positions with softening.

    Parameters
    ----------
    pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential at.
    other_pos : npt.NDArray[np.floating]
        The array of points to calculate the gravitational potential from.
    softening_length : float
        The softening length to use.

    Returns
    -------
    kernel : npt.NDArray[np.floating]
        The gravitational kernel. This is a 2D array of shape (N, M) where N is the number of points in `pos` and M is
        the number of points in `other_pos`.
    """
    distances = spatial.distance.cdist(pos, other_pos, metric="euclidean")
    distances[distances < softening_length] = np.inf
    return -np.reciprocal(distances)


SofteningKernelFunction: TypeAlias = Callable[
    [npt.NDArray[np.floating], npt.NDArray[np.floating], float],
    npt.NDArray[np.floating],
]
# A lookup table of kernel functions
KERNELS: dict[SofteningKernel, SofteningKernelFunction] = {
    SofteningKernel.NONE: calculate_kernel_none,
    SofteningKernel.PLUMMER: calculate_kernel_plummer,
    SofteningKernel.KERNEL_32: calculate_kernel_32,
    SofteningKernel.CUT: calculate_kernel_cut,
}


class RingSystem:
    """A system of concentric elliptic rings."""

    def __init__(
        self,
        radii: Sequence[float],
        eccentricity: float,
        scale_length: float,
        scale_mass: float,
        bend_amplitude: float,
        discretisation: RingDiscretisation,
    ):
        """Creates a system of concentric elliptic rings.

        Parameters
        ----------
        radii : Sequence[float]
            The radii of the rings. Assumed to be in units of scale length.
        eccentricity: float
            The eccentricity of the rings.
        scale_length : float
            The scale length of the disk.
        scale_mass : float
            The scale mass of the disk.
        """
        # List of rings
        self._rings: list[Ring] = []
        for _, current_radius in enumerate(radii):
            current_mass = scale_mass * current_radius * np.exp(-current_radius)
            current_ring = Ring(
                current_radius,
                eccentricity,
                (0, 0, 1),
                0,
                current_mass,
                bend_amplitude,
                scale_length,
                discretisation,
            )
            self._rings.append(current_ring)

        # Size to set for plot
        self._max_plot_extent: float = (
            PLOT_PADDING_FACTOR * self._rings[0].semi_major_axis * max(radii)
        )
        # Cache the radii
        self._radii = radii
        # Disk properties
        self._eccentricity = eccentricity
        self._scale_length = scale_length
        self._scale_mass = scale_mass
        self._bend_amplitude = bend_amplitude

    # Accessors
    @property
    def rings(self) -> Sequence[Ring]:
        """Sequence[Ring]: The rings in the system."""
        return self._rings

    @property
    def radii(self) -> Sequence[float]:
        """Sequence[float]: The radii of the rings in the system."""
        return self._radii

    @property
    def min_radius(self) -> float:
        """float: The minimum radius of the rings in the system."""
        return min(self._radii)

    @property
    def max_radius(self) -> float:
        """float: The maximum radius of the rings in the system."""
        return max(self._radii)

    def set_discretisation(self, discretisation: RingDiscretisation) -> None:
        for ring in self.rings:
            ring.discretisation = discretisation

    def get_total_num_particles(
        self, ring_idx: int | None = None
    ) -> tuple[int, Sequence[int]]:
        total_particles: int = 0
        particle_counts: list[int] = len(self._rings) * [0]
        for idx, ring in enumerate(self._rings):
            if ring_idx is not None and idx == ring_idx:
                continue
            particle_count = ring.get_num_particles()
            particle_counts[idx] = particle_count
            total_particles += particle_count
        return total_particles, particle_counts

    def get_all_points_and_masses(
        self, ring_idx: int | None = None
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        total_particles, particle_counts = self.get_total_num_particles(
            ring_idx=ring_idx
        )
        # Fill out the array of points and masses from the other rings
        all_points = np.full((3, total_particles), np.nan, dtype=NumpyFloatType)
        masses = np.full(total_particles, np.nan, dtype=NumpyFloatType)
        start_idx = 0
        for idx, ring in enumerate(self._rings):
            if ring_idx is not None and idx == ring_idx:
                continue
            num_particles = particle_counts[idx]
            end_idx = start_idx + num_particles
            all_points[:, start_idx:end_idx] = ring.autosample_points()
            masses[start_idx:end_idx] = ring.mass / num_particles
            start_idx += num_particles
        return all_points, masses

    def calculate_potential_on_ring_array(
        self,
        pos: npt.NDArray[np.floating],
        kernel_type: SofteningKernel,
        softening_length: float,
        ring_idx: int,
    ) -> npt.NDArray[np.floating]:
        """Calculate the potential on a ring due to the other rings, i.e. excludes the ring's potential to avoid
        self-gravitation.

        Note
        ----
        The points must be given as column vectors in the array. This must have a shape of (2, N) where N is the number
        of points to calculate the force at with the first row being the x coordinates and the second row being the
        y coordinates.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational potential at.
        kernel_type : SofteningKernel
            The type of kernel to use.
        softening_length : float
            The softening length to use.
        ring_idx : int
            The index of the ring to exclude from the potential.

        Returns
        -------
        potential : npt.NDArray[np.floating]
            The gravitational potential at the given points. This is a 1D array of length N.
        """
        # Initialise the potential array
        potential: npt.NDArray[np.floating] = np.zeros(
            pos.shape[1], dtype=NumpyFloatType
        )
        ring_points, masses = self.get_all_points_and_masses(ring_idx=ring_idx)
        total_particles = ring_points.shape[1]
        # Calculate the number of batches to split the calculation into
        num_batches = int(np.ceil(total_particles / CALCULATE_POTENTIAL_BATCH_SIZE))
        # Choose the kernel to use
        for batch_idx in range(num_batches):
            # Slice the points and masses
            start_idx = batch_idx * CALCULATE_POTENTIAL_BATCH_SIZE
            end_idx = min(
                (batch_idx + 1) * CALCULATE_POTENTIAL_BATCH_SIZE,
                total_particles,
            )
            current_mass_points = ring_points[:, start_idx:end_idx]
            current_masses = masses[start_idx:end_idx]
            # Call cdist to calculate the distances between the points
            kernel = KERNELS[kernel_type](
                pos.T, current_mass_points.T, softening_length
            )
            # Calculate the potential
            current_potential = (G_IN_SIMULATION_UNITS * current_masses * kernel).sum(
                axis=1
            )
            # Add to the total potential
            potential += current_potential

        return potential

    def calculate_force_on_ring_array(
        self,
        pos: npt.NDArray[np.floating],
        mass: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        ring_idx: int,
    ) -> npt.NDArray[np.floating]:
        """Calculate the force on a ring due to the other rings, i.e. excludes the ring's potential to avoid
        self-gravitation.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational force at.
        mass : float
            The mass of the particles.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        ring_idx : int
            The index of the ring to exclude from the force.

        Returns
        -------
        force : npt.NDArray[np.floating]
            The gravitational force at the given points. This is a 2D array of shape (2, N) where N is the
            number of points.
        """
        force: npt.NDArray[np.floating] = np.zeros_like(pos, dtype=NumpyFloatType)
        num_points = pos.shape[1]
        evaluation_points = np.tile(pos, 6)
        # Phi(x + epsilon, y, z)
        evaluation_points[0, :num_points] += EPSILON
        # Phi(x, y + epsilon, z)
        evaluation_points[1, num_points : (2 * num_points)] += EPSILON
        # Phi(x, y, z + epsilon)
        evaluation_points[2, (2 * num_points) : (3 * num_points)] += EPSILON
        # Phi(x - epsilon, y, z)
        evaluation_points[0, (3 * num_points) : (4 * num_points)] -= EPSILON
        # Phi(x, y - epsilon, z)
        evaluation_points[1, (4 * num_points) : (5 * num_points)] -= EPSILON
        # Phi(x, y, z - epsilon)
        evaluation_points[2, (5 * num_points) : (6 * num_points)] -= EPSILON
        potential = self.calculate_potential_on_ring_array(
            evaluation_points,
            kernel_type,
            softening_length,
            ring_idx,
        )
        forward_x_potential = potential[:num_points]
        forward_y_potential = potential[num_points : (2 * num_points)]
        forward_z_potential = potential[(2 * num_points) : (3 * num_points)]
        backward_x_potential = potential[(3 * num_points) : (4 * num_points)]
        backward_y_potential = potential[(4 * num_points) : (5 * num_points)]
        backward_z_potential = potential[(5 * num_points) : (6 * num_points)]
        acceleration_x = -(forward_x_potential - backward_x_potential) / (2 * EPSILON)
        acceleration_y = -(forward_y_potential - backward_y_potential) / (2 * EPSILON)
        acceleration_z = -(forward_z_potential - backward_z_potential) / (2 * EPSILON)
        force[0, :] = mass * acceleration_x
        force[1, :] = mass * acceleration_y
        force[2, :] = mass * acceleration_z
        return force

    def calculate_torque_on_ring_array(
        self,
        pos: npt.NDArray[np.floating],
        mass: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        ring_idx: int,
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        """Calculate the torque at a set of points exerted by rings and excludes the indexed ring's potential if the index is
        valid.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational potential at.
        mass : float
            The mass of the particles.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        ring_idx : int
            The index of the ring to exclude from the force.

        Returns
        -------
        total_torque : npt.NDArray[np.floating]
            The total torque on the ring.
        torques : npt.NDArray[np.floating]
            The torques on the ring at each sampled point.
        points : npt.NDArray[np.floating]
            The sampled points on the ring.
        """
        force = self.calculate_force_on_ring_array(
            pos, mass, kernel_type, softening_length, ring_idx
        )
        torques = np.cross(pos.T, force.T)
        total_torque: npt.NDArray[np.floating] = torques.sum(axis=0)
        assert total_torque.shape == (3,)
        return (total_torque, torques, pos)

    def calculate_torque_on_ring(
        self,
        mass: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        ring_idx: int,
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        """Calculate the torque on a ring due to the other rings, i.e. excludes the ring's potential to avoid
        self-gravitation.

        Parameters
        ----------
        mass : float
            The mass of the particles.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        ring_idx : int
            The index of the ring to exclude from the force.

        Returns
        -------
        total_torque : npt.NDArray[np.floating]
            The total torque on the ring.
        torques : npt.NDArray[np.floating]
            The torques on the ring at each sampled point.
        points : npt.NDArray[np.floating]
            The sampled points on the ring.
        """
        points = self._rings[ring_idx].autosample_points()
        return self.calculate_torque_on_ring_array(
            points, mass, kernel_type, softening_length, ring_idx
        )

    def tilt_rings(self, tilt_radius: float, seed: int):
        """Tilts the rings so that they are not all in the same plane.

        Parameters
        ----------
        tilt_radius : float
            The radius of the circle to tilt the rings on.
        seed : int
            The seed for the random number generator.

        Notes
        -----
        The tilted rotation axes are calculated by perturbing the z axis by a small amount given by generating a
        random (x,y) vector that lies within a circle of radius `tilt_radius` and then calculating the z component by
        enforcing unitarity.
        """
        # Seed the random number generator
        generator = np.random.default_rng(seed)
        num_rings: int = len(self._rings)
        # Generate radii such that almost all points are within the tilt radius
        radii: npt.NDArray[np.floating] = tilt_radius * generator.normal(
            loc=0.0, scale=tilt_radius / 3, size=num_rings
        )
        # Generate angles for each point inside the tilt radius
        theta: npt.NDArray[np.floating] = generator.uniform(
            low=0.0, high=2 * np.pi, size=num_rings
        )
        # Calculate the rotation axes
        rotation_axis: npt.NDArray[np.floating] = np.zeros(
            (3, len(self._rings)), dtype=NumpyFloatType
        )
        rotation_axis[0, :] = radii * np.cos(theta)
        rotation_axis[1, :] = radii * np.sin(theta)
        rotation_axis[2, :] = np.sqrt(1 - radii**2)
        # Generate random rotation angles
        rotation_angle: npt.NDArray[np.floating] = generator.uniform(
            low=0.0, high=2 * np.pi, size=num_rings
        )
        for ring_idx, ring in enumerate(self._rings):
            current_axis: tuple[float, float, float] = (
                rotation_axis[0, ring_idx],
                rotation_axis[1, ring_idx],
                rotation_axis[2, ring_idx],
            )
            current_angle: float = rotation_angle[ring_idx]
            new_rotation = Quaternion.from_axis_angle(current_axis, current_angle)
            ring.set_rotation_quaternion(new_rotation)

    def update_rotation_drift(
        self,
        omega: npt.NDArray[np.floating],
        time_step: float,
    ) -> None:
        """Calculates the next time step's orientations or "drift".
        This does mutate the system's orientations.

        Parameters
        ----------
        omega : npt.NDArray[np.floating]
            The angular velocity of the rings.
        time_step : float
            The time step.
        """
        next_rotation_quaternion: Sequence[Quaternion] = [
            Quaternion.from_axis_angle((0, 0, 1), 0) for _ in range(len(self._rings))
        ]
        for ring_idx, ring in enumerate(self._rings):
            current_omega = omega[:, ring_idx]
            velocity_quaternion = Quaternion.create_vector_quaternion(current_omega)
            quaternion_derivative = (
                0.5 * velocity_quaternion * ring.get_rotation_quaternion()
            )
            next_rotation_quaternion[ring_idx] = (
                ring.get_rotation_quaternion() + time_step * quaternion_derivative
            ).normalise()
        # Update the rings' rotation and patches
        for (
            ring_idx,
            (ring, next_rotation),
        ) in enumerate(zip(self._rings, next_rotation_quaternion)):
            # Update the ring's rotation
            ring.set_rotation_quaternion(next_rotation)

    def update_rotation_kick(
        self,
        omega: npt.NDArray[np.floating],
        time_step: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        with_gravity: bool,
    ) -> None:
        """Calculates the next time step's angular velocity or "kick".
        This does mutate the given array `omega`.

        Parameters
        ----------
        omega : npt.NDArray[np.floating]
            The angular velocity of the rings. This gets mutated by this function!
        time_step : float
            The time step.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        with_gravity : bool
            Whether to include gravity in the simulation.
        """
        next_omega: npt.NDArray[np.floating] = np.zeros_like(
            omega, dtype=NumpyFloatType
        )
        for ring_idx, ring in enumerate(self._rings):
            current_omega = omega[:, ring_idx]
            # Apply torque from gravity
            if with_gravity:
                particle_mass: float = self._rings[ring_idx].get_particle_mass()
                current_torque, _, _ = self.calculate_torque_on_ring(
                    particle_mass, kernel_type, softening_length, ring_idx
                )
                inertia_tensor = ring.get_rotated_inertia_tensor()
                inverse_inertia_tensor = np.linalg.inv(inertia_tensor)
                gyroscopic_term = -np.cross(
                    current_omega, inertia_tensor @ current_omega
                )
                current_angular_acceleration = inverse_inertia_tensor @ (
                    current_torque + gyroscopic_term
                )
            # Torqueless evolution
            else:
                current_angular_acceleration = np.zeros(3, dtype=NumpyFloatType)
            # Compute new angular velocity
            current_next_omega = (
                current_omega + current_angular_acceleration * time_step
            )
            # Evolve angular velocity
            next_omega[:, ring_idx] = current_next_omega
        omega[:] = next_omega

    def update_rotation_euler(
        self,
        omega: npt.NDArray[np.floating],
        time_step: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        with_gravity: bool,
        simple_addition: bool = False,
    ) -> None:
        """Calculates the next time step's rotation quaternion and angular velocity using an Euler numerical scheme.
        This does mutate the ring system's orientations.

        Parameters
        ----------
        omega : npt.NDArray[np.floating]
            The angular velocity of the rings. This is a 2D array of shape (3, N) where N is the number of rings.
        time_step : float
            The time step.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        with_gravity : bool
            Whether to include gravity in the simulation.
        """
        # Initialise the next rotation quaternion and angular velocity
        next_rotation_quaternion: list[Quaternion] = [
            Quaternion.from_axis_angle((0, 0, 1), 0) for _ in range(len(self._rings))
        ]
        next_omega: npt.NDArray[np.floating] = np.zeros_like(
            omega, dtype=NumpyFloatType
        )

        for ring_idx, ring in enumerate(self._rings):
            current_omega = omega[:, ring_idx]
            # Apply torque from gravity
            if with_gravity:
                particle_mass: float = self._rings[ring_idx].get_particle_mass()
                current_torque, _, _ = self.calculate_torque_on_ring(
                    particle_mass, kernel_type, softening_length, ring_idx
                )
                inertia_tensor = ring.get_rotated_inertia_tensor()
                inverse_inertia_tensor = np.linalg.inv(inertia_tensor)
                gyroscopic_term = -np.cross(
                    current_omega, inertia_tensor @ current_omega
                )
                current_angular_acceleration = inverse_inertia_tensor @ (
                    current_torque + gyroscopic_term
                )
            # Torqueless evolution
            else:
                current_angular_acceleration = np.zeros(3, dtype=NumpyFloatType)
            # Compute new angular velocity
            current_next_omega = (
                current_omega + current_angular_acceleration * time_step
            )
            # Evolve angular velocity
            next_omega[:, ring_idx] = current_next_omega
            if simple_addition:
                velocity_quaternion = Quaternion.create_vector_quaternion(
                    current_next_omega
                )
                quaternion_derivative = (
                    0.5 * velocity_quaternion * ring.get_rotation_quaternion()
                )
                next_rotation_quaternion[ring_idx] = (
                    ring.get_rotation_quaternion() + time_step * quaternion_derivative
                ).normalise()
            else:
                # Evolve rotation quaternion
                magnus_term_1 = 1 / 2 * (current_omega + current_next_omega) * time_step
                magnus_term_2 = (
                    1
                    / 12
                    * (np.cross(current_next_omega, current_omega))
                    * time_step**2
                )
                magnus_term_3 = (
                    1
                    / 240
                    * np.cross(
                        current_angular_acceleration,
                        np.cross(current_angular_acceleration, current_omega),
                    )
                    * time_step**5
                )
                magnus_exponent = Quaternion.create_vector_quaternion(
                    magnus_term_1 + magnus_term_2 + magnus_term_3
                )
                next_rotation_quaternion[ring_idx] = (
                    (0.5 * magnus_exponent).exponential()
                    * ring.get_rotation_quaternion()
                ).normalise()
        omega[:] = next_omega
        # Update the rings' rotation and patches
        for (
            ring_idx,
            (ring, next_rotation),
        ) in enumerate(zip(self._rings, next_rotation_quaternion)):
            # Update the ring's rotation
            ring.set_rotation_quaternion(next_rotation)

    def simulate_rings(
        self,
        output_path: str,
        num_time_steps: int,
        time_step: float,
        tangential_velocity: float,
        kernel_type: SofteningKernel,
        softening_length: float,
        save_interval: int,
        with_gravity: bool,
        with_tilt: bool,
        numerical_scheme: NumericalScheme,
        simulation_name: str,
        display: bool = True,
    ):
        """Simulates the ring system, saving the orientations and angular velocities in a hdf5 file.

        Parameters
        ----------
        output_path : str
            The path to save the animation to. The only supported file extension is `.hdf5`.
        num_particles : int
            The number of particles to sample from the ring as part of the particle approximation.
        num_time_steps : int
            The number of time steps to simulate.
        time_step : float
            The time step.
        tangential_velocity : float
            The tangential velocity of the rings.
        kernel_type : SofteningKernel
            The softening kernel to use.
        softening_length : float
            The softening length to use.
        fps : int
            The frame rate of the animation.
        with_gravity : bool
            Whether to include gravity in the simulation.
        with_tilt : bool
            Whether to tilt the rings.
        simulation_name : str
            The name of the simulation.
        display : bool
            Turns on logging, defaults to `True`.
        """
        log.info("Starting ring simulation...")

        # Setup initial conditions
        omega: npt.NDArray[np.floating] = np.zeros(
            (3, len(self._rings)), dtype=NumpyFloatType
        )
        omega[2, :] = tangential_velocity / np.array(self._radii, dtype=NumpyFloatType)

        if with_tilt:
            self.tilt_rings(DEFAULT_TILT_RADIUS, DEFAULT_TILT_SEED)

        log.info(f"Precession rates = {omega[2, :]}")
        total_saved_frames = 1 + num_time_steps // save_interval
        orientations: npt.NDArray[np.floating] = np.full(
            (4, len(self._rings), total_saved_frames), np.nan, dtype=NumpyFloatType
        )
        angular_velocities: npt.NDArray[np.floating] = np.full(
            (3, len(self._rings), total_saved_frames), np.nan, dtype=NumpyFloatType
        )
        # Set up initial conditions in orientations and angular velocities
        num_total_ring_particles: int = 0
        for ring_idx, ring in enumerate(self._rings):
            current_quaternion = ring.get_rotation_quaternion()
            orientations[0, ring_idx, 0] = current_quaternion.w
            orientations[1, ring_idx, 0] = current_quaternion.x
            orientations[2, ring_idx, 0] = current_quaternion.y
            orientations[3, ring_idx, 0] = current_quaternion.z
            angular_velocities[:, ring_idx, 0] = omega[:, ring_idx]
            particle_count = ring.get_num_particles()
            log.debug(f"Sampling ring {ring_idx} with {particle_count} particles")
            num_total_ring_particles += particle_count
        log.info(f"Simulating with {num_total_ring_particles} particles in total")

        total_values_used = orientations.size + angular_velocities.size
        total_bytes_used = orientations.nbytes + angular_velocities.nbytes
        log.info(
            f"{format_num_bytes(total_bytes_used)} of memory will be used to store the orientations and angular velocities."
        )
        log.info(f"{total_values_used:,} {NumpyFloatType.__name__}s will be used.")

        start_time = time.perf_counter()
        with CustomProgress(disable=not display, transient=True) as progress:
            task = progress.add_task(
                "[bold green]Simulating rings...",
                total=num_time_steps,
            )
            for frame in range(num_time_steps):
                match numerical_scheme:
                    case NumericalScheme.EULER:
                        self.update_rotation_euler(
                            omega,
                            time_step,
                            kernel_type,
                            softening_length,
                            with_gravity,
                            simple_addition=True,
                        )
                    case NumericalScheme.KICK_DRIFT_KICK:
                        # Kick
                        self.update_rotation_kick(
                            omega,
                            time_step / 2,
                            kernel_type,
                            softening_length,
                            with_gravity,
                        )
                        # Drift
                        self.update_rotation_drift(omega, time_step)
                        # Kick
                        self.update_rotation_kick(
                            omega,
                            time_step / 2,
                            kernel_type,
                            softening_length,
                            with_gravity,
                        )
                    case NumericalScheme.DRIFT_KICK_DRIFT:
                        # Drift
                        self.update_rotation_drift(omega, time_step / 2)
                        # Kick
                        self.update_rotation_kick(
                            omega,
                            time_step,
                            kernel_type,
                            softening_length,
                            with_gravity,
                        )
                        # Drift
                        self.update_rotation_drift(omega, time_step / 2)
                    case _ as scheme:
                        log.error(f"Unsupported numerical scheme {scheme}")
                        return
                # Save orientation and angular velocity
                for (
                    ring_idx,
                    (ring, next_omega),
                ) in enumerate(zip(self._rings, omega.T)):
                    # Update the ring's rotation
                    next_rotation = ring.get_rotation_quaternion()
                    if frame % save_interval == 0:
                        save_frame = 1 + frame // save_interval
                        orientations[0, ring_idx, save_frame] = next_rotation.w
                        orientations[1, ring_idx, save_frame] = next_rotation.x
                        orientations[2, ring_idx, save_frame] = next_rotation.y
                        orientations[3, ring_idx, save_frame] = next_rotation.z
                        angular_velocities[0, ring_idx, save_frame] = next_omega[0]
                        angular_velocities[1, ring_idx, save_frame] = next_omega[1]
                        angular_velocities[2, ring_idx, save_frame] = next_omega[2]
                progress.update(task, advance=1)

        elapsed_time = time.perf_counter() - start_time
        elapsed_time_str, iteration_rate_str = format_execution_time_s(
            elapsed_time, num_time_steps
        )
        log.info(f"Finished ring simulation in {elapsed_time_str}.")
        log.info(f"Iteration speed: {iteration_rate_str}.")
        # For v1.0.1 - Add softening length
        with h5py.File(output_path, "w") as output_file:
            output_file.create_dataset(
                "orientations",
                data=orientations,
            )
            output_file.create_dataset("angular_velocity", data=angular_velocities)
            output_file.attrs["tangential_velocity"] = tangential_velocity
            # Setting attributes
            output_file.attrs["save_interval"] = save_interval
            output_file.attrs["time_step"] = time_step
            output_file.attrs["radii"] = self._radii
            output_file.attrs["eccentricity"] = self._eccentricity
            output_file.attrs["scale_length"] = self._scale_length
            output_file.attrs["scale_mass"] = self._scale_mass
            output_file.attrs["bend_amplitude"] = self._bend_amplitude
            output_file.attrs["max_plot_extent"] = self._max_plot_extent
            output_file.attrs["with_gravity"] = with_gravity
            output_file.attrs["name"] = simulation_name
            # Set version
            output_file.attrs["major_version"] = CURRENT_MAJOR_VERSION
            output_file.attrs["minor_version"] = CURRENT_MINOR_VERSION
            output_file.attrs["patch_version"] = CURRENT_PATCH_VERSION

        log.info(f"Simulation data dumped to {output_path}")
