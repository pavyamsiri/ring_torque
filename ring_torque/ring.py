from __future__ import annotations

# Standard libraries
from enum import Enum
import logging

# External libraries
import numpy as np
import numpy.typing as npt

from scipy.special import ellipk, ellipe, ellipeinc
import scipy.optimize
import scipy.spatial as spatial

# Internal libraries
from ring_torque.common import EPSILON, NumpyFloatType
from ring_torque.quaternion import Quaternion

# Universal gravitational constant
G: float = 1.0

# Local logger
log = logging.getLogger(__name__)


class RingDiscretisationType(Enum):
    UNIFORM_NUM_PARTICLES = "UNIFORM_NUM_PARTICLES"
    UNIFORM_ARC_LENGTH = "UNIFORM_ARC_LENGTH"


class RingDiscretisation:
    def __init__(self, value: int | float, discretisation_type: RingDiscretisationType):
        self._value: int | float = value
        self._type: RingDiscretisationType = discretisation_type

    def get_type(self) -> RingDiscretisationType:
        return self._type

    def get_num_particles(self) -> int:
        assert self._type == RingDiscretisationType.UNIFORM_NUM_PARTICLES
        assert isinstance(self._value, int)
        return self._value

    def get_arc_length(self) -> float:
        assert self._type == RingDiscretisationType.UNIFORM_ARC_LENGTH
        assert isinstance(self._value, float)
        return self._value


class Ring:
    """Class representing an elliptic ring of uniform density centred at the origin.

    Attributes
    ----------
    radius : float
        The radius of the ring.
    scale_width : float
        The scale width of the ring.
    scale_height : float
        The scale height of the ring.
    rotation : float
        The angle of rotation of the ring in radians.
    mass : float
        The mass of the ring.
    """

    def __init__(
        self,
        radius: float,
        eccentricity: float,
        rotation_axis: tuple[float, float, float],
        rotation_angle: float,
        mass: float,
        bend_amplitude: float,
        disk_scale_length: float,
        discretisation: RingDiscretisation,
    ):
        """Initialises the ring object.

        Parameters
        ----------
        radius : float
            The radius of the ring.
        eccentricity : float
            The eccentricity of the ring.
        rotation : float
            The angle of rotation of the ring in radians.
        bend_amplitude : float
            The amplitude of the ring's bending mode.
        mass : float
            The mass of the ring.
        disk_scale_length : float
            The scale length of the disk in kpc.
        discretisation : RingDiscretisation
        """
        assert eccentricity >= 0 and eccentricity < 1
        self._radius: float = radius
        # Eccentricity
        self._eccentricity: float = eccentricity
        self._mass: float = mass
        self._bend_amplitude: float = bend_amplitude
        self._rotation: Quaternion = Quaternion.from_axis_angle(
            rotation_axis, rotation_angle
        )
        self._disk_scale_length: float = disk_scale_length
        self.discretisation: RingDiscretisation = discretisation

        # Major and minor axes
        self._scale_width: float = 1.0
        self._scale_height: float = 1 / np.sqrt(1 - eccentricity**2)
        self._semi_major_axis: float = max(self._scale_height, self._scale_width)
        self._semi_minor_axis: float = min(self._scale_height, self._scale_width)

        # FIXME: This does not take into account the bending mode
        # Special case e = 0 i.e. circle
        if self._semi_major_axis == self._semi_minor_axis:
            (
                moment_of_inertia_major,
                moment_of_inertia_minor,
                moment_of_inertia_z,
                perimeter,
            ) = Ring.calculate_moment_of_inertia_and_perimeter_circles(
                self._mass, self._radius, self._semi_major_axis, self._semi_minor_axis
            )
        # General case 0 < e < 1 i.e. ellipse
        else:
            (
                moment_of_inertia_major,
                moment_of_inertia_minor,
                moment_of_inertia_z,
                perimeter,
            ) = Ring.calculate_moment_of_inertia_and_perimeter_ellipses(
                self._mass, self._radius, self._semi_major_axis, self._semi_minor_axis
            )
        # Moments of inertia
        self._moment_of_inertia_x: float = (
            moment_of_inertia_major
            if self._scale_width >= self._scale_height
            else moment_of_inertia_minor
        )
        self._moment_of_inertia_y: float = (
            moment_of_inertia_minor
            if self._scale_width >= self._scale_height
            else moment_of_inertia_major
        )
        self._moment_of_inertia_z: float = moment_of_inertia_z
        self._inertia_tensor: npt.NDArray[np.floating] = np.array(
            [
                [self.moment_of_inertia_x, 0, 0],
                [0, self.moment_of_inertia_y, 0],
                [0, 0, self.moment_of_inertia_z],
            ],
            dtype=NumpyFloatType,
        )

        self._perimeter: float = perimeter

        # Cached attributes
        self._sampled_points: dict[int, npt.NDArray[np.floating]] = {}
        self._rotated_points: dict[int, npt.NDArray[np.floating]] = {}
        self._rotation_matrix: npt.NDArray[np.floating] | None = None
        self._rotated_inertia_tensor: npt.NDArray[np.floating] | None = None

        # DEBUG
        self.sample_point_count: int = 0

    @property
    def total_width(self) -> float:
        """float: The total width of the ring."""
        return 2 * self._radius * self._scale_width

    @property
    def total_height(self) -> float:
        """float: The total height of the ring."""
        return 2 * self._radius * self._scale_height

    @property
    def max_extent(self) -> float:
        """float: The extent of the ring along its major axis."""
        return max(self.total_width, self.total_height)

    @property
    def min_extent(self) -> float:
        """float: The extent of the ring along its minor axis."""
        return min(self.total_width, self.total_height)

    @property
    def mass(self) -> float:
        """float: The mass of the elliptic ring."""
        return self._mass

    @property
    def radius(self) -> float:
        """float: The radius of the elliptic ring."""
        return self._radius

    @property
    def scale_width(self) -> float:
        """float: The scale width of the elliptic ring."""
        return self._scale_width

    @property
    def scale_height(self) -> float:
        """float: The scale height of the elliptic ring."""
        return self._scale_height

    @property
    def eccentricity(self) -> float:
        """float: The eccentricity of the elliptic ring."""
        return self._eccentricity

    @property
    def semi_major_axis(self) -> float:
        """float: The semi-major axis of the elliptic ring."""
        return self._semi_major_axis

    @property
    def semi_minor_axis(self) -> float:
        """float: The semi-minor axis of the elliptic ring."""
        return self._semi_minor_axis

    @property
    def moment_of_inertia_x(self) -> float:
        """float: The moment of inertia about the x-axis of the elliptic ring."""
        return self._moment_of_inertia_x

    @property
    def moment_of_inertia_y(self) -> float:
        """float: The moment of inertia about the y-axis of the elliptic ring."""
        return self._moment_of_inertia_y

    @property
    def moment_of_inertia_z(self) -> float:
        """float: The moment of inertia about the z-axis of the elliptic ring."""
        return self._moment_of_inertia_z

    @property
    def perimeter(self) -> float:
        """float: The perimeter of the elliptic ring."""
        return self._perimeter

    def get_eccentricity_squared(self) -> float:
        """Calculates and returns the eccentricity squared of the ring.

        Returns
        -------
        eccentricity_squared : float
            The eccentricity squared of the ring.
        """
        return self._eccentricity**2

    def get_rotation_quaternion(self) -> Quaternion:
        """Returns the quaternion representing the ring's rotation.

        Returns
        -------
        rotation : Quaternion
            The quaternion representing the ring's rotation.
        """
        return self._rotation

    def get_rotation_matrix(self) -> npt.NDArray[np.floating]:
        """Calculates and returns the rotation matrix for the ring.

        Returns
        -------
        transform : npt.NDArray[np.floating]
            The rotation matrix for the ring.
        """
        if self._rotation_matrix is None:
            self._rotation_matrix = self._get_rotation_matrix()
        return self._rotation_matrix

    def _get_rotation_matrix(self) -> npt.NDArray[np.floating]:
        """Calculates and returns the rotation matrix for the ring.

        Returns
        -------
        transform : npt.NDArray[np.floating]
            The rotation matrix for the ring.
        """
        scalar_part = self._rotation.w
        x_part = self._rotation.x
        y_part = self._rotation.y
        z_part = self._rotation.z
        # Row 1
        matrix_xx = -1 + 2 * x_part**2 + 2 * scalar_part**2
        matrix_xy = 2 * (x_part * y_part - z_part * scalar_part)
        matrix_xz = 2 * (x_part * z_part + y_part * scalar_part)
        # Row 2
        matrix_yx = 2 * (x_part * y_part + z_part * scalar_part)
        matrix_yy = -1 + 2 * y_part**2 + 2 * scalar_part**2
        matrix_yz = 2 * (y_part * z_part - x_part * scalar_part)
        # Row 3
        matrix_zx = 2 * (x_part * z_part - y_part * scalar_part)
        matrix_zy = 2 * (y_part * z_part + x_part * scalar_part)
        matrix_zz = -1 + 2 * z_part**2 + 2 * scalar_part**2
        transform = np.array(
            [
                [matrix_xx, matrix_xy, matrix_xz],
                [matrix_yx, matrix_yy, matrix_yz],
                [matrix_zx, matrix_zy, matrix_zz],
            ],
            dtype=NumpyFloatType,
        )
        return transform

    def get_rotated_inertia_tensor(self) -> npt.NDArray[np.floating]:
        """Calculates and returns the inertia tensor of the rotated ring.

        Returns
        -------
        inertia_tensor : npt.NDArray[np.floating]
            The inertia tensor of the rotated ring.
        """
        if self._rotated_inertia_tensor is None:
            inertia_tensor_body = self._inertia_tensor
            rotation_matrix = self.get_rotation_matrix()
            self._rotated_inertia_tensor = (
                rotation_matrix @ inertia_tensor_body @ rotation_matrix.T
            )

        return self._rotated_inertia_tensor

    def __repr__(self) -> str:
        return (
            f"Ring(radius={self.radius:.2f}, scale_width={self.scale_width:.2f}, scale_height={self.scale_height:.2f}, "
            f"mass={self.mass:.2e}, perimeter={self.perimeter:.2e}, eccentricity={self.eccentricity:.3f}, "
            f"moment_of_inertia_x={self.moment_of_inertia_x:.2e}, moment_of_inertia_y={self.moment_of_inertia_y:.2e}, "
            f"moment_of_inertia_z={self.moment_of_inertia_z:.2e})"
        )

    @staticmethod
    def calculate_moment_of_inertia_and_perimeter_ellipses(
        mass: float, radius: float, semi_major_axis: float, semi_minor_axis: float
    ) -> tuple[float, float, float, float]:
        """Calculates the moment of inertia and perimeter of an elliptic ring.

        Parameters
        ----------
        mass : float
            The mass of the elliptic ring.
        radius : float
            The radius of the elliptic ring.
        semi_major_axis : float
            The length of the semi-major axis of the elliptic ring, divided by the `radius`.
        semi_minor_axis : float
            The length of the semi-minor axis of the elliptic ring, divided by the `radius`.

        Returns
        -------
        moment_of_inertia_x : float
            The moment of inertia of the elliptic ring about the x axis.
        moment_of_inertia_y : float
            The moment of inertia of the elliptic ring about the y axis.
        moment_of_inertia_z : float
            The moment of inertia of the elliptic ring about the z axis.
        perimeter : float
            The perimeter of the elliptic ring.

        Note
        ----
        The equations for the moments of inertia were calculated in part by Mathematica.
        """
        # Commonly used constants
        ecc_squared = 1 - semi_minor_axis**2 / semi_major_axis**2
        # Calculate integrals
        ellipk_ecc2 = ellipk(ecc_squared)
        ellipe_ecc2 = ellipe(ecc_squared)
        # Calculate perimeter
        perimeter = 4 * semi_major_axis * radius * ellipe_ecc2
        # Calculate moments of inertia
        moment_of_inertia_z = (
            mass * semi_major_axis**2 * radius**2 * ellipe_ecc2 / ellipk_ecc2
        )
        # Moment of inertia about the minor axis
        second_ecc_squared = ecc_squared / (1 - ecc_squared)
        moment_of_inertia_minor = (
            mass
            * (ellipe(-second_ecc_squared) - ellipk(-second_ecc_squared))
            / ellipk_ecc2
            * semi_major_axis
            * semi_minor_axis
            * radius**2
            / ecc_squared
        )
        moment_of_inertia_major = moment_of_inertia_z - moment_of_inertia_minor
        return (
            moment_of_inertia_major,
            moment_of_inertia_minor,
            moment_of_inertia_z,
            perimeter,
        )

    @staticmethod
    def calculate_moment_of_inertia_and_perimeter_circles(
        mass: float, radius: float, semi_major_axis: float, semi_minor_axis: float
    ) -> tuple[float, float, float, float]:
        """Calculates the moment of inertia and perimeter of a circular ring.

        Parameters
        ----------
        mass : float
            The mass of the circular ring.
        radius : float
            The radius of the circular ring.
        semi_major_axis : float
            The length of the semi-major axis of the circular ring, divided by the `radius`.
        semi_minor_axis : float
            The length of the semi-minor axis of the circular ring, divided by the `radius`.

        Returns
        -------
        moment_of_inertia_x : float
            The moment of inertia of the elliptic ring about the x axis.
        moment_of_inertia_y : float
            The moment of inertia of the elliptic ring about the y axis.
        moment_of_inertia_z : float
            The moment of inertia of the elliptic ring about the z axis.
        perimeter : float
            The perimeter of the elliptic ring.

        Notes
        -----
        This function assumes that the semi-major axis and semi-minor axis are equal, and will raise an assertion error
        if this is not true.
        """
        assert semi_major_axis == semi_minor_axis
        scaled_radius = semi_major_axis * radius
        moment_of_inertia_z = mass * scaled_radius**2
        perimeter = 2 * np.pi * scaled_radius
        return (
            moment_of_inertia_z / 2,
            moment_of_inertia_z / 2,
            moment_of_inertia_z,
            perimeter,
        )

    def calculate_particle_potential_array(
        self, pos: npt.NDArray[np.floating], num_particles: int, softening_length: float
    ) -> npt.NDArray[np.floating]:
        """Calculates the gravitational potential at the given array of points (x, y) due to the ring.

        Note
        ----
        The points must be given as column vectors in the array. This must have a shape of (3, N) where N is the number
        of points to calculate the force at with the first row being the x coordinates and the second row being the y
        coordinates.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational potential at.
        num_particles : int
            The number of particles to sample from the ring as part of the particle approximation.
        softening_length : float
            The softening length to use in the potential calculation.

        Returns
        -------
        potential : npt.NDArray[np.floating]
            The gravitational potential at the given points. This is a 1D array of length N.
        """
        # pos must be an array of postion column vectors
        assert pos.shape[0] == 3
        num_points = pos.shape[1]
        potential = np.zeros(num_points, dtype=NumpyFloatType)
        particle_mass = self._mass / num_particles
        particle_locations = self.autosample_points()
        if softening_length == 0.0:
            distances = spatial.distance.cdist(
                pos.T, particle_locations.T, metric="euclidean"
            )
            softening_kernel = -1 / distances
        else:
            squared_distances = spatial.distance.cdist(
                pos.T, particle_locations.T, metric="sqeuclidean"
            )
            softening_kernel = -1 / np.sqrt(squared_distances + softening_length**2)
        potential = (G * particle_mass * softening_kernel).sum(axis=1)
        return potential

    def calculate_particle_force_approx_array(
        self,
        pos: npt.NDArray[np.floating],
        test_mass: float,
        num_particles: int,
        softening_length: float,
    ) -> npt.NDArray[np.floating]:
        """Calculates the gravitational force at the given array of points (x, y) due to the ring.

        Note
        ----
        The points must be given as column vectors in the array. This must have a shape of (3, N) where N is the number
        of points to calculate the force at with the first row being the x coordinates and the second row being the
        y coordinates.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational force at.
        test_mass : float
            The mass of the test particle.
        num_particles : int
            The number of particles to sample from the ring as part of the particle approximation.
        softening_length : float
            The softening length to use.

        Returns
        -------
        force : npt.NDArray[np.floating]
            The gravitational force at the given points. This is a 2D array of shape (3, N).
        """
        # pos must be an array of postion column vectors
        assert pos.shape[0] == 3
        force: npt.NDArray[np.floating] = np.zeros_like(pos, dtype=NumpyFloatType)
        num_points = pos.shape[1]
        evaluation_points = np.tile(pos, 3)
        evaluation_points[0, num_points : (2 * num_points)] += EPSILON
        evaluation_points[1, (2 * num_points) : (3 * num_points)] += EPSILON
        potential = self.calculate_particle_potential_array(
            evaluation_points,
            num_particles,
            softening_length,
        )
        base_potential = potential[:num_points]
        forward_x_potential = potential[num_points : (2 * num_points)]
        forward_y_potential = potential[(2 * num_points) : (3 * num_points)]
        acceleration_x = -(forward_x_potential - base_potential) / EPSILON
        acceleration_y = -(forward_y_potential - base_potential) / EPSILON
        force[0, :] = test_mass * acceleration_x
        force[1, :] = test_mass * acceleration_y
        return force

    def calculate_particle_torque_approx_array(
        self,
        pos: npt.NDArray[np.floating],
        test_mass: float,
        num_particles: int,
        softening_length: float,
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        """Calculates the torque induced by gravity at the given array of points (x, y) due to the ring.

        Parameters
        ----------
        pos : npt.NDArray[np.floating]
            The array of points to calculate the gravitational torque at.
        test_mass : float
            The mass of the particles.
        num_particles : int
            The number of particles to sample from the ring as part of the particle approximation.
        softening_length : float
            The softening length to use.

        Returns
        -------
        total_torque : npt.NDArray[np.floating]
            The total torque on the ring.
        torques : npt.NDArray[np.floating]
            The torques on the ring at each sampled point.
        points : npt.NDArray[np.floating]
            The sampled points on the ring.
        """
        force = self.calculate_particle_force_approx_array(
            pos, test_mass, num_particles, softening_length
        )
        torques = np.cross(pos.T, force.T)
        total_torque: npt.NDArray[np.floating] = torques.sum(axis=0)
        assert total_torque.shape == (3,)
        return (total_torque, torques, pos)

    def get_num_particles(self) -> int:
        match self.discretisation.get_type():
            case RingDiscretisationType.UNIFORM_NUM_PARTICLES:
                num_particles = self.discretisation.get_num_particles()
            case RingDiscretisationType.UNIFORM_ARC_LENGTH:
                arc_length = self.discretisation.get_arc_length()
                num_particles: int = int(self.perimeter / arc_length)
            case _ as unexpected_type:
                raise ValueError(
                    f"Unexpected ring discretisation type {unexpected_type}"
                )
        return num_particles

    def get_particle_mass(self) -> float:
        return self.mass / self.get_num_particles()

    def sample_points(
        self, num_particles: int, as_kpc: bool = False
    ) -> npt.NDArray[np.floating]:
        """Samples particles uniformly on the elliptic ring.

        Note
        ----
        The points are uniformly distributed along the perimeter of the ellipse, that is, each point is separated by
        the same arc length.

        Parameters
        ----------
        num_particles : int
            The number of particles to sample on the ring.

        Returns
        -------
        position_vector : npt.NDArray[np.floating]
            The position of every sampled point on the elliptic ring. This is a 2D array of shape (3, N).
        tangent_vector : npt.NDArray[np.floating]
            The normalised tangent vector at every sampled point with respect to the elliptic ring. This is a 2D array
            of shape (3, N).
        """
        if num_particles not in self._sampled_points:
            self._sampled_points[num_particles] = self._sample_points(num_particles)
        if num_particles not in self._rotated_points:
            self._rotated_points[num_particles] = (
                self.get_rotation_matrix() @ self._sampled_points[num_particles]
            )
        conversion_factor = self._disk_scale_length if as_kpc else 1
        return conversion_factor * self._rotated_points[num_particles]

    # TODO: Removed tangent vector functionality as it is hard to implement right now for 3D
    def autosample_points(self, as_kpc: bool = False) -> npt.NDArray[np.floating]:
        """Samples points uniformly on the elliptic ring.

        Note
        ----
        The points are uniformly distributed along the perimeter of the ellipse, that is, each point is separated by
        the same arc length.

        Parameters
        ----------
        num_points : int
            The number of points to sample on the ring.

        Returns
        -------
        position_vector : npt.NDArray[np.floating]
            The position of every sampled point on the elliptic ring. This is a 2D array of shape (3, N).
        tangent_vector : npt.NDArray[np.floating]
            The normalised tangent vector at every sampled point with respect to the elliptic ring. This is a 2D array
            of shape (3, N).
        """
        num_particles = self.get_num_particles()
        return self.sample_points(num_particles, as_kpc=as_kpc)

    def _sample_points(self, num_points: int) -> npt.NDArray[np.floating]:
        """Samples points uniformly on the elliptic ring.

        Note
        ----
        The points are uniformly distributed along the perimeter of the ellipse, that is, each point is separated by
        the same arc length.

        Parameters
        ----------
        num_points : int
            The number of points to sample on the ring.

        Returns
        -------
        position_vector : npt.NDArray[np.floating]
            The position of every sampled point on the elliptic ring. This is a 2D array of shape (3, N).
        tangent_vector : npt.NDArray[np.floating]
            The normalised tangent vector at every sampled point with respect to the elliptic ring. This is a 2D array
            of shape (3, N).
        """
        if self.semi_major_axis != self.semi_minor_axis:
            normalised_perimeter = self.perimeter / (self.semi_major_axis * self.radius)
            arc_size = normalised_perimeter / num_points
            arc_divisions = np.arange(num_points) * arc_size
            estimate = (2 * np.pi / num_points) * np.arange(num_points)
            result = scipy.optimize.root(
                lambda phi: ellipeinc(phi, self.get_eccentricity_squared())
                - arc_divisions,
                estimate,
            )
            if not result.success:
                raise Exception("Could not find points on ellipse!")
            theta = result.x
        # Special case for circle
        else:
            theta = np.linspace(
                0, 2 * np.pi, num_points, dtype=NumpyFloatType, endpoint=False
            )

        # Position of every sampled point on elliptic ring
        unrotated_x = self._radius * self.scale_width * np.cos(theta)
        unrotated_y = self._radius * self.scale_height * np.sin(theta)
        unrotated_z = self._bend_amplitude * np.cos(2 * theta)
        position_vector = np.vstack((unrotated_x, unrotated_y, unrotated_z))
        # Normalised tangent vector at every sampled point with respect to the elliptic ring
        # TODO: Placeholder for z component of tangent vector. A correct version would account for rotations
        # tan_z = np.zeros_like(tan_x)
        # tan_mag = np.hypot(tan_x, tan_y)
        # tangent_vector = np.vstack((tan_x / tan_mag, tan_y / tan_mag, tan_z / tan_mag))

        # Sample points
        self.sample_point_count += 1

        return position_vector

    def rotate_about_axis_angle(
        self, axis: tuple[float, float, float] | npt.NDArray[np.floating], angle: float
    ):
        """Rotates the ring about the given axis by the given angle.

        Parameters
        ----------
        axis : tuple[float, float, float] | npt.NDArray[np.floating]
            The axis to rotate the ring about.
        angle : float
            The angle to rotate the ring by.
        """
        assert len(axis) == 3
        rotation_quaternion = Quaternion.from_axis_angle(axis, angle)
        self.set_rotation_quaternion(rotation_quaternion * self._rotation)

    def set_rotation_quaternion(self, rotation: Quaternion):
        """Sets the rotation quaternion of the ring.

        Parameters
        ----------
        rotation : Quaternion
            The rotation quaternion to set.
        """
        self._rotation = rotation
        # Invalidate cache attributes
        self._rotated_points = {}
        self._rotation_matrix = None
        self._rotated_inertia_tensor = None
