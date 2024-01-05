# Standard libraries
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib import axes
from matplotlib.lines import Line2D
import numpy as np
from numpy import typing as npt

# Internal libraries
from .common import (
    PanelInterface,
    LengthUnit,
    RingAnimationData,
    trace_density_wave,
)
from ring_torque.system import RingSystem
from ring_torque.common import NumpyFloatType


class PitchAnglePanel(PanelInterface):
    """A panel that displays the pitch angle of a spiral pattern."""

    def __init__(
        self,
        axis: axes.Axes,
        max_extent: float,
        unit: LengthUnit,
        rotate_clockwise: bool,
        show_axes: bool,
    ):
        """Initialise a new PitchAnglePanel.

        Parameters
        ----------
        axis : axes.Axes
            The axis to draw to.
        max_extent : float
            The maximum extent of the axis.
        unit : LengthUnit
            The length unit being used.
        rotate_clockwise : bool
            Whether the disc is rotating clockwise or anticlockwise.
        show_axes : bool
            Whether to show the axes.
        """
        super(PanelInterface, self).__init__()
        self._axis: axes.Axes = axis
        self._max_extent: float = max_extent
        self._line: Line2D | None = None
        self._rotate_clockwise: bool = rotate_clockwise
        self._show_axes: bool = show_axes
        self._unit: LengthUnit = unit

    def initialise(self, animation_data: RingAnimationData):
        disc: RingSystem = animation_data.ring_system
        self._axis.set_xlabel(rf"$r\,({self._unit.to_latex()})$")
        self._axis.set_ylabel(r"$\theta (\mathrm{deg})$")
        self._axis.set_xlim(0, 30)
        self._axis.set_ylim(0, 100)
        self._axis.axhline(90, color="black", linestyle="--")
        if self._show_axes:
            self._axis.set_title("Pitch angle")
        wave_points = trace_density_wave(
            disc.rings,
            self._rotate_clockwise,
            as_kpc=self._unit == LengthUnit.KPC,
        )
        mid_points, pitch_angles = calculate_pitch_angle(wave_points)
        (angle_line,) = self._axis.plot(
            mid_points, np.rad2deg(pitch_angles), color="red", linewidth=3
        )
        self._line = angle_line

    def draw(self, frame: int, animation_data: RingAnimationData) -> Sequence[Artist]:
        _ = frame
        disc: RingSystem = animation_data.ring_system
        assert self._line is not None
        wave_points = trace_density_wave(
            disc.rings,
            self._rotate_clockwise,
            as_kpc=self._unit == LengthUnit.KPC,
        )
        mid_points, pitch_angles = calculate_pitch_angle(wave_points)
        self._line.set_data(mid_points, np.rad2deg(pitch_angles))
        return [self._line]

    def turn_off_axes(self):
        return self._axis.set_axis_off()


def calculate_pitch_angle(
    points: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculates the pitch angle of a set of points.

    Parameters
    ----------
    points : npt.NDArray[np.floating]
        The points to calculate the pitch angle for.

    Returns
    -------
    radii : npt.NDArray[np.floating]
        The radii of the points.
    pitch_angles : npt.NDArray[np.floating]
        The pitch angles of the points.
    """
    num_points = points.shape[1]
    # Iterate through every pair of points
    radii: npt.NDArray[np.floating] = np.full(
        num_points - 1, np.nan, dtype=NumpyFloatType
    )
    pitch_angles: npt.NDArray[np.floating] = np.full(
        num_points - 1, np.nan, dtype=NumpyFloatType
    )
    for point_idx in range(num_points - 1):
        # Get the current and next point
        current_point = points[:, point_idx]
        next_point = points[:, (point_idx + 1)]
        # Gradient of the line between the two points
        segment_gradient = (next_point[1] - current_point[1]) / (
            next_point[0] - current_point[0]
        )
        mid_point = (current_point + next_point) / 2
        # Gradient of the tangent to the circle passing through the mid point
        tangent_gradient = -mid_point[0] / mid_point[1]
        # Calculate the acute angle between the two lines
        radii[point_idx] = np.linalg.norm(mid_point)
        pitch_angles[point_idx] = np.arctan2(
            np.abs(tangent_gradient - segment_gradient),
            1 + tangent_gradient * segment_gradient,
        )
    # Ensure pitch angles are acute
    obtuse_angles = pitch_angles > np.pi / 2
    pitch_angles[obtuse_angles] = np.pi - pitch_angles[obtuse_angles]
    return (radii, pitch_angles)
