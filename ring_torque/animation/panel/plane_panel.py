# Standard libraries
from enum import Enum
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib import axes
from matplotlib import colors as mplcolors
from matplotlib.lines import Line2D
import numpy as np
from numpy import typing as npt

# Internal libraries
from .common import (
    PanelInterface,
    LengthUnit,
    RingAnimationTraceOptions,
    RingAnimationData,
    trace_density_wave,
)
from ring_torque.system import RingSystem


class PhysicalPlane(Enum):
    """The different planes that can be plotted.

    Variants
    --------
    XY
        The XY plane.
    YZ
        The YZ plane.
    XZ
        The XZ plane.
    """

    XY = "XY"
    YZ = "YZ"
    XZ = "XZ"

    def to_title(self) -> str:
        """Returns the title of the plane."""
        match self:
            case PhysicalPlane.XY:
                return r"$x-y$ plane"
            case PhysicalPlane.YZ:
                return r"$y-z$ plane"
            case PhysicalPlane.XZ:
                return r"$x-z$ plane"

    def project(
        self, points: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Projects the given points (3D) onto the plane (2D).

        Parameters
        ----------
        points : npt.NDArray[np.floating]
            The points to project. This is assumed to be a 3xN array where N is the number of points.

        Returns
        -------
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
            The projected points. The first array is the 'horizontal' axis and the second array is the 'vertical' axis.
        """
        match self:
            case PhysicalPlane.XY:
                return (points[0, :], points[1, :])
            case PhysicalPlane.YZ:
                return (points[1, :], points[2, :])
            case PhysicalPlane.XZ:
                return (points[0, :], points[2, :])


class PlanePanel(PanelInterface):
    """A panel that displays a plane of a ring system."""

    LINE_WIDTH: float = 2.0

    def __init__(
        self,
        axis: axes.Axes,
        max_extent: float,
        plane: PhysicalPlane,
        unit: LengthUnit,
        show_points: bool,
        show_axes: bool,
        trace_options: RingAnimationTraceOptions,
        color_map: mplcolors.Colormap,
    ):
        """Initialise a new PlanePanel.

        Parameters
        ----------
        axis : axes.Axes
            The axis to draw to.
        max_extent : float
            The maximum extent of the axis.
        plane : str
            The plane to plot.
        unit : LengthUnit
            The unit of the axis.
        show_points : bool
            Whether to show the points.
        show_axes : bool
            Whether to show the axes.
        trace_options : RingAnimationTraceOptions
            The trace options.
        color_map : mplcolors.Colormap
            The color map to use.
        """
        super(PanelInterface, self).__init__()
        self._axis: axes.Axes = axis
        self._max_extent: float = max_extent
        self._plane: PhysicalPlane = plane
        self._unit: LengthUnit = unit
        self._show_points: bool = show_points
        self._show_axes: bool = show_axes
        self._ring_lines_most: list[Line2D] = []
        self._ring_lines_close: list[Line2D] = []
        self._cmap: mplcolors.Colormap = color_map
        self._trace_options: RingAnimationTraceOptions = trace_options
        self._trace_lines: tuple[Line2D, Line2D] | None = None

    def initialise(self, animation_data: RingAnimationData):
        disc: RingSystem = animation_data.ring_system
        self._axis.set_xlabel(rf"$x\,({self._unit.to_latex()})$")
        self._axis.set_ylabel(rf"$y\,({self._unit.to_latex()})$")
        self._axis.set_aspect("equal")
        self._axis.set_xlim(-self._max_extent, self._max_extent)
        self._axis.set_ylim(-self._max_extent, self._max_extent)
        if self._show_axes:
            self._axis.set_title(self._plane.to_title())
        # Draw the rings
        min_radius: float = disc.min_radius
        max_radius: float = disc.max_radius
        marker_style = "o" if self._show_points else ""
        for _, current_ring in reversed(list(enumerate(disc.rings))):
            color_value = (
                (current_ring.radius - min_radius) / (max_radius - min_radius)
                if not np.isclose(max_radius, min_radius)
                else 0.5
            )
            ring_color = self._cmap(color_value)
            points = current_ring.autosample_points(as_kpc=self._unit == LengthUnit.KPC)
            horizontal_points, vertical_points = self._plane.project(points)
            self._ring_lines_most.append(
                self._axis.plot(
                    horizontal_points,
                    vertical_points,
                    color=ring_color,
                    linestyle="-",
                    linewidth=PlanePanel.LINE_WIDTH,
                    marker=marker_style,
                )[0]
            )
            self._ring_lines_close.append(
                self._axis.plot(
                    (horizontal_points[-1], horizontal_points[0]),
                    (vertical_points[-1], vertical_points[0]),
                    color=ring_color,
                    linestyle="-",
                    linewidth=PlanePanel.LINE_WIDTH,
                    marker=marker_style,
                )[0]
            )
        # Reverse the order of the rings so that the innermost ring is on top
        self._ring_lines_most.reverse()
        self._ring_lines_close.reverse()

        # Trace is only supported for the XY plane
        if self._plane != PhysicalPlane.XY:
            return

        # Trace the rings
        match self._trace_options:
            case RingAnimationTraceOptions.TRACE_WITH_LINES:
                trace_marker = ""
                trace_style = "-"
            case RingAnimationTraceOptions.TRACE_WITH_POINTS:
                trace_marker = "o"
                trace_style = ""
            case _:
                return
        wave_points = trace_density_wave(disc.rings, self._unit == LengthUnit.KPC)
        (trace_line,) = self._axis.plot(
            wave_points[0, :],
            wave_points[1, :],
            color="red",
            marker=trace_marker,
            linestyle=trace_style,
            linewidth=3,
        )
        (other_trace_line,) = self._axis.plot(
            -wave_points[0, :],
            -wave_points[1, :],
            color="red",
            marker=trace_marker,
            linestyle=trace_style,
            linewidth=3,
        )
        self._trace_lines = (trace_line, other_trace_line)

    def draw(
        self,
        frame: int,
        animation_data: RingAnimationData,
    ) -> Sequence[Artist]:
        # Frame is not used
        _ = frame
        disc: RingSystem = animation_data.ring_system
        assert len(self._ring_lines_most) == len(disc.rings)
        assert len(self._ring_lines_close) == len(disc.rings)
        redrawn_artists: list[Artist] = []
        # Update the rings' rotation and patches
        for (
            _,
            (current_ring, most_line, close_line),
        ) in enumerate(zip(disc.rings, self._ring_lines_most, self._ring_lines_close)):
            # Draw new points
            points = current_ring.autosample_points(as_kpc=self._unit == LengthUnit.KPC)
            horizontal_points, vertical_points = self._plane.project(points)
            most_line.set_data(horizontal_points, vertical_points)
            close_line.set_data(
                (horizontal_points[-1], horizontal_points[0]),
                (vertical_points[-1], vertical_points[0]),
            )
        redrawn_artists.extend(self._ring_lines_most)
        redrawn_artists.extend(self._ring_lines_close)
        # Trace the rings
        if self._trace_options.will_trace() and self._trace_lines is not None:
            wave_points = trace_density_wave(disc.rings, self._unit == LengthUnit.KPC)
            self._trace_lines[0].set_data(wave_points[0, :], wave_points[1, :])
            self._trace_lines[1].set_data(-wave_points[0, :], -wave_points[1, :])
            redrawn_artists.extend(self._trace_lines)

        return redrawn_artists

    def turn_off_axes(self):
        self._axis.set_axis_off()
