# Standard libraries
from typing import Sequence

# External libraries
import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib import axes
from matplotlib import colors as mplcolors
from matplotlib import figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Internal libraries
from .common import DEFAULT_NUM_GRID_CELLS
from ring_torque.animation.panel.common import (
    PanelInterface,
    LengthUnit,
    RingAnimationTraceOptions,
    RingAnimationData,
    trace_density_wave,
)
from ring_torque.system import RingSystem


class PotentialGridPanel(PanelInterface):
    """A panel that displays the potential of a ring system in a histogram."""

    LINE_WIDTH: float = 2.0

    def __init__(
        self,
        fig: figure.Figure,
        axis: axes.Axes,
        max_extent: float,
        unit: LengthUnit,
        show_points: bool,
        show_axes: bool,
        trace_options: RingAnimationTraceOptions,
        skip_interpolation: bool,
    ):
        """Initialise a new PotentialGridPanel.

        Parameters
        ----------
        fig : figure.Figure
            The figure. This only used to draw the colorbar.
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
        """
        super(PanelInterface, self).__init__()
        self._fig: figure.Figure | None = fig
        self._axis: axes.Axes = axis
        self._max_extent: float = max_extent
        self._num_cells: int = DEFAULT_NUM_GRID_CELLS
        self._unit: LengthUnit = unit
        self._show_points: bool = show_points
        self._show_axes: bool = show_axes
        self._skip_interpolation: bool = skip_interpolation
        self._cmap: mplcolors.Colormap = mpl.colormaps["seismic"]
        self._potential_grid: AxesImage | None = None
        self._trace_options: RingAnimationTraceOptions = trace_options
        self._trace_lines: tuple[Line2D, Line2D] | None = None

        # Bin range
        self._bin_range: tuple[tuple[float, float], tuple[float, float]] = (
            (-self._max_extent, self._max_extent),
            (-self._max_extent, self._max_extent),
        )

    def initialise(self, animation_data: RingAnimationData):
        assert self._fig is not None
        disc: RingSystem = animation_data.ring_system
        self._axis.set_xlabel(rf"$x\,({self._unit.to_latex()})$")
        self._axis.set_ylabel(rf"$y\,({self._unit.to_latex()})$")
        self._axis.set_aspect("equal")
        self._axis.set_xlim(-self._max_extent, self._max_extent)
        self._axis.set_ylim(-self._max_extent, self._max_extent)
        if self._show_axes:
            self._axis.set_title(r"$x-y$ plane")
        # FIXME: Should be a parameter set by the animator or stored in animation_data
        FIXME_SOFTENING_LENGTH: float = 0.0

        point_grid = np.linspace(
            -self._max_extent, self._max_extent, DEFAULT_NUM_GRID_CELLS
        )
        mesh_x, mesh_y = np.meshgrid(point_grid, point_grid)
        pos_x_serial = mesh_x.flatten()
        pos_y_serial = mesh_y.flatten()
        pos_z_serial = np.zeros_like(pos_x_serial)
        pos = np.vstack((pos_x_serial, pos_y_serial, pos_z_serial))
        potential_map = disc.calculate_potential_on_ring_array(
            pos, FIXME_SOFTENING_LENGTH, -1
        ).reshape((DEFAULT_NUM_GRID_CELLS, DEFAULT_NUM_GRID_CELLS))
        # Plot the height map
        self._potential_grid = self._axis.imshow(
            potential_map.T,
            origin="lower",
            extent=(
                -self._max_extent,
                self._max_extent,
                -self._max_extent,
                self._max_extent,
            ),
            cmap="viridis",
            interpolation="nearest",
        )
        # Draw color bar
        divider = make_axes_locatable(self._axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        assert self._fig is not None
        cbar = self._fig.colorbar(self._potential_grid, cax=cax, orientation="vertical")
        cbar.set_label("Potential")
        # Remove fig handle
        self._fig = None
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
        _ = frame
        disc: RingSystem = animation_data.ring_system
        assert self._potential_grid is not None
        # Collect the redrawn artists
        redrawn_artists: list[Artist] = []
        # FIXME: Should be a parameter set by the animator or stored in animation_data
        FIXME_SOFTENING_LENGTH: float = 0.0

        point_grid = np.linspace(
            -self._max_extent, self._max_extent, DEFAULT_NUM_GRID_CELLS
        )
        mesh_x, mesh_y = np.meshgrid(point_grid, point_grid)
        pos_x_serial = mesh_x.flatten()
        pos_y_serial = mesh_y.flatten()
        pos_z_serial = np.zeros_like(pos_x_serial)
        pos = np.vstack((pos_x_serial, pos_y_serial, pos_z_serial))
        potential_map = disc.calculate_potential_on_ring_array(
            pos, FIXME_SOFTENING_LENGTH, -1
        ).reshape((DEFAULT_NUM_GRID_CELLS, DEFAULT_NUM_GRID_CELLS))

        self._potential_grid.set_data(potential_map.T)
        redrawn_artists.append(self._potential_grid)

        # Trace the rings
        if self._trace_options.will_trace() and self._trace_lines is not None:
            wave_points = trace_density_wave(disc.rings, self._unit == LengthUnit.KPC)
            self._trace_lines[0].set_data(wave_points[0, :], wave_points[1, :])
            self._trace_lines[1].set_data(-wave_points[0, :], -wave_points[1, :])
            redrawn_artists.extend(self._trace_lines)

        return redrawn_artists

    def turn_off_axes(self):
        self._axis.set_axis_off()
