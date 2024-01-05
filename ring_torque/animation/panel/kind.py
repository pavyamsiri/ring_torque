# Standard libraries
from enum import Enum

# External libraries
from matplotlib.gridspec import SubplotSpec
from matplotlib.figure import Figure
from matplotlib import colors as mplcolors

# Internal libraries
from .common import (
    PanelInterface,
    LengthUnit,
    RingAnimationData,
    RingAnimationTraceOptions,
)
from .grid import DensityGridPanel, HeightGridPanel, PotentialGridPanel
from .plane_panel import PhysicalPlane, PlanePanel
from .pitch_angle_panel import PitchAnglePanel
from .precession_panel import PrecessionRatePanel


class RingAnimationPanelKind(Enum):
    """All the different variants of panels that display ring animation data.

    Variants
    --------
    XY
        The XY plane.
    YZ
        The YZ plane.
    XZ
        The XZ plane.
    OMEGA
        The precession rate.
    ANGLE
        The pitch angle.
    GRID_XYZ
        The height histogram of the XY plane.
    GRID_XYD
        The density histogram of the XY plane.
    GRID_XYP
        The potential in the XY plane.
    """

    XY = "XY"
    YZ = "YZ"
    XZ = "XZ"
    OMEGA = "OMEGA"
    ANGLE = "ANGLE"
    GRID_XYZ = "GRID_XYZ"
    GRID_XYD = "GRID_XYD"
    GRID_XYP = "GRID_XYP"

    def create(
        self,
        fig: Figure,
        plot_spec: SubplotSpec,
        animation_data: RingAnimationData,
        max_plot_extent: float,
        unit: LengthUnit,
        show_points: bool,
        show_axes: bool,
        skip_interpolation: bool,
        trace_options: RingAnimationTraceOptions,
        color_map: mplcolors.Colormap,
    ) -> PanelInterface:
        """Creates the panel.

        Parameters
        ----------
        fig : Figure
            The figure to add the panel to. This is needed to create axis.
        plot_spec : SubplotSpec
            The subplot spec for the panel.
        animation_data : RingAnimationData
            The animation data.
        max_plot_extent : float
            The maximum extent of the panel.
        unit : LengthUnit
            The unit to use for the panel.
        show_points : bool
            Whether to show the points.
        show_axes : bool
            Whether to show the axes.
        skip_interpolation : bool
            Skips grid interpolation. This greatly increases the animation speed.
        trace_options : RingAnimationTraceOptions
            The trace options.
        color_map : mplcolors.Colormap
            The color map to use.

        Returns
        -------
        PanelInterface
            The panel.
        """
        match self:
            case RingAnimationPanelKind.XY:
                axis = fig.add_subplot(plot_spec)
                plane = PlanePanel(
                    axis,
                    max_plot_extent,
                    PhysicalPlane.XY,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    color_map,
                )
                return plane
            case RingAnimationPanelKind.XZ:
                axis = fig.add_subplot(plot_spec)
                plane = PlanePanel(
                    axis,
                    max_plot_extent,
                    PhysicalPlane.XZ,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    color_map,
                )
                return plane
            case RingAnimationPanelKind.YZ:
                axis = fig.add_subplot(plot_spec)
                plane = PlanePanel(
                    axis,
                    max_plot_extent,
                    PhysicalPlane.YZ,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    color_map,
                )
                return plane
            case RingAnimationPanelKind.ANGLE:
                axis = fig.add_subplot(plot_spec)
                rotate_clockwise = animation_data.tangential_velocity < 0
                panel = PitchAnglePanel(
                    axis,
                    max_plot_extent,
                    unit,
                    rotate_clockwise,
                    show_axes,
                )
                return panel
            case RingAnimationPanelKind.OMEGA:
                axis = fig.add_subplot(plot_spec)
                panel = PrecessionRatePanel(
                    axis,
                    unit,
                    show_axes,
                )
                return panel
            case RingAnimationPanelKind.GRID_XYZ:
                axis = fig.add_subplot(plot_spec)
                panel = HeightGridPanel(
                    axis,
                    max_plot_extent,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    skip_interpolation,
                )
                return panel
            case RingAnimationPanelKind.GRID_XYD:
                axis = fig.add_subplot(plot_spec)
                panel = DensityGridPanel(
                    fig,
                    axis,
                    max_plot_extent,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    skip_interpolation,
                )
                return panel
            case RingAnimationPanelKind.GRID_XYP:
                axis = fig.add_subplot(plot_spec)
                panel = PotentialGridPanel(
                    fig,
                    axis,
                    max_plot_extent,
                    unit,
                    show_points,
                    show_axes,
                    trace_options,
                    skip_interpolation,
                )
                return panel
            case _:
                raise ValueError(f"The {self} panel is currently not supported.")
