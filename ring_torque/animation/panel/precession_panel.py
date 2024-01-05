# Standard libraries
from typing import cast, Sequence

# External libraries
from matplotlib.artist import Artist
from matplotlib import axes
from matplotlib.lines import Line2D
import numpy as np
from numpy import typing as npt


# Internal libraries
from .common import PanelInterface, LengthUnit, RingAnimationData
from ring_torque.system import RingSystem


class PrecessionRatePanel(PanelInterface):
    """A panel that displays the precession rate of a series of rings."""

    PADDING_FACTOR: float = 0.1

    def __init__(
        self,
        axis: axes.Axes,
        unit: LengthUnit,
        show_axes: bool,
    ):
        self._axis: axes.Axes = axis
        self._show_axes: bool = show_axes
        self._unit: LengthUnit = unit
        self._line: Line2D | None = None

    def initialise(self, animation_data: RingAnimationData):
        disc: RingSystem = animation_data.ring_system
        min_radius: float = min(disc.radii)
        max_radius: float = max(disc.radii)
        # TODO: Maybe we can parameterise the units?
        angular_velocities: npt.NDArray[np.floating] = (
            animation_data.angular_velocities / (2 * np.pi)
        )
        initial_omega: npt.NDArray[np.floating] = angular_velocities[2, :, 0]
        min_omega: float = cast(float, np.nanmin(angular_velocities[2, :]))
        max_omega: float = cast(float, np.nanmax(angular_velocities[2, :]))
        radius_extent: float = max_radius - min_radius
        omega_extent: float = max_omega - min_omega
        self._axis.set_xlabel(rf"$r\,({self._unit.to_latex()})$")
        self._axis.set_ylabel(r"$\Omega_{p}\,{}(\text{cycles}/\mathrm{Gyr})$")
        radius_padding: float = PrecessionRatePanel.PADDING_FACTOR * radius_extent
        self._axis.set_xlim(
            min_radius - radius_padding,
            max_radius + radius_padding,
        )
        omega_padding: float = PrecessionRatePanel.PADDING_FACTOR * omega_extent
        self._axis.set_ylim(
            min_omega - omega_padding,
            max_omega + omega_padding,
        )
        self._axis.axhline(0, color="black", linestyle="--")
        if self._show_axes:
            self._axis.set_title("Precession rate")

        # Current precession rate of rings
        (precession_line,) = self._axis.plot(
            disc.radii,
            initial_omega,
            color="tab:red",
            label="Precession rate",
        )
        self._line = precession_line
        # Initial precession rate of rings
        (initial_precession,) = self._axis.plot(
            disc.radii,
            initial_omega,
            color="tab:blue",
            linestyle="--",
            label="Initial precession",
        )
        if self._show_axes:
            self._axis.legend(
                handles=[
                    precession_line,
                    initial_precession,
                ],
                loc="upper right",
            )

    def draw(self, frame: int, animation_data: RingAnimationData) -> Sequence[Artist]:
        assert self._line is not None
        actual_frame = frame // animation_data.save_interval
        # Convert from rad/Gyr to cycles/Gyr
        current_precession = animation_data.angular_velocities[2, :, actual_frame] / (
            2 * np.pi
        )
        self._line.set_ydata(current_precession)
        return [self._line]

    def turn_off_axes(self):
        return self._axis.set_axis_off()
