"""This module contains the Panel class, which is used to draw to a single subplot."""

# Standard libraries
from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import Sequence

# External libraries
from matplotlib.artist import Artist
import numpy as np
from numpy import typing as npt

# Internal libraries
from ring_torque.animation.common import RingAnimationData
from ring_torque.ring import Ring
from ring_torque.common import NumpyFloatType


log = logging.getLogger(__name__)


class PanelInterface(ABC):
    """An interface for a panel in a multi-panel animation."""

    @abstractmethod
    def initialise(self, animation_data: RingAnimationData) -> None:
        """Initialises the panel. This is where the axes limits and labels should be set and artists should be created.

        Parameters
        ----------
        animation_data : RingAnimationData
            The animation data.
        """
        raise NotImplementedError("initialise() not implemented")

    @abstractmethod
    def draw(self, frame: int, animation_data: RingAnimationData) -> Sequence[Artist]:
        """Draws the panel for the given frame.

        Parameters
        ----------
        frame : int
            The frame to draw.
        animation_data : RingAnimationData
            The animation data.

        Returns
        -------
        Sequence[Artist]
            The artists that were drawn.
        """
        raise NotImplementedError("draw() not implemented")

    @abstractmethod
    def turn_off_axes(self) -> None:
        """Turns off the axes for the panel so that only the artists are visible."""
        raise NotImplementedError("turn_off_axes() not implemented")


class RingAnimationTraceOptions(Enum):
    """Enum class for defining trace options for ring animation."""

    NO_TRACE = "NO_TRACE"
    TRACE_WITH_LINES = "TRACE_WITH_LINES"
    TRACE_WITH_POINTS = "TRACE_WITH_POINTS"

    def will_trace(self) -> bool:
        """Whether or not the option traces the density wave.

        Returns
        -------
        bool
            Whether or not the option traces the density wave.
        """
        return self != RingAnimationTraceOptions.NO_TRACE

    def __str__(self) -> str:
        return self.value


class LengthUnit(Enum):
    """An enumeration of the different length units."""

    KPC = "kpc"
    SCALE_LENGTH = "scale length"

    def to_latex(self) -> str:
        """Returns the LaTeX representation of the unit.

        Returns
        -------
        str
            The LaTeX representation of the unit.
        """
        match self:
            case LengthUnit.KPC:
                return r"\mathrm{kpc}"
            case LengthUnit.SCALE_LENGTH:
                return r"R_{\mathrm{d}}"
            case _:
                raise ValueError(f"The {self} unit is currently not supported.")


# NOTE: This function was implemented incorectly which resulted in the points being returned always being the 0th
# particle.
# This function has been changed to reflect that i.e. this behaviour is now explicit.
# Even though this function is incorrect, it does trace out the density wave nicely for the case of no gravity and
# planar motion.
# There is a further optimisation where we can compute the distance between two paired points i.e. they have the same
# index and roughly the same angle for the two innermost rings, and then from that compute the shortest distance from
# the outer point for neighbouring rings. This allows us to generalise over ellipses that might have been rotated so
# that their particle indices are different. This does not help with the tilted and no gravity cases and in our
# configurations, this optimisation is redundant as it should always return the index we have hardcoded.
def trace_density_wave(
    rings: Sequence[Ring],
    rotate_clockwise: bool,
    as_kpc: bool = False,
) -> npt.NDArray[np.floating]:
    """Returns points that make up a density wave in a system of rings.

    Parameters
    ----------
    rings : Sequence[Ring]
        The rings to trace the density wave for. These rings are assumed to be ordered from innermost to outermost.
    rotate_clockwise : bool
        Whether or not the rings are rotating clockwise or not.
    as_kpc : bool
        Whether to return the points in kpc or in disk scale lengths.

    Returns
    -------
    npt.NDArray[np.floating]
        The points that make up the density wave.

    Notes
    -----
    This function assumes that the rings are ordered from innermost to outermost.
    """
    # Iterate over pairs of rings
    wave_points: npt.NDArray[np.floating] = np.full(
        (3, len(rings)), np.nan, dtype=NumpyFloatType
    )
    # NOTE: This is a hack as tracing out the density wave turns out to be very complicated in this setup.
    for ring_idx in range(len(rings)):
        ring = rings[ring_idx]
        num_particles = ring.get_num_particles()
        if rotate_clockwise:
            select_idx = num_particles // 8
        else:
            select_idx = num_particles - num_particles // 8
        points = ring.autosample_points(as_kpc=as_kpc)
        wave_points[:, ring_idx] = points[:, select_idx]
    return wave_points
