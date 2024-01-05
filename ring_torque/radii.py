"""Functions to generate radii for concentric elliptic rings."""

# External libraries
import numpy as np

# Internal libraries
from ring_torque.common import NumpyFloatType
from ring_torque.common import setup_logging


def calculate_noncrossing_radii(
    num_rings: int,
    base_radius: float,
    separation_ratio: float,
    scale_width: float,
    scale_height: float,
) -> list[float]:
    """Calculate radii that produce non-crossing rings.

    Parameters
    ----------
    num_rings : int
        Number of rings to generate.
    base_radius : float
        Radius of the innermost ring.
    separation_ratio : float
        Ratio of the separation between the semi-major axis length of an inner ring the semi-minor axis length of its next
        outermost ring.
    scale_width : float
        The scale width of the elliptic rings.
    scale_height : float
        The scale height of the elliptic rings.

    Returns
    -------
    radii : list[float]
        A list of radii that produce non-crossing rings of the given eccentricity.
    """
    exponent_base = (
        separation_ratio
        * max(scale_width, scale_height)
        / min(scale_width, scale_height)
        * np.ones(num_rings, dtype=NumpyFloatType)
    )
    radii = list(base_radius * np.power(exponent_base, np.arange(num_rings)))
    return radii


def calculate_uniform_radii(
    num_rings: int, base_radius: float, radius_separation: float
) -> list[float]:
    """Generate radii that are uniformly spaced, starting from the base radius.

    Parameters
    ----------
    num_rings : int
        Number of rings to generate.
    base_radius : float
        Radius of the innermost ring.
    radius_separation : float
        Separation between the radii of adjacent rings.

    Returns
    -------
    radii : list[float]
        A list of radii that are evenly spaced from `base_radius` to `base_radius + num_rings * radius_separation`.
    """
    radii = list(
        np.arange(
            base_radius, base_radius + num_rings * radius_separation, radius_separation
        )
    )
    assert len(radii) == num_rings
    return radii


def calculate_uniform_radii_to_max_radius(
    num_rings: int, base_radius: float, max_radius: float
) -> list[float]:
    """Generate radii that are uniformly spaced in the range [`base_radius`, `max_radius`].

    Parameters
    ----------
    num_rings : int
        Number of rings to generate.
    base_radius : float
        Radius of the innermost ring.
    max_radius : float
        Radius of the outermost ring.

    Returns
    -------
    radii : list[float]
        A list of radii that are evenly spaced from `base_radius` to `max_radius`.
    """
    radii = list(np.linspace(base_radius, max_radius, num_rings, endpoint=True))
    return radii


def main():
    rconsole = setup_logging(None)
    num_rings = 5
    base_radius = 3
    scale_width = 1
    scale_height = 2
    rconsole.print(
        f"Generating {num_rings} rings with a base radius = {base_radius} and a = {scale_width}, b = {scale_height}"
    )

    radius_ratio = 1
    noncrossing_radii = calculate_noncrossing_radii(
        num_rings, base_radius, radius_ratio, scale_width, scale_height
    )
    rconsole.print(f"Noncrossing rings with radius ratio = {radius_ratio}")
    rconsole.print(noncrossing_radii)

    radius_separation = base_radius
    uniform_radii = calculate_uniform_radii(num_rings, base_radius, radius_separation)
    rconsole.print(f"Uniform rings with radius separation = {radius_separation}")
    rconsole.print(uniform_radii)


if __name__ == "__main__":
    main()
