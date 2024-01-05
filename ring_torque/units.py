import numpy as np

M_TO_KPC: float = 3.24077929e-20
"""The number of kpc in 1 m. Multiply a value in m by this to convert to kpc."""
KM_TO_KPC: float = 3.24077929e-17
"""The number of kpc in 1 km. Multiply a value in km by this to convert to kpc."""
S_TO_GYR: float = 3.1709792e-17
"""The number of Gyr in 1 second. Multiply a value in seconds by this to convert to Gyr."""
KG_TO_MSUN: float = 5.027854e-31
"""The number of solar masses in 1 kg. Multiply a value in kg by this to convert to solar masses."""
G_IN_SI: float = 6.67430e-11
"""The gravitational constant in SI units m^3/(kg * s^2)."""
G_IN_SIMULATION_UNITS: float = 1.0
"""The gravitational constant in simulation units. Set to 1."""
KM_PER_S_TO_KPC_PER_GYR: float = KM_TO_KPC / S_TO_GYR
"""The number of kpc/Gyr in 1 km/s. Multiply a value in km/s by this to convert to kpc/Gyr."""


def convert_velocity_to_simulation_units(velocity: float, scale_length: float) -> float:
    """Converts the given velocity in km/s to simulation units.

    Parameters
    ----------
    velocity : float
        The velocity in km/s to convert to simulation units.
    scale_length : float
        The scale length of the disk in kpc.

    Returns
    -------
    float
        The converted velocity.

    Notes
    -----
    Simulation units are defined as follows:
        - Length: 1 kpc
        - Time: 1e9 years
        - G: 1
    """
    # Convert km/s to Rd/Gyr
    conversion_factor = KM_PER_S_TO_KPC_PER_GYR / scale_length
    return conversion_factor * velocity


def convert_mass_to_simulation_units(mass: float, scale_length: float) -> float:
    """Converts the given mass in solar masses to simulation units.

    Parameters
    ----------
    mass : float
        The mass in solar masses to convert to simulation units.
    scale_length : float
        The scale length of the disk in kpc.

    Returns
    -------
    float
        The converted mass.

    Notes
    -----
    Simulation units are defined as follows:
        - Length: 1 kpc
        - Time: 1e9 years
        - G: 1
    """
    # Gravitational constant in Rd^3/(Msun * Gyr^2)
    m_to_rd = M_TO_KPC / scale_length
    gravitational_constant = G_IN_SI * m_to_rd**3 / (KG_TO_MSUN * S_TO_GYR**2)
    # Convert km to Rd
    distance_conversion_factor = KM_TO_KPC / scale_length
    # Convert solar masses from Msun to Rd^3/Gyr^2
    conversion_factor = np.sqrt(
        distance_conversion_factor**3 / (gravitational_constant * S_TO_GYR**2)
    )
    return conversion_factor * mass


def convert_velocity_to_kms(velocity: float, scale_length: float) -> float:
    """Converts the given velocity in simulation units to km/s.

    Parameters
    ----------
    velocity : float
        The velocity in simulation units to convert to km/s.
    scale_length : float
        The scale length of the disk in kpc.

    Returns
    -------
    float
        The converted velocity.

    Notes
    -----
    Simulation units are defined as follows:
        - Length: 1 kpc
        - Time: 1e9 years
        - G: 1
    """
    # Convert Rd/Gyr to km/s
    conversion_factor = scale_length / KM_PER_S_TO_KPC_PER_GYR
    return conversion_factor * velocity
