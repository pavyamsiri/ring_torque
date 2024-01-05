"""This module contains helper functions that deal with simulation specific tasks."""


# Internal libraries
from ring_torque.ring import RingDiscretisation, RingDiscretisationType


def parse_discretisation(
    num_particles: int | None, arc_length: float | None
) -> RingDiscretisation:
    # As arc length has a default, we must test num particles first
    if num_particles is not None:
        discretisation = RingDiscretisation(
            num_particles, RingDiscretisationType.UNIFORM_NUM_PARTICLES
        )
    elif arc_length is not None:
        discretisation = RingDiscretisation(
            arc_length, RingDiscretisationType.UNIFORM_ARC_LENGTH
        )
    else:
        raise ValueError(
            "A uniform number of particles or a uniform arc length must be given to the simulate command!"
        )
    return discretisation
