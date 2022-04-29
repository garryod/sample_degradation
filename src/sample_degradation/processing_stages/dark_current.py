from typing import Tuple, TypeVar

from numpy import dtype, empty, expand_dims, floating, ndarray
from numpy.ma import MaskedArray, masked_array
from pandas import array

from sample_degradation.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    subtract_uncertain,
    uncertain,
)

NumFrames = TypeVar("NumFrames", bound=int)
FrameWidth = TypeVar("FrameWidth", bound=int)
FrameHeight = TypeVar("FrameHeight", bound=int)


def correct_dark_current(
    frames: MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], Uncertain],
    count_times: ndarray[Tuple[NumFrames], dtype[floating]],
    base_dark_current: float,
    temporal_dark_current: float,
    flux_dependant_dark_current: float,
) -> MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], Uncertain]:
    """Correct by subtracting base, temporal and flux-dependant dark currents.

    Correct for incident dark current by subtracting a baselike, time dependant and a
    flux dependant count rate from frames, as detailed in section 3.3.6 of 'Everything
    SAXS: small-angle scattering pattern collection and correction'
    [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], Uncertain]): A
            stack of frames to be corrected and their uncertaintities.
        atemporal_dark_current (float): The dark current flux, irrespective of time.
        temporal_dark_current (float): The dark current flux, as a factor of time.
        flux_dependant_dark_current (float): The dark current flux, as a factor of
            incident flux.
        count_times (ndarray[Tuple[NumFrames], dtype[floating]]): The period over which
            photons are counted for each frame.

    Returns:
        MaskedArray[Tuple[NumFrames, FrameWidth, FrameHeight], Uncertain]: The
            corrected stack of frames and their updated uncertainties.
    """
    base = array([(base_dark_current, 0.0)], dtype=uncertain)
    temporal = expand_dims(empty(count_times.shape, dtype=uncertain), (-2, -1))
    temporal["nominal"] = expand_dims(temporal_dark_current * count_times, (-2, -1))
    temporal["uncertainty"] = 0
    flux_dependant = multiply_uncertain(
        array([(flux_dependant_dark_current, 0)], dtype=uncertain), frames
    )
    return masked_array(
        subtract_uncertain(
            subtract_uncertain(subtract_uncertain(frames.data, base), temporal),
            flux_dependant,
        ),
        frames.mask,
    )
