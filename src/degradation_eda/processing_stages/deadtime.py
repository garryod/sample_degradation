from typing import Any, TypeVar

from numpy import atleast_1d, dtype, empty, exp, expand_dims, floating, ndarray
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import Uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def correct_deadtime(
    frames: MaskedArray[FramesShape, Uncertain],
    count_times: ndarray[TimesShape, dtype[floating]],
    minimum_pulse_separation: float,
    minimum_arrival_separation: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Correct for detector deadtime by scaling counts to account for overlapping events.

    Correct for detector deadtime by scaling photon counts according to the liklihood
    of overlapping events, as detailed in section 3.3.4 of 'Everything SAXS: small-
    angle scattering pattern collection and correction'
    [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of frames to be corrected.
        count_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are counted for each frame.
        minimum_pulse_separation (float): The minimum time difference required between
            a prior pulse and the current pulse for the current pulse to be recorded
            correctly.
        minimum_arrival_separation (float): The minimum time difference required
            between the current pulse and a subsequent pulse for the current pulse to
            be recorded correctly.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    overlap_likelihood = expand_dims(
        atleast_1d(
            (minimum_pulse_separation + minimum_arrival_separation) / count_times
        ),
        (1, 2),
    )
    corrected_frames = empty(frames.shape, dtype=uncertain)
    corrected_frames["nominal"] = frames["nominal"].data * exp(
        frames["nominal"].data * overlap_likelihood
    )
    corrected_frames["uncertainty"] = 0
    return masked_array(corrected_frames, frames.mask)
