from typing import Any, TypeVar

from numpy import atleast_1d, dtype, empty, expand_dims, floating, ndarray
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def normalize_frame_time(
    frames: MaskedArray[FramesShape, Uncertain],
    count_times: ndarray[TimesShape, dtype[floating]],
) -> MaskedArray[FramesShape, Uncertain]:
    """Normalize for detector frame rate by scaling photon counts according to count time.

    Normalize for detector frame rate by scaling photon counts according to count time,
    as detailed in section 3.4.3 of 'Everything SAXS: small-angle scattering pattern
    collection and correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of frames to be
            normalized.
        count_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are counted for each frame.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The normalized stack of frames and their
            updated uncertainties.
    """
    times = expand_dims(
        atleast_1d(empty(count_times.shape, uncertain)),
        (-2, -1),
    )
    times["nominal"] = expand_dims(atleast_1d(count_times), (-2, -1))
    times["uncertainty"] = 0
    return masked_array(divide_uncertain(frames.data, times), frames.mask)
