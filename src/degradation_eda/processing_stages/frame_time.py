from typing import Any, TypeVar

from numpy import (
    add,
    atleast_1d,
    broadcast_shapes,
    dtype,
    empty,
    expand_dims,
    floating,
    ndarray,
)
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def normalize_frame_time(
    frames: MaskedArray[FramesShape, Uncertain],
    count_times: ndarray[TimesShape, dtype[floating]],
    dead_times: ndarray[TimesShape, dtype[floating]],
) -> MaskedArray[FramesShape, Uncertain]:
    """Normalize for detector frame rate by scaling photon counts according to frame time.

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of frames to be
            normalized.
        count_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are counted for each frame.
        dead_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are not counted for each frame.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The normalized stack of frames and their
            updated uncertainties.
    """
    scale_factors = expand_dims(
        atleast_1d(
            empty(broadcast_shapes(count_times.shape, dead_times.shape), uncertain)
        ),
        (1, 2),
    )
    scale_factors["nominal"] = expand_dims(
        atleast_1d(add(count_times, dead_times)),
        (1, 2),
    )
    scale_factors["uncertainty"] = 0
    return masked_array(divide_uncertain(frames.data, scale_factors), frames.mask)
