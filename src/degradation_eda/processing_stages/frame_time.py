from typing import Any, TypeVar

from numpy import add, broadcast_shapes, dtype, empty, floating, ndarray

from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def correct_frame_time(
    frames: ndarray[FramesShape, Uncertain],
    count_times: ndarray[TimesShape, dtype[floating]],
    dead_times: ndarray[TimesShape, dtype[floating]],
) -> ndarray[FramesShape, Uncertain]:
    """Correct for detector frame rate by scaling photon counts according to frame time.

    Args:
        frames (ndarray[FramesShape, Uncertain]): A stack of frames to be corrected.
        count_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are counted for each frame.
        dead_times (ndarray[TimesShape, dtype[floating]]): The period over which
            photons are not counted for each frame.

    Returns:
        ndarray[FramesShape, Uncertain]: The corrected stack of frames and their
            updated uncertainties.
    """
    scale_factors = empty(
        broadcast_shapes(count_times.shape, dead_times.shape), uncertain
    )[:, None, None]
    scale_factors["nominal"] = add(count_times, dead_times)[:, None, None]
    scale_factors["uncertainty"] = 0
    return divide_uncertain(frames, scale_factors)
