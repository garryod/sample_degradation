from typing import Any, TypeVar

from numpy import (
    atleast_1d,
    broadcast_shapes,
    divide,
    dtype,
    empty,
    expand_dims,
    floating,
    ndarray,
)

from degradation_eda.utils.uncertain_maths import (
    Uncertain,
    multiply_uncertain,
    uncertain,
)

FramesShape = TypeVar("FramesShape", bound=Any)
TimesShape = TypeVar("TimesShape", bound=Any)


def correct_deadtime(
    frames: ndarray[FramesShape, Uncertain],
    count_times: ndarray[TimesShape, dtype[floating]],
    dead_times: ndarray[TimesShape, dtype[floating]],
) -> ndarray[FramesShape, Uncertain]:
    """Correct for detector deadtime by scaling photon counts according to the duty cycle.

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
    scale_factors = expand_dims(
        atleast_1d(
            empty(broadcast_shapes(count_times.shape, dead_times.shape), uncertain)
        ),
        (1, 2),
    )
    scale_factors["nominal"] = expand_dims(
        atleast_1d(divide(count_times + dead_times, count_times, dtype=floating)),
        (1, 2),
    )
    scale_factors["uncertainty"] = 0
    return multiply_uncertain(frames, scale_factors)
