from typing import Any, TypeVar

from numpy import bool_, broadcast_to, dtype, empty_like, maximum, ndarray, number, sqrt
from numpy.ma import masked_where

from degradation_eda.utils.uncertain_maths import Uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)


def estimate_poisson_uncertainty(
    frames: ndarray[FramesShape, dtype[number]],
    mask: ndarray[Any, dtype[bool_]],
) -> ndarray[FramesShape, Uncertain]:
    """Estimates the poisson uncertainty of photon counts.

    Estimates the poisson uncertainty of photon counts by taking the square root of
    real numbers greater than one, or one in place of numbers less than one.

    Args:
        frames (ndarray[FramesShape, dtype[number]]): A stack of frames from which the
            uncertainties can be computed.
        mask (ndarray[Any, dtype[bool_]]): The boolean mask to apply to each frame.

    Returns:
        ndarray[FramesShape, Uncertain]: A stack of frames and their pixel-wise
            uncertainties.
    """
    results = empty_like(frames, dtype=uncertain)
    results["nominal"] = frames
    results["uncertainty"] = masked_where(
        broadcast_to(mask, frames.shape), maximum(sqrt(frames), 1.0)
    ).filled(fill_value=0)
    return results
