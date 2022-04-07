from typing import Any, TypeVar

from numpy import dtype, empty_like, maximum, ndarray, number, sqrt

from degradation_eda.utils.uncertain_maths import Uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)


def estimate_poisson_uncertainty(
    frames: ndarray[FramesShape, dtype[number]]
) -> ndarray[FramesShape, Uncertain]:
    """Estimates the poisson uncertainty of photon counts.

    Estimates the poisson uncertainty of photon counts by taking the square root of
    real numbers greater than one, or one in place of numbers less than one.

    Args:
        frames (ndarray[FramesShape, dtype[number]]): A stack of frames from which the
            uncertainties can be computed.

    Returns:
        ndarray[FramesShape, Uncertain]: A stack of frames and their pixel-wise
            uncertainties.
    """
    results = empty_like(frames, dtype=uncertain)
    results["nominal"] = frames
    results["uncertainty"] = maximum(sqrt(frames), 1.0)
    return results
