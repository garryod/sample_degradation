from typing import Any, TypeVar

from numpy import dtype, empty_like, maximum, number, sqrt
from numpy.ma import MaskedArray, masked_array

from sample_degradation.utils.uncertain_maths import Uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Any)


def estimate_poisson_uncertainty(
    frames: MaskedArray[FramesShape, dtype[number]]
) -> MaskedArray[FramesShape, Uncertain]:
    """Estimates the poisson uncertainty of photon counts.

    Estimates the poisson uncertainty of photon counts by taking the square root of
    real numbers greater than one, or one in place of numbers less than one.

    Args:
        frames (MaskedArray[FramesShape, dtype[number]]): A stack of frames from which
            the uncertainties can be computed.

    Returns:
        MaskedArray[FramesShape, Uncertain]: A stack of frames and their pixel-wise
            uncertainties.
    """
    results = empty_like(frames, dtype=uncertain)
    results["nominal"] = frames.data
    results["uncertainty"] = sqrt(maximum(frames, 1.0))
    return masked_array(results, frames.mask)
