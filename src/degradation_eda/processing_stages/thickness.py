from typing import Tuple, TypeVar

from numpy import empty
from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import Uncertain, divide_uncertain, uncertain

FramesShape = TypeVar("FramesShape", bound=Tuple[int, int, int])


def normalize_thickness(
    frames: MaskedArray[FramesShape, Uncertain],
    sample_thickness: float,
) -> MaskedArray[FramesShape, Uncertain]:
    """Normailizes pixel intensities by dividing by the sample thickness.

    Args:
        frames (MaskedArray[FramesShape, Uncertain]): A stack of uncertain frames to be
            corrected.
        sample_thickness (float): The thickness of the exposed sample.

    Returns:
        MaskedArray[FramesShape, Uncertain]: The normalized stack of frames and their
            updated uncertainties.
    """
    uncertain_thickness = empty((1,), dtype=uncertain)
    uncertain_thickness["nominal"] = sample_thickness
    uncertain_thickness["uncertainty"] = 0
    return masked_array(divide_uncertain(frames.data, uncertain_thickness), frames.mask)
