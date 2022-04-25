from typing import Any, Tuple, TypeVar

from numpy.ma import MaskedArray, masked_array

from degradation_eda.utils.uncertain_maths import Uncertain, subtract_uncertain

FramesShape = TypeVar("FramesShape", bound=Any)


def subtract_background(
    foreground_frames: MaskedArray[FramesShape, Uncertain],
    background_frame: MaskedArray[Tuple[int, int], Uncertain],
) -> MaskedArray[FramesShape, Uncertain]:
    """Subtract a background frame from a sequence of foreground frames.

    Subtract a background frame from a sequence of foreground frames, as detailed in
    section 3.4.6 of 'Everything SAXS: small-angle scattering pattern collection and
    correction' [https://doi.org/10.1088/0953-8984/25/38/383201].

    Args:
        foreground_frames (MaskedArray[FramesShape, Uncertain]): A sequence of
            foreground frames to be corrected.
        background_frame (MaskedArray[Tuple[int, int], Uncertain]): The background
            which is to be corrected for.

    Returns:
        MaskedArray[FramesShape, Uncertain]: A sequence of corrected frames and their
            updated uncertainities.
    """
    return masked_array(
        subtract_uncertain(foreground_frames.data, background_frame.data),
        foreground_frames.mask,
    )
