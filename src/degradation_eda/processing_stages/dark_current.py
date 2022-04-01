from typing import Tuple, TypeVar

from numpy import floating, number, subtract
from numpy.typing import NDArray

UncertaintyType = TypeVar("UncertaintyType", bound=number)


def correct_dark_current(
    frames: NDArray[number],
    uncertainties: NDArray[UncertaintyType],
    dark_current: float,
) -> Tuple[NDArray[floating], NDArray[UncertaintyType]]:
    """Correct for dark current by subtracting a constant.

    Args:
        frames (NDArray[number]): A stack of frames to be corrected.
        uncertainties (NDArray[UncertaintyType]): Pixel uncertainties to be updated.
        dark_current (float): The dark current flux.

    Returns:
        Tuple[NDArray[floating], NDArray[floating]]: The corrected stack of frames and
            their updated uncertainties.
    """
    return (subtract(frames, dark_current, dtype=floating), uncertainties)
