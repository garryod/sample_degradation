from typing import Tuple

from numpy import divide, floating, multiply, number
from numpy.typing import NDArray


def correct_deadtime(
    frames: NDArray[number],
    uncertainties: NDArray[number],
    count_times: NDArray[floating],
    dead_times: NDArray[floating],
) -> Tuple[NDArray[floating], NDArray[floating]]:
    """Correct for detector deadtime by scaling photon counts according to the duty cycle.

    Args:
        frames (NDArray[number]): A stack of frames to be corrected.
        uncertainties (NDArray[number]): Pixel uncertainties to be updated.
        count_times (NDArray[number]): The period over which photons are counted for
            each frame.
        dead_times (NDArray[number]): The period over which photons are not counted for
            each frame.

    Returns:
        NDArray[floating]: The corrected stack of frames.
        NDArray[floating]: The updated uncertainties.
    """
    scale_factors = divide(
        count_times + dead_times, count_times, dtype=floating
    ).squeeze()[:, None, None]
    return (
        multiply(scale_factors, frames, dtype=floating),
        multiply(scale_factors, uncertainties, dtype=floating),
    )
