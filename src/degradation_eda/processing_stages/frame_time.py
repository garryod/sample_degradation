from typing import Tuple

from numpy import add, divide, floating, number
from numpy.typing import NDArray


def correct_frame_time(
    frames: NDArray[number],
    uncertainties: NDArray[number],
    count_times: NDArray[floating],
    dead_times: NDArray[floating],
) -> Tuple[NDArray[floating], NDArray[floating]]:
    """Correct for detector frame rate by scaling photon counts according to frame time.

    Args:
        frames (NDArray[number]): A stack of frames to be corrected.
        count_times (NDArray[number]): The period over which photons are counted for
            each frame.
        dead_times (NDArray[number]): The period over which photons are not counted for
            each frame.

    Returns:
        Tuple[NDArray[floating], NDArray[floating]]: The corrected stack of frames and
            their updated uncertainties.
    """
    scale_factors = add(count_times, dead_times).squeeze()[:, None, None]
    return (
        divide(frames, scale_factors, dtype=floating),
        divide(uncertainties, scale_factors, dtype=floating),
    )
