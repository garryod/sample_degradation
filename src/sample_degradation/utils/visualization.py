from typing import Any

from numpy import dtype, ndarray, number
from plotly.express import imshow
from plotly.express.colors import get_colorscale


def show_frame(frame: ndarray[Any, dtype[number]], log_z: float = 1.0, **kwargs):
    """Generates and displays a plotly heatmap from a 2 dimensional frame.

    Args:
        frame (ndarray[Any, dtype[number]]): A 2 dimensional frame.
        log_z (float): The exponential scaling to apply on the z axis. Default 1.0.
    """
    BASE_SCALE = get_colorscale("viridis")
    color_scale = (
        [
            [(log_z**val - log_z**0) / (log_z**1 - log_z**0), color]
            for val, color in BASE_SCALE
        ]
        if log_z > 1
        else BASE_SCALE
    )
    imshow(
        frame, color_continuous_scale=color_scale, binary_string=True, **kwargs
    ).show()
