from typing import Optional, cast

from nexusformat.nexus import NXFile


def get_background(path: str) -> Optional[str]:
    """Retrieves the path to the background image if one exists.

    Args:
        path (str): The path of the foreground image nexus file.

    Returns:
        Optional[str]: The path to the background image nexus file.
    """
    with NXFile(path) as file:
        return (
            cast(bytes, file["entry1"]["sample"]["background"][()]).decode()
            if "background" in file["entry1"]["sample"]
            else None
        )
