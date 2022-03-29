import subprocess
import sys

from degradation_eda import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "degradation_eda", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
