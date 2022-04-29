import subprocess
import sys

from sample_degradation import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "sample_degradation", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
