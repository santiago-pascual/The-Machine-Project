import subprocess
import sys


def test_main_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=".",
    )
    assert result.returncode == 0
    assert "La Máquina project CLI" in result.stdout


def test_main_smoke_command() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "smoke"],
        capture_output=True,
        text=True,
        check=False,
        cwd=".",
    )
    assert result.returncode == 0
    assert "smoke-ok" in result.stdout
