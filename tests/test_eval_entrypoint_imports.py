"""Exercise oracle imports with the paths used by direct script launches."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("module_name", ["helpers", "scripts.helpers"])
def test_oracle_teacher_loads_in_script_and_package_contexts(tmp_path, module_name):
    repo = Path(__file__).resolve().parents[1]
    # Expose the library without exposing the repository's scripts package,
    # independent of whether active_adaptation has an editable installation.
    (tmp_path / "active_adaptation").symlink_to(
        repo / "active_adaptation", target_is_directory=True
    )
    code = textwrap.dedent(
        """
        import importlib
        import importlib.util
        from pathlib import Path
        import sys

        repo = Path(sys.argv[1])
        dependencies = sys.argv[2]
        module_name = sys.argv[3]
        sys.path[:] = [path for path in sys.path if Path(path).resolve() != repo]
        launch_path = repo / "scripts" if module_name == "helpers" else repo
        sys.path[:0] = [str(launch_path), dependencies]
        if module_name == "helpers":
            assert importlib.util.find_spec("scripts") is None

        helpers = importlib.import_module(module_name)
        import torch
        from torch import nn

        owner = nn.Module()
        owner.encoder_priv = nn.Linear(2, 2)
        expected = {
            key: torch.full_like(value, 7)
            for key, value in owner.encoder_priv.state_dict().items()
        }
        helpers.load_oracle_teacher(owner, {"encoder_priv": expected})
        for key, value in owner.encoder_priv.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        if module_name == "helpers":
            assert importlib.util.find_spec("scripts") is None
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(repo), str(tmp_path), module_name],
        cwd=repo / "scripts",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
