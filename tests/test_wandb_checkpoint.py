import os
from types import SimpleNamespace

from active_adaptation.utils import wandb as wandb_utils


class _RemoteFile:
    def __init__(self, name, downloads):
        self.name = name
        self._downloads = downloads

    def download(self, root, replace):
        self._downloads.append((self.name, root, replace))


def test_replay_name_containing_checkpoint_is_not_model_checkpoint(
    monkeypatch, tmp_path
):
    downloads = []
    remote_files = [
        _RemoteFile("files/checkpoint_12.pt", downloads),
        _RemoteFile("files/checkpoint_final.pt", downloads),
        _RemoteFile("files/my_checkpoint_replay.h5", downloads),
    ]
    run = SimpleNamespace(id="stable-run-id", files=lambda: remote_files)
    monkeypatch.setattr(
        wandb_utils.wandb,
        "Api",
        lambda: SimpleNamespace(run=lambda _: run),
    )
    monkeypatch.setattr(wandb_utils, "__file__", str(tmp_path / "wandb.py"))

    path = wandb_utils.parse_checkpoint_path(
        "run:entity/project/run-id",
        download_replay=True,
        replay_filename="my_checkpoint_replay.h5",
    )

    assert path == os.path.join(
        str(tmp_path), "wandb", "stable-run-id", "files", "checkpoint_final.pt"
    )
    assert [name for name, _, _ in downloads] == [
        "files/checkpoint_final.pt",
        "files/my_checkpoint_replay.h5",
    ]
