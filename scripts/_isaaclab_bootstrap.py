"""Import Isaac Lab safely when TorchRL has selected multiprocessing spawn."""

import multiprocessing
import platform


_IS_LINUX_ARM = (
    platform.system() == "Linux"
    and platform.machine().lower() in {"aarch64", "arm64"}
)
_START_METHOD = multiprocessing.get_start_method(allow_none=True)

if _IS_LINUX_ARM:
    # Isaac Sim 5.1 runs a short preload check during import. It must use fork
    # so a TorchRL-selected spawn context does not re-execute the entrypoint.
    multiprocessing.set_start_method("fork", force=True)

try:
    from isaaclab.app import AppLauncher
finally:
    if _IS_LINUX_ARM:
        multiprocessing.set_start_method(_START_METHOD, force=True)


__all__ = ["AppLauncher"]
