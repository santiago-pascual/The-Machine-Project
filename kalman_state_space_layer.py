"""Auto-generated shim to preserve import paths after repo reorg.

Original module moved to src/kalman_state_space_layer.py — this shim re-exports public API.
"""

import importlib.util
import os
from importlib import import_module

_mod = None
try:
    _mod = import_module("src.kalman_state_space_layer")
except Exception:
    _path = os.path.join(os.path.dirname(__file__), "src", "kalman_state_space_layer.py")
    if os.path.isfile(_path):
        spec = importlib.util.spec_from_file_location("src.kalman_state_space_layer", _path)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    else:
        raise


for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)

if __name__ == "__main__":
    if hasattr(_mod, "main"):
        _mod.main()
