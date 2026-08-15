"""Auto-generated shim to preserve import paths after repo reorg.

Original module moved to src/growth_live_tracking_monitor.py — this shim re-exports public API.
"""
from importlib import import_module

_mod = import_module("src.growth_live_tracking_monitor")

for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)

if __name__ == "__main__":
    if hasattr(_mod, "main"):
        _mod.main()
