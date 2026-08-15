"""Auto-generated shim to preserve import paths after repo reorg.

Original module moved to src/quant_blend_comparison_test.py — this shim re-exports public API.
"""

from importlib import import_module
import importlib, importlib.util, importlib.machinery, os
n_mod = None
try:
    _mod = import_module("src.quant_blend_comparison_test")
except Exception:
    # Fallback: load directly from src/quant_blend_comparison_test.py if available
    _path = os.path.join(os.path.dirname(__file__), "src", "quant_blend_comparison_test.py")
    if os.path.isfile(_path):
        spec = importlib.util.spec_from_file_location("src.quant_blend_comparison_test", _path)
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
