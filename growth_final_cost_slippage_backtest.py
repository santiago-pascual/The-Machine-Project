# Auto-generated shim to preserve import paths after repo reorg
from importlib import import_module
_mod = import_module("src.growth_final_cost_slippage_backtest")
# re-export public attributes
for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)
# If moved module defines a main(), call it when executed as a script
if __name__ == '__main__':
    if hasattr(_mod, 'main'):
        _mod.main()
