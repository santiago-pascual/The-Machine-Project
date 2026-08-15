import importlib


def test_core_modules_import():
    for module_name in [
        "src.quant_research_features",
        "src.quant_blend_comparison_test",
        "src.hmm_incremental_portfolio_test",
        "portfolio_optimizer",
        "financial_data_system",
    ]:
        module = importlib.import_module(module_name)
        assert module is not None


def test_legacy_top_level_shims_still_resolve():
    for module_name in [
        "quant_research_features",
        "quant_blend_comparison_test",
        "hmm_incremental_portfolio_test",
    ]:
        module = importlib.import_module(module_name)
        assert module is not None
