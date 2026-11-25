import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('portfolio_audit', Path(__file__).resolve().parents[1]/'scripts/portfolio_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_compound_and_initial_peak():
    m = audit.metrics([-0.2, 0.25], periods=2)
    assert m['cumulative'] == pytest.approx(0)
    assert m['drawdown'] == pytest.approx(-0.2)
    assert m['annualized_geometric'] == pytest.approx(0)


def test_risk_conventions():
    r = np.array([0.1, -0.05, 0.02])
    m = audit.metrics(r, periods=12)
    assert m['volatility'] == pytest.approx(r.std(ddof=1)*np.sqrt(12))
    assert m['sortino'] == pytest.approx(r.mean()*np.sqrt(12)/np.sqrt(np.mean(np.minimum(r,0)**2)))


@pytest.mark.parametrize('bad', [[0.1], [0.1, np.nan], [0.1,-1]])
def test_invalid_returns(bad):
    with pytest.raises(ValueError):
        audit.metrics(bad)


def test_output_cannot_write_input(tmp_path):
    with pytest.raises(ValueError):
        audit.run(tmp_path, tmp_path/'output')
