from __future__ import annotations

from risk_sensitivity_analysis import run_grid


def main():
    grid, daily = run_grid()
    daily.to_csv("risk_constraint_shadow_daily.csv", index=False)
    grid.to_csv("risk_constraint_grid.csv", index=False)
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
