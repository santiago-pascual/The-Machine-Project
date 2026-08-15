from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


class PortfolioOptimizer:
    def __init__(
        self,
        returns_df: pd.DataFrame,
        rf_daily: float,
        expected_daily_returns: pd.Series | None = None,
        use_expected_returns: bool = False,
        alpha: float = 0.5,
        no_opportunity: bool = False,
        defensive_mode: bool = False,
        regime_score: float = 0.0,
        regime_type: str = "neutral",
        regime_confidence: float = 1.0,
        population_size: int = 30,
        elite_size: int = 5,
        mutation_rate: float = 0.15,
        min_weight: float = 0.0,
        max_weight: float = 0.50,
        herf_lambda: float = 0.15,
        n_generations: int = 500,
        random_seed: int | None = 42,
        stagnation_limit: int = 50,
        soft_constraints: bool = True,
        covariance_matrix: pd.DataFrame | None = None,
    ) -> None:
        if returns_df.empty:
            raise ValueError("returns_df cannot be empty.")
        if population_size <= 0:
            raise ValueError("population_size must be greater than 0.")
        if elite_size <= 0 or elite_size > population_size:
            raise ValueError("elite_size must be greater than 0 and less than or equal to population_size.")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if min_weight < 0 or max_weight <= 0 or min_weight > max_weight:
            raise ValueError("Invalid min_weight/max_weight configuration.")
        if stagnation_limit <= 0:
            raise ValueError("stagnation_limit must be greater than 0.")

        self.returns_df = returns_df.copy()
        self.rf_daily = float(rf_daily)
        self.use_expected_returns = use_expected_returns
        self.alpha = float(alpha)
        self.no_opportunity = no_opportunity
        self.defensive_mode = defensive_mode
        self.regime_score = float(regime_score)
        self.regime_type = str(regime_type)
        self.regime_confidence = float(regime_confidence)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.herf_lambda = float(herf_lambda)
        self.n_generations = n_generations
        self.random_seed = random_seed
        self.stagnation_limit = stagnation_limit
        self.soft_constraints = soft_constraints
        self.n_assets = self.returns_df.shape[1]
        self.asset_names = list(self.returns_df.columns)
        self.historical_mean_returns = self.returns_df.mean()
        self.covariance_matrix = (
            covariance_matrix.reindex(index=self.returns_df.columns, columns=self.returns_df.columns)
            if covariance_matrix is not None
            else None
        )

        if self.n_assets == 0:
            raise ValueError("returns_df must contain at least one asset column.")
        if self.min_weight * self.n_assets > 1:
            raise ValueError("min_weight is too large for the number of assets.")
        if self.max_weight * self.n_assets < 1:
            raise ValueError("max_weight is too small for the number of assets.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if expected_daily_returns is None:
            self.expected_daily_returns = self.historical_mean_returns.copy()
        else:
            self.expected_daily_returns = (
                pd.Series(expected_daily_returns, dtype=float)
                .reindex(self.returns_df.columns)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.historical_mean_returns)
            )

        self.population: list[dict[str, np.ndarray | float | None]] = []

        self.effective_max_weight = self._compute_effective_max_weight()
        self.effective_herf_lambda = self._compute_effective_herf_lambda()

    def _compute_effective_max_weight(self) -> float:
        feasible_floor = min(0.95, max(1.0 / self.n_assets + 0.02, self.min_weight + 0.02))
        if self.regime_type == "high_volatility":
            boosted = self.max_weight * 1.20
            return float(_clip(boosted, feasible_floor, 0.95))
        if self.regime_type == "neutral":
            reduced = self.max_weight * 0.75
            return float(_clip(reduced, feasible_floor, self.max_weight))
        return float(_clip(self.max_weight, feasible_floor, 0.95))

    def _compute_effective_herf_lambda(self) -> float:
        if self.regime_type == "high_volatility":
            return float(max(0.01, self.herf_lambda * 0.75))
        if self.regime_type == "neutral":
            return float(self.herf_lambda * 1.35)
        return self.herf_lambda

    def get_asset_return_vector(self) -> pd.Series:
        if not self.use_expected_returns:
            return self.historical_mean_returns

        blended_returns = (
            self.alpha * self.expected_daily_returns
            + (1 - self.alpha) * self.historical_mean_returns
        )
        return blended_returns

    def portfolio_return(self, weights: np.ndarray) -> float:
        asset_returns = self.get_asset_return_vector()
        return float(np.dot(weights, asset_returns))

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        if self.covariance_matrix is not None:
            cov = self.covariance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            variance = float(weights.T @ cov @ weights)
            return float(np.sqrt(max(variance, 0.0)))
        portfolio_returns = self.returns_df.dot(weights)
        return float(portfolio_returns.std())

    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        portfolio_mean = self.portfolio_return(weights)
        portfolio_std = self.portfolio_volatility(weights)

        if portfolio_std == 0:
            return float("-inf")

        return (portfolio_mean - self.rf_daily) / portfolio_std

    def herfindahl_penalty(self, weights: np.ndarray) -> float:
        herfindahl_index = float(np.sum(np.square(weights)))
        baseline = 1 / self.n_assets
        return self.effective_herf_lambda * (herfindahl_index - baseline)

    def fitness(self, weights: np.ndarray) -> float:
        sharpe = self.portfolio_sharpe(weights)
        if not np.isfinite(sharpe):
            return float("-inf")
        adjusted_sharpe = sharpe * self.regime_confidence
        return (
            adjusted_sharpe
            - self.herfindahl_penalty(weights)
            - self.constraint_penalty(weights)
            - self.expected_return_penalty(weights)
            - self.negative_asset_penalty(weights)
            - self.defensive_penalty(weights)
        )

    def expected_return_penalty(self, weights: np.ndarray) -> float:
        portfolio_expected_return = self.portfolio_return(weights)
        if portfolio_expected_return <= 0:
            return abs(portfolio_expected_return)
        return 0.0

    def negative_asset_penalty(self, weights: np.ndarray) -> float:
        asset_returns = self.get_asset_return_vector().to_numpy()
        negative_exposure = weights[asset_returns < 0].sum()
        return float(5.0 * negative_exposure)

    def defensive_penalty(self, weights: np.ndarray) -> float:
        if not self.defensive_mode:
            return 0.0
        return 0.5 * self.portfolio_volatility(weights)

    def constraint_penalty(self, weights: np.ndarray) -> float:
        if not self.soft_constraints:
            return 0.0

        lower_gap = np.maximum(0.0, (self.min_weight + 0.02) - weights)
        upper_gap = np.maximum(0.0, weights - (self.max_weight - 0.02))
        upper_gap = np.maximum(upper_gap, np.maximum(0.0, weights - (self.effective_max_weight - 0.02)))
        return float(0.5 * np.sum(np.square(lower_gap) + np.square(upper_gap)))

    def random_weights(self) -> np.ndarray:
        weights = np.random.uniform(self.min_weight, self.max_weight, self.n_assets)
        return self.normalize(weights)

    def normalize(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != self.n_assets:
            raise ValueError("weights must be a one-dimensional array with length equal to the number of assets.")

        weights = np.clip(weights, self.min_weight, self.effective_max_weight)

        for _ in range(200):
            total = weights.sum()
            if total <= 0:
                weights = np.full(self.n_assets, 1 / self.n_assets)
            else:
                weights = weights / total

            weights = np.clip(weights, self.min_weight, self.effective_max_weight)
            residual = 1.0 - weights.sum()

            if abs(residual) < 1e-10:
                break

            if residual > 0:
                eligible = np.where(weights < self.effective_max_weight - 1e-12)[0]
            else:
                eligible = np.where(weights > self.min_weight + 1e-12)[0]

            if len(eligible) == 0:
                break

            weights[eligible] += residual / len(eligible)

        weights = np.clip(weights, self.min_weight, self.effective_max_weight)
        total = weights.sum()
        if total <= 0:
            return np.full(self.n_assets, 1 / self.n_assets)
        return weights / total

    def initialize_population(self) -> list[dict[str, np.ndarray | float | None]]:
        self.population = [
            {"weights": self.random_weights(), "fitness": None}
            for _ in range(self.population_size)
        ]
        return self.population

    def evaluate_population(self) -> None:
        for individual in self.population:
            individual["fitness"] = self.fitness(individual["weights"])

    def select_elite(self) -> list[dict[str, np.ndarray | float | None]]:
        ranked_population = sorted(
            self.population,
            key=lambda member: float(member["fitness"]),
            reverse=True,
        )
        return [
            {"weights": member["weights"].copy(), "fitness": member["fitness"]}
            for member in ranked_population[: self.elite_size]
        ]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return self.normalize(child)

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        mutated = weights.copy()

        if np.random.rand() < self.mutation_rate:
            index = np.random.randint(0, self.n_assets)
            mutated[index] += np.random.normal(0, 0.05)

        return self.normalize(mutated)

    def evolve(
        self,
        elite: list[dict[str, np.ndarray | float | None]],
    ) -> list[dict[str, np.ndarray | float | None]]:
        new_population = [
            {"weights": member["weights"].copy(), "fitness": None}
            for member in elite
        ]

        elite_weights = [member["weights"] for member in elite]
        random_slots = max(1, int(self.population_size * 0.2))
        target_children = self.population_size - len(new_population) - random_slots

        while len(new_population) < len(elite) + max(0, target_children):
            parent_indices = np.random.choice(len(elite_weights), size=2, replace=True)
            parent1 = elite_weights[parent_indices[0]]
            parent2 = elite_weights[parent_indices[1]]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append({"weights": child, "fitness": None})

        while len(new_population) < self.population_size:
            new_population.append({"weights": self.random_weights(), "fitness": None})

        return new_population

    def partial_restart(
        self,
        elite: list[dict[str, np.ndarray | float | None]],
    ) -> list[dict[str, np.ndarray | float | None]]:
        restarted_population = [
            {"weights": member["weights"].copy(), "fitness": None}
            for member in elite
        ]

        while len(restarted_population) < self.population_size:
            restarted_population.append({"weights": self.random_weights(), "fitness": None})

        return restarted_population

    def optimize(self) -> tuple[np.ndarray, float, float, float, dict[str, list[float]]]:
        equal_weight = np.full(self.n_assets, 1 / self.n_assets)
        equal_weight_sharpe = self.portfolio_sharpe(equal_weight)
        equal_weight_return = self.portfolio_return(equal_weight)
        equal_weight_volatility = self.portfolio_volatility(equal_weight)
        history = {"best": [], "mean": []}

        if self.no_opportunity:
            print("No clear opportunities detected")
            print("Final Portfolio:")
            print(f"Return: {equal_weight_return:.6f}")
            print(f"Volatility: {equal_weight_volatility:.6f}")
            print(f"Sharpe: {equal_weight_sharpe:.6f}")
            return equal_weight, equal_weight_sharpe, equal_weight_return, equal_weight_volatility, history

        self.initialize_population()

        best_global_weights: np.ndarray | None = None
        best_global_fitness = float("-inf")
        best_global_sharpe = float("-inf")
        stagnant_generations = 0

        for generation in range(self.n_generations):
            self.evaluate_population()

            current_fitness_values = np.array(
                [float(member["fitness"]) for member in self.population],
                dtype=float,
            )
            current_best_index = int(np.argmax(current_fitness_values))
            current_best = self.population[current_best_index]
            current_best_fitness = float(current_best["fitness"])
            current_mean_fitness = float(np.mean(current_fitness_values))
            history["best"].append(current_best_fitness)
            history["mean"].append(current_mean_fitness)

            if current_best_fitness > best_global_fitness:
                best_global_fitness = current_best_fitness
                best_global_weights = current_best["weights"].copy()
                best_global_sharpe = self.portfolio_sharpe(best_global_weights)
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if generation % 25 == 0 or generation == self.n_generations - 1:
                print(f"Gen {generation}: Best {best_global_fitness:.6f} | Mean {current_mean_fitness:.6f}")

            elite = self.select_elite()

            if stagnant_generations >= self.stagnation_limit:
                self.population = self.partial_restart(elite)
                stagnant_generations = 0
            else:
                self.population = self.evolve(elite)

        if best_global_weights is None:
            raise ValueError("Optimization failed to produce a valid portfolio.")

        best_return = self.portfolio_return(best_global_weights)
        best_volatility = self.portfolio_volatility(best_global_weights)

        if np.sum(
            (best_global_weights <= self.min_weight + 1e-3)
            | (best_global_weights >= self.max_weight - 1e-3)
        ) > 0.8 * len(best_global_weights):
            print("WARNING: Solution is hitting constraints heavily")

        if np.isfinite(equal_weight_sharpe) and abs(equal_weight_sharpe) >= 0.05:
            improvement_pct = ((best_global_sharpe - equal_weight_sharpe) / abs(equal_weight_sharpe)) * 100
            print(f"Sharpe improvement vs equal weight: {improvement_pct:.2f}%")
        elif np.isfinite(equal_weight_sharpe):
            print(f"Sharpe improved from {equal_weight_sharpe:.4f} to {best_global_sharpe:.4f}")
        else:
            print("Sharpe improvement vs equal weight: not available")

        print("Final Portfolio:")
        print(f"Return: {best_return:.6f}")
        print(f"Volatility: {best_volatility:.6f}")
        print(f"Sharpe: {best_global_sharpe:.6f}")

        contribution = best_global_weights * self.get_asset_return_vector().to_numpy()
        contribution_ranking = pd.Series(contribution, index=self.asset_names).sort_values(ascending=False)
        print("Contribution Ranking:")
        print(contribution_ranking)

        return best_global_weights, best_global_sharpe, best_return, best_volatility, history
