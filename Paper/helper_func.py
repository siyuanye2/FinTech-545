from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252
UNIVERSE = ["ACWI", "TLT", "LQD", "TIP", "DBC", "VNQ", "QAI"]


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    target_weights: pd.DataFrame
    pre_trade_weights: pd.DataFrame
    turnover: pd.Series


def load_prices(data_dir: Path, tickers: list[str]) -> pd.DataFrame:
    price_frames = []

    for ticker in tickers:
        path = data_dir / f"{ticker}.csv"
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame[["Date", "Close/Last"]].copy()
        frame["Close/Last"] = pd.to_numeric(frame["Close/Last"], errors="coerce")
        frame = frame.sort_values("Date").set_index("Date")
        frame.rename(columns={"Close/Last": ticker}, inplace=True)
        price_frames.append(frame)

    prices = pd.concat(price_frames, axis=1).dropna().sort_index()
    return prices


def portfolio_drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    return wealth / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(portfolio_drawdown(returns).min())


def annualized_return(returns: pd.Series) -> float:
    growth = (1.0 + returns).prod()
    periods = len(returns)
    return growth ** (TRADING_DAYS / periods) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def effective_n(weights: pd.DataFrame) -> pd.Series:
    return 1.0 / (weights.pow(2).sum(axis=1))


def monthly_rebalance_dates(returns: pd.DataFrame) -> pd.DatetimeIndex:
    return returns.groupby(returns.index.to_period("M")).tail(1).index


def solve_mvo(window_returns: pd.DataFrame, risk_aversion: float = 3.0) -> np.ndarray:
    mu = (window_returns.mean() * TRADING_DAYS).to_numpy()
    cov = (window_returns.cov() * TRADING_DAYS).to_numpy()
    n_assets = len(mu)

    def objective(weights: np.ndarray) -> float:
        port_return = mu @ weights
        port_var = weights @ cov @ weights
        utility = port_return - 0.5 * risk_aversion * port_var
        return -utility

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        return x0

    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    marginal = cov @ weights / portfolio_vol
    return weights * marginal


def solve_erc(window_returns: pd.DataFrame) -> np.ndarray:
    cov = (window_returns.cov() * TRADING_DAYS).to_numpy()
    n_assets = cov.shape[0]
    x0 = np.full(n_assets, 1.0 / n_assets)
    target = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        contributions = risk_contributions(weights, cov)
        total_risk = contributions.sum()
        if total_risk <= 0:
            return 1e12
        contribution_shares = contributions / total_risk
        return float(np.sum((contribution_shares - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n_assets

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        return x0

    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def _run_backtest(
    returns: pd.DataFrame,
    weight_solver,
    lookback: int = 252,
) -> BacktestResult:
    rebalance_dates = monthly_rebalance_dates(returns)
    daily_portfolio_returns: dict[pd.Timestamp, float] = {}
    target_weights: list[pd.Series] = []
    pre_trade_weights: list[pd.Series] = []
    turnover_rows: list[tuple[pd.Timestamp, float]] = []

    current_weights = None
    columns = list(returns.columns)

    for idx, rebalance_date in enumerate(rebalance_dates[:-1]):
        date_loc = returns.index.get_loc(rebalance_date)
        if date_loc < lookback - 1:
            continue

        window = returns.iloc[date_loc - lookback + 1 : date_loc + 1]
        new_weights = weight_solver(window)

        if current_weights is None:
            prior_weights = np.zeros(len(columns))
        else:
            prior_weights = current_weights.copy()

        target_weights.append(pd.Series(new_weights, index=columns, name=rebalance_date))
        pre_trade_weights.append(pd.Series(prior_weights, index=columns, name=rebalance_date))
        turnover_rows.append((rebalance_date, 0.5 * np.abs(new_weights - prior_weights).sum()))

        hold_start = date_loc + 1
        hold_end = returns.index.get_loc(rebalance_dates[idx + 1])
        current_weights = new_weights.copy()

        for current_date, asset_returns in returns.iloc[hold_start : hold_end + 1].iterrows():
            asset_vector = asset_returns.to_numpy()
            port_return = float(current_weights @ asset_vector)
            daily_portfolio_returns[current_date] = port_return
            current_weights = current_weights * (1.0 + asset_vector)
            current_weights /= current_weights.sum()

    return BacktestResult(
        daily_returns=pd.Series(daily_portfolio_returns).sort_index(),
        target_weights=pd.DataFrame(target_weights),
        pre_trade_weights=pd.DataFrame(pre_trade_weights),
        turnover=pd.Series(dict(turnover_rows)).sort_index(),
    )


def performance_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        rows.append(
            {
                "Strategy": name,
                "Annual Return": annualized_return(result.daily_returns),
                "Volatility": annualized_volatility(result.daily_returns),
                "Sharpe Ratio": sharpe_ratio(result.daily_returns),
                "Max Drawdown": max_drawdown(result.daily_returns),
                "Average Monthly Turnover": float(result.turnover.mean()),
                "Effective N": float(effective_n(result.target_weights).mean()),
                "Gross Leverage": float(result.target_weights.abs().sum(axis=1).mean()),
            }
        )

    return pd.DataFrame(rows).set_index("Strategy")


def weight_instability_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        rows.append(
            {
                "Strategy": name,
                "Average Turnover": float(result.turnover.mean()),
                "Median Turnover": float(result.turnover.median()),
                "Weight Std Dev": float(result.target_weights.std().mean()),
                "Average Effective N": float(effective_n(result.target_weights).mean()),
            }
        )

    return pd.DataFrame(rows).set_index("Strategy")


def bootstrap_weight_sensitivity(
    returns: pd.DataFrame,
    solver,
    lookback: int = 252,
    n_bootstrap: int = 20,
    max_dates: int = 24,
    seed: int = 7,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rebalance_dates = monthly_rebalance_dates(returns)
    sampled_dates = rebalance_dates[-max_dates:]
    distances = []

    for rebalance_date in sampled_dates:
        date_loc = returns.index.get_loc(rebalance_date)
        if date_loc < lookback - 1:
            continue

        window = returns.iloc[date_loc - lookback + 1 : date_loc + 1]
        baseline = solver(window)

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, len(window), len(window))
            boot_window = window.iloc[sample_idx].reset_index(drop=True)
            boot_window.columns = window.columns
            boot_weights = solver(boot_window)
            distances.append(float(np.abs(boot_weights - baseline).sum()))

    if not distances:
        return np.nan, np.nan

    distances_array = np.array(distances)
    return float(distances_array.mean()), float(np.quantile(distances_array, 0.9))


def rolling_parameter_instability(returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    rolling_mu = returns.rolling(lookback).mean() * TRADING_DAYS
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)

    return pd.DataFrame(
        {
            "Average Annualized Mean Std": rolling_mu.std().mean(),
            "Average Annualized Vol Std": rolling_vol.std().mean(),
        },
        index=["Across Assets"],
    )


def save_plots(
    output_dir: Path,
    results: dict[str, BacktestResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cumulative = pd.DataFrame(
        {name: (1.0 + result.daily_returns).cumprod() for name, result in results.items()}
    )
    drawdowns = pd.DataFrame(
        {name: portfolio_drawdown(result.daily_returns) for name, result in results.items()}
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative.plot(ax=ax, linewidth=2)
    ax.set_title("Out-of-Sample Cumulative Performance")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    drawdowns.plot(ax=ax, linewidth=2)
    ax.set_title("Out-of-Sample Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "drawdowns.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, (name, result) in zip(axes, results.items()):
        result.target_weights.plot.area(ax=axis, stacked=True, linewidth=0)
        axis.set_title(f"{name} Target Weights")
        axis.set_ylabel("Weight")
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[-1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "weights.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_percent_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    return formatted


def run_analysis(
    data_dir: str | Path,
    lookback: int = 252,
    risk_aversion: float = 3.0,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    base_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_dir / "output"

    prices = load_prices(base_dir, UNIVERSE)
    returns = prices.pct_change().dropna()

    mvo_result = _run_backtest(
        returns,
        weight_solver=lambda window: solve_mvo(window, risk_aversion=risk_aversion),
        lookback=lookback,
    )
    erc_result = _run_backtest(returns, weight_solver=solve_erc, lookback=lookback)

    results = {"Mean-Variance": mvo_result, "Risk Parity": erc_result}

    summary = performance_table(results)
    instability = weight_instability_table(results)
    parameter_instability = rolling_parameter_instability(returns, lookback=lookback)

    sensitivity_rows = []
    for name, solver in [
        ("Mean-Variance", lambda window: solve_mvo(window, risk_aversion=risk_aversion)),
        ("Risk Parity", solve_erc),
    ]:
        mean_l1, p90_l1 = bootstrap_weight_sensitivity(
            returns,
            solver=solver,
            lookback=lookback,
        )
        sensitivity_rows.append(
            {
                "Strategy": name,
                "Mean L1 Weight Change": mean_l1,
                "90th Pct L1 Weight Change": p90_l1,
            }
        )

    sensitivity = pd.DataFrame(sensitivity_rows).set_index("Strategy")

    output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_dir / "prices.csv")
    returns.to_csv(output_dir / "asset_returns.csv")
    summary.to_csv(output_dir / "performance_summary.csv")
    instability.to_csv(output_dir / "weight_instability.csv")
    sensitivity.to_csv(output_dir / "sensitivity_summary.csv")
    parameter_instability.to_csv(output_dir / "parameter_instability.csv")
    mvo_result.target_weights.to_csv(output_dir / "mvo_target_weights.csv")
    erc_result.target_weights.to_csv(output_dir / "erc_target_weights.csv")
    mvo_result.daily_returns.to_csv(output_dir / "mvo_daily_returns.csv", header=["return"])
    erc_result.daily_returns.to_csv(output_dir / "erc_daily_returns.csv", header=["return"])

    save_plots(output_dir, results)

    formatted_summary = format_percent_columns(
        summary,
        ["Annual Return", "Volatility", "Max Drawdown", "Average Monthly Turnover"],
    )
    formatted_sensitivity = format_percent_columns(
        sensitivity,
        ["Mean L1 Weight Change", "90th Pct L1 Weight Change"],
    )
    formatted_instability = format_percent_columns(
        instability,
        ["Average Turnover", "Median Turnover", "Weight Std Dev"],
    )
    formatted_parameter_instability = format_percent_columns(
        parameter_instability,
        ["Average Annualized Mean Std", "Average Annualized Vol Std"],
    )

    return {
        "prices": prices,
        "returns": returns,
        "summary": summary,
        "summary_formatted": formatted_summary,
        "instability": instability,
        "instability_formatted": formatted_instability,
        "sensitivity": sensitivity,
        "sensitivity_formatted": formatted_sensitivity,
        "parameter_instability": parameter_instability,
        "parameter_instability_formatted": formatted_parameter_instability,
        "mvo_result": mvo_result,
        "erc_result": erc_result,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory that contains the ETF CSV files.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Rolling estimation window in trading days.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=3.0,
        help="Risk-aversion parameter for mean-variance optimization.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for CSV and chart outputs.",
    )
    args = parser.parse_args()

    results = run_analysis(
        Path(args.data_dir),
        lookback=args.lookback,
        risk_aversion=args.risk_aversion,
        output_dir=args.output_dir,
    )

    print("\nPerformance summary")
    print(results["summary_formatted"])

    print("\nWeight instability")
    print(results["instability_formatted"])

    print("\nBootstrap sensitivity")
    print(results["sensitivity_formatted"])

    print("\nParameter instability")
    print(results["parameter_instability_formatted"])

    print(f"\nSaved outputs to: {results['output_dir']}")
