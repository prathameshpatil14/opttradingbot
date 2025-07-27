# backtest/backtest_runner.py

import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("BacktestRunner")

class BacktestMetrics:
    """Helper for tracking backtest stats and risk metrics."""
    def __init__(self, env_class, agent, data, log_dir, audit, market_config=None, risk_config=None, features_config=None):
        self.env_class = env_class
        self.agent = agent
        self.data = data
        self.log_dir = log_dir
        self.audit = audit
        self.market_config = market_config or {}
        self.risk_config = risk_config or {}
        self.features_config = features_config or {}
        self.rewards = []
        self.pnl = []
        self.actions = []
        self.positions = []
        self.q_values = []
        self.confidences = []
        self.equity_curve = [1.0]  # Start with 1.0 as initial equity

    def step(self, reward, pnl, action, position, q=None, conf=None):
        self.rewards.append(reward)
        self.pnl.append(pnl)
        self.actions.append(action)
        self.positions.append(position)
        self.q_values.append(q)
        self.confidences.append(conf)
        # Update equity
        self.equity_curve.append(self.equity_curve[-1] + pnl)

    def summary(self):
        arr = np.array(self.pnl)
        eq = np.array(self.equity_curve)
        returns = np.diff(eq) / eq[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 6.5 * 60) if len(returns) > 0 else 0
        max_dd = max_drawdown(eq)
        cvar_95 = cvar(arr, alpha=0.95)
        win_rate = (np.array(self.pnl) > 0).mean() if len(self.pnl) > 0 else 0
        total_return = eq[-1] - eq[0]
        return {
            "total_return": total_return,
            "mean_reward": np.mean(self.rewards) if self.rewards else 0,
            "std_reward": np.std(self.rewards) if self.rewards else 0,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "cvar_95": cvar_95,
            "win_rate": win_rate,
            "final_equity": eq[-1],
            "num_trades": len(self.actions),
        }

def max_drawdown(equity_curve):
    # Drawdown: max peak-to-valley loss
    equity_curve = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / (peaks + 1e-8)
    return np.max(drawdowns) if len(drawdowns) > 0 else 0

def cvar(returns, alpha=0.95):
    """
    Conditional Value at Risk (CVaR): mean loss beyond the quantile at 1-alpha.
    """
    returns = np.sort(returns)
    n = int((1 - alpha) * len(returns))
    if n == 0:
        return 0
    return -np.mean(returns[:n])

class BacktestRunner:
    """
    Full historical backtester for RL agents, tracks all metrics and logs every step.
    """
    def __init__(self, env_class, agent, data, log_dir="logs", audit=True,
                 market_config=None, risk_config=None, features_config=None):
        self.env_class = env_class
        self.agent = agent
        self.data = data
        self.log_dir = log_dir
        self.audit = audit
        self.market_config = market_config or {}
        self.risk_config = risk_config or {}
        self.features_config = features_config or {}
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_path = os.path.join(log_dir, "backtest_audit.csv")

    def run(self, max_steps=None, risk_manager=None, verbose=True):
        env = self.env_class(
            data=self.data,
            config=self.market_config,
            risk_config=self.risk_config,
            features_config=self.features_config,
            mode="backtest"
        )
        metrics = BacktestMetrics(
            env_class=self.env_class,
            agent=self.agent,
            data=self.data,
            log_dir=self.log_dir,
            audit=self.audit,
            market_config=self.market_config,
            risk_config=self.risk_config,
            features_config=self.features_config
        )
        state = env.reset()
        done = False
        step = 0

        # Prepare audit log
        if self.audit and not os.path.exists(self.audit_path):
            with open(self.audit_path, "w") as f:
                f.write("step,state,action,reward,pnl,q_value,confidence,position,info\n")

        while not done and (max_steps is None or step < max_steps):
            if hasattr(self.agent, 'select_action'):
                out = self.agent.select_action(state, eval_mode=True)
                # Handle agent output (action only or (action, conf))
                if isinstance(out, tuple):
                    action, conf = out
                else:
                    action, conf = out, None
            else:
                raise ValueError("Agent must have select_action method.")

            next_state, reward, done, info = env.step(action)
            pnl = info.get("pnl", reward)
            q_value = info.get("q_value", None)
            position = info.get("position", None)

            # Step metrics
            metrics.step(reward, pnl, action, position, q=q_value, conf=conf)

            # Audit log each step
            if self.audit:
                with open(self.audit_path, "a") as f:
                    state_str = np.array2string(np.array(state), precision=3, separator=",")
                    q_str = np.array2string(q_value, precision=3, separator=",") if q_value is not None else ""
                    conf_str = f"{conf:.3f}" if conf is not None else ""
                    info_str = str(info)
                    f.write(f"{step},{state_str},{action},{reward},{pnl},{q_str},{conf_str},{position},{info_str}\n")

            # Risk management (stop if circuit breaker triggered)
            if risk_manager and hasattr(risk_manager, "__call__"):
                # Pass the environment so risk checks can access positions and equity
                if not risk_manager(env, state):
                    logger.warning("Risk manager triggered circuit breaker. Stopping backtest.")
                    break

            state = next_state
            step += 1

            if verbose and step % 500 == 0:
                logger.info(f"Backtest step {step}: Eq={metrics.equity_curve[-1]:.3f} Return={metrics.rewards[-1]:.2f}")

        summary = metrics.summary()
        logger.info(f"Backtest complete. Total return: {summary['total_return']:.2f}, Sharpe: {summary['sharpe']:.2f}, MaxDD: {summary['max_drawdown']:.2%}")
        return summary
