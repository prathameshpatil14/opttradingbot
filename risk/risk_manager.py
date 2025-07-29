# risk/risk_manager.py

import numpy as np
from utils.logger import get_logger

logger = get_logger("RiskManager")

class RiskManager:
    """
    Implements robust risk controls for both live and backtest trading:
    - Max drawdown circuit breaker
    - Per-leg stop loss
    - Transaction cost/slippage
    - Dynamic position sizing
    - Forced flatten before expiry/close
    """
    def __init__(self, risk_config, audit_log_path=None):
       
        self.max_drawdown = risk_config.get("max_drawdown", 0.10)
        self.per_leg_stop_loss = risk_config.get("per_leg_stop_loss", 0.05)
        self.transaction_cost = risk_config.get("transaction_cost", 0.001)
        self.slippage = risk_config.get("slippage", 0.0005)
        self.flatten_before_close = risk_config.get("flatten_before_close_minutes", 15)
        self.min_equity = risk_config.get("min_equity", 0.2)
        self.audit_log_path = audit_log_path

    def check_risk(self, env, state):
        """
        Main risk check.
        Returns True if safe to continue, False to halt/flatten all.
        - env: trading environment (must provide .equity, .max_equity, .position, .entry_price, .current_step, .df)
        - state: current state vector (for custom risk, e.g., confidence-based sizing)
        """
        triggered = False
        reason = None
        step = getattr(env, "current_step", None)

        # Max drawdown check
        equity = getattr(env, "equity", 1.0)
        max_equity = getattr(env, "max_equity", 1.0)
        drawdown = (max_equity - equity) / (max_equity + 1e-8)
        if drawdown > self.max_drawdown:
            triggered = True
            reason = f"Drawdown {drawdown:.3f} > {self.max_drawdown:.3f}"

        # Min equity circuit breaker
        if equity < self.min_equity:
            triggered = True
            reason = f"Equity {equity:.3f} < min_equity {self.min_equity:.3f}"

        # Per-leg stop loss: flatten if unrealized loss > threshold
        position = getattr(env, "position", 0)
        entry_price = getattr(env, "entry_price", 0)
        price = env.df.iloc[step]["close"] if step is not None and "close" in env.df.columns else entry_price
        if position != 0 and entry_price > 0:
            if position == 1:
                unrealized = price - entry_price
            else:
                unrealized = entry_price - price
            if unrealized < -self.per_leg_stop_loss * entry_price:
                triggered = True
                reason = f"Per-leg stop loss: unrealized {unrealized:.3f} < -{self.per_leg_stop_loss * entry_price:.3f}"

        # Force flatten before market close/expiry (handled in env/strategy but can audit here)
        minutes_left = env.df.iloc[step]["minutes_to_close"] if step is not None and "minutes_to_close" in env.df.columns else 30
        if position != 0 and minutes_left <= self.flatten_before_close:
            triggered = True
            reason = f"Flatten before close: {minutes_left} min left"

        # (Optional) Add custom risk logic: high realized volatility regime, etc.

        # Audit risk event if triggered
        if triggered:
            logger.warning(f"RiskManager: Circuit breaker triggered: {reason}")
            if self.audit_log_path:
                with open(self.audit_log_path, "a") as f:
                    f.write(f"Step {step}: RISK EVENT: {reason}\n")
        return not triggered

    def transaction_cost_model(self, price, volume=1, lot_size=1, lot_multiplier=1):
        
        total_qty = volume * lot_size * lot_multiplier
        return price * total_qty * (self.transaction_cost + self.slippage)

    def dynamic_position_size(self, confidence, base_size=1, min_size=0.25, max_size=2.0):

        conf = np.clip(confidence, -1, 1)
        if conf >= 0:
            size = base_size + (max_size - base_size) * conf
        else:
            # When confidence is negative, scale position towards the minimum size
            size = base_size + (min_size - base_size) * (-conf)
        return np.clip(size, min_size, max_size)


