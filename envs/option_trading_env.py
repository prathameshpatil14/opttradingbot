# envs/option_trading_env.py

import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque
from features.feature_engineering import compute_features
from utils.logger import get_logger

logger = get_logger("OptionTradingEnv")

class OptionTradingEnv(gym.Env):
    """
    Custom Gym environment for options trading with RL.
    State includes OHLCV, technicals, IV, Greeks, regime, etc.
    Supports multi-leg, flatten-before-expiry, risk controls.
    """

    def __init__(self,
                 data=None,
                 data_path=None,
                 config=None,
                 risk_config=None,
                 features_config=None,
                 mode="train"):
        super().__init__()
        self.mode = mode
        self.config = config or {}
        self.risk_config = risk_config or {}
        self.features_config = features_config or {}

        # Load data
        if data is not None:
            self.raw_data = data.copy()
        elif data_path is not None:
            self.raw_data = pd.read_csv(data_path)
        else:
            raise ValueError("Either data or data_path must be provided.")

        # Compute features (see features/feature_engineering.py)
        self.df = compute_features(self.raw_data, features_config or {})
        self.df = self.df.bfill().ffill().reset_index(drop=True)
        if self.df.empty:
            raise ValueError("OptionTradingEnv: Feature DataFrame is empty after feature engineering.")
        
        self.n_steps = len(self.df)
        self.current_step = 0

        # ---- Action & Observation space ----
        self.action_space = spaces.Discrete(4)
        # Ensure only numeric features are included as state
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.state_columns = [col for col in numeric_cols if col not in ("date", "timestamp")]
        self.state_dim = len(self.state_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # ---- Trading state ----
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.peak_price = 0  # For trailing stops
        self.equity = 1.0   # Starting equity
        self.max_equity = 1.0
        self.trade_history = []
        self.drawdown_triggered = False

        # ---- Risk/Reward params ----
        self.transaction_cost = risk_config.get("transaction_cost", 0.001)
        self.slippage = risk_config.get("slippage", 0.0005)
        self.max_drawdown = risk_config.get("max_drawdown", 0.10)
        self.per_leg_stop_loss = risk_config.get("per_leg_stop_loss", 0.05)
        self.flatten_before_close = risk_config.get("flatten_before_close_minutes", 15)
        self.min_equity = risk_config.get("min_equity", 0.2)

        # ---- For audit ----
        self._last_info = {}

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.peak_price = 0
        self.equity = 1.0
        self.max_equity = 1.0
        self.trade_history = []
        self.drawdown_triggered = False
        if self.df.empty or self.current_step >= len(self.df):
            raise ValueError("OptionTradingEnv: DataFrame is empty or current_step out of bounds on reset!")
        state = self._get_state()
        return state

    def _get_state(self):
        # Always use only numeric state columns (excluding date)
        row = self.df.iloc[self.current_step]
        state = row[self.state_columns].values.astype(np.float32)
        return state

    def step(self, action):
        """
        action: 0=hold, 1=long, 2=short, 3=flatten
        """
        info = {}
        done = False
        prev_price = self.df.iloc[self.current_step]["close"]
        prev_equity = self.equity

        # ---- Advance to next step ----
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
            # Avoid stepping beyond available data
            self.current_step = self.n_steps - 1

        row = self.df.iloc[self.current_step]
        price = row["close"]

        # ---- Synthetic event/crash handling (if injected in state) ----
        crash = row.get("synthetic_crash", False)
        vix_spike = row.get("synthetic_vix", False)
        if crash or vix_spike:
            info["synthetic_event"] = True

        # ---- Flatten before expiry or forced close ----
        minutes_left = row.get("minutes_to_close", 30)
        if self.position != 0 and minutes_left <= self.flatten_before_close:
            action = 3  # flatten

        # ---- Execute Action ----
        reward, trade_pnl, position_change = 0, 0, False

        if action == 0:  # Hold
            if self.position == 1:
                trade_pnl = price - prev_price
            elif self.position == -1:
                trade_pnl = prev_price - price
            reward = trade_pnl

        elif action == 1:  # Long
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                self.peak_price = price
                trade_pnl = 0
                position_change = True
            elif self.position == -1:
                reward = self.entry_price - price
                self.position = 1
                self.entry_price = price
                self.peak_price = price
                position_change = True

        elif action == 2:  # Short
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                self.peak_price = price
                trade_pnl = 0
                position_change = True
            elif self.position == 1:
                reward = price - self.entry_price
                self.position = -1
                self.entry_price = price
                self.peak_price = price
                position_change = True

        elif action == 3:  # Flatten
            if self.position == 1:
                reward = price - self.entry_price
            elif self.position == -1:
                reward = self.entry_price - price
            else:
                reward = 0
            self.position = 0
            self.entry_price = 0
            self.peak_price = 0
            position_change = True

        # Transaction cost/slippage (on any position change)
        if position_change:
            cost = self.transaction_cost * price + self.slippage * price
            reward -= cost
            info["transaction_cost"] = cost
            info["position_change"] = True

        # Per-leg stop loss: if unrealized loss > per_leg_stop_loss, force flatten
        if self.position != 0:
            if self.position == 1:
                unrealized = price - self.entry_price
            else:
                unrealized = self.entry_price - price
            if unrealized < -self.per_leg_stop_loss * self.entry_price:
                reward += unrealized  # Realize loss
                self.position = 0
                self.entry_price = 0
                self.peak_price = 0
                info["per_leg_stop"] = True

        # Track peak price for trailing stops
        if self.position == 1:
            self.peak_price = max(self.peak_price, price)
        elif self.position == -1:
            self.peak_price = min(self.peak_price, price)

        # Update equity and max drawdown tracking
        self.equity += reward
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-8)
        info["drawdown"] = drawdown
        info["equity"] = self.equity

        # Drawdown circuit breaker
        if drawdown > self.max_drawdown or self.equity < self.min_equity:
            done = True
            info["drawdown_trigger"] = True
            self.drawdown_triggered = True

        # Save audit info
        info.update({
            "position": self.position,
            "entry_price": self.entry_price,
            "reward": reward,
            "price": price,
            "minutes_left": minutes_left,
            "step": self.current_step,
        })
        self._last_info = info

        next_state = self._get_state()
        return next_state, reward, done, info

    def render(self, mode="human"):
        info = self._last_info
        print(f"Step {info.get('step')}: Position={info.get('position')} Price={info.get('price'):.2f} "
              f"Equity={info.get('equity'):.2f} Drawdown={info.get('drawdown'):.3f}")
