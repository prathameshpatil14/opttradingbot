# live/live_trading.py

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.angel_one_api import AngelOneAPI
from utils.logger import get_logger

logger = get_logger("LiveTrading")

class LiveTradingEngine:

    def __init__(self, env_class, agent_class, config, data_fetch_fn, risk_manager=None):
        self.env_class = env_class
        self.agent_class = agent_class
        self.config = config
        self.data_fetch_fn = data_fetch_fn
        self.risk_manager = risk_manager
        self.broker = AngelOneAPI(config_path="config/keys.yml")

        self.market = config.get("market", {})
        self.live_cfg = config.get("live", {})
        self.risk_cfg = config.get("risk", {})
        self.features_cfg = config.get("features", {})
        self.checkpoint_dir = config.get("agent", {}).get("checkpoint_dir", "models")
        self.trading_symbols = self.market.get("symbols", ["NIFTY"])
        self.start_time = self.market.get("start_time", "09:20")
        self.end_time = self.market.get("end_time", "15:25")
        self.flatten_before_close = self.risk_cfg.get("flatten_before_close_minutes", 15)
        self.save_audit = config.get("backtest", {}).get("save_audit", True)
        self.log_dir = config.get("backtest", {}).get("log_dir", "logs")
        self.online_finetune_every = config.get("agent", {}).get("retrain_every_minutes", 30)
        self.last_finetune = datetime.now()
        self.batch_steps = 0

        self.agent = None
        self.env = None

        self.audit_path = os.path.join(self.log_dir, "live_trading_audit.csv")
        if self.save_audit and not os.path.exists(self.audit_path):
            with open(self.audit_path, "w") as f:
                f.write("dt,symbol,expiry,strike,opt_type,state,action,reward,position,entry,price,equity,drawdown,order_status,info\n")

    def load_agent(self):
        logger.info("Loading agent and checkpoint...")
        self.agent = self.agent_class(**self.config.get("agent", {}))
        try:
            self.agent.load_checkpoint()
            logger.info("Loaded latest agent checkpoint.")
        except Exception as e:
            logger.warning(f"Could not load agent checkpoint: {e}")

    def get_market_time(self):
        now = datetime.now()
        start = datetime.strptime(self.start_time, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        end = datetime.strptime(self.end_time, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        return now, start, end

    def should_flatten(self, minutes_left):
        return minutes_left <= self.flatten_before_close

    def run(self):
        logger.info("Starting live trading engine.")
        self.load_agent()

        while True:
            now, start, end = self.get_market_time()
            if now < start or now > end:
                logger.info("Outside trading hours. Sleeping 60s...")
                time.sleep(60)
                continue

            try:
                df = self.data_fetch_fn()
                if df is None or len(df) == 0:
                    logger.warning("No market data available; retrying...")
                    time.sleep(15)
                    continue

                for symbol in self.trading_symbols:
                    symbol_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df

                    self.env = self.env_class(
                        data=symbol_df,
                        config=self.config.get("market", {}),
                        risk_config=self.risk_cfg,
                        features_config=self.features_cfg,
                        mode="live",
                    )
                    state = self.env.reset()
                    done = False
                    step = 0
                    total_reward = 0
                    equity = 1.0

                    while not done:
                        row = symbol_df.iloc[self.env.current_step]
                        minutes_left = row.get("minutes_to_close", 30)

                        # Option metadata: must be in DataFrame, or default/fallback logic
                        expiry = row.get("expiry") if "expiry" in row else None
                        strike = row.get("strike") if "strike" in row else None
                        option_type = row.get("option_type") if "option_type" in row else None

                        # --- RISK MANAGEMENT ---
                        if self.risk_manager and not self.risk_manager(self.env, state):
                            logger.warning("Risk/circuit breaker triggered; flattening all positions.")
                            self.env.position = 0
                            done = True
                            break

                        # --- RL POLICY ---
                        action = self.agent.select_action(state, eval_mode=True)
                        if self.should_flatten(minutes_left):
                            action = 3  # flatten

                        # --- STEP ENV ---
                        next_state, reward, done, info = self.env.step(action)
                        total_reward += reward
                        equity = info.get("equity", equity)
                        self.batch_steps += 1

                        # --- OPTION TOKEN RESOLUTION ---
                        # If not already available in info, resolve using AngelOne API
                        token = info.get("symboltoken")
                        if not token and expiry and strike and option_type:
                            token = self.broker.resolve_option_token(
                                symbol=symbol,
                                expiry=expiry,
                                strike=strike,
                                option_type=option_type,
                            )
                            info["symboltoken"] = token

                        # --- PLACE ORDER IF ENABLED ---
                        order_status = None
                        if self.live_cfg.get("enable_live_trading", False):
                            order_status = self.place_real_order(symbol, token, expiry, strike, option_type, info, action)

                        # --- LOGGING ---
                        if self.save_audit:
                            with open(self.audit_path, "a") as f:
                                dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                f.write(
                                    f"{dt_str},{symbol},{expiry},{strike},{option_type},{np.array2string(state, precision=3, separator=',')},{action},"
                                    f"{reward},{info.get('position')},{info.get('entry_price')},{info.get('price')},"
                                    f"{equity},{info.get('drawdown')},{order_status},{info}\n"
                                )

                        state = next_state
                        step += 1
                        time.sleep(0.5)

                    logger.info(f"Live session for {symbol} complete. Steps: {step} Reward: {total_reward:.2f}")

                if (datetime.now() - self.last_finetune).total_seconds() > self.online_finetune_every * 60:
                    logger.info("Performing online fine-tuning on recent data...")
                    if hasattr(self.agent, "online_finetune"):
                        self.agent.online_finetune(self.env, steps=3000)
                    self.last_finetune = datetime.now()

                logger.info("Sleeping 60s until next batch...")
                time.sleep(60)

            except Exception as e:
                logger.error(f"Exception in live trading loop: {e}", exc_info=True)
                logger.info("Sleeping 30s and will retry...")
                time.sleep(30)

    def place_real_order(self, symbol, token, expiry, strike, option_type, info, action):
        if not self.live_cfg.get("enable_live_trading", False):
            return "SKIP"
        if not token:
            logger.error(f"Cannot place order: missing option token for {symbol} {expiry} {strike} {option_type}")
            return "NO_TOKEN"
        if action == 0:  # Hold
            return "HOLD"
        qty = self.live_cfg.get("lot_size", 25)

        try:
            if action == 1:  # Buy (Long)
                resp = self.broker.place_order(symbol, token, "BUY", qty, expiry, strike, option_type)
            elif action == 2:  # Sell (Short)
                resp = self.broker.place_order(symbol, token, "SELL", qty, expiry, strike, option_type)
            elif action == 3:  # Flatten
                pos = info.get("position", 0)
                if pos == 1:
                    resp = self.broker.place_order(symbol, token, "SELL", qty, expiry, strike, option_type)
                elif pos == -1:
                    resp = self.broker.place_order(symbol, token, "BUY", qty, expiry, strike, option_type)
                else:
                    resp = "NO_POSITION"
            else:
                resp = "NO_ACTION"
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            resp = "ERROR"
        return str(resp)

