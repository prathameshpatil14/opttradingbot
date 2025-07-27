import os
import argparse
import yaml
import time
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.utils import load_yaml, set_seed
from features.feature_engineering import compute_features
from envs.option_trading_env import OptionTradingEnv
from risk.risk_manager import RiskManager
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.ensemble_manager import EnsembleManager
from agents.optuna_tuner import OptunaTuner
from backtest.backtest_runner import BacktestRunner
from live.live_trading import LiveTradingEngine
from utils.angel_one_api import AngelOneAPI

logger = get_logger("Main")

def parse_args():
    parser = argparse.ArgumentParser(description="Modular RL Option Trading Bot")
    parser.add_argument("--mode", choices=["train", "backtest", "live", "tune", "eval"], default="train")
    parser.add_argument("--config", default="config/config.yml", help="Config YAML file")
    parser.add_argument("--hyperparams", default="config/hyperparams.yml", help="Hyperparams YAML file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--autoloop", action="store_true", help="Run train->backtest->tune->eval in a loop")
    return parser.parse_args()

def instantiate_agent(agent_cfg, env):
    t = agent_cfg.get("type", "dqn").lower()
    if t == "dqn":
        return DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n, config=agent_cfg)
    elif t == "ppo":
        agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_space.n, config=agent_cfg)
        agent.set_env(env)
        return agent
    elif t == "ensemble":
        agents = {}
        dqn = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n, config=agent_cfg)
        ppo = PPOAgent(state_dim=env.state_dim, action_dim=env.action_space.n, config=agent_cfg)
        ppo.set_env(env)
        agents["dqn"] = dqn
        agents["ppo"] = ppo
        ens_cfg = agent_cfg.get("ensemble", {})
        return EnsembleManager(
            agent_dict=agents,
            voting=ens_cfg.get("voting", "weighted"),
            weights=ens_cfg.get("weights", {"dqn": 0.5, "ppo": 0.5})
        )
    else:
        raise ValueError(f"Unknown agent type: {t}")

def load_data(config):
    data_path = config.get("data", {}).get("historical_data_path", "data/historical_data.csv")
    logger.info(f"Loading historical data from {data_path}")
    df = pd.read_csv(data_path)
    return df

def fetch_live_data_from_angel_one(config):
    market_cfg = config.get("market", {})
    symbols = market_cfg.get("symbols", ["NIFTY"])
    expiry_days_before = market_cfg.get("expiry_days_before", 2)
    angel = AngelOneAPI(config_path=config.get("live", {}).get("api_keys_path", "config/keys.yml"))
    dfs = []
    for symbol in symbols:
        oc = angel.get_option_chain(symbol, None)
        if oc.empty:
            logger.warning(f"No option chain for {symbol}, skipping...")
            continue
        expiries = sorted(oc['expiry'].unique())
        if not expiries:
            logger.warning(f"No expiries found for {symbol}, skipping...")
            continue
        nearest_expiry = expiries[0]
        df_exp = oc[oc['expiry'] == nearest_expiry].copy()
        spot_ltp = None
        try:
            spot_ltp = angel.get_ltp(symbol, token=symbol, exchange="NSE")
        except Exception as e:
            logger.warning(f"Could not fetch spot LTP for {symbol}: {e}")
        if spot_ltp is None:
            spot_ltp = df_exp["strike"].iloc[df_exp['open_interest'].idxmax()]
        atm_strike = min(df_exp["strike"], key=lambda x: abs(x - spot_ltp))
        for option_type in ["CE", "PE"]:
            row = df_exp[(df_exp["strike"] == atm_strike) & (df_exp["option_type"] == option_type)]
            if not row.empty:
                row = row.iloc[0]
                data = {
                    "symbol": symbol,
                    "date": pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S"),
                    "open": row.get("open", 0),
                    "high": row.get("high", 0),
                    "low": row.get("low", 0),
                    "close": row.get("close", 0),
                    "volume": row.get("volume", 0),
                    "strike": row.get("strike", 0),
                    "expiry": row.get("expiry"),
                    "option_type": option_type,
                    "iv": row.get("implied_volatility", row.get("iv", 0.15)),
                    "days_to_expiry": (pd.to_datetime(row.get("expiry")) - pd.to_datetime("now")).days,
                    "risk_free_rate": 0.06,
                    "open_interest": row.get("open_interest", 0),
                }
                dfs.append(pd.DataFrame([data]))
    if not dfs:
        logger.error("No live market data fetched from Angel One.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df

# ---- Modular Step Functions ----

def do_train(env, agent):
    logger.info("Starting training loop...")
    if isinstance(agent, EnsembleManager):
        for name, subagent in agent.agents.items():
            logger.info(f"Training ensemble agent: {name} ...")
            if hasattr(subagent, "train"):
                if isinstance(subagent, PPOAgent):
                    subagent.train(total_timesteps=100_000)
                elif isinstance(subagent, DQNAgent):
                    subagent.train(num_steps=100_000, env=env)
                else:
                    try:
                        subagent.train(env=env)
                    except TypeError:
                        subagent.train()
                logger.info(f"Training complete for {name}.")
            else:
                logger.error(f"{name} does not support .train() method.")
        logger.info("All ensemble agents trained.")
    elif hasattr(agent, "train"):
        if isinstance(agent, PPOAgent):
            agent.train(total_timesteps=100_000, env=env)
        elif isinstance(agent, DQNAgent):
            agent.train(num_steps=100_000, env=env)
        else:
            try:
                agent.train(env=env)
            except TypeError:
                agent.train()
        logger.info("Training complete.")
    else:
        logger.error("Agent does not support .train() method.")

def load_agent_checkpoints(agent):
    """
    Ensures DQN/PPO checkpoints are loaded before backtest/tune/eval
    """
    if isinstance(agent, EnsembleManager):
        if "ppo" in agent.agents and hasattr(agent.agents["ppo"], "load_checkpoint"):
            agent.agents["ppo"].load_checkpoint("ppo_agent_ppo.zip")
        if "dqn" in agent.agents and hasattr(agent.agents["dqn"], "load_checkpoint"):
            agent.agents["dqn"].load_checkpoint("dqn_agent.pth")
    elif isinstance(agent, PPOAgent):
        agent.load_checkpoint("ppo_agent_ppo.zip")
    elif isinstance(agent, DQNAgent):
        agent.load_checkpoint("dqn_agent.pth")

def do_backtest(env, agent, feat_df, config, args, risk_mgr):
    logger.info("Running backtest...")
    load_agent_checkpoints(agent)
    runner = BacktestRunner(
        env_class=OptionTradingEnv,
        agent=agent,
        data=feat_df,
        log_dir=config.get("backtest", {}).get("log_dir", "logs"),
        audit=config.get("backtest", {}).get("save_audit", True),
        market_config=config.get("market", {}),
        risk_config=config.get("risk", {}),
        features_config=config.get("features", {})
    )
    summary = runner.run(max_steps=args.max_steps, risk_manager=risk_mgr.check_risk)
    logger.info(f"Backtest summary: {summary}")

def do_tune(feat_df, config, args, risk_mgr):
    logger.info("Starting Optuna hyperparameter tuning...")
    def param_space(trial):
        return {
            "agent": {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "epsilon_decay": trial.suggest_float("epsilon_decay", 0.990, 0.9999),
                "num_quantiles": trial.suggest_categorical("num_quantiles", [32, 51, 101]),
                "buffer_size": trial.suggest_int("buffer_size", 20000, 200000),
            },
            "env": {
                "transaction_cost": trial.suggest_float("transaction_cost", 0.0005, 0.0015),
            }
        }
    def backtest_fn(agent, env, params):
        load_agent_checkpoints(agent)
        market_cfg = dict(config.get("market", {}))
        risk_cfg = dict(config.get("risk", {}))
        features_cfg = dict(config.get("features", {}))
        # Inject transaction_cost from params['env'] if present
        if "transaction_cost" in params["env"]:
            risk_cfg["transaction_cost"] = params["env"]["transaction_cost"]

        runner = BacktestRunner(
            env_class=OptionTradingEnv,
            agent=agent,
            data=feat_df,
            log_dir=config.get("backtest", {}).get("log_dir", "logs"),
            audit=False,
            market_config=market_cfg,
            risk_config=risk_cfg,
            features_config=features_cfg
        )
        summary = runner.run(max_steps=args.max_steps, risk_manager=risk_mgr.check_risk, verbose=False)
        return summary[config.get("optuna", {}).get("metric", "total_return")]
    tuner = OptunaTuner(
        agent_class=DQNAgent,
        env_class=OptionTradingEnv,
        param_space=param_space,
        backtest_fn=backtest_fn,
        n_trials=config.get("optuna", {}).get("n_trials", 30),
        direction=config.get("optuna", {}).get("direction", "maximize"),
        storage_path=config.get("optuna", {}).get("storage_path", "optuna_trials.db"),
        log_dir=config.get("backtest", {}).get("log_dir", "logs")
    )
    best_trial = tuner.optimize()
    logger.info(f"Best Optuna trial: {best_trial}")

def do_eval(env, agent):
    logger.info("Evaluating agent performance (deterministic, no exploration)...")
    load_agent_checkpoints(agent)
    if hasattr(agent, "evaluate"):
        # Accepts either argument, but use num_episodes for EnsembleManager compatibility
        agent.evaluate(env, num_episodes=5)
    else:
        logger.error("Agent does not support .evaluate() method.")

def main():
    args = parse_args()
    set_seed(args.seed)
    config = load_yaml(args.config)
    hyperparams = load_yaml(args.hyperparams)

    # ---- Load data and compute features ----
    df = load_data(config)
    features_cfg = config.get("features", {})
    feat_df = compute_features(df, features_cfg)
    logger.info(f"Loaded raw data shape: {df.shape}")
    logger.info(f"Features DataFrame shape after engineering: {feat_df.shape}")

    # ---- Instantiate env & risk manager ----
    env = OptionTradingEnv(
        data=feat_df,
        config=config.get("market", {}),
        risk_config=config.get("risk", {}),
        features_config=features_cfg,
        mode="train"
    )
    risk_mgr = RiskManager(config.get("risk", {}))
    agent_cfg = config.get("agent", {})
    agent = instantiate_agent(agent_cfg, env)

    if getattr(args, "autoloop", False):
        logger.info("Starting AUTO-LOOP: train -> backtest -> tune -> eval (Ctrl+C to stop)")
        while True:
            try:
                do_train(env, agent)
                do_backtest(env, agent, feat_df, config, args, risk_mgr)
                do_tune(feat_df, config, args, risk_mgr)
                do_eval(env, agent)
                logger.info("Cycle complete. Sleeping 60 seconds before next cycle.")
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Auto-loop stopped by user.")
                break
            except Exception as e:
                logger.error(f"Exception in auto-loop: {e}", exc_info=True)
                logger.info("Sleeping 60 seconds before retrying cycle.")
                time.sleep(60)
    else:
        mode = args.mode.lower()
        if mode == "train":
            do_train(env, agent)
        elif mode == "backtest":
            do_backtest(env, agent, feat_df, config, args, risk_mgr)
        elif mode == "tune":
            do_tune(feat_df, config, args, risk_mgr)
        elif mode == "eval":
            do_eval(env, agent)
        elif mode == "live":
            logger.info("Launching live trading loop...")
            def fetch_live_data():
                df = fetch_live_data_from_angel_one(config)
                if df.empty:
                    logger.error("No live market data.")
                return df
            engine = LiveTradingEngine(
                env_class=OptionTradingEnv,
                agent_class=type(agent),
                config=config,
                data_fetch_fn=fetch_live_data,
                risk_manager=risk_mgr.check_risk
            )
            engine.run()
        else:
            logger.error(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
