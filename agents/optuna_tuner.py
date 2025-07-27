# agents/optuna_tuner.py

import os
import optuna
import yaml
import numpy as np
import sys
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("OptunaTuner")

class OptunaTuner:
    """
    Hyperparameter search using Optuna with full agent/env creation, logging, checkpointing.
    """
    def __init__(self, agent_class, env_class, param_space, backtest_fn,
                 study_name="rl_hyperparam_search", storage_path=None,
                 n_trials=25, direction="maximize", checkpoint_dir="models", log_dir="logs"):
        self.agent_class = agent_class
        self.env_class = env_class
        self.param_space = param_space
        self.backtest_fn = backtest_fn
        self.n_trials = n_trials
        self.direction = direction
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if storage_path:
            self.storage = f"sqlite:///{storage_path}"
        else:
            self.storage = None

        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            load_if_exists=True
        )
        self.best_params_path = os.path.join(log_dir, "optuna_best_params.yml")
        self.trials_log_path = os.path.join(log_dir, "optuna_trials.csv")

    def objective(self, trial):
        # Sample hyperparameters
        params = self.param_space(trial)
        logger.info(f"Trial {trial.number}: Hyperparameters: {params}")
        # Create agent and env with sampled hyperparams
        try:
            env = self.env_class(**params.get('env', {}))
            agent = self.agent_class(env.state_dim, env.action_dim, params.get('agent', {}))
            score = self.backtest_fn(agent, env, params)
            logger.info(f"Trial {trial.number} result: {score:.4f}")
            # Optionally save agent checkpoint if best so far
            if self.direction == "maximize":
                is_best = (self.study.best_value is None) or (score > self.study.best_value)
            else:
                is_best = (self.study.best_value is None) or (score < self.study.best_value)
            if is_best:
                agent.save_checkpoint(f"optuna_best_{self.agent_class.__name__}.pth")
                # Save BOTH the param dict and Optuna's flat trial.params
                self.save_best_params({"sampled_params": params, "flat_params": trial.params}, trial)
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            score = float('-inf') if self.direction == "maximize" else float('inf')
        self.log_trial(trial, params, score)
        return score

    def save_best_params(self, params, trial):
        """Save the best hyperparameters and Optuna trial info to YAML."""
        with open(self.best_params_path, "w") as f:
            yaml.dump({"trial": trial.number, "params": params}, f)
        logger.info(f"Saved best params (trial {trial.number}) to {self.best_params_path}")

    def log_trial(self, trial, params, score):
        """Append trial info to CSV log."""
        write_header = not os.path.exists(self.trials_log_path)
        with open(self.trials_log_path, "a") as f:
            if write_header:
                f.write("trial,params,score,time\n")
            f.write(f"{trial.number},{params},{score},{datetime.now()}\n")

    def optimize(self, show_progress=True):
        logger.info(f"Starting Optuna hyperparameter search: {self.n_trials} trials")
        try:
            use_bar = show_progress and sys.stdout.isatty()
        except Exception:
            use_bar = False
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=use_bar)
        logger.info(f"Optuna optimization complete. Best value: {self.study.best_value}")
        # Save best params for reference (flat dict and YAML)
        best_trial = self.study.best_trial
        self.save_best_params({"flat_params": best_trial.params}, best_trial)
        return best_trial

    def get_best_params(self):
        """Get the best found hyperparameters (from Optuna)."""
        return self.study.best_trial.params

    def plot_optimization_history(self):
        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self.study)
        except Exception as e:
            logger.error(f"Could not plot optimization history: {e}")
            return None
