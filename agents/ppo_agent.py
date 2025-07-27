# agents/ppo_agent.py

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.logger import get_logger

logger = get_logger("PPOAgent")

class PPOAgent:
    """
    PPOAgent wraps Stable Baselines3 PPO, with support for audit logging, 
    checkpointing, dynamic position sizing, and plug-and-play ensemble usage.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        config,
        env=None,
        checkpoint_dir="models",
        log_dir="logs",
        save_audit=True,
        name="ppo_agent",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_audit = save_audit
        self.name = name
        self.audit_path = os.path.join(log_dir, f"{name}_audit.csv")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Ensure log and checkpoint directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Stable Baselines3 PPO parameters
        self.learning_rate = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.n_steps = config.get("n_steps", 2048)
        self.policy_kwargs = config.get("policy_kwargs", dict(net_arch=[256, 256]))

        # Use DummyVecEnv for SB3 compatibility
        if env is not None:
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = None  # You must set_env() before training

        # The PPO model (created on first train if env provided)
        self.model = None

        # Prepare audit log
        if self.save_audit and not os.path.exists(self.audit_path):
            with open(self.audit_path, "w") as f:
                f.write("step,state,action,reward,prob,value,entropy,info\n")

    def set_env(self, env):
        self.env = DummyVecEnv([lambda: env])

    def train(self, total_timesteps=100_000, save_every=5000, callback=None):
        """
        Trains PPO agent. Supports online/offline training. 
        Periodically saves checkpoint and logs.
        """
        if self.env is None:
            raise ValueError("Environment not set. Call set_env(env) first.")

        logger.info(f"Training PPO for {total_timesteps} timesteps...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            batch_size=self.batch_size,
            ent_coef=self.ent_coef,
            n_steps=self.n_steps,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            verbose=1,
        )

        # Optionally provide callback for live checkpointing/logging
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.save_checkpoint()

    def online_finetune(self, env, steps=10000):
        """
        Fine-tune PPO agent on recent data (used during market hours).
        """
        self.set_env(env)
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                batch_size=self.batch_size,
                ent_coef=self.ent_coef,
                n_steps=self.n_steps,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                verbose=0,
            )
        logger.info(f"Online finetuning PPO for {steps} timesteps...")
        self.model.learn(total_timesteps=steps)
        self.save_checkpoint()

    def select_action(self, state, eval_mode=True, position_scaling=True):
        """
        Returns action from policy, optionally with dynamic sizing (confidence-based).
        """
        if self.model is None:
            raise RuntimeError("PPO model not initialized. Train or load a checkpoint first.")
        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=eval_mode)
        # Optionally use value estimate as confidence for dynamic sizing
        if position_scaling:
            with torch.no_grad():
                tensor_obs = torch.from_numpy(obs).float().to(self.device)
                value = self.model.policy.predict_values(tensor_obs).cpu().numpy()[0]
                conf = float(np.tanh(value))  # Scale [-1,1]
                logger.debug(f"Confidence (value): {conf:.3f}")
        else:
            conf = 1.0
        return int(action[0]), conf  # Return action and confidence

    def act_and_log(self, state, step, reward=None, info=None):
        """
        Returns action and logs state, action, reward, probability, value, entropy.
        """
        if self.model is None:
            raise RuntimeError("PPO model not initialized. Train or load a checkpoint first.")
        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        # Get policy details for explainability
        with torch.no_grad():
            tensor_obs = torch.from_numpy(obs).float().to(self.device)
            dist = self.model.policy.get_distribution(tensor_obs)
            value = self.model.policy.predict_values(tensor_obs).cpu().numpy()[0]
            try:
                entropy = dist.entropy().cpu().numpy()[0]
            except Exception:
                entropy = 0.0
            try:
                prob = np.exp(dist.distribution.logits.cpu().numpy())[0]
            except Exception:
                prob = 0.0
        # Log everything
        if self.save_audit:
            with open(self.audit_path, "a") as f:
                state_str = np.array2string(np.array(state), precision=3, separator=",")
                prob_str = np.array2string(prob, precision=3, separator=",")
                f.write(f"{step},{state_str},{action[0]},{reward},{prob_str},{value},{entropy},{info}\n")
        return int(action[0])

    def save_checkpoint(self, name=None):
        if not name:
            name = f"{self.name}_ppo.zip"
        save_path = os.path.join(self.checkpoint_dir, name)
        self.model.save(save_path)
        logger.info(f"PPO checkpoint saved to {save_path}")

    def load_checkpoint(self, name=None):
        if not name:
            name = f"{self.name}_ppo.zip"
        load_path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(load_path):
            self.model = PPO.load(load_path, env=self.env, device=self.device)
            logger.info(f"PPO checkpoint loaded from {load_path}")

    def evaluate(self, env, episodes=5, position_scaling=True, regime_fn=None):
        """
        Runs evaluation loop, logs return and risk.
        """
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                regime = regime_fn(state) if regime_fn else None
                action, conf = self.select_action(state, eval_mode=True, position_scaling=position_scaling)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                self.act_and_log(state, step, reward, info)
                state = next_state
                step += 1
            returns.append(total_reward)
            logger.info(f"PPO Eval Episode {ep+1}: Return={total_reward:.2f}")
        mean_return = np.mean(returns)
        logger.info(f"PPO Eval Mean Return: {mean_return:.2f}")
        return mean_return

