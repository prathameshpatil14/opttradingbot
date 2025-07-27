# agents/ensemble_manager.py

import os
import numpy as np
import torch
from collections import defaultdict, Counter
from utils.logger import get_logger

logger = get_logger('EnsembleManager')

class EnsembleManager:
    """
    RL Ensemble agent supporting various voting strategies and regime-based agent selection.
    """

    def __init__(self, agent_dict, voting='majority', weights=None, audit_log=True, log_dir="logs"):
        self.agents = agent_dict
        self.agent_names = list(agent_dict.keys())
        self.voting = voting
        self.audit_log = audit_log
        self.weights = weights or {name: 1.0 for name in self.agent_names}
        self.log_path = f"{log_dir}/ensemble_audit.csv"
        self.last_regime = None  # For regime switching

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Add header for audit log (with extra_info)
        if audit_log and not self._file_exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("step,state,actions,q_values,selected,regime,extra_info\n")

    def _file_exists(self, path):
        try:
            open(path, "r").close()
            return True
        except FileNotFoundError:
            return False

    def update_weights(self, new_weights):
        """Update agent voting weights dynamically (e.g., by recent Sharpe, regime, etc.)"""
        self.weights.update(new_weights)

    def select_action(self, state, step=None, eval_mode=False, regime=None, extra_info=None):
        """
        Returns final action by combining all agent suggestions according to voting method.
        Optionally supports regime switching (only some agents vote in some regimes).
        """
        action_votes = []
        q_values_dict = {}

        # Regime-based voting (optional)
        agent_names = self.agent_names
        if regime and hasattr(self, "regime_agents"):
            agent_names = self.regime_agents.get(regime, self.agent_names)
            self.last_regime = regime

        # Get each agent's action & (if available) q_values
        for name in agent_names:
            agent = self.agents[name]
            action = agent.select_action(state, eval_mode=eval_mode)
            action_votes.append(action)
            # Try to get Q values for explainability
            if hasattr(agent, "q_net"):
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    quantiles = agent.q_net(state_t)
                    q_values = quantiles.mean(dim=2).cpu().numpy()[0]
                q_values_dict[name] = q_values.tolist()
            else:
                q_values_dict[name] = None

        # Voting logic
        if self.voting == 'majority':
            selected = Counter(action_votes).most_common(1)[0][0]
        elif self.voting == 'weighted':
            action_weight_sum = defaultdict(float)
            for i, action in enumerate(action_votes):
                w = self.weights.get(agent_names[i], 1.0)  # More robust
                action_weight_sum[action] += w
            selected = max(action_weight_sum.items(), key=lambda x: x[1])[0]
        elif self.voting == 'mean_q':
            all_qs = [np.array(q_values_dict[name]) for name in agent_names if q_values_dict[name] is not None]
            if all_qs:
                mean_q = np.mean(all_qs, axis=0)
                selected = np.argmax(mean_q)
            else:
                selected = Counter(action_votes).most_common(1)[0][0]
        else:
            selected = Counter(action_votes).most_common(1)[0][0]

        # Audit voting trace for explainability
        if self.audit_log:
            self._log_vote(step, state, action_votes, q_values_dict, selected, regime, extra_info)

        return selected

    def _log_vote(self, step, state, action_votes, q_values_dict, selected, regime, extra_info):
        with open(self.log_path, "a") as f:
            state_str = np.array2string(np.array(state), precision=3, separator=",")
            votes_str = "|".join(map(str, action_votes))
            q_str = "|".join(str(q_values_dict.get(k)) for k in self.agent_names)
            regime_str = regime if regime else ""
            info_str = str(extra_info) if extra_info else ""
            f.write(f"{step},{state_str},{votes_str},{q_str},{selected},{regime_str},{info_str}\n")

    def evaluate(self, env, episodes=5, eval_mode=True, regime_fn=None, **kwargs):
        """
        Evaluate ensemble agent over multiple episodes, log returns.
        regime_fn: function(state) -> regime, if using regime-based voting
        """
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            t = 0
            while not done:
                regime = regime_fn(state) if regime_fn else None
                action = self.select_action(state, step=t, eval_mode=eval_mode, regime=regime)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                t += 1
            logger.info(f"Ensemble Evaluation Ep {ep+1}: Return={total_reward:.2f}")
            returns.append(total_reward)
        mean_return = np.mean(returns)
        logger.info(f"Ensemble Eval Mean Return over {episodes} episodes: {mean_return:.2f}")
        return mean_return

    def set_regime_agents(self, regime_agents_dict):
        """
        Allows you to set regime-specific agent voting:
        Example: {"bull": ["dqn", "ppo"], "bear": ["ppo", "a2c"], ...}
        """
        self.regime_agents = regime_agents_dict
