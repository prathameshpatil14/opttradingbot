# agents/dqn_agent.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from utils.logger import get_logger

logger = get_logger('DQNAgent')

# ------- QR-DQN: Quantile Regression DQN Network -------
class QuantileDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=51, hidden_dim=256):
        super(QuantileDQN, self).__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * num_quantiles)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        q = q.view(-1, self.action_dim, self.num_quantiles)
        return q  # Shape: [batch, action, quantile]

# ------- Experience Replay Buffer (with optional Prioritization) -------
class ReplayBuffer:
    def __init__(self, capacity, prioritized=False, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.prioritized = prioritized
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.alpha = alpha

    def push(self, transition):
        self.memory.append(transition)
        if self.prioritized:
            # Fix: If buffer is empty, initialize priorities with 1.0
            if len(self.priorities) == 0:
                self.priorities.append(1.0)
            else:
                self.priorities.append(max(self.priorities, default=1.0))

    def sample(self, batch_size, beta=0.4):
        if not self.prioritized:
            batch = random.sample(self.memory, batch_size)
            weights = np.ones(batch_size)
            idxs = range(batch_size)
        else:
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()
            idxs = np.random.choice(len(self.memory), batch_size, p=probs)
            batch = [self.memory[idx] for idx in idxs]
            # Importance-sampling weights
            weights = (len(self.memory) * probs[idxs]) ** (-beta)
            weights /= weights.max()
        return batch, weights, idxs

    def update_priorities(self, idxs, td_errors):
        # Make sure idxs are within bounds (can happen with buffer wraparound)
        for idx, td in zip(idxs, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.memory)

# ------- DQN Agent -------
class DQNAgent:
    """
    Distributional QR-DQN agent with support for prioritized replay and double DQN.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        config,
        device=None,
        log_dir="logs",
        checkpoint_dir="models",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configs
        self.lr = config.get("lr", 1e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 1e-3)  # for soft update
        self.batch_size = config.get("batch_size", 64)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.update_target_every = config.get("update_target_every", 500)
        self.checkpoint_dir = checkpoint_dir
        self.num_quantiles = config.get("num_quantiles", 51)
        self.prioritized_replay = config.get("prioritized_replay", False)
        self.double_dqn = config.get("double_dqn", True)
        self.save_audit = config.get("save_audit", True)
        self.audit_path = os.path.join(log_dir, "dqn_audit.csv")
        self.quantile_embed_dim = config.get("quantile_embed_dim", 64)

        # Networks
        self.q_net = QuantileDQN(state_dim, action_dim, self.num_quantiles).to(self.device)
        self.target_net = QuantileDQN(state_dim, action_dim, self.num_quantiles).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.get("buffer_size", 100_000),
            prioritized=self.prioritized_replay,
        )
        self.transition = namedtuple(
            "Transition",
            ("state", "action", "reward", "next_state", "done", "info"),
        )

        self.steps = 0

        # Audit log setup
        if self.save_audit and not os.path.exists(self.audit_path):
            with open(self.audit_path, "w") as f:
                f.write("step,state,action,reward,q_value,info\n")

        # For resuming training
        self.load_checkpoint_if_exists()

    def select_action(self, state, eval_mode=False):
        """
        Epsilon-greedy action selection.
        In eval_mode, disables exploration.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                quantiles = self.q_net(state)
                q_values = quantiles.mean(dim=2)  # Average over quantiles
                action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done, info=None):
        """Store a transition for replay."""
        tr = self.transition(state, action, reward, next_state, done, info)
        self.replay_buffer.push(tr)

    def train_step(self, beta=0.4):
        """Train one batch of QR-DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions, weights, idxs = self.replay_buffer.sample(self.batch_size, beta=beta)
        batch = self.transition(*zip(*transitions))

        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Compute quantile Q-values for current and next states
        quantiles = self.q_net(state)  # [batch, action, num_quantiles]
        current_quantiles = quantiles.gather(1, action.unsqueeze(2).expand(-1, 1, self.num_quantiles)).squeeze(1)

        # Double DQN for next action selection
        with torch.no_grad():
            if self.double_dqn:
                next_quantiles_online = self.q_net(next_state)
                next_q_values_online = next_quantiles_online.mean(dim=2)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_quantiles_target = self.target_net(next_state)
                next_quantiles = next_quantiles_target.gather(1, next_actions.unsqueeze(2).expand(-1, 1, self.num_quantiles)).squeeze(1)
            else:
                next_quantiles = self.target_net(next_state).max(1)[0]
            target_quantiles = reward + (1 - done) * self.gamma * next_quantiles

        # Quantile Huber Loss
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber = torch.where(td_errors.abs() < 1, 0.5 * td_errors.pow(2), td_errors.abs() - 0.5)
        # Ensure tau shape is [1, num_quantiles]
        tau = torch.linspace(0.0, 1.0, self.num_quantiles + 1, device=self.device)[1:].view(1, -1)
        quantile_loss = (tau - (td_errors.detach() < 0).float()).abs() * huber
        quantile_loss = quantile_loss.mean(dim=2).mean(dim=1)
        # Importance sampling weights for prioritized replay
        quantile_loss = quantile_loss * torch.FloatTensor(weights).to(self.device)
        loss = quantile_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        # Update priorities
        if self.prioritized_replay:
            td_errs = quantile_loss.detach().cpu().numpy()
            self.replay_buffer.update_priorities(idxs, td_errs)

        # Soft target network update
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.soft_update(self.q_net, self.target_net)

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def soft_update(self, net, target_net):
        """Polyak averaging for target network."""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, name="dqn_agent.pth"):
        """Save agent weights and optimizer state."""
        save_path = os.path.join(self.checkpoint_dir, name)
        # Ensure the checkpoint directory exists before saving
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
        }, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    def load_checkpoint_if_exists(self, name="dqn_agent.pth"):
        """Resume training if checkpoint exists."""
        save_path = os.path.join(self.checkpoint_dir, name)
        if os.path.exists(save_path):
            data = torch.load(save_path, map_location=self.device)
            self.q_net.load_state_dict(data["q_net"])
            self.target_net.load_state_dict(data["target_net"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.steps = data.get("steps", 0)
            self.epsilon = data.get("epsilon", self.epsilon)
            logger.info(f"Checkpoint loaded: {save_path}")

    def audit_log(self, step, state, action, reward, q_value, info=None):
        """Log each step for explainability."""
        if not self.save_audit:
            return
        with open(self.audit_path, "a") as f:
            s_str = np.array2string(np.array(state), precision=3, separator=",")
            q_str = np.array2string(q_value, precision=3, separator=",")
            info_str = str(info) if info else ""
            f.write(f"{step},{s_str},{action},{reward},{q_str},{info_str}\n")

    def train(self, num_steps, env, online_finetune=False):
        """
        Main training loop.
        If online_finetune=True, use for live/paper online adaptation.
        """
        state = env.reset()
        for step in range(num_steps):
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)

            self.store_transition(state, action, reward, next_state, done, info)
            quantiles = self.q_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            q_value = quantiles.mean(dim=2).detach().cpu().numpy()[0]
            self.audit_log(step, state, action, reward, q_value, info)

            self.train_step()

            state = next_state
            if done:
                state = env.reset()

            if step % 500 == 0:
                self.save_checkpoint()

        self.save_checkpoint()

    def online_finetune(self, env, steps=1000):
        """
        Alias for live online training during market hours.
        """
        self.train(num_steps=steps, env=env, online_finetune=True)

    def evaluate(self, env, num_episodes=10):
        """
        Evaluate agent with greedy policy (no exploration).
        """
        returns = []
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            ep_return = 0
            while not done:
                action = self.select_action(state, eval_mode=True)
                next_state, reward, done, info = env.step(action)
                ep_return += reward
                state = next_state
            returns.append(ep_return)
            logger.info(f"Eval Episode {ep+1}: Return={ep_return:.2f}")
        mean_return = np.mean(returns)
        logger.info(f"Eval Mean Return over {num_episodes} episodes: {mean_return:.2f}")
        return mean_return