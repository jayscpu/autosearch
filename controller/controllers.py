"""
Controller implementations for ECHO model-switching simulation.

All controllers implement the same interface: select_model() returns a model
tier index {0=nano, 1=small, 2=medium} given the predicted miss rate and
current state. Controllers operate sequentially on the prediction stream —
they cannot look ahead (except MPC which extrapolates from history).
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import deque

from .models import MODELS, required_model, energy_per_window, switching_energy


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class Controller(ABC):
    """Base class for all model-switching controllers."""

    @abstractmethod
    def select_model(self, pred_miss_rate: float, t: int, state: dict) -> int:
        """Select model tier {0, 1, 2} for current decision window.

        Args:
            pred_miss_rate: LSTM-predicted miss rate for this window.
            t: Frame/window index in the sequence.
            state: Mutable dict carried across steps (for controllers with memory).

        Returns:
            Model tier index: 0=nano, 1=small, 2=medium.
        """
        ...

    def reset(self):
        """Reset internal state for a new evaluation run."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Human-readable controller name."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Baselines (1–5)
# ═══════════════════════════════════════════════════════════════════════════════

class AlwaysNano(Controller):
    """Always selects the cheapest model (nano). Minimum energy, worst detection."""

    def select_model(self, pred_miss_rate, t, state):
        return 0

    def name(self):
        return "AlwaysNano"


class AlwaysMedium(Controller):
    """Always selects the most capable model (medium). Best detection, max energy."""

    def select_model(self, pred_miss_rate, t, state):
        return 2

    def name(self):
        return "AlwaysMedium"


class BestFixed(Controller):
    """Always selects a fixed model tier (default: small)."""

    def __init__(self, model_idx: int = 1):
        self.model_idx = model_idx

    def select_model(self, pred_miss_rate, t, state):
        return self.model_idx

    def name(self):
        return f"BestFixed({MODELS[self.model_idx]['name']})"


class RandomController(Controller):
    """Uniform random model selection, seeded for reproducibility."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def select_model(self, pred_miss_rate, t, state):
        return self.rng.randint(0, 3)

    def reset(self):
        self.rng = np.random.RandomState(self.seed)

    def name(self):
        return "Random"


class OracleController(Controller):
    """Optimal controller using ground-truth miss rates.

    Picks the cheapest model that is adequate for the true scene difficulty.
    No real system can achieve this — it serves as the theoretical lower bound.
    """

    def __init__(self, t1: float = 0.15, t2: float = 0.35):
        self.t1 = t1
        self.t2 = t2
        self.true_miss_rates = None

    def set_ground_truth(self, true_miss_rates: np.ndarray):
        self.true_miss_rates = true_miss_rates

    def select_model(self, pred_miss_rate, t, state):
        return required_model(self.true_miss_rates[t], self.t1, self.t2)

    def name(self):
        return "Oracle"


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Controllers (6–10)
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdController(Controller):
    """Simple threshold-based model selection.

    Maps predicted miss rate to model tier using two thresholds:
      pred < t1       -> nano   (easy scene)
      t1 <= pred < t2 -> small  (moderate)
      pred >= t2      -> medium (hard scene)

    Zero overhead, fully interpretable.
    """

    def __init__(self, t1: float = 0.15, t2: float = 0.35):
        self.t1 = t1
        self.t2 = t2

    def select_model(self, pred_miss_rate, t, state):
        if pred_miss_rate < self.t1:
            return 0
        elif pred_miss_rate < self.t2:
            return 1
        else:
            return 2

    def name(self):
        return f"Threshold(t1={self.t1:.3f}, t2={self.t2:.3f})"


class ThresholdHysteresisController(Controller):
    """Threshold controller with hysteresis to prevent oscillation.

    Only switches model after hysteresis_n consecutive windows recommend the
    same new model tier. Implements Algorithm 2 from the K-Dense review.
    """

    def __init__(self, t1: float = 0.15, t2: float = 0.35, hysteresis_n: int = 3):
        self.t1 = t1
        self.t2 = t2
        self.hysteresis_n = hysteresis_n
        self._current_model = 1  # start with small
        self._pending_model = None
        self._pending_count = 0

    def select_model(self, pred_miss_rate, t, state):
        # Compute what threshold logic would recommend
        if pred_miss_rate < self.t1:
            recommended = 0
        elif pred_miss_rate < self.t2:
            recommended = 1
        else:
            recommended = 2

        # If recommendation matches current model, reset pending counter
        if recommended == self._current_model:
            self._pending_model = None
            self._pending_count = 0
            return self._current_model

        # Track consecutive recommendations for a different model
        if recommended == self._pending_model:
            self._pending_count += 1
        else:
            self._pending_model = recommended
            self._pending_count = 1

        # Switch only after hysteresis_n consecutive same recommendations
        if self._pending_count >= self.hysteresis_n:
            self._current_model = recommended
            self._pending_model = None
            self._pending_count = 0

        return self._current_model

    def reset(self):
        self._current_model = 1
        self._pending_model = None
        self._pending_count = 0

    def name(self):
        return (f"ThreshHyst(t1={self.t1:.3f}, t2={self.t2:.3f}, "
                f"n={self.hysteresis_n})")


class BayesRiskMPCController(Controller):
    """Model Predictive Control using Bayes risk minimization.

    From thesis Chapter 4: converts predicted miss rate to class probabilities,
    computes Bayes risk for each model candidate, then uses MPC lookahead to
    find the optimal model sequence over a short horizon.

    With only 3 models and small horizons (2-5), exhaustive enumeration of all
    3^H sequences is tractable (max 243 sequences for H=5).
    """

    def __init__(self, horizon: int = 3, lambda_under: float = 1.0,
                 lambda_over: float = 0.3, w_switch: float = 0.01,
                 t1: float = 0.15, t2: float = 0.35):
        self.horizon = horizon
        self.lambda_under = lambda_under
        self.lambda_over = lambda_over
        self.w_switch = w_switch
        self.t1 = t1
        self.t2 = t2
        self._history = deque(maxlen=5)
        self._current_model = 1

        # Precompute all 3^H model sequences for the given horizon
        self._sequences = self._enumerate_sequences(horizon)

        # Normalized energy costs (relative to medium)
        e_med = energy_per_window(2)
        self._energy_norm = {m: energy_per_window(m) / e_med for m in range(3)}

    @staticmethod
    def _enumerate_sequences(h):
        """Generate all 3^H model sequences as a numpy array."""
        n = 3 ** h
        seqs = np.zeros((n, h), dtype=int)
        for i in range(n):
            val = i
            for j in range(h - 1, -1, -1):
                seqs[i, j] = val % 3
                val //= 3
        return seqs

    def _pred_to_probs(self, pred: float) -> np.ndarray:
        """Convert a point prediction to soft class probabilities [P(easy), P(mod), P(hard)].

        Uses a triangular soft-assignment: each class gets probability proportional
        to proximity to thresholds, with smooth blending near boundaries.
        """
        t1, t2 = self.t1, self.t2
        mid = (t1 + t2) / 2.0

        if pred <= 0:
            return np.array([1.0, 0.0, 0.0])
        elif pred >= 1:
            return np.array([0.0, 0.0, 1.0])
        elif pred < t1:
            # Blend between easy and moderate near t1
            alpha = max(0.0, (t1 - pred) / t1)
            return np.array([alpha, 1.0 - alpha, 0.0])
        elif pred < mid:
            # Lower half of moderate zone — blend easy/moderate
            alpha = (pred - t1) / (mid - t1) if mid > t1 else 1.0
            return np.array([max(0, 0.3 * (1 - alpha)), 1.0 - 0.3 * (1 - alpha), 0.0])
        elif pred < t2:
            # Upper half of moderate zone — blend moderate/hard
            alpha = (pred - mid) / (t2 - mid) if t2 > mid else 1.0
            return np.array([0.0, 1.0 - 0.3 * alpha, 0.3 * alpha])
        else:
            # Above t2 — blend moderate/hard
            alpha = min(1.0, (pred - t2) / (1.0 - t2)) if t2 < 1.0 else 1.0
            return np.array([0.0, 1.0 - alpha, alpha])

    def _bayes_risk(self, model_idx: int, probs: np.ndarray) -> float:
        """Compute Bayes risk for selecting a given model under class probabilities."""
        p_easy, p_mod, p_hard = probs

        if model_idx == 0:  # nano
            loss = self.lambda_under * (p_mod + p_hard)
        elif model_idx == 1:  # small
            loss = self.lambda_over * p_easy + self.lambda_under * p_hard
        else:  # medium
            loss = self.lambda_over * (p_easy + p_mod)

        # Add normalized energy cost
        loss += self._energy_norm[model_idx]
        return loss

    def _extrapolate(self, steps: int) -> np.ndarray:
        """Linear extrapolation of future miss rates from recent history."""
        hist = list(self._history)
        if len(hist) < 2:
            # Not enough history — repeat last known value
            last = hist[-1] if hist else 0.5
            return np.full(steps, last)

        # Linear fit on last 3 values (or fewer)
        n = min(len(hist), 3)
        recent = hist[-n:]
        x = np.arange(n)
        slope = np.polyfit(x, recent, 1)[0]
        future = np.array([recent[-1] + slope * (i + 1) for i in range(steps)])
        return np.clip(future, 0.0, 1.0)

    def select_model(self, pred_miss_rate, t, state):
        self._history.append(pred_miss_rate)

        # Extrapolate future predictions for the horizon
        future_preds = self._extrapolate(self.horizon)
        # Overwrite first step with actual current prediction
        future_preds[0] = pred_miss_rate

        # Compute class probabilities for each step in the horizon
        probs_seq = np.array([self._pred_to_probs(p) for p in future_preds])

        # Evaluate all 3^H model sequences
        best_cost = float("inf")
        best_first = self._current_model

        for seq in self._sequences:
            cost = 0.0
            prev = self._current_model
            for step in range(self.horizon):
                m = seq[step]
                # Bayes risk at this step
                cost += self._bayes_risk(m, probs_seq[step])
                # Switching penalty
                if m != prev:
                    cost += self.w_switch
                prev = m

            if cost < best_cost:
                best_cost = cost
                best_first = seq[0]

        self._current_model = best_first
        return best_first

    def reset(self):
        self._history.clear()
        self._current_model = 1

    def name(self):
        return (f"BayesMPC(H={self.horizon}, λu={self.lambda_under}, "
                f"λo={self.lambda_over}, ws={self.w_switch})")


class DQNController(Controller):
    """Deep Q-Network controller that learns a switching policy from predictions.

    Trains on the LSTM prediction stream itself, learning what model to pick
    given the predicted miss rates. Ground truth is only used for reward
    computation during training. Uses experience replay and a target network.

    Architecture: 2-layer MLP (state_dim -> 32 -> ReLU -> 32 -> ReLU -> 3).
    """

    def __init__(self, lambda_tradeoff: float = 0.5, alpha_switch: float = 0.01,
                 lr: float = 1e-3, state_dim: int = 8, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 500, buffer_size: int = 5000,
                 batch_size: int = 64, target_update: int = 200):
        self.lambda_tradeoff = lambda_tradeoff
        self.alpha_switch = alpha_switch
        self.lr = lr
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update

        self._trained = False
        self._current_model = 1
        self._pred_history = deque(maxlen=5)
        self._step = 0

        # Lazily import torch
        self._torch = None
        self._policy_net = None
        self._target_net = None

    def _import_torch(self):
        if self._torch is None:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self._torch = torch
            self._nn = nn
            self._optim = optim

    def _build_network(self):
        """Build Q-network: state_dim -> 32 -> ReLU -> 32 -> ReLU -> 3."""
        self._import_torch()
        nn = self._nn

        class QNet(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, 32)
                self.fc2 = nn.Linear(32, 32)
                self.fc3 = nn.Linear(32, 3)

            def forward(self, x):
                x = nn.functional.relu(self.fc1(x))
                x = nn.functional.relu(self.fc2(x))
                return self.fc3(x)

        self._policy_net = QNet(self.state_dim)
        self._target_net = QNet(self.state_dim)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._optimizer = self._optim.Adam(self._policy_net.parameters(), lr=self.lr)

    def _make_state(self, pred_miss_rate: float, current_model: int) -> np.ndarray:
        """Construct the 8-dim state vector."""
        hist = list(self._pred_history)
        # Pad history with zeros if not enough
        while len(hist) < 4:
            hist.insert(0, 0.0)
        hist = hist[-4:]  # last 4 predictions

        # One-hot encoding of current model
        one_hot = [0.0, 0.0, 0.0]
        one_hot[current_model] = 1.0

        return np.array([pred_miss_rate] + hist + one_hot, dtype=np.float32)

    def _compute_reward(self, selected_model: int, true_miss_rate: float,
                        prev_model: int, t1: float, t2: float) -> float:
        """Reward: lambda * adequate - (1-lambda) * energy_norm - alpha * switch."""
        req = required_model(true_miss_rate, t1, t2)
        adequate = 1.0 if selected_model >= req else 0.0

        # Normalized energy (relative to medium)
        from .models import energy_per_window
        e_norm = energy_per_window(selected_model) / energy_per_window(2)

        switch = 1.0 if selected_model != prev_model else 0.0

        return (self.lambda_tradeoff * adequate
                - (1.0 - self.lambda_tradeoff) * e_norm
                - self.alpha_switch * switch)

    def train(self, pred_miss_rates: np.ndarray, true_miss_rates: np.ndarray,
              t1: float, t2: float, n_episodes: int = 10):
        """Train the DQN on the prediction stream.

        Args:
            pred_miss_rates: Array of LSTM-predicted miss rates (training portion).
            true_miss_rates: Array of ground-truth miss rates (for reward only).
            t1, t2: Threshold parameters for required_model().
            n_episodes: Number of training passes over the data.
        """
        self._build_network()
        torch = self._torch

        # Experience replay buffer
        replay_buffer = deque(maxlen=self.buffer_size)
        total_steps = 0
        n_frames = len(pred_miss_rates)

        for episode in range(n_episodes):
            current_model = 1  # start with small
            pred_history = deque(maxlen=5)

            for t in range(n_frames):
                pred_history.append(pred_miss_rates[t])

                # Build state
                hist = list(pred_history)
                while len(hist) < 5:
                    hist.insert(0, 0.0)
                hist = hist[-5:]
                one_hot = [0.0, 0.0, 0.0]
                one_hot[current_model] = 1.0
                state = np.array([pred_miss_rates[t]] + hist[-4:] + one_hot,
                                 dtype=np.float32)

                # Epsilon-greedy action selection
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-total_steps / self.epsilon_decay)

                if np.random.random() < epsilon:
                    action = np.random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = self._policy_net(torch.FloatTensor(state))
                        action = q_vals.argmax().item()

                # Compute reward
                reward = self._compute_reward(action, true_miss_rates[t],
                                              current_model, t1, t2)

                # Build next state
                prev_model = current_model
                current_model = action
                next_hist = list(pred_history)
                while len(next_hist) < 5:
                    next_hist.insert(0, 0.0)
                next_hist = next_hist[-5:]
                next_one_hot = [0.0, 0.0, 0.0]
                next_one_hot[current_model] = 1.0
                next_pred = pred_miss_rates[min(t + 1, n_frames - 1)]
                next_state = np.array([next_pred] + next_hist[-4:] + next_one_hot,
                                      dtype=np.float32)

                done = (t == n_frames - 1)
                replay_buffer.append((state, action, reward, next_state, done))

                # Training step
                if len(replay_buffer) >= self.batch_size:
                    indices = np.random.choice(len(replay_buffer), self.batch_size,
                                               replace=False)
                    batch = [replay_buffer[i] for i in indices]

                    states_b = torch.FloatTensor(np.array([b[0] for b in batch]))
                    actions_b = torch.LongTensor([b[1] for b in batch])
                    rewards_b = torch.FloatTensor([b[2] for b in batch])
                    next_states_b = torch.FloatTensor(np.array([b[3] for b in batch]))
                    dones_b = torch.FloatTensor([b[4] for b in batch])

                    # Current Q-values
                    q_values = self._policy_net(states_b).gather(
                        1, actions_b.unsqueeze(1)).squeeze(1)

                    # Target Q-values
                    with torch.no_grad():
                        next_q = self._target_net(next_states_b).max(1)[0]
                        target = rewards_b + self.gamma * next_q * (1 - dones_b)

                    loss = self._nn.functional.mse_loss(q_values, target)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                # Update target network periodically
                total_steps += 1
                if total_steps % self.target_update == 0:
                    self._target_net.load_state_dict(self._policy_net.state_dict())

        self._trained = True
        self._current_model = 1
        self._pred_history.clear()

    def select_model(self, pred_miss_rate, t, state):
        if not self._trained:
            return 1  # fallback to small if untrained

        self._pred_history.append(pred_miss_rate)
        s = self._make_state(pred_miss_rate, self._current_model)

        with self._torch.no_grad():
            q_vals = self._policy_net(self._torch.FloatTensor(s))
            action = q_vals.argmax().item()

        self._current_model = action
        return action

    def reset(self):
        self._current_model = 1
        self._pred_history.clear()
        self._step = 0

    def name(self):
        return (f"DQN(λ={self.lambda_tradeoff}, α={self.alpha_switch}, "
                f"lr={self.lr})")


class ProxyController(Controller):
    """Uncertainty-based switching without LSTM miss-rate predictions.

    Maps epistemic uncertainty directly to model tier. Only usable when
    the prediction stream includes an epistemic_unc column.

    Tests whether raw model uncertainty is sufficient for switching
    without explicit miss-rate prediction.
    """

    def __init__(self, unc_threshold_low: float = 0.05,
                 unc_threshold_high: float = 0.15):
        self.unc_low = unc_threshold_low
        self.unc_high = unc_threshold_high
        self._unc_values = None

    def set_uncertainty(self, unc_values: np.ndarray):
        """Provide the epistemic uncertainty stream."""
        self._unc_values = unc_values

    def select_model(self, pred_miss_rate, t, state):
        if self._unc_values is None:
            return 1  # fallback
        unc = self._unc_values[t]
        if unc < self.unc_low:
            return 0
        elif unc < self.unc_high:
            return 1
        else:
            return 2

    def name(self):
        return f"Proxy(lo={self.unc_low:.3f}, hi={self.unc_high:.3f})"
