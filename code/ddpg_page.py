"""
Deep Deterministic Policy Gradient (DDPG) implementation for PAGE algorithm.
This module implements the DDPG agents used by both clients and server in the PAGE framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


class Actor(nn.Module):
    """
    Actor network for DDPG that outputs continuous actions.
    Maps states to actions using a deterministic policy.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden1=400,
        hidden2=300,
        action_bounds=None,
        init_w=3e-3,
    ):
        """
        Initialize Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
            action_bounds: List of tuples [(low, high)] for each action dimension
            init_w: Initial weight range for output layer
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

        # Initialize weights
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.action_bounds = action_bounds
        self.action_dim = action_dim

        # Pre-compute bounds tensors for efficient vectorized operations
        self._bounds_tensor = None
        if action_bounds is not None:
            self._setup_bounds_tensors()

    def _setup_bounds_tensors(self):
        """
        Pre-compute bounds tensors for efficient vectorized operations.
        This avoids repeated tensor creation during forward passes.
        """
        if self.action_bounds is not None:
            low_bounds = [b[0] for b in self.action_bounds]
            high_bounds = [b[1] for b in self.action_bounds]

            # Store as lists initially, will convert to tensors on first forward pass
            # when we know the device
            self._low_bounds_list = low_bounds
            self._high_bounds_list = high_bounds

    def forward(self, state):
        """
        Forward pass through actor network.

        Args:
            state: Input state tensor

        Returns:
            actions: Continuous action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        # Apply action bounds if specified
        if self.action_bounds is not None:
            # Use sigmoid to map to [0, 1], then scale to action bounds
            actions = torch.sigmoid(actions)

            # Setup bounds tensors on first forward pass (when device is known)
            if self._bounds_tensor is None:
                device = actions.device
                dtype = actions.dtype
                low_bounds = torch.tensor(
                    self._low_bounds_list, device=device, dtype=dtype
                )
                high_bounds = torch.tensor(
                    self._high_bounds_list, device=device, dtype=dtype
                )
                self._bounds_tensor = (low_bounds, high_bounds)

            # Vectorized scaling operation (no inplace operations, no loops)
            low_bounds, high_bounds = self._bounds_tensor
            actions = actions * (high_bounds - low_bounds) + low_bounds
        else:
            # Default: use tanh activation for bounded actions [-1, 1]
            actions = torch.tanh(actions)

        return actions


class Critic(nn.Module):
    """
    Critic network for DDPG that estimates Q-values.
    Maps (state, action) pairs to Q-values.
    """

    def __init__(self, state_dim, action_dim, hidden1=400, hidden2=300, init_w=3e-3):
        """
        Initialize Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
            init_w: Initial weight range for output layer
        """
        super(Critic, self).__init__()

        # First layer processes state
        self.fc1 = nn.Linear(state_dim, hidden1)

        # Second layer processes state features + actions
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)

        # Output layer produces Q-value
        self.fc3 = nn.Linear(hidden2, 1)

        # Initialize output layer weights
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Forward pass through critic network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            q_value: Estimated Q-value for (state, action) pair
        """
        # Process state through first layer
        x = F.relu(self.fc1(state))

        # Concatenate state features with actions
        x = torch.cat([x, action], dim=1)

        # Process through remaining layers
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise.
    Generates temporally correlated noise for continuous action spaces.
    """

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        """
        Initialize OU noise process.

        Args:
            action_dim: Dimension of action space
            mu: Long-term mean
            theta: Mean reversion rate
            sigma: Volatility parameter
            dt: Time step
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset noise to initial state."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """
        Generate next noise sample.

        Returns:
            noise: Correlated noise vector
        """
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """
    Experience replay buffer for DDPG.
    Stores transitions and samples mini-batches for training.
    """

    def __init__(self, capacity=1000000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample random mini-batch from buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        # Convert each element to numpy array, handling both torch tensors and numpy arrays
        def to_numpy_safe(x):
            """Convert tensor or array to numpy array on CPU."""
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    return x.cpu().detach().numpy()
                else:
                    return x.detach().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.array(x)

        # Apply conversion to each component of the batch
        converted_batch = []
        for transition in batch:
            converted_transition = tuple(to_numpy_safe(item) for item in transition)
            converted_batch.append(converted_transition)

        # Stack the converted data
        state, action, reward, next_state, done = map(np.stack, zip(*converted_batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class DDPG:
    """
    Deep Deterministic Policy Gradient agent for PAGE algorithm.
    Implements both client and server agents with appropriate state/action spaces.
    """

    def __init__(
        self,
        nb_states,
        nb_actions,
        nb_agents,
        args,
        hidden1=400,
        hidden2=300,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        buffer_size=1000000,
        action_bounds=None,
        device="cuda",
    ):
        """
        Initialize DDPG agent.

        Args:
            nb_states: Dimension of state space
            nb_actions: Dimension of action space
            nb_agents: Number of agents (for client agent, this is W; for server, this is 1)
            args: Command line arguments
            hidden1: Size of first hidden layer
            hidden2: Size of second hidden layer
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor
            tau: Soft update parameter for target networks
            batch_size: Mini-batch size for training
            buffer_size: Replay buffer capacity
            action_bounds: Action bounds for each dimension
            device: Device to run on ('cuda' or 'cpu')
        """
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.nb_agents = nb_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Set default action bounds based on PAGE requirements
        if action_bounds is None:
            if nb_actions == 2:  # Client agent (lr, epochs)
                action_bounds = [
                    (0.001, 0.1),
                    (1, 5),
                ]  # lr: [0.001, 0.5], epochs: [1, 10]
            else:  # Server agent (weights)
                action_bounds = [(0.0, 1.0)] * nb_actions  # weights: [0, 1]
        self.action_bounds = action_bounds

        # Create actor and critic networks
        self.actor = Actor(nb_states, nb_actions, hidden1, hidden2, action_bounds).to(
            self.device
        )
        self.actor_target = Actor(
            nb_states, nb_actions, hidden1, hidden2, action_bounds
        ).to(self.device)
        self.critic = Critic(nb_states, nb_actions, hidden1, hidden2).to(self.device)
        self.critic_target = Critic(nb_states, nb_actions, hidden1, hidden2).to(
            self.device
        )

        # Initialize target networks
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Create replay buffer and noise process
        self.memory = ReplayBuffer(buffer_size)
        self.noise = OrnsteinUhlenbeckNoise(nb_actions)

        # Store current observations for each agent
        self.observations = np.zeros((nb_agents, nb_states))

    def _hard_update(self, target, source):
        """
        Copy parameters from source to target network.

        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        """
        Soft update target network parameters.

        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def select_action(self, state, add_noise=True):
        """
        Select action using actor network.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            action: Selected action (numpy array)
        """
        # Ensure state is numpy array before converting to tensor
        if isinstance(state, torch.Tensor):
            if state.is_cuda:
                state = state.cpu().detach().numpy()
            else:
                state = state.detach().numpy()
        else:
            state = np.array(state)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().detach().numpy()[0]
        self.actor.train()

        # Add exploration noise if specified
        if add_noise:
            noise = self.noise.sample()
            action = action + noise

            # Clip actions to bounds
            if self.action_bounds is not None:
                for i in range(self.nb_actions):
                    low, high = self.action_bounds[i]
                    action[i] = np.clip(action[i], low, high)

        # Ensure return value is numpy array
        return np.array(action)

    def random_action(self):
        """
        Generate random action within bounds.

        Returns:
            action: Random action vector
        """
        action = np.zeros(self.nb_actions)
        for i in range(self.nb_actions):
            if self.action_bounds is not None:
                low, high = self.action_bounds[i]
                action[i] = np.random.uniform(low, high)
            else:
                action[i] = np.random.uniform(-1, 1)
        return action

    def observe(self, rewards, next_observations, done):
        """
        Store transitions in replay buffer.

        Args:
            rewards: Array of rewards for each agent
            next_observations: Array of next states for each agent
            done: Episode termination flag
        """
        # Store transitions for each agent
        for i in range(self.nb_agents):
            if hasattr(self, "last_actions") and self.last_actions is not None:
                self.memory.push(
                    self.observations[i],
                    self.last_actions[i] if self.nb_agents > 1 else self.last_actions,
                    rewards[i] if self.nb_agents > 1 else rewards[0],
                    next_observations[i],
                    done,
                )

        # Update observations
        self.observations = next_observations.copy()

    def update_policy(self):
        """
        Update actor and critic networks using replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def select_a(self, observations):
        """
        Select actions for all agents (client-side).

        Args:
            observations: Array of observations for all agents

        Returns:
            actions: Array of actions for all agents
        """
        actions = []
        for i in range(self.nb_agents):
            action = self.select_action(observations[i], add_noise=True)
            actions.append(action)

        self.last_actions = np.array(actions)
        return self.last_actions

    def select_sa(self, observation):
        """
        Select action for server agent.

        Args:
            observation: Server observation (all client accuracies)

        Returns:
            weights: Normalized weights for all clients
        """
        # Server has single observation containing all client accuracies
        action = self.select_action(observation[0], add_noise=True)

        # Ensure weights are non-negative and normalized
        weights = np.abs(action)
        weights = weights / (np.sum(weights) + 1e-8)

        self.last_actions = action
        return weights

    def random_a(self):
        """
        Generate random actions for all client agents.

        Returns:
            actions: Array of random actions
        """
        actions = []
        for i in range(self.nb_agents):
            actions.append(self.random_action())
        return np.array(actions)

    def random_sa(self):
        """
        Generate random weights for server agent.

        Returns:
            weights: Random normalized weights
        """
        # Generate random weights and normalize
        weights = np.random.rand(self.nb_actions)
        weights = weights / np.sum(weights)
        return weights
