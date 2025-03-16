import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
import multiprocessing as mp
import datetime
import os
import time
import logging
from collections import defaultdict, deque
import pytz  # For timezone handling
import subprocess
import re

# Function to get current time in Amsterdam timezone
def get_amsterdam_time():
    """Get the current time in Amsterdam timezone"""
    utc_now = datetime.datetime.now(pytz.utc)
    amsterdam_tz = pytz.timezone('Europe/Amsterdam')
    amsterdam_time = utc_now.astimezone(amsterdam_tz)
    return amsterdam_time

# Fixed hyperparameters as specified
fixed_hyperparams = {
    'update_frequency': 5,        # How often to perform a learning update
    'max_steps': 500,             # Maximum steps per episode (CartPole limit)
    'total_steps': 1000000,        # Total environment steps to run each experiment
    'gamma': 0.99,                # Discount factor
    'epsilon_start': 1.0,         # Starting exploration rate
    'epsilon_min': 0.1,           # Minimum exploration rate
    'epsilon_stop_percentage': 0.1, # When to stop epsilon decay (% of total steps)
    'learning_rate': 0.001,       # Learning rate
    'layer_size': 64,             # Neurons per layer
    'nr_layers': 3,               # Number of layers
    'target_update_frequency': 50, # How often to update target network (in steps) - Reduced from 100
    'replay_buffer_size': 100000,  # Capacity of replay buffer
    'batch_size': 16,             # Batch size for experience replay
}

# The four methods to compare
experiment_methods = {
    'baseline': {
        'description': 'Baseline DQN (without Target Network or Experience Replay)',
        'use_target_network': False,
        'use_experience_replay': False
    },
    'target_network_only': {
        'description': 'DQN with Target Network only',
        'use_target_network': True,
        'use_experience_replay': False
    },
    'experience_replay_only': {
        'description': 'DQN with Experience Replay only',
        'use_target_network': False,
        'use_experience_replay': True
    },
    'both_methods': {
        'description': 'DQN with both Target Network and Experience Replay',
        'use_target_network': True,
        'use_experience_replay': True
    }
}

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.state_size = None  # Will be set on first add
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer, ensuring consistent state shapes"""
        # First, determine the state size if not already set
        if self.state_size is None:
            if isinstance(state, np.ndarray):
                if len(state.shape) > 1:
                    # If state is multi-dimensional (e.g., [1, 4]), flatten it
                    self.state_size = state.size
                    state = state.flatten()
                    next_state = next_state.flatten()
                else:
                    # If state is already flat
                    self.state_size = state.shape[0]
            else:
                # If state is a list
                self.state_size = len(state)
        else:
            # Ensure consistent state shape
            if isinstance(state, np.ndarray):
                if len(state.shape) > 1:
                    state = state.flatten()
                    next_state = next_state.flatten()
        
        # Ensure action is a single scalar value
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Store the experience
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch_size experiences, ensuring consistent shapes"""
        # Sample batch_size experiences from buffer randomly
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return batch
    
    def __len__(self):
        return len(self.buffer)

# Define the Q-Network model using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size=64, nr_layers=3):
        super(QNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, layer_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(nr_layers - 2):  # -2 because we already have input and will add output layer
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(layer_size, action_size))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Define the baseline agent (same as NaiveAgent in Assignment1.py)
class NaiveAgent:
    def __init__(self, state_size, action_size, hyperparams, 
                 use_target_network=False, use_experience_replay=False):
        # Basic agent properties
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhancement flags
        self.use_target_network = use_target_network
        self.use_experience_replay = use_experience_replay
        
        # Set hyperparameters from passed dictionary
        self.gamma = hyperparams.get('gamma', 0.99)
        
        # Epsilon parameters
        self.epsilon_start = hyperparams.get('epsilon_start', 1.0)
        self.epsilon_min = hyperparams.get('epsilon_min', 0.1)
        self.epsilon_stop_percentage = hyperparams.get('epsilon_stop_percentage', 0.1)
        self.total_steps = hyperparams.get('total_steps', 100000)
        self.stop_step = int(self.total_steps * self.epsilon_stop_percentage)
        
        # Current state
        self.epsilon = self.epsilon_start
        self.current_step = 0
        
        # Learning parameters
        self.learning_rate = hyperparams.get('learning_rate', 0.001)
        self.batch_size = hyperparams.get('batch_size', 32)
        self.target_update_frequency = hyperparams.get('target_update_frequency', 1000)
        
        # Get neural network hyperparameters
        self.layer_size = hyperparams.get('layer_size', 64)
        self.nr_layers = hyperparams.get('nr_layers', 3)
        
        # Setup device and models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main Q-Network
        self.model = QNetwork(state_size, action_size, self.layer_size, self.nr_layers).to(self.device)
        
        # Target Q-Network (if used)
        if self.use_target_network:
            self.target_model = QNetwork(state_size, action_size, self.layer_size, self.nr_layers).to(self.device)
            self.update_target_network()  # Initialize target with same weights
        
        # Experience Replay Buffer (if used)
        if self.use_experience_replay:
            buffer_size = hyperparams.get('replay_buffer_size', 10000)
            self.memory = ReplayBuffer(buffer_size)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.losses = []  # Track losses
    
    def update_epsilon(self):
        """Update epsilon using linear decay with a stopping point"""
        if self.current_step < self.stop_step:
            # Linear decay from epsilon_start to epsilon_min
            decay_progress = self.current_step / self.stop_step
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * decay_progress
        else:
            # Fixed at minimum value
            self.epsilon = self.epsilon_min
    
    def update_target_network(self):
        """Copy weights from main model to target model"""
        if self.use_target_network:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        """Select an action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if self.use_experience_replay:
            self.memory.add(state, action, reward, next_state, done)
    
    def learn(self, experiences=None):
        """Learn from experiences based on agent's configuration"""
        # Always increment step counter and update epsilon
        self.current_step += 1
        self.update_epsilon()
        
        # Method 1: Using Experience Replay
        if self.use_experience_replay:
            # Only learn if we have enough samples
            if len(self.memory) >= max(self.batch_size, 1000):  # At least 1000 samples
                # Sample batch from replay buffer
                batch = self.memory.sample(self.batch_size)
                self._learn_from_batch(batch)
        
        # Method 2: Traditional Learning (without Experience Replay)
        elif experiences is not None and len(experiences) > 0:
            self._learn_from_batch(experiences)
        
        # Update target network if it's time
        if self.use_target_network and self.current_step % self.target_update_frequency == 0:
            self.update_target_network()
    
    def _learn_from_batch(self, experiences):
        """Internal method for learning from a batch of experiences with robust dimension handling"""
        if len(experiences) == 0:
            return
        
        try:
            # Extract batch components
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            
            # Process each experience individually to ensure consistent shapes
            for state, action, reward, next_state, done in experiences:
                # Handle state and next_state dimensions
                if isinstance(state, np.ndarray):
                    state = state.flatten()
                if isinstance(next_state, np.ndarray):
                    next_state = next_state.flatten()
                
                # Handle action to ensure it's a scalar
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(float(done))
            
            # Convert to numpy arrays
            states_array = np.array(batch_states, dtype=np.float32)
            actions_array = np.array(batch_actions, dtype=np.int64)
            rewards_array = np.array(batch_rewards, dtype=np.float32)
            next_states_array = np.array(batch_next_states, dtype=np.float32)
            dones_array = np.array(batch_dones, dtype=np.float32)
            
            # Print shapes for debugging
            # print(f"States shape: {states_array.shape}, Actions shape: {actions_array.shape}")
            
            # Convert to PyTorch tensors
            states_tensor = torch.FloatTensor(states_array).to(self.device)
            actions_tensor = torch.LongTensor(actions_array).unsqueeze(1).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards_array).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states_array).to(self.device)
            dones_tensor = torch.FloatTensor(dones_array).to(self.device)
            
            # Get current Q values
            current_q_values = self.model(states_tensor)
            
            # Check shapes before gather operation
            # print(f"Q values shape: {current_q_values.shape}, Actions tensor shape: {actions_tensor.shape}")
            
            # Handle possible dimension issues
            if current_q_values.dim() == 1 and actions_tensor.dim() == 2:
                current_q_values = current_q_values.unsqueeze(0)  # Add batch dimension
            elif current_q_values.dim() == 2 and actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(1)  # Add action dimension
            
            # Gather Q values for the actions taken
            current_q_actions = current_q_values.gather(1, actions_tensor)
            
            # Compute next Q values
            with torch.no_grad():
                if self.use_target_network:
                    next_q_values = self.target_model(next_states_tensor)
                else:
                    next_q_values = self.model(next_states_tensor)
                
                # Handle case where next_q_values is 1D
                if next_q_values.dim() == 1:
                    max_next_q = next_q_values
                else:
                    max_next_q = next_q_values.max(1)[0]
            
            # Compute target Q values
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q
            
            # Ensure dimensions match for loss calculation
            if current_q_actions.dim() > target_q_values.dim():
                current_q_actions = current_q_actions.squeeze()
            elif target_q_values.dim() > current_q_actions.dim():
                target_q_values = target_q_values.unsqueeze(1)
            
            # Compute loss and update weights
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_actions, target_q_values)
            loss.backward()
            self.optimizer.step()
            
            # Store the loss value
            self.losses.append(loss.item())
        
        except Exception as e:
            print(f"Error in _learn_from_batch: {e}")
            import traceback
            traceback.print_exc()
            # Continue execution to avoid crashing the whole experiment
            self.losses.append(0.0)

# Add the original NaiveAgent from Assignment1.py
class OriginalNaiveAgent:
    def __init__(self, state_size, action_size, hyperparams=None):
        if hyperparams is None:
            hyperparams = fixed_hyperparams
            
        self.state_size = state_size
        self.action_size = action_size
        
        # Set hyperparameters from passed dictionary or use defaults
        self.gamma = hyperparams.get('gamma', 0.99)
        
        # Epsilon parameters
        self.epsilon_start = hyperparams.get('epsilon_start', 1.0)
        self.epsilon_min = hyperparams.get('epsilon_min', 0.1)
        self.epsilon_stop_percentage = hyperparams.get('epsilon_stop_percentage', 0.1)
        self.total_steps = hyperparams.get('total_steps', 10000)
        self.stop_step = int(self.total_steps * self.epsilon_stop_percentage)
        
        # Current state
        self.epsilon = self.epsilon_start
        self.current_step = 0
        
        self.learning_rate = hyperparams.get('learning_rate', 0.001)
        
        # Get neural network hyperparameters
        layer_size = hyperparams.get('layer_size', 64)
        nr_layers = hyperparams.get('nr_layers', 3)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size, layer_size, nr_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.losses = []  # Track losses
    
    def update_epsilon(self):
        """Update epsilon using linear decay with a stopping point"""
        if self.current_step < self.stop_step:
            # Linear decay from epsilon_start to epsilon_min
            decay_progress = self.current_step / self.stop_step
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * decay_progress
        else:
            # Fixed at minimum value
            self.epsilon = self.epsilon_min
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self, experiences):
        # Skip if no experiences
        if len(experiences) == 0:
            return
            
        # Process all experiences in the batch
        batch_loss = 0
        for state, action, reward, next_state, done in experiences:
            # Convert numpy arrays to PyTorch tensors
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # Get current Q value
            q_values = self.model(state_tensor)
            
            # Get next Q value
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor)
                max_next_q = torch.max(next_q_values)
            
            # Calculate target Q value
            target = reward
            if not done:
                target += self.gamma * max_next_q.item()
            
            # Update only the Q value for the action taken
            target_q_values = q_values.clone()
            target_q_values[0, action] = target
            
            # Compute loss and update weights
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
            
            batch_loss += loss.item()
            
            # Increment step counter and update epsilon
            self.current_step += 1
            self.update_epsilon()
        
        # Store the average loss value
        avg_loss = batch_loss / len(experiences)
        self.losses.append(avg_loss)

def get_gpu_with_most_free_memory():
    """
    Returns the index of the GPU with the most free memory based on nvidia-smi.
    If no GPUs are available, returns 0.
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', 
                                         '--format=csv,nounits,noheader'], 
                                         encoding='utf-8')
        lines = output.strip().split('\n')
        gpu_memory = {}
        for line in lines:
            values = line.split(',')
            gpu_idx = int(values[0])
            total_memory = int(values[1])
            used_memory = int(values[2])
            free_memory = total_memory - used_memory
            gpu_memory[gpu_idx] = free_memory
        
        if not gpu_memory:
            return 0
        
        # Find GPU with most free memory
        gpu_idx = max(gpu_memory.items(), key=lambda x: x[1])[0]
        free_memory_mb = gpu_memory[gpu_idx]
        
        print(f"Selected GPU {gpu_idx} with {free_memory_mb} MiB free memory")
        return gpu_idx
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0

def run_experiment(experiment_config):
    """Run a single experiment with the given configuration"""
    try:
        # Extract experiment configuration
        method_name = experiment_config['method_name']
        method_config = experiment_config['method_config']
        use_target_network = method_config['use_target_network']
        use_experience_replay = method_config['use_experience_replay']
        hyperparams = experiment_config['hyperparams']
        run_id = experiment_config['run_id']
        gpu_id = experiment_config['gpu_id']
        
        # Set CUDA device if available
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        # Extract hyperparameters
        update_frequency = hyperparams.get('update_frequency', 1)
        max_steps = hyperparams.get('max_steps', 500)
        total_steps = hyperparams.get('total_steps', 10000)
        
        # Create a local environment for this process
        local_env = gym.make("CartPole-v1", render_mode="rgb_array")

        # Set parameters
        state_size = local_env.observation_space.shape[0]
        action_size = local_env.action_space.n
        
        # Create the agent with the specified method enhancements
        if method_name == "baseline":
            # Use the original NaiveAgent for baseline
            agent = OriginalNaiveAgent(
                state_size=state_size, 
                action_size=action_size, 
                hyperparams=hyperparams
            )
            # Special training loop for baseline agent
            return run_baseline_experiment(agent, local_env, hyperparams, method_name, run_id)
        else:
            # Use the enhanced agent for other methods
            agent = NaiveAgent(
                state_size=state_size, 
                action_size=action_size, 
                hyperparams=hyperparams,
                use_target_network=use_target_network,
                use_experience_replay=use_experience_replay
            )

        # Training variables
        episode_rewards = []
        average_rewards = []
        episode_losses = []
        env_steps = []
        total_env_steps = 0
        episodes_completed = 0
        
        # Buffer for experiences (used by non-experience-replay methods)
        experiences_buffer = []
        
        # Training loop
        while total_env_steps < total_steps:
            # Start a new episode
            episodes_completed += 1
            state, info = local_env.reset()
            state = np.reshape(state, [1, state_size])
            episode_reward = 0
            start_loss_idx = len(agent.losses)
            
            # Episode loop
            for step in range(max_steps):
                # Choose an action
                action = agent.act(state)
                
                # Take the action and observe the next state and reward
                next_state, reward, terminated, truncated, info = local_env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                done = terminated or truncated
                
                # Store experience (either in buffer or replay memory)
                if use_experience_replay:
                    agent.store_experience(state, action, reward, next_state, done)
                else:
                    experiences_buffer.append((state, action, reward, next_state, done))
                
                # Update state and tracking variables
                state = next_state
                episode_reward += reward
                total_env_steps += 1
                
                # Learning step
                if use_experience_replay:
                    min_buffer_size = 1000
                    # Only learn every few steps and after minimum buffer size
                    if len(agent.memory) >= min_buffer_size and total_env_steps % update_frequency == 0:
                        agent.learn()
                elif len(experiences_buffer) >= update_frequency:
                    # Without experience replay, learn after collecting several experiences
                    agent.learn(experiences_buffer)
                    experiences_buffer = []
                
                # End episode if done or reached total steps limit
                if done or total_env_steps >= total_steps:
                    break
            
            # Process any remaining experiences at end of episode
            if not use_experience_replay and experiences_buffer:
                agent.learn(experiences_buffer)
                experiences_buffer = []
            
            # Calculate average loss for this episode
            if len(agent.losses) > start_loss_idx:
                episode_loss = np.mean(agent.losses[start_loss_idx:])
                episode_losses.append(episode_loss)
            else:
                episode_losses.append(0)
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
            average_rewards.append(avg_reward)
            env_steps.append(total_env_steps)
            
            # Progress reporting (less frequent to reduce output)
            if episodes_completed % 50 == 0:
                print(f"[{method_name} Run {run_id}] Episode {episodes_completed}: Reward={episode_reward}, Avg={avg_reward:.2f}, Steps={total_env_steps}/{total_steps}")
        
        # Close the environment
        local_env.close()
        
        # Print final results
        final_avg_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
        print(f"[{method_name} Run {run_id}] Completed: Episodes={episodes_completed}, Avg Reward={final_avg_reward:.2f}")
        
        # Return results
        return {
            'method_name': method_name,
            'run_id': run_id,
            'episode_rewards': episode_rewards,
            'average_rewards': average_rewards,
            'episode_losses': episode_losses,
            'env_steps': env_steps,
            'episodes_completed': episodes_completed,
            'use_target_network': use_target_network,
            'use_experience_replay': use_experience_replay,
            'hyperparams': hyperparams
        }
    except Exception as e:
        import traceback
        print(f"Error in {experiment_config['method_name']} run {experiment_config['run_id']}:")
        print(traceback.format_exc())  # Print full stack trace
        # Return a minimal result structure so the experiment doesn't completely fail
        return {
            'method_name': experiment_config['method_name'],
            'run_id': experiment_config['run_id'],
            'error': str(e),
            'episode_rewards': [],
            'average_rewards': [],
            'episode_losses': [],
            'env_steps': [],
            'episodes_completed': 0,
            'use_target_network': experiment_config['method_config']['use_target_network'],
            'use_experience_replay': experiment_config['method_config']['use_experience_replay'],
            'hyperparams': experiment_config['hyperparams']
        }

def run_baseline_experiment(agent, local_env, hyperparams, method_name, run_id):
    """Run experiment with the original NaiveAgent (from Assignment1.py)"""
    # Extract hyperparameters
    update_frequency = hyperparams.get('update_frequency', 1)
    max_steps = hyperparams.get('max_steps', 500)
    total_steps = hyperparams.get('total_steps', 10000)
    
    # Set parameters
    state_size = local_env.observation_space.shape[0]
    action_size = local_env.action_space.n
    
    # Training variables
    episode_rewards = []
    average_rewards = []
    epsilon_values = []
    episode_losses = []
    env_steps = []
    total_env_steps = 0
    episodes_completed = 0
    
    # Buffer for experiences
    experiences_buffer = []
    
    # Training loop - follows the exact structure from Assignment1.py
    while total_env_steps < total_steps:
        # Start a new episode
        episodes_completed += 1
        state, info = local_env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        start_loss_idx = len(agent.losses)
        episode_steps = 0
        episode_epsilon_values = []
        
        for step in range(max_steps):
            # Record current epsilon for this step
            episode_epsilon_values.append(agent.epsilon)
            
            # Choose an action
            action = agent.act(state)
            
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = local_env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience
            done = terminated or truncated
            experiences_buffer.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_env_steps += 1
            
            # Learn after collecting N experiences (1:N ratio)
            if len(experiences_buffer) >= update_frequency:
                agent.learn(experiences_buffer)
                experiences_buffer = []  # Clear buffer after learning
            
            # End episode if done or reached total steps limit
            if done or total_env_steps >= total_steps:
                break
        
        # Process any remaining experiences
        if experiences_buffer:
            agent.learn(experiences_buffer)
            experiences_buffer = []
        
        # Calculate average loss for this episode
        if len(agent.losses) > start_loss_idx:
            episode_loss = np.mean(agent.losses[start_loss_idx:])
            episode_losses.append(episode_loss)
        else:
            episode_losses.append(0)
        
        # Store average epsilon for this episode
        if episode_epsilon_values:
            avg_epsilon = np.mean(episode_epsilon_values)
            epsilon_values.append(avg_epsilon)
        else:
            epsilon_values.append(agent.epsilon)
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
        average_rewards.append(avg_reward)
        env_steps.append(total_env_steps)
        
        # Progress reporting
        if episodes_completed % 50 == 0:
            print(f"[{method_name} Run {run_id}] Episode {episodes_completed}: Reward={episode_reward}, Avg={avg_reward:.2f}, Steps={total_env_steps}/{total_steps}")
    
    # Close the environment
    local_env.close()
    
    # Print final results
    final_avg_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
    print(f"[{method_name} Run {run_id}] Completed: Episodes={episodes_completed}, Avg Reward={final_avg_reward:.2f}")
    
    # Return results in the same format
    return {
        'method_name': method_name,
        'run_id': run_id,
        'episode_rewards': episode_rewards,
        'average_rewards': average_rewards,
        'episode_losses': episode_losses,
        'env_steps': env_steps,
        'episodes_completed': episodes_completed,
        'use_target_network': False,
        'use_experience_replay': False,
        'hyperparams': hyperparams
    }

def update_progress_bar(result, pbar):
    """Callback function to update the progress bar"""
    pbar.update(1)
    return result

def plot_comparison_results(all_results, output_dir):
    """Create plots comparing the different methods"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by method
    results_by_method = defaultdict(list)
    for result in all_results:
        results_by_method[result['method_name']].append(result)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Colors for different methods
    colors = {
        'baseline': '#CC78BC',
        'target_network_only': '#0173B2',
        'experience_replay_only': '#029E73',
        'both_methods': '#D55E00',
        'random_baseline': '#000000'  # Black for random baseline
    }
    
    # Method labels for legend
    method_labels = {
        'baseline': 'Baseline DQN',
        'target_network_only': 'Target Network Only',
        'experience_replay_only': 'Experience Replay Only',
        'both_methods': 'Both Methods',
        'random_baseline': 'Random Baseline'  # Label for random baseline
    }
    
    # Find maximum steps across all runs for common x-axis
    max_steps = 0
    for method_results in results_by_method.values():
        for result in method_results:
            if result['env_steps']:
                max_steps = max(max_steps, max(result['env_steps']))
    
    # Create common x-axis for interpolation
    common_x = np.linspace(0, max_steps, 1000)
    
    # Try to load the random baseline data
    try:
        random_baseline_file = "RandomBaselineCartPole.csv"
        if os.path.exists(random_baseline_file):
            # Load the random baseline data
            random_data = np.genfromtxt(random_baseline_file, delimiter=',', names=True)
            
            # Extract the columns we need
            random_rewards = random_data['Episode_Return']
            random_steps = random_data['env_step']
            
            # Interpolate random baseline data to common x-axis
            # Only use points within our x range
            valid_indices = random_steps <= max_steps
            if np.any(valid_indices):
                random_steps_valid = random_steps[valid_indices]
                random_rewards_valid = random_rewards[valid_indices]
                
                # Interpolate to common x
                if len(random_steps_valid) > 1:
                    random_rewards_interp = np.interp(
                        common_x, 
                        random_steps_valid, 
                        random_rewards_valid,
                        left=random_rewards_valid[0],  # Use first value for extrapolation
                        right=random_rewards_valid[-1]  # Use last value for extrapolation
                    )
                    
                    # Plot random baseline
                    ax1.plot(common_x, random_rewards_interp, color=colors['random_baseline'], 
                             linewidth=2.5, linestyle='--', label=method_labels['random_baseline'], alpha=0.2)
                    
                    print(f"Successfully loaded and plotted random baseline data from {random_baseline_file}")
                else:
                    print(f"Not enough valid data points in {random_baseline_file}")
            else:
                print(f"No valid data points in {random_baseline_file} for our x range")
        else:
            print(f"Random baseline file {random_baseline_file} not found")
    except Exception as e:
        print(f"Error loading random baseline data: {e}")
    
    # Plot rewards by steps for each method
    for method_name, method_results in results_by_method.items():
        # Interpolate each run to the common x-axis
        interpolated_rewards = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['average_rewards']
            if len(x) > 1 and len(y) > 1:
                interpolated_rewards.append(np.interp(common_x, x, y))
        
        if not interpolated_rewards:
            continue
            
        # Calculate mean and std of rewards across runs
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        
        # Plot mean reward with std deviation
        ax1.plot(common_x, mean_rewards, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
        ax1.fill_between(common_x, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        color=colors[method_name], alpha=0.2)
    
    # Configure reward plot
    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward vs. Environment Steps', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 500)  # CartPole max reward is 500
    
    # Plot loss curves for each method
    for method_name, method_results in results_by_method.items():
        # Interpolate each run to the common x-axis
        interpolated_losses = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['episode_losses']
            if len(x) > 1 and len(y) > 1:
                interpolated_losses.append(np.interp(common_x, x, y))
        
        if not interpolated_losses:
            continue
            
        # Calculate mean and std of losses across runs
        mean_losses = np.mean(interpolated_losses, axis=0)
        std_losses = np.std(interpolated_losses, axis=0)
        
        # Plot mean loss with std deviation
        ax2.plot(common_x, mean_losses, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
        ax2.fill_between(common_x, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[method_name], alpha=0.2)
    
    # Configure loss plot
    ax2.set_xlabel('Environment Steps', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Training Loss vs. Environment Steps (Log Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Set y-axis to log scale for better visualization of loss values
    ax2.set_yscale('symlog')  # Use symlog to handle potential negative values
    
    # Set reasonable y-limits to focus on the relevant range
    # This helps avoid extreme values dominating the plot
    y_min = 0.1  # Minimum positive value to show
    y_max = max(100, ax2.get_ylim()[1])  # Use at least 100 or the current max
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=300)
    plt.close()

    # Also create a version with linear scale for comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Recreate the reward plot (same as before)
    # Try to load the random baseline data
    try:
        random_baseline_file = "RandomBaselineCartPole.csv"
        if os.path.exists(random_baseline_file):
            # Load the random baseline data
            random_data = np.genfromtxt(random_baseline_file, delimiter=',', names=True)
            
            # Extract the columns we need
            random_rewards = random_data['Episode_Return']
            random_steps = random_data['env_step']
            
            # Interpolate random baseline data to common x-axis
            valid_indices = random_steps <= max_steps
            if np.any(valid_indices):
                random_steps_valid = random_steps[valid_indices]
                random_rewards_valid = random_rewards[valid_indices]
                
                if len(random_steps_valid) > 1:
                    random_rewards_interp = np.interp(
                        common_x, 
                        random_steps_valid, 
                        random_rewards_valid,
                        left=random_rewards_valid[0],
                        right=random_rewards_valid[-1]
                    )
                    
                    ax1.plot(common_x, random_rewards_interp, color=colors['random_baseline'], 
                             linewidth=2.5, linestyle='--', label=method_labels['random_baseline'], alpha=0.2)
    except Exception as e:
        pass
    
    # Plot rewards by steps for each method
    for method_name, method_results in results_by_method.items():
        interpolated_rewards = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['average_rewards']
            if len(x) > 1 and len(y) > 1:
                interpolated_rewards.append(np.interp(common_x, x, y))
        
        if not interpolated_rewards:
            continue
            
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        
        ax1.plot(common_x, mean_rewards, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
        ax1.fill_between(common_x, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        color=colors[method_name], alpha=0.2)
    
    # Configure reward plot
    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward vs. Environment Steps', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 500)
    
    # Plot loss curves with normalized scale
    for method_name, method_results in results_by_method.items():
        interpolated_losses = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['episode_losses']
            if len(x) > 1 and len(y) > 1:
                # Clip extreme values for better visualization
                y_clipped = np.clip(y, 0, 100)  # Clip to range [0, 100]
                interpolated_losses.append(np.interp(common_x, x, y_clipped))
        
        if not interpolated_losses:
            continue
            
        mean_losses = np.mean(interpolated_losses, axis=0)
        std_losses = np.std(interpolated_losses, axis=0)
        
        ax2.plot(common_x, mean_losses, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
        ax2.fill_between(common_x, 
                        mean_losses - std_losses, 
                        mean_losses + std_losses, 
                        color=colors[method_name], alpha=0.2)
    
    # Configure normalized loss plot
    ax2.set_xlabel('Environment Steps', fontsize=12)
    ax2.set_ylabel('Loss (clipped to [0,100])', fontsize=12)
    ax2.set_title('Training Loss vs. Environment Steps (Normalized)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the normalized version
    plt.savefig(os.path.join(output_dir, "method_comparison_normalized.png"), dpi=300)
    plt.close()

    # Create a summary statistics file
    stats_file = os.path.join(output_dir, "method_comparison_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Comparison of DQN Methods\n")
        f.write("========================\n\n")
        
        f.write("Fixed Hyperparameters:\n")
        for key, value in fixed_hyperparams.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nResults Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} {'Avg Reward':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10} {'Avg Episodes':<15}\n")
        f.write("-" * 80 + "\n")
        
        for method_name, method_results in results_by_method.items():
            # Calculate statistics across runs
            all_rewards = [np.mean(result['episode_rewards']) for result in method_results]
            avg_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            min_reward = np.min(all_rewards)
            max_reward = np.max(all_rewards)
            avg_episodes = np.mean([result['episodes_completed'] for result in method_results])
            
            # Write to file
            f.write(f"{method_labels[method_name]:<20} {avg_reward:<15.2f} {std_reward:<15.2f} {min_reward:<10.2f} {max_reward:<10.2f} {avg_episodes:<15.1f}\n")
        
        # Add random baseline stats if available
        try:
            if os.path.exists("RandomBaselineCartPole.csv"):
                random_data = np.genfromtxt("RandomBaselineCartPole.csv", delimiter=',', names=True)
                random_rewards = random_data['Episode_Return']
                avg_random = np.mean(random_rewards)
                std_random = np.std(random_rewards)
                min_random = np.min(random_rewards)
                max_random = np.max(random_rewards)
                
                f.write(f"{'Random Baseline':<20} {avg_random:<15.2f} {std_random:<15.2f} {min_random:<10.2f} {max_random:<10.2f} {'N/A':<15}\n")
        except Exception as e:
            print(f"Error adding random baseline stats: {e}")
        
        f.write("\nDetailed Analysis:\n")
        
        # Find best method based on average reward
        method_avg_rewards = {method_name: np.mean([np.mean(result['episode_rewards']) for result in method_results]) 
                            for method_name, method_results in results_by_method.items()}
        best_method = max(method_avg_rewards.items(), key=lambda x: x[1])[0]
        
        f.write(f"\nBest Method: {method_labels[best_method]} with average reward {method_avg_rewards[best_method]:.2f}\n\n")
        
        # Compare the impact of each enhancement
        f.write("Enhancement Impact Analysis:\n")
        
        # Impact of target network (both_methods vs experience_replay_only)
        if 'both_methods' in method_avg_rewards and 'experience_replay_only' in method_avg_rewards:
            target_impact = method_avg_rewards['both_methods'] - method_avg_rewards['experience_replay_only']
            f.write(f"Impact of adding Target Network to Experience Replay: {target_impact:+.2f}\n")
        
        # Impact of experience replay (both_methods vs target_network_only)
        if 'both_methods' in method_avg_rewards and 'target_network_only' in method_avg_rewards:
            er_impact = method_avg_rewards['both_methods'] - method_avg_rewards['target_network_only']
            f.write(f"Impact of adding Experience Replay to Target Network: {er_impact:+.2f}\n")
        
        # Impact of target network alone (target_network_only vs baseline)
        if 'target_network_only' in method_avg_rewards and 'baseline' in method_avg_rewards:
            target_only_impact = method_avg_rewards['target_network_only'] - method_avg_rewards['baseline']
            f.write(f"Impact of Target Network alone: {target_only_impact:+.2f}\n")
        
        # Impact of experience replay alone (experience_replay_only vs baseline)
        if 'experience_replay_only' in method_avg_rewards and 'baseline' in method_avg_rewards:
            er_only_impact = method_avg_rewards['experience_replay_only'] - method_avg_rewards['baseline']
            f.write(f"Impact of Experience Replay alone: {er_only_impact:+.2f}\n")

def run_comparison_experiments():
    """Run all 4 types of experiments, 5 runs each, and compare the results"""
    # Record start time
    start_time = get_amsterdam_time()
    
    # Create a sequentially numbered results directory
    os.makedirs("results", exist_ok=True)
    
    # Find the next run number
    existing_runs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d)) and d.startswith("methods_comparison_")]
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.split("_")[-1])
            run_numbers.append(num)
        except ValueError:
            continue
    
    next_run_number = 1 if not run_numbers else max(run_numbers) + 1
    results_dir = os.path.join("results", f"methods_comparison_{next_run_number}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)")
    print(f"Results will be saved to: {results_dir}")
    
    # Save the hyperparameters used
    with open(os.path.join(results_dir, "hyperparameters.txt"), 'w') as f:
        f.write("Fixed Hyperparameters:\n")
        for key, value in fixed_hyperparams.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nExperiment Methods:\n")
        for method_name, method_config in experiment_methods.items():
            f.write(f"{method_name}: {method_config['description']}\n")
            f.write(f"  use_target_network: {method_config['use_target_network']}\n")
            f.write(f"  use_experience_replay: {method_config['use_experience_replay']}\n")
    
    # Select GPU to use
    best_gpu_id = 0  # Use first GPU by default
    if torch.cuda.is_available():
        best_gpu_id = get_gpu_with_most_free_memory()
        print(f"Selected GPU {best_gpu_id} for all experiments")
    else:
        print("CUDA not available, using CPU")
    
    # Prepare all experiment configurations
    all_experiment_configs = []
    
    for method_name, method_config in experiment_methods.items():
        for run_id in range(1, 6d):  # 5 runs per method (1-indexed)
            experiment_config = {
                'method_name': method_name,
                'method_config': method_config,
                'hyperparams': fixed_hyperparams.copy(),
                'run_id': run_id,
                'gpu_id': best_gpu_id  # Use the same GPU for all experiments
            }
            all_experiment_configs.append(experiment_config)
    
    total_experiments = len(all_experiment_configs)
    print(f"Total experiments to run: {total_experiments}")
    
    # Create a master progress bar
    with tqdm.tqdm(total=total_experiments, desc="Overall Progress") as master_pbar:
        # Run experiments sequentially to avoid CUDA multiprocessing issues
        all_results = []
        
        for config in all_experiment_configs:
            print(f"\nRunning {config['method_name']} (run {config['run_id']}) experiment")
            
            try:
                # Run the experiment
                result = run_experiment(config)
                all_results.append(result)
                
                # Update progress bar
                master_pbar.update(1)
                
                # Clear GPU cache after each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to run {config['method_name']} (run {config['run_id']}): {e}")
                import traceback
                traceback.print_exc()
    
    # Calculate elapsed time
    end_time = get_amsterdam_time()
    elapsed_time = (end_time - start_time).total_seconds() / 3600  # in hours
    
    print(f"\nAll experiments completed in {elapsed_time:.2f} hours")
    
    # Save the raw results
    results_file = os.path.join(results_dir, "all_results.pkl")
    with open(results_file, 'wb') as f:
        import pickle
        pickle.dump(all_results, f)
    
    # Plot and analyze results
    plot_comparison_results(all_results, results_dir)
    
    # Create a summary file
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"DQN Methods Comparison - Run {next_run_number}\n")
        f.write("============================\n\n")
        f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)\n")
        f.write(f"Duration: {elapsed_time:.2f} hours\n\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"GPU used: {best_gpu_id}\n")
        f.write(f"Methods compared: {', '.join(experiment_methods.keys())}\n")
        f.write(f"Runs per method: 5\n\n")
        f.write("See method_comparison_stats.txt for detailed analysis.\n")
    
    print(f"Results saved to {results_dir}")
    print(f"Experiment complete! (Run {next_run_number})")

# Main execution
if __name__ == "__main__":
    # Fix for CUDA multiprocessing issue
    mp.set_start_method('spawn', force=True)
    
    print("Training Enhanced DQN agents for CartPole-v1")
    print("Comparing 4 different methods: Baseline, Target Network, Experience Replay, and Both Combined")
    
    # Run all experiment types and compare them
    run_comparison_experiments()
    
    # Close the environment
    env.close()

