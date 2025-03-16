import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm  # Import tqdm for progress bar
import multiprocessing as mp
from functools import partial
import datetime
import os
import time
import logging
from collections import defaultdict
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

# Define default hyperparameters
default_hyperparams = {
    'update_frequency': 1,
    'max_steps': 500,
    'total_steps': 100000,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_min': 0.1,
    'epsilon_stop_percentage': 0.1, 
    'learning_rate': 0.001,
    'layer_size': 32,
    'nr_layers': 3,
}

# Hyperparameters to study in ablation studies and their values
ablation_params = {
    'gamma': [0.99, 0.95, 0.9, 0.8, 0.5],
    'epsilon_stop_percentage': [0.01, 0.05, 0.1, 0.25, 0.5],  # When to stop epsilon decay (% of total steps)
    'learning_rate': [0.01, 0.005, 0.001, 0.0005],
    'update_frequency': [1, 3, 5, 10, 50],
    'layer_size': [32, 64, 128, 256],
    'nr_layers': [3, 5, 10]
}

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Define the Q-Network model using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size=32, nr_layers=3):
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

# Define the naive function approximation agent
class NaiveAgent:
    def __init__(self, state_size, action_size, hyperparams=None):
        if hyperparams is None:
            hyperparams = default_hyperparams
            
        self.state_size = state_size
        self.action_size = action_size
        
        # Set hyperparameters from passed dictionary or use defaults
        self.gamma = hyperparams.get('gamma', 0.95)
        
        # Epsilon parameters
        self.epsilon_start = hyperparams.get('epsilon_start', 1.0)
        self.epsilon_min = hyperparams.get('epsilon_min', 0.1)
        self.epsilon_stop_percentage = hyperparams.get('epsilon_stop_percentage', 0.25)
        self.total_steps = hyperparams.get('total_steps', 1000000)
        self.stop_step = int(self.total_steps * self.epsilon_stop_percentage)
        
        # Current state
        self.epsilon = self.epsilon_start
        self.current_step = 0
        
        self.learning_rate = hyperparams.get('learning_rate', 0.001)
        
        # Get neural network hyperparameters
        layer_size = hyperparams.get('layer_size', 32)
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

def run_experiment(experiment_config):
    """Run a single experiment with the given hyperparameters"""
    # Extract experiment configuration
    hyperparams = experiment_config['hyperparams']
    param_name = experiment_config['param_name']
    param_value = experiment_config['param_value']
    experiment_id = experiment_config['experiment_id']
    gpu_id = experiment_config['gpu_id']
    logger = experiment_config['logger']
    
    # Log experiment start with experiment identifier
    logger.info(f"[Exp {experiment_id}] Starting experiment: {param_name}={param_value}, GPU {gpu_id}")
    
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        logger.info(f"[Exp {experiment_id}] Using GPU {gpu_id}")
    
    # Extract hyperparameters
    update_frequency = hyperparams.get('update_frequency', 5)
    max_steps = hyperparams.get('max_steps', 100)
    total_steps = hyperparams.get('total_steps', 1000)
    
    # Create a local environment for this process
    local_env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Set parameters
    state_size = local_env.observation_space.shape[0]
    action_size = local_env.action_space.n
    agent = NaiveAgent(state_size, action_size, hyperparams)

    # Training variables
    episode_rewards = []
    average_rewards = []
    epsilon_values = []
    episode_losses = []  # To store average loss per episode
    env_steps = []       # Track total environment steps
    total_env_steps = 0  # Counter for total environment steps
    
    # Disable progress bar for individual experiments
    # Just log progress periodically
    
    episode = 0
    experiences_buffer = []  # Buffer to collect experiences before learning
    
    while total_env_steps < total_steps:
        episode += 1
        state, info = local_env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        start_loss_idx = len(agent.losses)  # Track starting index for this episode's losses
        episode_steps = 0  # Track steps in this episode
        episode_epsilon_values = []  # Track epsilon values during this episode
        
        for step in range(max_steps):
            # Record current epsilon for this step
            episode_epsilon_values.append(agent.epsilon)
            
            # Choose an action
            action = agent.act(state)
            
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = local_env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience instead of learning immediately
            done = terminated or truncated
            experiences_buffer.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_env_steps += 1  # Increment total environment steps
            
            # Log progress periodically (less frequently to reduce log size)
            if total_env_steps % 500 == 0:
                logger.info(f"[Exp {experiment_id}] Steps: {total_env_steps}/{total_steps}, Episode: {episode}, Reward: {episode_reward}")
            
            # Learn after collecting N experiences (1:N ratio)
            if len(experiences_buffer) >= update_frequency:
                agent.learn(experiences_buffer)
                experiences_buffer = []  # Clear buffer after learning
            
            # Check if we've reached the total steps target
            if total_env_steps >= total_steps:
                break
            
            # Check if episode is done
            if done:
                break
        
        # Calculate average loss for this episode
        if len(agent.losses) > start_loss_idx:
            episode_loss = np.mean(agent.losses[start_loss_idx:])
            episode_losses.append(episode_loss)
        else:
            episode_losses.append(0)  # No learning happened
        
        # Store average epsilon for this episode
        if episode_epsilon_values:
            avg_epsilon = np.mean(episode_epsilon_values)
            epsilon_values.append(avg_epsilon)
        else:
            epsilon_values.append(agent.epsilon)
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards)
        average_rewards.append(avg_reward)
        env_steps.append(total_env_steps)  # Store the current total steps
        
        # Log episode results (only for significant episodes to reduce log size)
        if episode % 10 == 0 or episode == 1:
            logger.info(f"[Exp {experiment_id}] Episode {episode} completed: Reward={episode_reward}, Avg={avg_reward:.2f}, Loss={episode_losses[-1]:.6f}")
    
    # Process any remaining experiences
    if experiences_buffer:
        agent.learn(experiences_buffer)
    
    local_env.close()
    logger.info(f"[Exp {experiment_id}] Experiment completed: {param_name}={param_value}, GPU {gpu_id}")
    
    # Return results
    return {
        'param_name': param_name,
        'param_value': param_value,
        'experiment_id': experiment_id,
        'episode_rewards': episode_rewards,
        'average_rewards': average_rewards,
        'episode_losses': episode_losses,
        'env_steps': env_steps,
        'epsilon_values': epsilon_values,
        'hyperparams': hyperparams
    }

def process_and_plot_ablation_results(all_results, param_name, output_dir, start_time, duration):
    """Process results and create plots for the ablation study"""
    # Define better colors and line styles
    colors = ['#0173B2', '#029E73', '#D55E00', '#CC78BC', 'orange', 'brown', 'pink', 'gray']
    
    # Group results by parameter value
    results_by_param = defaultdict(list)
    for result in all_results:
        if result['param_name'] == param_name:
            results_by_param[result['param_value']].append(result)
    
    # Get parameter values and sort them
    param_values = sorted(list(results_by_param.keys()))
    
    # Determine how many columns we need (up to 5)
    num_columns = min(5, len(param_values))
    
    # Create a figure with a grid of subplots (2 rows, up to 5 columns)
    # First row: Loss plots for each parameter value
    # Second row: Reward plots for each parameter value
    fig, axs = plt.subplots(2, num_columns, figsize=(4*num_columns, 10))
    
    # Create output file paths
    output_file = os.path.join(output_dir, f"{param_name}_ablation_cartpole.png")
    stats_file = os.path.join(output_dir, f"{param_name}_ablation_stats_cartpole.txt")
    
    # Create a text file to save statistics
    with open(stats_file, 'w') as f:
        f.write(f"Ablation Study Results for {param_name}\n")
        f.write("=" * 50 + "\n\n")
        current_time = get_amsterdam_time()
        f.write(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)\n")
        f.write(f"Experiment started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)\n")
        f.write(f"Duration: {duration:.2f} hours\n\n")
        f.write(f"Parameter studied: {param_name}\n")
        f.write(f"Values tested: {param_values}\n\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Value':<10} {'Avg Reward':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10} {'Final Avg Loss':<15}\n")
        f.write("-" * 80 + "\n")
    
    # Process each parameter value
    summary_stats = {}
    
    for i, param_value in enumerate(param_values):
        # Skip if beyond our plotting capacity
        if i >= num_columns:
            continue
            
        param_results = results_by_param[param_value]
        
        # Interpolate results to common x-axis
        max_steps = max([max(result['env_steps']) for result in param_results])
        common_x = np.linspace(0, max_steps, 100)
        
        # Collect interpolated losses
        interpolated_losses = []
        for result in param_results:
            x = result['env_steps']
            y = result['episode_losses']
            interpolated_losses.append(np.interp(common_x, x, y))
        
        # Calculate mean and std of losses
        mean_losses = np.mean(interpolated_losses, axis=0)
        std_losses = np.std(interpolated_losses, axis=0)
        
        # --- LOSS SUBPLOT (Top Row) ---
        ax_loss = axs[0, i]
        
        # Plot mean loss with std deviation
        ax_loss.plot(common_x, mean_losses, color=colors[i], linewidth=2.5)
        ax_loss.fill_between(common_x, 
                            mean_losses - std_losses, 
                            mean_losses + std_losses, 
                            color=colors[i], alpha=0.2)
        ax_loss.set_xlabel('Environment Steps', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title(f'Loss for {param_name}={param_value}', fontsize=12)
        ax_loss.grid(True, alpha=0.3)
        
        # Collect interpolated rewards
        interpolated_rewards = []
        for result in param_results:
            x = result['env_steps']
            y = result['episode_rewards']
            interpolated_rewards.append(np.interp(common_x, x, y))
        
        # Calculate mean and std of rewards
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        
        # --- REWARD SUBPLOT (Bottom Row) ---
        ax_reward = axs[1, i]
        
        # Plot mean reward with std deviation
        ax_reward.plot(common_x, mean_rewards, color=colors[i], linewidth=2.5)
        ax_reward.fill_between(common_x, 
                              mean_rewards - std_rewards, 
                              mean_rewards + std_rewards, 
                              color=colors[i], alpha=0.2)
        ax_reward.set_xlabel('Environment Steps', fontsize=10)
        ax_reward.set_ylabel('Reward', fontsize=10)
        ax_reward.set_title(f'Reward for {param_name}={param_value}', fontsize=12)
        ax_reward.grid(True, alpha=0.3)
        
        # Set the y-axis limit for reward plots to be consistent (0 to 500)
        ax_reward.set_ylim(0, 500)
        
        # Collect statistics for this parameter value
        all_rewards = [item for sublist in [result['episode_rewards'] for result in param_results] for item in sublist]
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        min_reward = np.min(all_rewards)
        max_reward = np.max(all_rewards)
        final_loss = mean_losses[-1] if len(mean_losses) > 0 else 0
        
        # Store statistics
        summary_stats[param_value] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'final_loss': final_loss,
            'mean_rewards': mean_rewards,
            'std_rewards': std_rewards,
            'mean_losses': mean_losses,
            'std_losses': std_losses
        }
        
        # Write to stats file
        with open(stats_file, 'a') as f:
            f.write(f"{param_value:<10} {avg_reward:<15.2f} {std_reward:<15.2f} {min_reward:<10.2f} {max_reward:<10.2f} {final_loss:<15.6f}\n")
    
    # Add a main title for the entire figure
    fig.suptitle(f'Ablation Study Results for {param_name}', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    
    # Save the figure
    plt.savefig(output_file, dpi=300)  # Higher DPI for better quality
    plt.close()

def update_master_progress(result, pbar):
    """Callback function to update the master progress bar"""
    pbar.update(1)
    return result

def get_gpu_memory_map():
    """
    Get the GPU memory usage using nvidia-smi command.
    Returns a dictionary with GPU indices as keys and free memory in MiB as values.
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
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        # Return empty dict if nvidia-smi fails
        return {}

def get_gpu_with_most_free_memory():
    """
    Returns the index of the GPU with the most free memory based on nvidia-smi.
    If no GPUs are available, returns 0.
    """
    gpu_memory = get_gpu_memory_map()
    
    if not gpu_memory:
        print("Warning: Could not get GPU memory info, defaulting to GPU 0")
        return 0
    
    # Find GPU with most free memory
    gpu_idx = max(gpu_memory.items(), key=lambda x: x[1])[0]
    free_memory_mb = gpu_memory[gpu_idx]
    
    return gpu_idx

def monitor_gpu_memory(interval=30):
    """
    Monitor GPU memory usage and print warnings if memory is low.
    """
    while True:
        gpu_memory = get_gpu_memory_map()
        for gpu_idx, free_memory in gpu_memory.items():
            if free_memory < 1000:  # Less than 1GB free
                print(f"WARNING: GPU {gpu_idx} has only {free_memory} MiB free memory!")
        time.sleep(interval)

def run_massively_parallel_ablation_studies():
    """Run all ablation studies with maximum parallelization"""
    # Record start time in Amsterdam timezone
    start_time = get_amsterdam_time()
    
    # Create a sequentially numbered directory for this run
    os.makedirs("results", exist_ok=True)
    
    # Find the next run number
    existing_runs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d)) and d.startswith("run_")]
    run_numbers = [int(run.split("_")[1]) for run in existing_runs if run.split("_")[1].isdigit()]
    next_run_number = 1 if not run_numbers else max(run_numbers) + 1
    
    # Create the new run directory
    results_dir = os.path.join("results", f"run_{next_run_number}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Print minimal initial information
    print(f"Run {next_run_number} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)")
    print(f"Results will be saved to: {results_dir}")
    
    # Set up a single logger for all experiments
    log_file = os.path.join(results_dir, "logs.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    # Get logger instance
    logger = logging.getLogger()
    
    # Log start of run
    logger.info(f"Starting ablation studies - Run {next_run_number}")
    logger.info(f"All logs will be saved to: {log_file}")
    
    # Start GPU memory monitoring in a separate thread
    import threading
    monitor_thread = threading.Thread(target=monitor_gpu_memory, daemon=True)
    monitor_thread.start()
    
    # Save a copy of the hyperparameters used for this run
    with open(os.path.join(results_dir, "hyperparameters.txt"), 'w') as f:
        f.write("Default Hyperparameters:\n")
        for key, value in default_hyperparams.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nAblation Parameters:\n")
        for key, values in ablation_params.items():
            f.write(f"{key}: {values}\n")
        
        # Also save the timestamp for reference
        f.write(f"\nRun started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)\n")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs available")
    
    # Calculate GPU capacity
    gpu_capacity_per_experiment = 0.05  # Each experiment uses 5% of a GPU
    gpu_utilization = 0.85  # GPUs run at 85% capacity
    experiments_per_gpu = int(gpu_utilization / gpu_capacity_per_experiment)
    total_parallel_experiments = experiments_per_gpu * num_gpus
    
    logger.info(f"Each GPU can run {experiments_per_gpu} experiments in parallel at {gpu_utilization*100}% utilization")
    logger.info(f"Total parallel experiments possible: {total_parallel_experiments}")
    
    # Prepare all experiment configurations
    all_experiment_configs = []
    experiment_id = 0
    
    for param_name, param_values in ablation_params.items():
        for param_value in param_values:
            # Create hyperparameters for this experiment
            hyperparams = default_hyperparams.copy()
            hyperparams[param_name] = param_value
            
            # Create multiple experiment configs for statistical significance
            for i in range(5):  # 5 experiments per parameter value
                # Create experiment configuration
                experiment_config = {
                    'param_name': param_name,
                    'param_value': param_value,
                    'hyperparams': hyperparams,
                    'experiment_id': experiment_id,
                    'gpu_id': None,  # Will be assigned right before execution
                    'logger': logger
                }
                
                all_experiment_configs.append(experiment_config)
                experiment_id += 1
    
    total_experiments = len(all_experiment_configs)
    logger.info(f"Total experiments to run: {total_experiments}")
    
    
    # Create a pool of workers
    num_processes = min(total_parallel_experiments, mp.cpu_count(), total_experiments)
    logger.info(f"Using {num_processes} processes")
    
    # Print minimal info before starting
    print(f"Running {total_experiments} experiments using {num_processes} processes")
    print("Progress:")
    
    # Run experiments in parallel with a single master progress bar
    run_start_time = time.time()
    
    # Create a master progress bar - THE ONLY THING VISIBLE IN TERMINAL
    master_pbar = tqdm.tqdm(
        total=total_experiments, 
        desc="Overall Progress",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    # Use apply_async with callback to update the master progress bar
    results = []
    with mp.Pool(processes=num_processes) as pool:
        # Create a list to hold all the async results
        async_results = []
        
        # Submit all tasks
        for config in all_experiment_configs:
            # Dynamically assign GPU right before execution
            config['gpu_id'] = get_gpu_with_most_free_memory()
            
            async_result = pool.apply_async(
                run_experiment, 
                args=(config,),
                callback=lambda result: update_master_progress(result, master_pbar)
            )
            async_results.append(async_result)
        
        # Wait for all tasks to complete
        for async_result in async_results:
            result = async_result.get()  # This will block until the result is ready
            results.append(result)
    
    # Close the master progress bar
    master_pbar.close()
    
    run_end_time = time.time()
    elapsed_time = run_end_time - run_start_time
    elapsed_hours = elapsed_time / 3600
    
    # Print completion message
    print(f"\nAll experiments completed in {elapsed_hours:.2f} hours")
    print(f"Results saved to {results_dir}")
    
    # Log completion
    logger.info(f"All experiments completed in {elapsed_hours:.2f} hours")
    
    # Save the raw results
    results_file = os.path.join(results_dir, "all_results.pkl")
    with open(results_file, 'wb') as f:
        import pickle
        pickle.dump(results, f)
    logger.info(f"Raw results saved to {results_file}")
    
    # Process results for each parameter
    for param_name in ablation_params.keys():
        process_and_plot_ablation_results(results, param_name, results_dir, start_time, elapsed_hours)
        logger.info(f"Results for {param_name} processed and saved to {results_dir}")
    
    # Create a summary file
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Ablation Study Summary - Run {next_run_number}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Amsterdam time)\n")
        f.write(f"Duration: {elapsed_hours:.2f} hours\n\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"GPUs used: {num_gpus}\n")
        f.write(f"Processes used: {num_processes}\n\n")
        
        f.write("Parameters studied:\n")
        for param_name, param_values in ablation_params.items():
            f.write(f"- {param_name}: {param_values}\n")
    
    logger.info(f"All ablation studies complete! Results saved to {results_dir} (Run {next_run_number})")
    
    # Final message
    print(f"Ablation studies complete! Run {next_run_number} finished in {elapsed_hours:.2f} hours")

# Main execution
if __name__ == "__main__":
    print("Training Naive Function Approximation agent for CartPole-v1")
    print("Running massively parallel ablation studies")
    
    # Run all ablation studies with maximum parallelization
    run_massively_parallel_ablation_studies()
    
    # Close the environment
    env.close()

