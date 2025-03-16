print("Starting script...")

# Force matplotlib to use a non-interactive backend
import os
print("Setting matplotlib backend...")
os.environ['MPLBACKEND'] = 'Agg'  # Use the Agg backend (non-interactive)

import numpy as np
print("Importing matplotlib...")
import matplotlib
matplotlib.use('Agg')  # Set the backend again to be sure
import matplotlib.pyplot as plt
print("Matplotlib imported successfully")

import pickle
from collections import defaultdict
import time

print("All imports successful")

# Directory containing the results
results_dir = "results/methods_comparison_17"
pickle_file = os.path.join(results_dir, "all_results.pkl")

print("Loading pickle file: {}".format(pickle_file))
start_time = time.time()

# Load the pickle file with a timeout check
try:
    with open(pickle_file, 'rb') as f:
        print("File opened, loading data...")
        all_results = pickle.load(f)
        print("Data loaded successfully in {:.2f} seconds".format(time.time() - start_time))
        print("Found {} experiment results".format(len(all_results)))
except Exception as e:
    print("Error loading pickle file: {}".format(e))
    exit(1)

# Fixed hyperparameters (copied from 4methods.py)
fixed_hyperparams = {
    'update_frequency': 5,
    'max_steps': 500,
    'total_steps': 1000000,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.1,
    'epsilon_stop_percentage': 0.1,
    'learning_rate': 0.001,
    'layer_size': 64,
    'nr_layers': 3,
    'target_update_frequency': 50,
    'replay_buffer_size': 100000,
    'batch_size': 16,
}

def plot_comparison_results_no_std(all_results, output_dir):
    """Create plots comparing the different methods without standard deviation bands"""
    print("Grouping results by method...")
    # Group results by method
    results_by_method = defaultdict(list)
    for result in all_results:
        results_by_method[result['method_name']].append(result)
    
    print("Creating figure...")
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
        'random_baseline': 'Random Baseline'
    }
    
    print("Finding maximum steps...")
    # Find maximum steps across all runs for common x-axis
    max_steps = 0
    for method_results in results_by_method.values():
        for result in method_results:
            if result['env_steps']:
                max_steps = max(max_steps, max(result['env_steps']))
    
    print("Creating common x-axis with {} points...".format(max_steps))
    # Create common x-axis for interpolation
    common_x = np.linspace(0, max_steps, 1000)
    
    # Define smoothing function
    def smooth(data, window=100):
        """Apply moving average smoothing to data"""
        smoothed = np.convolve(data, np.ones(window)/window, mode='same')
        # Fix the edges affected by the convolution
        smoothed[:window//2] = data[:window//2]
        smoothed[-window//2:] = data[-window//2:]
        return smoothed
    
    print("Loading random baseline data...")
    # Try to load the random baseline data
    try:
        random_baseline_file = "RandomBaselineCartPole.csv"
        if os.path.exists(random_baseline_file):
            print("Random baseline file found, loading...")
            # Load the random baseline data
            random_data = np.genfromtxt(random_baseline_file, delimiter=',', names=True)
            
            # Extract the columns we need
            random_rewards = random_data['Episode_Return']
            random_steps = random_data['env_step']
            
            print("Processing random baseline data...")
            # Use all data points regardless of range
            valid_indices = np.ones_like(random_steps, dtype=bool)
            
            if np.any(valid_indices):
                random_steps_valid = random_steps[valid_indices]
                random_rewards_valid = random_rewards[valid_indices]
                
                if len(random_steps_valid) > 1:
                    print("Interpolating random baseline data...")
                    random_rewards_interp = np.interp(
                        common_x, 
                        random_steps_valid, 
                        random_rewards_valid,
                        left=random_rewards_valid[0],
                        right=random_rewards_valid[-1]
                    )
                    
                    # Apply smoothing to random baseline
                    random_rewards_smoothed = smooth(random_rewards_interp, window=100)
                    
                    print("Plotting random baseline...")
                    # Plot random baseline with dashed line and full opacity
                    ax1.plot(common_x, random_rewards_smoothed, color=colors['random_baseline'], 
                             linewidth=2.0, linestyle='--', label=method_labels['random_baseline'])
                    
                    print("Successfully loaded and plotted random baseline data")
                else:
                    print("Not enough valid data points in random baseline")
            else:
                print("No valid indices in random baseline data")
        else:
            print("Random baseline file not found")
    except Exception as e:
        print("Error loading random baseline data: {}".format(e))
    
    print("Plotting rewards for each method...")
    # Plot rewards by steps for each method (mean only, no std deviation)
    for method_name, method_results in results_by_method.items():
        print("Processing method: {}".format(method_name))
        # Interpolate each run to the common x-axis
        interpolated_rewards = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['average_rewards']
            if len(x) > 1 and len(y) > 1:
                interpolated_rewards.append(np.interp(common_x, x, y))
        
        if not interpolated_rewards:
            print("No valid reward data for method: {}".format(method_name))
            continue
            
        # Calculate mean of rewards across runs
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        
        # Apply strong smoothing
        mean_rewards_smoothed = smooth(mean_rewards, window=100)
        
        print("Plotting reward data for method: {}".format(method_name))
        # Plot smoothed mean reward
        ax1.plot(common_x, mean_rewards_smoothed, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
    
    print("Configuring reward plot...")
    # Configure reward plot
    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward vs. Environment Steps (Smoothed)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 500)  # CartPole max reward is 500
    
    print("Plotting loss curves...")
    # Plot loss curves for each method (mean only, no std deviation)
    for method_name, method_results in results_by_method.items():
        print("Processing loss data for method: {}".format(method_name))
        # Interpolate each run to the common x-axis
        interpolated_losses = []
        
        for result in method_results:
            x = result['env_steps']
            y = result['episode_losses']
            if len(x) > 1 and len(y) > 1:
                interpolated_losses.append(np.interp(common_x, x, y))
        
        if not interpolated_losses:
            print("No valid loss data for method: {}".format(method_name))
            continue
            
        # Calculate mean of losses across runs
        mean_losses = np.mean(interpolated_losses, axis=0)
        
        # No smoothing for loss data
        print("Plotting loss data for method: {}".format(method_name))
        # Plot raw mean loss (no smoothing)
        ax2.plot(common_x, mean_losses, color=colors[method_name], linewidth=2.5, label=method_labels[method_name])
    
    print("Configuring loss plot...")
    # Configure loss plot
    ax2.set_xlabel('Environment Steps', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Training Loss vs. Environment Steps (Log Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Set y-axis to log scale for better visualization of loss values
    ax2.set_yscale('symlog')  # Use symlog to handle potential negative values
    
    # Set reasonable y-limits to focus on the relevant range
    y_min = 0.1  # Minimum positive value to show
    y_max = max(100, ax2.get_ylim()[1])  # Use at least 100 or the current max
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    print("Saving figure...")
    # Save the figure
    output_file = os.path.join(output_dir, "method_comparison_smoothed.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print("Figure saved to: {}".format(output_file))

def create_stats_file(all_results, output_dir):
    """Create a summary statistics file"""
    print("Creating statistics file...")
    # Group results by method
    results_by_method = defaultdict(list)
    for result in all_results:
        results_by_method[result['method_name']].append(result)
    
    # Method labels for display
    method_labels = {
        'baseline': 'Baseline DQN',
        'target_network_only': 'Target Network Only',
        'experience_replay_only': 'Experience Replay Only',
        'both_methods': 'Both Methods',
        'random_baseline': 'Random Baseline'
    }
    
    # Create a summary statistics file
    stats_file = os.path.join(output_dir, "method_comparison_stats.txt")
    print("Writing statistics to: {}".format(stats_file))
    
    with open(stats_file, 'w') as f:
        f.write("Comparison of DQN Methods\n")
        f.write("========================\n\n")
        
        f.write("Fixed Hyperparameters:\n")
        for key, value in fixed_hyperparams.items():
            f.write("  {}: {}\n".format(key, value))
        
        f.write("\nResults Summary:\n")
        f.write("-" * 160 + "\n")
        f.write("{:<20} {:<15} {:<15} {:<15} {:<10} {:<10} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Method", "Avg Reward", "Std Dev (Runs)", "Std Dev (Eps)", "Min", "Max", "Avg Episodes", 
            "Last Ep Reward", "Last 200K Avg", "Max Episode Rew"))
        f.write("-" * 160 + "\n")
        
        for method_name, method_results in results_by_method.items():
            print("Calculating statistics for method: {}".format(method_name))
            # Calculate statistics across runs
            all_rewards = [np.mean(result['episode_rewards']) for result in method_results]
            avg_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)  # Std dev across runs
            min_reward = np.min(all_rewards)
            max_reward = np.max(all_rewards)
            avg_episodes = np.mean([result['episodes_completed'] for result in method_results])
            
            # Calculate standard deviation across all episodes (concatenating all runs)
            all_episodes_rewards = []
            for result in method_results:
                all_episodes_rewards.extend(result['episode_rewards'])
            
            std_all_episodes = np.std(all_episodes_rewards) if all_episodes_rewards else float('nan')
            
            # Calculate average of last episode rewards across runs
            last_episode_rewards = []
            for result in method_results:
                if result['episode_rewards'] and len(result['episode_rewards']) > 0:
                    last_episode_rewards.append(result['episode_rewards'][-1])
            
            avg_last_episode_reward = np.mean(last_episode_rewards) if last_episode_rewards else float('nan')
            
            # Calculate maximum episode reward across all runs
            max_episode_rewards = []
            for result in method_results:
                if result['episode_rewards'] and len(result['episode_rewards']) > 0:
                    max_episode_rewards.append(np.max(result['episode_rewards']))
            
            max_episode_reward = np.mean(max_episode_rewards) if max_episode_rewards else float('nan')
            
            # Calculate average reward from last 200,000 environment steps
            last_200k_rewards = []
            for result in method_results:
                if result['env_steps'] and result['episode_rewards']:
                    # Find episodes that occurred in the last 200K steps
                    total_steps = result['env_steps'][-1]
                    last_200k_start = max(0, total_steps - 200000)
                    
                    # Find the index of the first episode that starts after last_200k_start
                    episode_start_steps = [0] + [result['env_steps'][i-1] for i in range(1, len(result['env_steps']))]
                    last_200k_episode_indices = [i for i, step in enumerate(episode_start_steps) if step >= last_200k_start]
                    
                    if last_200k_episode_indices:
                        # Get rewards for episodes in the last 200K steps
                        last_200k_episode_rewards = [result['episode_rewards'][i] for i in last_200k_episode_indices]
                        last_200k_rewards.append(np.mean(last_200k_episode_rewards))
            
            avg_last_200k_reward = np.mean(last_200k_rewards) if last_200k_rewards else float('nan')
            
            # Write to file
            f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<15.1f} {:<15.2f} {:<15.2f} {:<15.2f}\n".format(
                method_labels[method_name], avg_reward, std_reward, std_all_episodes, min_reward, max_reward, 
                avg_episodes, avg_last_episode_reward, avg_last_200k_reward, max_episode_reward))
        
        # Add random baseline stats if available
        try:
            print("Adding random baseline statistics...")
            if os.path.exists("RandomBaselineCartPole.csv"):
                random_data = np.genfromtxt("RandomBaselineCartPole.csv", delimiter=',', names=True)
                random_rewards = random_data['Episode_Return']
                random_steps = random_data['env_step']
                avg_random = np.mean(random_rewards)
                std_random = np.std(random_rewards)
                std_all_episodes_random = np.std(random_rewards)  # For random, this is the same as std_random
                min_random = np.min(random_rewards)
                max_random = np.max(random_rewards)
                last_random = random_rewards[-1] if len(random_rewards) > 0 else float('nan')
                max_episode_random = np.max(random_rewards)
                
                # Calculate last 200K steps average for random baseline
                if len(random_steps) > 0:
                    total_steps = random_steps[-1]
                    last_200k_start = max(0, total_steps - 200000)
                    last_200k_indices = random_steps >= last_200k_start
                    last_200k_avg = np.mean(random_rewards[last_200k_indices]) if np.any(last_200k_indices) else float('nan')
                else:
                    last_200k_avg = float('nan')
                
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<15} {:<15.2f} {:<15.2f} {:<15.2f}\n".format(
                    "Random Baseline", avg_random, std_random, std_all_episodes_random, min_random, max_random, "N/A", 
                    last_random, last_200k_avg, max_episode_random))
        except Exception as e:
            print("Error adding random baseline stats: {}".format(e))
        
        # Add a section explaining the different standard deviation metrics
        f.write("\nStandard Deviation Explanation:\n")
        f.write("- Std Dev (Runs): Standard deviation of average rewards across the 5 runs (measures run-to-run consistency)\n")
        f.write("- Std Dev (Eps): Standard deviation across all episodes in all runs (measures episode-to-episode variability)\n\n")
        
        f.write("\nDetailed Analysis:\n")
        
        # Find best method based on average reward
        print("Finding best method...")
        method_avg_rewards = {method_name: np.mean([np.mean(result['episode_rewards']) for result in method_results]) 
                            for method_name, method_results in results_by_method.items()}
        best_method = max(method_avg_rewards.items(), key=lambda x: x[1])[0]
        
        f.write("\nBest Method (Overall Avg): {} with average reward {:.2f}\n".format(
            method_labels[best_method], method_avg_rewards[best_method]))
        
        # Find best method based on last episode reward
        method_last_ep_rewards = {}
        for method_name, method_results in results_by_method.items():
            last_rewards = []
            for result in method_results:
                if result['episode_rewards'] and len(result['episode_rewards']) > 0:
                    last_rewards.append(result['episode_rewards'][-1])
            if last_rewards:
                method_last_ep_rewards[method_name] = np.mean(last_rewards)
        
        if method_last_ep_rewards:
            best_last_ep_method = max(method_last_ep_rewards.items(), key=lambda x: x[1])[0]
            f.write("Best Method (Last Episode): {} with average last episode reward {:.2f}\n".format(
                method_labels[best_last_ep_method], method_last_ep_rewards[best_last_ep_method]))
        
        # Find best method based on last 200K steps
        method_last_200k_rewards = {}
        for method_name, method_results in results_by_method.items():
            last_200k_avgs = []
            for result in method_results:
                if result['env_steps'] and result['episode_rewards']:
                    # Find episodes that occurred in the last 200K steps
                    total_steps = result['env_steps'][-1]
                    last_200k_start = max(0, total_steps - 200000)
                    
                    # Find the index of the first episode that starts after last_200k_start
                    episode_start_steps = [0] + [result['env_steps'][i-1] for i in range(1, len(result['env_steps']))]
                    last_200k_episode_indices = [i for i, step in enumerate(episode_start_steps) if step >= last_200k_start]
                    
                    if last_200k_episode_indices:
                        # Get rewards for episodes in the last 200K steps
                        last_200k_episode_rewards = [result['episode_rewards'][i] for i in last_200k_episode_indices]
                        last_200k_avgs.append(np.mean(last_200k_episode_rewards))
            
            if last_200k_avgs:
                method_last_200k_rewards[method_name] = np.mean(last_200k_avgs)
        
        if method_last_200k_rewards:
            best_last_200k_method = max(method_last_200k_rewards.items(), key=lambda x: x[1])[0]
            f.write("Best Method (Last 200K Steps): {} with average reward {:.2f}\n\n".format(
                method_labels[best_last_200k_method], method_last_200k_rewards[best_last_200k_method]))
        
        # Find method with lowest episode-to-episode variability
        method_episode_std = {}
        for method_name, method_results in results_by_method.items():
            all_episodes_rewards = []
            for result in method_results:
                all_episodes_rewards.extend(result['episode_rewards'])
            
            if all_episodes_rewards:
                method_episode_std[method_name] = np.std(all_episodes_rewards)
        
        if method_episode_std:
            most_consistent_method = min(method_episode_std.items(), key=lambda x: x[1])[0]
            f.write("Most Consistent Method (Lowest Episode-to-Episode Variability): {} with std dev {:.2f}\n\n".format(
                method_labels[most_consistent_method], method_episode_std[most_consistent_method]))
        
        # Compare the impact of each enhancement
        print("Analyzing enhancement impacts...")
        f.write("Enhancement Impact Analysis:\n")
        
        # Impact of target network (both_methods vs experience_replay_only)
        if 'both_methods' in method_avg_rewards and 'experience_replay_only' in method_avg_rewards:
            target_impact = method_avg_rewards['both_methods'] - method_avg_rewards['experience_replay_only']
            f.write("Impact of adding Target Network to Experience Replay: {:+.2f}\n".format(target_impact))
        
        # Impact of experience replay (both_methods vs target_network_only)
        if 'both_methods' in method_avg_rewards and 'target_network_only' in method_avg_rewards:
            er_impact = method_avg_rewards['both_methods'] - method_avg_rewards['target_network_only']
            f.write("Impact of adding Experience Replay to Target Network: {:+.2f}\n".format(er_impact))
        
        # Impact of target network alone (target_network_only vs baseline)
        if 'target_network_only' in method_avg_rewards and 'baseline' in method_avg_rewards:
            target_only_impact = method_avg_rewards['target_network_only'] - method_avg_rewards['baseline']
            f.write("Impact of Target Network alone: {:+.2f}\n".format(target_only_impact))
        
        # Impact of experience replay alone (experience_replay_only vs baseline)
        if 'experience_replay_only' in method_avg_rewards and 'baseline' in method_avg_rewards:
            er_only_impact = method_avg_rewards['experience_replay_only'] - method_avg_rewards['baseline']
            f.write("Impact of Experience Replay alone: {:+.2f}\n".format(er_only_impact))
        
        # Add section for last episode reward analysis
        f.write("\nLast Episode Reward Analysis:\n")
        f.write("-" * 80 + "\n")
        f.write("{:<20} {:<15} {:<15} {:<15}\n".format(
            "Method", "Last Ep Reward", "Std Dev", "% of Max Possible"))
        f.write("-" * 80 + "\n")
        
        for method_name, method_results in results_by_method.items():
            last_rewards = []
            for result in method_results:
                if result['episode_rewards'] and len(result['episode_rewards']) > 0:
                    last_rewards.append(result['episode_rewards'][-1])
            
            if last_rewards:
                avg_last = np.mean(last_rewards)
                std_last = np.std(last_rewards)
                pct_of_max = (avg_last / 500.0) * 100  # CartPole max is 500
                
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    method_labels[method_name], avg_last, std_last, pct_of_max))
        
        # Add section for last 200K steps analysis
        f.write("\nLast 200K Steps Reward Analysis:\n")
        f.write("-" * 80 + "\n")
        f.write("{:<20} {:<15} {:<15} {:<15}\n".format(
            "Method", "Last 200K Avg", "Std Dev", "% of Max Possible"))
        f.write("-" * 80 + "\n")
        
        for method_name, method_results in results_by_method.items():
            last_200k_avgs = []
            for result in method_results:
                if result['env_steps'] and result['episode_rewards']:
                    # Find episodes that occurred in the last 200K steps
                    total_steps = result['env_steps'][-1]
                    last_200k_start = max(0, total_steps - 200000)
                    
                    # Find the index of the first episode that starts after last_200k_start
                    episode_start_steps = [0] + [result['env_steps'][i-1] for i in range(1, len(result['env_steps']))]
                    last_200k_episode_indices = [i for i, step in enumerate(episode_start_steps) if step >= last_200k_start]
                    
                    if last_200k_episode_indices:
                        # Get rewards for episodes in the last 200K steps
                        last_200k_episode_rewards = [result['episode_rewards'][i] for i in last_200k_episode_indices]
                        last_200k_avgs.append(np.mean(last_200k_episode_rewards))
            
            if last_200k_avgs:
                avg_last_200k = np.mean(last_200k_avgs)
                std_last_200k = np.std(last_200k_avgs)
                pct_of_max = (avg_last_200k / 500.0) * 100  # CartPole max is 500
                
                f.write("{:<20} {:<15.2f} {:<15.2f} {:<15.2f}%\n".format(
                    method_labels[method_name], avg_last_200k, std_last_200k, pct_of_max))
    
    print("Statistics file created successfully")

# Main execution
try:
    # Generate plots without standard deviation bands
    print("\nGenerating plots...")
    plot_comparison_results_no_std(all_results, results_dir)
    
    # Generate statistics file
    print("\nGenerating statistics...")
    create_stats_file(all_results, results_dir)
    
    print("\nAll tasks completed successfully!")
except Exception as e:
    print("\nError during execution: {}".format(e))
    import traceback
    traceback.print_exc()
