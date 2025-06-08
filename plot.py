import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

def find_training_folders(base_dir="./training") -> List[str]:
    """Find all folders in /training that contain experiment_summary.json files."""
    training_folders = []
    
    # Walk through all subdirectories in the /training folder
    print(f"Scanning {base_dir} for training experiments...")
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains an experiment_summary.json file
        if "experiment_summary.json" in files:
            # Store the full path to the folder
            training_folders.append(root)
    
    return sorted(training_folders)

def load_experiment_summary(folder_path: str) -> Dict:
    """Load the experiment summary JSON from a folder."""
    json_path = os.path.join(folder_path, "experiment_summary.json")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": "Could not load experiment summary"}

def display_training_folders(folders: List[str]) -> None:
    """Display the list of training folders with their complete experiment summaries."""
    print("\nAvailable Training Experiments:")
    print("=" * 80)
    
    for i, folder in enumerate(folders, 1):
        summary = load_experiment_summary(folder)
        print(f"{i}. {os.path.basename(folder)}")
        
        # Display the entire summary JSON content
        print("   Summary Content:")
        for key, value in summary.items():
            # Format the output for readability
            if isinstance(value, (int, float)) and key in ["accuracy", "f1_score"]:
                print(f"   - {key}: {value:.2f}%")
            else:
                print(f"   - {key}: {value}")
        print("-" * 80)

def get_user_selections(folders: List[str]) -> List[int]:
    """Get the user's selection of training folders by index."""
    while True:
        try:
            selection = input("\nSelect training folders by number (comma-separated, e.g., 1,3,4): ").strip()
            indices = [int(idx.strip()) for idx in selection.split(",") if idx.strip()]
            
            # Check if all indices are valid
            if all(1 <= idx <= len(folders) for idx in indices):
                return indices
            else:
                print(f"Error: Please enter valid indices between 1 and {len(folders)}")
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

def get_plot_parameters(selected_folders: List[str], indices: List[int]) -> Tuple[str, List[str], Tuple[float, float]]:
    """Get the metric choice, custom labels, and y-axis range for the plot."""
    # Choose metric
    while True:
        metric = input("\nChoose metric to plot (accuracy or f1): ").strip().lower()
        if metric in ["accuracy", "f1"]:
            break
        print("Error: Please enter either 'accuracy' or 'f1'")
    
    metric_key = "accuracy" if metric == "accuracy" else "f1_score"
    
    # Get y-axis range customization
    custom_range = input("\nWould you like to set a custom y-axis range? (y/n): ").strip().lower()
    y_min, y_max = None, None
    
    if custom_range == 'y':
        while True:
            try:
                range_input = input("Enter range as 'min-max' (e.g., 94.0-94.9): ").strip()
                if '-' not in range_input:
                    print("Error: Please use format 'min-max' (e.g., 94.0-94.9)")
                    continue
                    
                parts = range_input.split('-')
                y_min = float(parts[0].strip())
                y_max = float(parts[1].strip())
                
                if y_min >= y_max:
                    print("Error: Min value must be less than max value")
                    continue
                    
                break
            except ValueError:
                print("Error: Please enter valid numbers")
    
    # Get custom labels for each selected training
    labels = []
    print("\nProvide custom labels for each training:")
    for i, folder_idx in enumerate(range(len(selected_folders))):
        folder = selected_folders[folder_idx]
        summary = load_experiment_summary(folder)
        
        print(f"\nTraining {i+1}: {os.path.basename(folder)}")
        
        # Display the entire summary JSON content
        print("   Complete Summary Content:")
        for key, value in summary.items():
            # Format the output for readability
            if isinstance(value, (int, float)) and key in ["accuracy", "f1_score"]:
                print(f"   - {key}: {value:.2f}%")
            else:
                print(f"   - {key}: {value}")
        
        label = input(f"Label for this training (press Enter to use folder name): ").strip()
        if not label:
            label = os.path.basename(folder)
        labels.append(label)
    
    return metric_key, labels, (y_min, y_max)

def create_plot(selected_folders: List[str], indices: List[int], metric: str, labels: List[str], y_range: Tuple[float, float] = (None, None)) -> None:
    """Create and display the plot."""
    # Extract the metric values
    values = []
    for i in range(len(selected_folders)):
        folder = selected_folders[i]
        summary = load_experiment_summary(folder)
        values.append(summary.get(metric, 0))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, width=0.6, color='skyblue', edgecolor='navy')
    
    # Add labels and title
    metric_name = "Accuracy" if metric == "accuracy" else "F1 Score"
    plt.xlabel("Training Experiments", fontsize=12)
    plt.ylabel(f"{metric_name} (%)", fontsize=12)
    plt.title(f"{metric_name} Comparison of Selected Training Experiments", fontsize=14)
    
    # Set custom y-axis limits if provided
    y_min, y_max = y_range
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
        # Adjust the position of the value labels based on custom range
        label_offset = (y_max - y_min) * 0.02
    else:
        # Default offset
        label_offset = 0.5
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + label_offset,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    # Add x-tick labels
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create plots directory if it doesn't exist
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot to the plots folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_filename = f"training_comparison_{metric_name.lower().replace(' ', '_')}_{timestamp}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    
    # Display plot
    plt.show()

def main():
    """Main function to orchestrate the workflow."""
    training_folders = find_training_folders()
    
    if not training_folders:
        print("No valid training experiments found with experiment_summary.json files.")
        return
    
    print(f"Found {len(training_folders)} training experiments.")
    display_training_folders(training_folders)
    selected_indices = get_user_selections(training_folders)
    
    if not selected_indices:
        print("No valid selections. Exiting.")
        return
    
    # Create list of selected folder paths
    selected_folders = [training_folders[i-1] for i in selected_indices]
    
    # Pass the selected folders and their sequential indices (1, 2, 3, etc.)
    metric, labels, y_range = get_plot_parameters(selected_folders, list(range(1, len(selected_folders) + 1)))
    create_plot(selected_folders, list(range(1, len(selected_folders) + 1)), metric, labels, y_range)
    
    print("Plot visualization complete.")

if __name__ == "__main__":
    main()