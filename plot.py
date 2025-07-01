import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def find_training_folders(base_dir="./training") -> List[str]:
    training_folders = []
    print(f"Scanning {base_dir} for training experiments...")
    for root, dirs, files in os.walk(base_dir):
        if "predictions.npy" in files and "labels.npy" in files:
            training_folders.append(root)
    return sorted(training_folders)

def load_experiment_summary(folder_path: str) -> Dict:
    # Only for display, not for metrics
    json_path = os.path.join(folder_path, "experiment_summary.json")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": "Could not load experiment summary"}

def display_training_folders(folders: List[str]) -> None:
    print("\nAvailable Training Experiments:")
    print("=" * 80)
    for i, folder in enumerate(folders, 1):
        # Show the parent folder name (experiment folder)
        exp_folder = os.path.basename(os.path.dirname(folder)) if os.path.basename(folder) == "predictions" else os.path.basename(folder)
        print(f"{i}. {exp_folder}")
        summary_path = os.path.join(os.path.dirname(folder), "experiment_summary.json") if os.path.basename(folder) == "predictions" else os.path.join(folder, "experiment_summary.json")
        metrics = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                for key in ["accuracy", "precision", "recall", "f1", "f1_score"]:
                    if key in summary:
                        metrics[key] = summary[key]
            except Exception:
                pass
        # If metrics missing, compute from predictions
        preds_path = os.path.join(folder, "predictions.npy")
        labels_path = os.path.join(folder, "labels.npy")
        if (not metrics or len(metrics) < 4) and os.path.exists(preds_path) and os.path.exists(labels_path):
            try:
                preds = np.load(preds_path)
                labels = np.load(labels_path)
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                acc = accuracy_score(labels, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
                metrics = {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            except Exception:
                pass
        if metrics:
            print("   - accuracy: {:.2f}%".format(metrics.get("accuracy", 0)*100 if metrics.get("accuracy", 0) < 1 else metrics.get("accuracy", 0)))
            print("   - precision: {:.2f}%".format(metrics.get("precision", 0)*100 if metrics.get("precision", 0) < 1 else metrics.get("precision", 0)))
            print("   - recall: {:.2f}%".format(metrics.get("recall", 0)*100 if metrics.get("recall", 0) < 1 else metrics.get("recall", 0)))
            print("   - f1: {:.2f}%".format(metrics.get("f1", metrics.get("f1_score", 0))*100 if metrics.get("f1", metrics.get("f1_score", 0)) < 1 else metrics.get("f1", metrics.get("f1_score", 0))))
        print("-" * 80)

def get_user_selections(folders: List[str]) -> List[int]:
    while True:
        try:
            selection = input("\nSelect training folders by number (comma-separated, e.g., 1,3,4): ").strip()
            indices = [int(idx.strip()) for idx in selection.split(",") if idx.strip()]
            if all(1 <= idx <= len(folders) for idx in indices):
                return indices
            else:
                print(f"Error: Please enter valid indices between 1 and {len(folders)}")
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

def get_plot_parameters(selected_folders: List[str], indices: List[int]) -> Tuple[List[str], Tuple[float, float]]:
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
        print("   Complete Summary Content:")
        for key, value in summary.items():
            if isinstance(value, (int, float)) and key in ["accuracy", "f1_score"]:
                print(f"   - {key}: {value:.2f}%")
            else:
                print(f"   - {key}: {value}")
        label = input(f"Label for this training (press Enter to use folder name): ").strip()
        if not label:
            label = os.path.basename(folder)
        labels.append(label)
    return labels, (y_min, y_max)

def get_metrics_from_folder(folder: str) -> Dict[str, float]:
    preds_path = os.path.join(folder, "predictions.npy")
    labels_path = os.path.join(folder, "labels.npy")
    if os.path.exists(preds_path) and os.path.exists(labels_path):
        preds = np.load(preds_path)
        y_true = np.load(labels_path)
        acc = accuracy_score(y_true, preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='weighted')
        return {
            "accuracy": acc,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100
        }
    else:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }

def create_multi_metric_plot(selected_folders: List[str], labels: List[str], y_range: Tuple[float, float] = (None, None)) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    values = {metric: [] for metric in metrics}

    # Get all metrics from .npy files, not from json
    for folder in selected_folders:
        folder_metrics = get_metrics_from_folder(folder)
        for metric in metrics:
            values[metric].append(folder_metrics[metric])

    x = np.arange(len(labels))
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axs = axs.flatten()
    y_min, y_max = y_range
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        axs[i].bar(x, values[metric], color=color, alpha=0.8)
        axs[i].set_title(name, fontsize=14, fontweight='bold')
        axs[i].set_ylim([y_min if y_min is not None else 91, y_max if y_max is not None else 100])
        axs[i].set_ylabel("Score (%)")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
        for xi, val in zip(x, values[metric]):
            axs[i].text(xi, val + 0.1, f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

def main():
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
    selected_folders = [training_folders[i-1] for i in selected_indices]
    labels, y_range = get_plot_parameters(selected_folders, list(range(1, len(selected_folders) + 1)))
    create_multi_metric_plot(selected_folders, labels, y_range)
    print("Plot visualization complete.")

if __name__ == "__main__":
    main()