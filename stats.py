import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from statsmodels.stats.contingency_tables import mcnemar

def find_training_folders(base_dir="./training") -> List[str]:
    training_folders = []
    for root, dirs, files in os.walk(base_dir):
        if "predictions.npy" in files and "labels.npy" in files:
            training_folders.append(root)
    return sorted(training_folders)

def get_user_selections(folders: List[str]) -> List[int]:
    while True:
        try:
            selection = input("\nEnter EVEN number of training folder numbers (comma-separated, e.g., 14,11,20,18,2,1): ").strip()
            indices = [int(idx.strip()) for idx in selection.split(",") if idx.strip()]
            if len(indices) >= 2 and len(indices) % 2 == 0 and all(1 <= idx <= len(folders) for idx in indices):
                return indices
            else:
                print(f"Error: Please enter an even number of valid indices between 1 and {len(folders)}")
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

def get_labels_for_trainings(selected_folders: List[str]) -> List[str]:
    labels = []
    print("\nEnter a custom label for each selected experiment (press Enter to use the folder name):")
    for folder in selected_folders:
        exp_folder = os.path.basename(os.path.dirname(folder)) if os.path.basename(folder) == "predictions" else os.path.basename(folder)
        label = input(f"Label for '{exp_folder}': ").strip()
        if not label:
            label = exp_folder
        labels.append(label)
    return labels

def run_mcnemar_test(folder1: str, folder2: str):
    preds1 = np.load(os.path.join(folder1, "predictions.npy"))
    preds2 = np.load(os.path.join(folder2, "predictions.npy"))
    labels1 = np.load(os.path.join(folder1, "labels.npy"))
    labels2 = np.load(os.path.join(folder2, "labels.npy"))
    if not np.array_equal(labels1, labels2):
        raise ValueError("Test set labels do not match between the two experiments.")
    tb = np.zeros((2, 2))
    for a, b, y in zip(preds1, preds2, labels1):
        tb[int(a == y), int(b == y)] += 1
    result = mcnemar(tb, exact=True)
    return {
        "contingency_table": tb,
        "statistic": result.statistic,
        "pvalue": result.pvalue
    }

def plot_multiple_mcnemar(pairs, pair_labels, pvalues):
    n = len(pairs)
    ncols = 2
    nrows = (n + 1) // 2  # ensures enough rows for all pairs

    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 3.5 * nrows))
    axs = np.array(axs).reshape(-1)  # flatten in case nrows=1

    for i, (table, labels, pvalue) in enumerate(zip(pairs, pair_labels, pvalues)):
        ax = axs[i]
        im = ax.imshow(table, cmap="Blues", vmin=0)
        vmax = table.max() if table.max() > 0 else 1
        for x in range(2):
            for y in range(2):
                value = int(table[x, y])
                # Choose white text for dark blue, black for light blue
                color = "white" if table[x, y] > vmax / 2 else "black"
                ax.text(y, x, value, ha="center", va="center", color=color, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"], fontsize=10)
        ax.set_yticklabels(["Incorrect", "Correct"], fontsize=10)
        ax.set_xlabel(f"{labels[1]}", fontsize=10, wrap=True)
        ax.set_ylabel(f"{labels[0]}", fontsize=10, wrap=True)
        ax.set_title(f"Pair {i+1}\np={pvalue:.5f}", fontsize=13)
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def display_training_folders(folders: List[str]) -> None:
    print("\nAvailable Training Experiments:")
    print("=" * 80)
    for i, folder in enumerate(folders, 1):
        # Always use the folder name for clarity
        exp_folder = os.path.basename(os.path.dirname(folder)) if os.path.basename(folder) == "predictions" else os.path.basename(folder)
        print(f"{i}. {exp_folder}")
        # Only use .npy files for metrics
        preds_path = os.path.join(folder, "predictions.npy")
        labels_path = os.path.join(folder, "labels.npy")
        metrics = {}
        if os.path.exists(preds_path) and os.path.exists(labels_path):
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
            print("   - accuracy: {:.2f}%".format(metrics.get("accuracy", 0)*100))
            print("   - precision: {:.2f}%".format(metrics.get("precision", 0)*100))
            print("   - recall: {:.2f}%".format(metrics.get("recall", 0)*100))
            print("   - f1: {:.2f}%".format(metrics.get("f1", 0)*100))
        else:
            print("   - Metrics: Not available")
        print("-" * 80)

def main():
    folders = find_training_folders()
    if len(folders) < 2:
        print("Need at least two experiments with predictions to run McNemar's test.")
        return
    display_training_folders(folders)
    selected_indices = get_user_selections(folders)
    selected_folders = [folders[i-1] for i in selected_indices]
    labels = get_labels_for_trainings(selected_folders)
    pairs = []
    pair_labels = []
    pvalues = []
    for i in range(0, len(selected_folders), 2):
        result = run_mcnemar_test(selected_folders[i], selected_folders[i+1])
        pairs.append(result["contingency_table"])
        pair_labels.append([labels[i], labels[i+1]])
        pvalues.append(result["pvalue"])
        print(f"\nPair {i//2+1}: {labels[i]} vs {labels[i+1]}")
        print("Contingency Table:")
        print(result["contingency_table"])
        print(f"p-value: {result['pvalue']:.5f}")
    plot_multiple_mcnemar(pairs, pair_labels, pvalues)

if __name__ == "__main__":
    main()