import matplotlib.pyplot as plt
import numpy as np

# Encoded values from the image (rounded to one decimal)
labels = [
    "Distilbert without metadata",
    "Distilbert with metadata (early fusion)",
    "RoBERTa without metadata",
    "RoBERTa with metadata (early fusion)",
    "BERT+PTP without metadata",
    "BERT+PTP with metadata (early fusion)",
    "TF-IDF without metadata",
    "TF-IDF with metadata (early fusion)"
]

accuracy =   [93.3, 93.6, 95.0, 95.2, 94.1, 94.4, 88.6, 89.5]
precision =  [92.6, 92.7, 94.0, 94.9, 93.2, 94.0, 88.6, 89.5]
recall =     [94.2, 94.5, 96.2, 95.5, 95.1, 94.7, 88.6, 89.5]
f1_score =   [93.4, 93.6, 95.1, 95.2, 94.1, 94.4, 88.6, 89.5]

metrics = [accuracy, precision, recall, f1_score]
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Define a color for each pair (8 bars, 4 pairs)
pair_colors = [
    ('#1f77b4', '#aec7e8'),  # DistilBERT
    ('#ff7f0e', '#ffbb78'),  # RoBERTa
    ('#2ca02c', '#98df8a'),  # BERT+PTP
    ('#d62728', '#ff9896')   # TF-IDF
]

x = np.arange(len(labels))

fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
axs = axs.flatten()

# Indices of pairs: (0,1), (2,3), (4,5), (6,7)
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    # Build bar colors for this subplot
    bar_colors = []
    for p, (idx1, idx2) in enumerate(pairs):
        bar_colors.extend([pair_colors[p][0], pair_colors[p][1]])
    axs[i].bar(x, metric, color=bar_colors, alpha=0.8)
    axs[i].set_title(name, fontsize=14, fontweight='bold')
    axs[i].set_ylim([85, 98])  # Changed from 97 to 98
    axs[i].set_ylabel("Score (%)")
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    for xi, val in zip(x, metric):
        axs[i].text(xi, val + 0.1, f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    # Draw horizontal lines and annotate differences for each pair
    for p, (idx1, idx2) in enumerate(pairs):
        y1 = metric[idx1]
        y2 = metric[idx2]
        y_max = max(y1, y2)
        # Draw horizontal line between the tops of the two bars
        axs[i].plot([idx1, idx2], [y_max + 0.7, y_max + 0.7], color=pair_colors[p][0], linewidth=2)
        # Draw vertical ticks
        axs[i].plot([idx1, idx1], [y_max + 0.5, y_max + 0.7], color=pair_colors[p][0], linewidth=2)
        axs[i].plot([idx2, idx2], [y_max + 0.5, y_max + 0.7], color=pair_colors[p][1], linewidth=2)
        # Annotate the difference
        diff = y2 - y1
        axs[i].text((idx1 + idx2) / 2, y_max + 0.8, f"Δ={diff:.2f}%", ha='center', va='bottom', fontsize=10, color=pair_colors[p][0])

plt.tight_layout()
plt.show()