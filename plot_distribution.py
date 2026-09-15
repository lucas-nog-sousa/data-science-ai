import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_distribution_plot():
    np.random.seed(42)
    active_ht = np.random.lognormal(mean=2.1, sigma=0.75, size=1500)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5), dpi=300)

    # Plot Histogram and KDE
    ax = sns.histplot(active_ht, kde=True, bins=50, color="#2b5c8f", alpha=0.5, edgecolor="none")

    mean_val = np.mean(active_ht)
    median_val = np.median(active_ht)

    # Reference lines
    plt.axvline(median_val, color='#2ca02c', linestyle='--', linewidth=2.5, label=f'Median: {median_val:.2f} min (Typical Customer)')
    plt.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f} min (Distorted by Outliers)')

    # Annotations
    plt.annotate('Long Right Tail\n(Outliers / Extreme Tickets)', 
                 xy=(mean_val + 8, 30), 
                 xytext=(mean_val + 15, 60),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', color='#333333')

    plt.title('Right-Skewed Distribution of Support Handling Time', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Handling Time (Minutes)', fontsize=12)
    plt.ylabel('Ticket Frequency', fontsize=12)
    plt.xlim(0, 50)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('right_skewed_handling_time.png', dpi=300)
    print("Plot saved successfully as 'right_skewed_handling_time.png'")

if __name__ == "__main__":
    generate_distribution_plot()
