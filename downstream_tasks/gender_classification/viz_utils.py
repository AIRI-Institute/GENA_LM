import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_error_bar(sample_ids_labels, Ns, accs, stds, X_probs, Y_probs, species_names, chrX_ratio, chrY_ratio):
    # Multi-row title for the figure
    fig = plt.figure(figsize=(15, 15))  # Larger figure
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.5) 
    fig.suptitle(
        f"Analysis of Sampled Species: {", ".join(species_names)}\n"
        f"Sampling chromosome ratio of X: {chrX_ratio}\n"
        f"Sampling chromosome ratio of Y: {chrY_ratio}\n",
        fontsize=16,
        fontweight='bold',
        y=0.95,  # Adjust the vertical position of the title
        ha='center'
    )


    # Top plot: Errorbar plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(Ns, accs, yerr=stds, marker='o', linestyle='-', color='b', capsize=5)
    ax1.hlines(np.mean(list(sample_ids_labels.values())), xmin=min(Ns), xmax=max(Ns), color='black', linestyle='dotted')
    ax1.set_title(f'Accuracy vs. Number of sampled subsequences (Nx16x3072bp)')
    ax1.set_xlabel('Number of sampled subsequences Nx16x3072bp')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.set_xscale('log')  # Optional: Use logarithmic scale for better visualization if needed
    ax1.set_xticks(Ns)
    ax1.set_xticklabels([str(n) for n in Ns]) # Ensure that all N values are labeled

    # Calculate dynamic bar widths proportional to differences in log-scaled x values
    bar_widths = [0.2*10**np.log10(elem) for i, elem in enumerate(Ns)]
    # Plot with dynamic bar widths
    # Bottom left bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(Ns, X_probs, width=bar_widths, color='orange', alpha=0.8, align='center')
    ax2.set_title('Probabilities of X chromosome')
    ax2.set_xlabel('Number of sampled subsequences Nx16x3072bp')
    ax2.set_ylabel('Probability of X chromosome')
    ax2.set_xscale('log')
    ax2.set_xticks(Ns)
    # ax2.set_xticklabels([str(n) for n in Ns])
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Bottom right bar chart
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(Ns, Y_probs, width=bar_widths, color='green', alpha=0.8, align='center')
    ax3.set_title('Probabilities of Y chromosome')
    ax3.set_xlabel('Number of sampled subsequences Nx16x3072bp')
    ax3.set_ylabel('Probability of Y chromosome')
    ax3.set_xscale('log')
    ax3.set_xticks(Ns)
    ax3.set_xticklabels([str(n) for n in Ns])
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_violins(sample_ids_probs, sample_ids_labels, species_names, chrX_ratio, chrY_ratio):
    # Prepare DataFrame for plotting
    data_list = []
    for sample_id, probs in sample_ids_probs.items():
        # if sample_id in mouses_strains:
        # if sample_id in human_ids:

            for prob in probs:
                data_list.append({
                    'Sample ID': sample_id,  # Convert to string for consistent key usage
                    'Probability': prob[0],
                    'Label': sample_ids_labels[sample_id]
                })

    df = pd.DataFrame(data_list)

    # Sort data by label for plotting
    label_order = sorted(df['Label'].unique())
    sample_order = df.sort_values(by='Label')['Sample ID'].unique()

    # Plotting
    plt.figure(figsize=(20, 10))  # Adjusted size for better visibility
    plt.suptitle(
        f"Analysis of Sampled Species: {", ".join(species_names)}\n"
        f"Sampling chromosome ratio of X: {chrX_ratio}\n"
        f"Sampling chromosome ratio of Y: {chrY_ratio}\n",
        fontsize=16,
        fontweight='bold',
        y=0.95,  # Adjust the vertical position of the title
        ha='center'
    )

    plt.grid()
    ax = sns.violinplot(x='Sample ID', y='Probability', hue='Label', data=df, 
                        inner='box',
                        order=sample_order,
                        hue_order=label_order
                        )              
    # sns.stripplot(x='Sample ID', y='Probability', data=df, color='black', size=1, jitter=True, order=sample_order)

    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2)  # Add horizontal line at y=0.5

    plt.title('Sex: 0 - M, 1 - F')
    plt.xticks(rotation=90)
    plt.xlabel('Sample ID')
    plt.ylabel('Probability')
    plt.legend(title='Label')
    # plt.tight_layout()
    plt.show()