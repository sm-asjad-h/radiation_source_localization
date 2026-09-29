import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
def detectordistancebins(df):
    """
    Plots the frequency of source-to-detector distances.
    Shows the overall distribution and the breakdown by sorted detector.
    """
    # 1. Calculate the true distance for all three detectors across the entire dataframe
    dist1 = np.sqrt((df['det1_x'] - df['source_x'])**2 + (df['det1_y'] - df['source_y'])**2)
    dist2 = np.sqrt((df['det2_x'] - df['source_x'])**2 + (df['det2_y'] - df['source_y'])**2)
    dist3 = np.sqrt((df['det3_x'] - df['source_x'])**2 + (df['det3_y'] - df['source_y'])**2)

    # Combine them for the overall view
    all_distances = pd.concat([dist1, dist2, dist3])

    # 2. Set up the plotting grid (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Frequency vs. True Distance to Source', fontsize=18, fontweight='bold')

    # --- Plot A: The Overall Distance Distribution ---
    # Using a binwidth of 0.5 meters for clean, readable bars
    sns.histplot(all_distances, binwidth=0.5, ax=axes[0], color='mediumpurple', edgecolor='black', alpha=0.8)
    axes[0].set_title('Overall Distance Distribution (All Detectors Combined)', fontsize=14)
    axes[0].set_xlabel('Distance from Source to Detector (m)', fontsize=12)
    axes[0].set_ylabel('Frequency (Number of Occurrences)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot B: Breakdown by Sorted Detector ---
    # This will prove that your intensity sorting worked geometrically!
    sns.histplot(dist1, binwidth=0.5, ax=axes[1], color='royalblue', label='DET1 (Highest Reading)', alpha=0.5, edgecolor='black')
    sns.histplot(dist2, binwidth=0.5, ax=axes[1], color='forestgreen', label='DET2 (Middle Reading)', alpha=0.5, edgecolor='black')
    sns.histplot(dist3, binwidth=0.5, ax=axes[1], color='firebrick', label='DET3 (Lowest Reading)', alpha=0.5, edgecolor='black')

    axes[1].set_title('Distance Distribution Breakdown by Sorted Detector', fontsize=14)
    axes[1].set_xlabel('Distance from Source to Detector (m)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("/home/lus04/trainee4/radiation_source_localization/v1.6_poissondata/dataset/dataset.png", bbox_inches='tight')
    plt.close(fig)

    # 3. Print hard statistics
    print("\n--- Geometric Distance Statistics ---")
    print(f"Absolute Minimum Distance: {all_distances.min():.2f} m")
    print(f"Absolute Maximum Distance: {all_distances.max():.2f} m")
    print(f"Average Distance across all sensors: {all_distances.mean():.2f} m")
def visualize_locations(df):
  plt.figure()
  sample = df.iloc[2500]
  plt.scatter(sample['source_x'], sample['source_y'],s=5, c='red', marker='*')
  plt.scatter([sample['det1_x'], sample['det2_x'], sample['det3_x']],
                  [sample['det1_y'], sample['det2_y'], sample['det3_y']],s=5, c='blue')
  plt.title("Constraint Check: Spatial Node Placement (100 Sample Sets)")
  plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.legend(); plt.grid(True, alpha=0.3)
  plt.show()
def visualize_sample_geometries(df, num_samples_to_plot=5, area_size=18):
    """
    Plots the physical 2D layout (source and 3 detectors) for a random subset of samples.
    Each sample gets a unique color. Source = Star, Detectors = Dots.
    """
    plt.figure(figsize=(10, 10))
    plt.title(f'Spatial Geometry of {num_samples_to_plot} Random Samples', fontsize=16, fontweight='bold')

    # Use a distinct categorical colormap to ensure each sample looks different
    cmap = plt.get_cmap('tab10')

    # Randomly select a few rows from the dataframe so it's readable
    sample_indices = np.random.choice(df.index, size=num_samples_to_plot, replace=False)

    for idx, i in enumerate(sample_indices):
        row = df.loc[i]
        color = cmap(idx % 7) # Cycle through distinct colors

        # Extract coordinates
        sx, sy = row['source_x'], row['source_y']
        d1x, d1y = row['det1_x'], row['det1_y']
        d2x, d2y = row['det2_x'], row['det2_y']
        d3x, d3y = row['det3_x'], row['det3_y']

        # 1. Plot Source (Large Star)
        plt.scatter(sx, sy, color=color, marker='*', s=400, edgecolor='black',
                    label=f'Sample {i}')

        # 2. Plot Detectors (Dots)
        plt.scatter([d1x, d2x, d3x], [d1y, d2y, d3y], color=color, marker='o', s=100, edgecolor='black')

        # 3. Draw the "Detector Net" (Triangle)
        # This gives you visual proof of your >10m^2 area constraint!
        det_xs = [d1x, d2x, d3x, d1x]
        det_ys = [d1y, d2y, d3y, d1y]
        plt.plot(det_xs, det_ys, color=color, linestyle='--', alpha=0.5, linewidth=2)

        # 4. Draw faint lines from the source to the detectors
        # This helps visualize the "radial distance" logic you just built
        plt.plot([sx, d1x], [sy, d1y], color=color, linestyle=':', alpha=0.3)
        plt.plot([sx, d2x], [sy, d2y], color=color, linestyle=':', alpha=0.3)
        plt.plot([sx, d3x], [sy, d3y], color=color, linestyle=':', alpha=0.3)

    # Lock the axes to your exact room size
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)

    # Formatting
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.3)

    # Put the legend outside the plot so it doesn't cover your data
    plt.legend(title="Random Samples", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()
    
df=pd.read_csv("/home/lus04/trainee4/radiation_source_localization/v1.6_poissondata/dataset/dataset.csv")
detectordistancebins(df)