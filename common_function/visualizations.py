import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
def visualize_loc_results(models_preds, y_test_loc, X_test_unscaled):
    actual_locations = np.array(y_test_loc)

    print("\n--- Standard Location Metrics ---")
    for name, preds in models_preds.items():
        mae = mean_absolute_error(y_test_loc, preds, multioutput="raw_values")
        mse = mean_squared_error(y_test_loc, preds, multioutput='raw_values')
        print(f"{name} -> MAE: {mae} | MSE: {mse}")

    error_distributions_loc = {}
    for name, preds in models_preds.items():
        error_distributions_loc[name] = np.sqrt(np.sum((actual_locations - np.array(preds))**2, axis=1))

    print("\n--- Location Error Statistics (Euclidean Distance) ---")
    for name, errors in error_distributions_loc.items():
        print(f"{name}:")
        print(f"  MSE:     {np.mean(errors**2)}")
        print(f"  Mean:    {np.mean(errors)}")
        print(f"  Median:  {np.median(errors)}")
        print(f"  Minimum: {np.min(errors)}")
        print(f"  Maximum: {np.max(errors)}\n")

    plt.figure(figsize=(10, 6))
    data_to_plot = list(error_distributions_loc.values())
    labels = list(error_distributions_loc.keys())
    box = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, medianprops=dict(color='red', linewidth=2))
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'thistle']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel('Location Error (Distance)')
    plt.title('Comparison of Location Estimation Errors Across Models')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n--- Generating Individual Model Error Distributions ---")
    for name, preds in models_preds.items():
        preds_arr = np.array(preds)
        err_x = actual_locations[:, 0] - preds_arr[:, 0]
        err_y = actual_locations[:, 1] - preds_arr[:, 1]
        abs_err_x = np.abs(err_x)
        abs_err_y = np.abs(err_y)
        sq_err_x = err_x ** 2
        sq_err_y = err_y ** 2
        euclidean_dist = np.sqrt(sq_err_x + sq_err_y)
        squared_dist = euclidean_dist ** 2

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Detailed Error Distribution: {name}", fontsize=16, fontweight='bold')
        sns.histplot(euclidean_dist, kde=True, ax=axes[0, 0], color='royalblue', bins=50)
        axes[0, 0].set_title('Overall Distance Error Frequency')
        axes[0, 0].set_xlabel('Euclidean Distance Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        sns.histplot(abs_err_x, kde=True, ax=axes[0, 1], color='mediumseagreen', alpha=0.5, label='X Abs Error', bins=50)
        sns.histplot(abs_err_y, kde=True, ax=axes[0, 1], color='coral', alpha=0.5, label='Y Abs Error', bins=50)
        axes[0, 1].set_title('Absolute Coordinate Errors (MAE View)')
        axes[0, 1].set_xlabel('Absolute Error (m)')
        axes[0, 1].legend()
        sns.histplot(err_x, kde=True, ax=axes[1, 0], color='mediumpurple', alpha=0.5, label='X Error', bins=50)
        sns.histplot(err_y, kde=True, ax=axes[1, 0], color='gold', alpha=0.5, label='Y Error', bins=50)
        axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1.5)
        axes[1, 0].set_title('Directional Errors (Actual - Predicted)')
        axes[1, 0].set_xlabel('Directional Error (m)')
        axes[1, 0].legend()
        sns.histplot(squared_dist, kde=True, ax=axes[1, 1], color='firebrick', bins=50)
        axes[1, 1].set_title('Squared Distance Error (MSE View)')
        axes[1, 1].set_xlabel('Squared Distance (m²)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    print("\n--- Generating Detector-Specific Error vs True Distance Plots ---")
    sx_true = np.array(y_test_loc['source_x'])
    sy_true = np.array(y_test_loc['source_y'])
    d1x, d1y = np.array(X_test_unscaled['det1_x']), np.array(X_test_unscaled['det1_y'])
    d2x, d2y = np.array(X_test_unscaled['det2_x']), np.array(X_test_unscaled['det2_y'])
    d3x, d3y = np.array(X_test_unscaled['det3_x']), np.array(X_test_unscaled['det3_y'])
    dist_true_1 = np.sqrt((d1x - sx_true)**2 + (d1y - sy_true)**2)
    dist_true_2 = np.sqrt((d2x - sx_true)**2 + (d2y - sy_true)**2)
    dist_true_3 = np.sqrt((d3x - sx_true)**2 + (d3y - sy_true)**2)
    all_true_distances = np.concatenate([dist_true_1, dist_true_2, dist_true_3])

    max_d = int(np.ceil(np.max(all_true_distances)))
    bins = np.arange(0, max_d + 2, 2)
    labels = [f"{bins[i]}-{bins[i+1]}m" for i in range(len(bins)-1)]
    binned_dist = pd.cut(all_true_distances, bins=bins, labels=labels, include_lowest=True)

    for name, preds in models_preds.items():
        preds_arr = np.array(preds)
        px, py = preds_arr[:, 0], preds_arr[:, 1]
        dist_pred_1 = np.sqrt((d1x - px)**2 + (d1y - py)**2)
        dist_pred_2 = np.sqrt((d2x - px)**2 + (d2y - py)**2)
        dist_pred_3 = np.sqrt((d3x - px)**2 + (d3y - py)**2)
        all_pred_distances = np.concatenate([dist_pred_1, dist_pred_2, dist_pred_3])
        err_abs = np.abs(all_true_distances - all_pred_distances)
        err_sq = (all_true_distances - all_pred_distances)**2

        metrics_df = pd.DataFrame({'Bin': binned_dist, 'MAE_Detector_Dist': err_abs, 'MSE_Detector_Dist': err_sq})
        grouped = metrics_df.groupby('Bin', observed=False)
        bin_means = grouped.mean().reset_index()
        bin_stds = grouped.std().reset_index()

        valid_mask = bin_means['MAE_Detector_Dist'].notna()
        bin_means = bin_means[valid_mask].reset_index(drop=True)
        bin_stds = bin_stds[valid_mask].reset_index(drop=True)
        bin_stds['MAE_Detector_Dist'] = bin_stds['MAE_Detector_Dist'].fillna(0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Detector-Specific Error vs Distance to Source: {name}", fontsize=14, fontweight='bold')
        axes[0].errorbar(bin_means['Bin'], bin_means['MAE_Detector_Dist'],
                             yerr=bin_stds['MAE_Detector_Dist']/np.sqrt(len(metrics_df)),
                             marker='o', color='royalblue', lw=2, capsize=4, label='Mean Absolute Error')
        axes[0].set_title('MAE (|True Dist - Pred Dist|) vs Distance')
        axes[0].set_xlabel('True Distance between Source & Detector')
        axes[0].set_ylabel('Mean Absolute Error (m)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        axes[1].plot(bin_means['Bin'], bin_means['MSE_Detector_Dist'], marker='s', color='firebrick', lw=2, label='Mean Squared Error')
        axes[1].set_title('MSE ((True Dist - Pred Dist)²) vs Distance')
        axes[1].set_xlabel('True Distance between Source & Detector')
        axes[1].set_ylabel('Mean Squared Error (m²)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    print("\n--- Generating Location Error vs Source Distance Plots ---")
    dist_1 = np.sqrt((d1x - sx_true)**2 + (d1y - sy_true)**2)
    dist_2 = np.sqrt((d2x - sx_true)**2 + (d2y - sy_true)**2)
    dist_3 = np.sqrt((d3x - sx_true)**2 + (d3y - sy_true)**2)
    avg_true_distance = (dist_1 + dist_2 + dist_3) / 3.0

    max_d = int(np.ceil(np.max(avg_true_distance)))
    bins = np.arange(0, max_d + 2, 2)
    labels = [f"{bins[i]}-{bins[i+1]}m" for i in range(len(bins)-1)]
    binned_dist = pd.cut(avg_true_distance, bins=bins, labels=labels, include_lowest=True)

    for name, preds in models_preds.items():
        preds_arr = np.array(preds)
        px, py = preds_arr[:, 0], preds_arr[:, 1]
        loc_error = np.sqrt((sx_true - px)**2 + (sy_true - py)**2)

        metrics_df = pd.DataFrame({'Bin': binned_dist, 'Loc_Error': loc_error})
        grouped = metrics_df.groupby('Bin', observed=False)
        bin_means = grouped.mean().reset_index()
        bin_stds = grouped.std().reset_index()
        bin_counts_full = grouped.count().reset_index()

        valid_mask = bin_means['Loc_Error'].notna()
        bin_means = bin_means[valid_mask].reset_index(drop=True)
        bin_stds = bin_stds[valid_mask].reset_index(drop=True)
        bin_stds['Loc_Error'] = bin_stds['Loc_Error'].fillna(0)
        bin_counts = bin_counts_full[valid_mask].reset_index(drop=True)['Loc_Error']

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Location Error vs Distance to Sensors: {name}", fontsize=14, fontweight='bold')
        ax.errorbar(bin_means['Bin'], bin_means['Loc_Error'],
                         yerr=bin_stds['Loc_Error'] / np.sqrt(bin_counts),
                         marker='o', color='mediumpurple', lw=2, capsize=4, label='Mean Location Error')
        ax.set_xlabel('Average Distance from Source to Detectors (m)')
        ax.set_ylabel('Location Error (m)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()
def visualize_int_results(models_preds, y_test_int, y_test_loc, X_test_unscaled, SI_min=400.0, SI_max=8000.0):
    def calculate_nrmse(actuals, predictions, si_min, si_max):
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"  MSE: {mean_squared_error(actuals, predictions)} µCi*µCi")
        nrmse = rmse / (si_max - si_min)
        return nrmse

    print("\n--- Intensity Prediction Metrics ---")
    for name, preds in models_preds.items():
        mae = mean_absolute_error(y_test_int, preds)
        print(f"{name}:")
        nrmse = calculate_nrmse(y_test_int, preds, SI_min, SI_max)
        print(f"  MAE:   {mae} µCi")
        print(f"  NRMSE: {nrmse}\n")

    actual_intensities = np.array(y_test_int)
    error_distributions = {name: np.abs(actual_intensities - preds) for name, preds in models_preds.items()}

    plt.figure(figsize=(10, 6))
    data_to_plot = list(error_distributions.values())
    labels = list(error_distributions.keys())
    box = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, medianprops=dict(color='red', linewidth=2))
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'thistle']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel('Absolute Intensity Error (µCi)')
    plt.title('Comparison of Source Intensity Estimation Errors')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n--- Generating Individual Intensity Error Distributions ---")
    for name, preds_arr in models_preds.items():
        err = actual_intensities - preds_arr
        abs_err = np.abs(err)
        sq_err = err ** 2
        per_err = (abs_err/actual_intensities)*100

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f"Detailed Error Distribution: {name}", fontsize=16, fontweight='bold')
        sns.histplot(err, kde=True, ax=axes[0, 0], color='mediumpurple', bins=50)
        axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=1.5)
        axes[0, 0].set_title('Directional Error (Checking for Bias)')
        axes[0, 0].set_xlabel('Error (Actual - Predicted) [µCi]')
        axes[0, 0].set_ylabel('Frequency')
        sns.histplot(abs_err, kde=True, ax=axes[0, 1], color='mediumseagreen', bins=50)
        axes[0, 1].set_title('Absolute Error Frequency (MAE View)')
        axes[0, 1].set_xlabel('Absolute Error [µCi]')
        axes[1, 0].scatter(actual_intensities, preds_arr, alpha=0.3, color='royalblue', s=10)
        min_val = min(np.min(actual_intensities), np.min(preds_arr))
        max_val = max(np.max(actual_intensities), np.max(preds_arr))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
        axes[1, 0].set_title('Actual vs. Predicted Intensity')
        axes[1, 0].set_xlabel('Actual Intensity [µCi]')
        axes[1, 0].set_ylabel('Predicted Intensity [µCi]')
        axes[1, 0].legend()
        sns.histplot(sq_err, kde=True, ax=axes[1, 1], color='firebrick', bins=50)
        axes[1, 1].set_title('Squared Error Frequency (MSE View)')
        axes[1, 1].set_xlabel('Squared Error [µCi²]')
        sns.histplot(per_err, kde=True, ax=axes[2, 0], color='firebrick', binwidth=5)
        axes[2, 0].set_title('percentage error Frequency (MSE View)')
        axes[2, 0].set_xlabel('percentage Error [%]')
        axes[2, 0].set_ylabel('Frequency')
        axes[2,0].set_xlim(0, 100)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    print("\n--- Generating Detector-Specific Error vs True Distance Plots ---")
    sx_true = np.array(y_test_loc['source_x'])
    sy_true = np.array(y_test_loc['source_y'])
    d1x, d1y = np.array(X_test_unscaled['det1_x']), np.array(X_test_unscaled['det1_y'])
    d2x, d2y = np.array(X_test_unscaled['det2_x']), np.array(X_test_unscaled['det2_y'])
    d3x, d3y = np.array(X_test_unscaled['det3_x']), np.array(X_test_unscaled['det3_y'])
    dist_true_1 = np.sqrt((d1x - sx_true)**2 + (d1y - sy_true)**2)
    dist_true_2 = np.sqrt((d2x - sx_true)**2 + (d2y - sy_true)**2)
    dist_true_3 = np.sqrt((d3x - sx_true)**2 + (d3y - sy_true)**2)

    all_true_distances = np.concatenate([dist_true_1, dist_true_2, dist_true_3])

    max_d = np.max(all_true_distances)
    bins = np.arange(0, max_d + 2,2)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}m" for i in range(len(bins)-1)]

    binned_dist = pd.cut(all_true_distances, bins=bins, labels=labels, include_lowest=True)

    for name, preds in models_preds.items():
        preds_arr = np.array(preds)
        err_abs = np.abs(actual_intensities-preds_arr)
        err_sq = (actual_intensities-preds_arr)**2
        all_err_abs = np.concatenate([err_abs, err_abs, err_abs])
        all_err_sq = np.concatenate([err_sq, err_sq, err_sq])

        metrics_df = pd.DataFrame({'Bin': binned_dist, 'MAE_Detector_Intensity': all_err_abs, 'MSE_Detector_Intensity': all_err_sq})
        grouped = metrics_df.groupby('Bin', observed=False)
        bin_means = grouped.mean().reset_index()
        bin_stds = grouped.std().reset_index()
        bin_counts_full = grouped.count().reset_index()

        valid_mask = bin_means['MAE_Detector_Intensity'].notna()
        bin_means = bin_means[valid_mask].reset_index(drop=True)
        bin_stds = bin_stds[valid_mask].reset_index(drop=True)
        bin_stds['MAE_Detector_Intensity'] = bin_stds['MAE_Detector_Intensity'].fillna(0)
        bin_counts = bin_counts_full[valid_mask].reset_index(drop=True)['MAE_Detector_Intensity']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Detector-Specific Error vs Distance to Source: {name}", fontsize=14, fontweight='bold')
        axes[0].errorbar(bin_means['Bin'].astype(str), bin_means['MAE_Detector_Intensity'],
                             yerr=bin_stds['MAE_Detector_Intensity']/np.sqrt(bin_counts),
                             marker='o', color='royalblue', lw=2, capsize=4, label='Mean Absolute Error')
        axes[0].set_title('MAE (|True Intensity - Pred Intensity|) vs Distance')
        axes[0].set_xlabel('True Distance between Source & Detector')
        axes[0].set_ylabel('Mean Absolute Error (µCi)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        axes[1].plot(bin_means['Bin'].astype(str), bin_means['MSE_Detector_Intensity'], marker='s', color='firebrick', lw=2, label='Mean Squared Error')
        axes[1].set_title('MSE ((True Intensity - Pred Intensity)²) vs Distance')
        axes[1].set_xlabel('True Distance between Source & Detector')
        axes[1].set_ylabel('Mean Squared Error (µCi²)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    print("\n--- Generating Intensity Error vs Source Distance Plots ---")
    dist_1 = np.sqrt((d1x - sx_true)**2 + (d1y - sy_true)**2)
    dist_2 = np.sqrt((d2x - sx_true)**2 + (d2y - sy_true)**2)
    dist_3 = np.sqrt((d3x - sx_true)**2 + (d3y - sy_true)**2)

    avg_true_distance = (dist_1 + dist_2 + dist_3) / 3.0
    max_d_avg = np.max(avg_true_distance)
    step_avg = max(1.0, max_d_avg / 15.0)
    bins_avg = np.arange(0, max_d_avg + step_avg, step_avg)
    labels_avg = [f"{bins_avg[i]:.1f}-{bins_avg[i+1]:.1f}m" for i in range(len(bins_avg)-1)]

    binned_dist_avg = pd.cut(avg_true_distance, bins=bins_avg, labels=labels_avg, include_lowest=True)

    for name, preds in models_preds.items():
        preds_arr = np.array(preds)
        err_abs = np.abs(actual_intensities - preds_arr)
        err_sq = (actual_intensities - preds_arr)**2

        metrics_df = pd.DataFrame({'Bin': binned_dist_avg, 'MAE_Intensity': err_abs, 'MSE_Intensity': err_sq})
        grouped = metrics_df.groupby('Bin', observed=False)
        bin_means = grouped.mean().reset_index()
        bin_stds = grouped.std().reset_index()
        bin_counts_full = grouped.count().reset_index()

        valid_mask = bin_means['MAE_Intensity'].notna()
        bin_means = bin_means[valid_mask].reset_index(drop=True)
        bin_stds = bin_stds[valid_mask].reset_index(drop=True)
        bin_stds['MAE_Intensity'] = bin_stds['MAE_Intensity'].fillna(0)
        bin_counts = bin_counts_full[valid_mask].reset_index(drop=True)['MAE_Intensity']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Intensity Error vs Distance to Sensors: {name}", fontsize=14, fontweight='bold')
        axes[0].errorbar(bin_means['Bin'].astype(str), bin_means['MAE_Intensity'],
                         yerr=bin_stds['MAE_Intensity'] / np.sqrt(bin_counts),
                         marker='o', color='royalblue', lw=2, capsize=4, label='Mean Absolute Error')
        axes[0].set_xlabel('Average Distance from Source to Detectors (m)')
        axes[0].set_ylabel('Intensity MAE (µCi)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        axes[1].plot(bin_means['Bin'].astype(str), bin_means['MSE_Intensity'], marker='s', color='firebrick', lw=2, label='Mean Squared Error')
        axes[1].set_xlabel('Average Distance from Source to Detectors (m)')
        axes[1].set_ylabel('Intensity MSE (µCi²)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        plt.show()