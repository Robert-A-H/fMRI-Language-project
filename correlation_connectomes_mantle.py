import os
import numpy as np
import pandas as pd
import nibabel as nib
import ants
import json

from nilearn import image, plotting, datasets, connectome
from nilearn.maskers import NiftiLabelsMasker
from skbio.stats.distance import mantel
from scipy.spatial.distance import squareform


def process_roi_time_series_and_plot(fmri_img_path, roi_folder, anat_img_path, output_dir_timeseries, output_dir_plots, tsv_file_path=None):

    # Create output directories if they do not exist
    os.makedirs(output_dir_timeseries, exist_ok=True)
    os.makedirs(output_dir_plots, exist_ok=True)

    # List all ROI files in the folder and combine them to create a label image
    roi_files = [os.path.join(roi_folder, f) for f in os.listdir(roi_folder) if
                 f.endswith('.nii') or f.endswith('.nii.gz')]

    # Load fMRI data and anatomical image
    fmri_img = image.load_img(fmri_img_path)
    anat_img = image.load_img(anat_img_path)

    # Initialize an empty list for ROIs and a mask for labels
    label_data = None
    label_count = 1
    labels_dict = {}
    roi_names = []

    for roi_file in roi_files:
        # Load the ROI mask
        roi_mask = image.load_img(roi_file)

        # Resample ROI to match the resolution of the fMRI data
        resampled_roi_mask = image.resample_to_img(roi_mask, fmri_img, interpolation='nearest')

        # Extract the filename without extension
        roi_name = os.path.basename(roi_file).split('.')[0]

        # Create a mask with a unique label for each ROI
        if label_data is None:
            label_data = resampled_roi_mask.get_fdata() * label_count
        else:
            label_data += (resampled_roi_mask.get_fdata() * label_count)

        # Store label names in a dictionary
        labels_dict[label_count] = roi_name
        roi_names.append(roi_name)
        label_count += 1

    # Create a label image from the masks
    labels_img = image.new_img_like(fmri_img, label_data)

    # Generate filename for the label image based on the output directory
    labels_img_filename = os.path.join(output_dir_timeseries, 'labels_img.nii.gz')

    # Save the label image
    labels_img.to_filename(labels_img_filename)
    print(f"Labels image saved as {labels_img_filename}")

    # Save the mapping between labels and names in a JSON file
    labels_json_filename = os.path.join(output_dir_timeseries, 'labels_info.json')
    with open(labels_json_filename, 'w') as json_file:
        json.dump(labels_dict, json_file, indent=4)
    print(f"Labels information saved as {labels_json_filename}")

    # Define the masker to extract and standardize time series from the ROIs in labels_img
    masker = NiftiLabelsMasker(labels_img=labels_img, standardize=False)

    if tsv_file_path is None:
        roi_time_series = masker.fit_transform(fmri_img)
    else:
        confounds_df = pd.read_csv(tsv_file_path, sep='\t')

        # List of required confounds
        required_confounds = [
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
            "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
            "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
            "global_signal"
        ]

        # Extract the required confounds
        confound_dataframe = confounds_df[required_confounds]

        # Check for NaN or inf values and replace them (e.g., with 0 or linear interpolation)
        confound_dataframe = confound_dataframe.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
        confound_dataframe = confound_dataframe.fillna(method='ffill').fillna(method='bfill')  # Fill NaNs using forward and backward fill

        # Alternatively, replace all NaNs with the mean value:
        # confound_dataframe = confound_dataframe.fillna(confound_dataframe.mean())
        roi_time_series = masker.fit_transform(fmri_img, confounds=confound_dataframe)

    # Save time series in a DataFrame
    roi_time_series_df = pd.DataFrame(roi_time_series, columns=[f"ROI_{label}" for label in roi_names])

    # Save time series as a CSV file with a sequential index in output_dir_timeseries
    output_csv = os.path.join(output_dir_timeseries, 'roi_time_series.csv')
    roi_time_series_df.to_csv(output_csv, index=False)
    print(f"Time series saved in {output_csv}")

    return roi_time_series_df, labels_img_filename, roi_names  # Return the path to the label image



def compute_correlation_matrix_thresholded(roi_time_series, threshold=0):
    # Compute the correlation matrix
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation', standardize=False)
    correlation_matrix = correlation_measure.fit_transform([roi_time_series])[0]

    thresholded_matrix = np.copy(correlation_matrix)

    # Set values below the threshold to 0
    thresholded_matrix[np.abs(thresholded_matrix) < threshold] = 0

    thresholded_matrix = thresholded_matrix.squeeze()

    return thresholded_matrix



def plot_correlation_matrix(correlation_matrix, output_dir_timeseries, labels=None, matrix_threshold='0'):
    # Define a larger font size for ROI labels
    label_fontsize = 23  # Adjust as needed

    # Plot the correlation matrix with Nilearn using 'reorder'
    fig, ax = plt.subplots(figsize=(20, 20))
    display = plotting.plot_matrix(correlation_matrix, figure=fig,
                                   labels=labels, vmax=1, vmin=-1, reorder='average')

    # Rotate labels and adjust font size
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)  # X-axis labels vertical with larger font
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, rotation=0, fontsize=label_fontsize)  # Y-axis labels horizontal with larger font

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout(rect=[0, 0, 1, 1])  # Ensure tight layout
    fig.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)  # Add padding for labels

    # Save the figure
    fig.savefig(
        os.path.join(output_dir_timeseries, f'roi_correlation_matrix_threshold_{matrix_threshold}_averaged.png'),
        dpi=400, bbox_inches='tight')

    # Plot the correlation matrix with Nilearn without 'reorder'
    fig, ax = plt.subplots(figsize=(20, 20))
    display = plotting.plot_matrix(correlation_matrix, figure=fig,
                                   labels=labels, vmax=1, vmin=-1)

    # Rotate labels and adjust font size
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, rotation=0, fontsize=label_fontsize)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)

    # Save the figure
    fig.savefig(os.path.join(output_dir_timeseries, f'roi_correlation_matrix_threshold_{matrix_threshold}.png'),
                dpi=400, bbox_inches='tight')
    plt.show()



def plot_correlation_matrix_difference(correlation_matrix_01, correlation_matrix_02, output_dir_timeseries, labels=None, matrix_threshold='0'):
    # Plot the correlation matrix difference with Nilearn
    fig = plt.figure(figsize=(20, 20))

    correlation_matrix = np.abs(correlation_matrix_01 - correlation_matrix_02)

    plotting.plot_matrix(correlation_matrix, figure=fig,
                         labels=labels, vmax=1, vmin=0, cmap='Blues', reorder='average')
    # Save the figure
    fig.savefig(os.path.join(output_dir_timeseries, f'roi_difference_matrix_threshold_{matrix_threshold}_averaged.png'), dpi=300)
    plt.show()



def plot_connectome(matrix, labels_img_path, output_dir_timeseries, matrix_type="correlation", threshold="70%", transform_path=None):
    """
    Plot the connectome for a given matrix (correlation or inverse covariance) on the MNI152 template, ensuring the coordinates are correctly aligned.

    :param matrix: The connectivity matrix to be plotted (either correlation or inverse covariance matrix).
    :param labels_img_path: Path to the labels image in native space.
    :param output_dir_timeseries: Directory where the output image file will be saved.
    :param matrix_type: Type of the matrix being plotted ("correlation" or "inverse_covariance").
    :param threshold: Edge threshold for plotting, can be a percentage (e.g., "80%") or a float value.
    :param transform_path: Path to the transformation matrix from native to MNI space.
    """

    # Path to the MNI152 template
    mni_template = datasets.load_mni152_template()
    mni_template_path = os.path.join(output_dir_timeseries, 'mni_template.nii.gz')

    # Save the MNI template as a Nifti file
    nib.save(mni_template, mni_template_path)

    # Define the path where the transformed labels image will be saved
    transformed_labels_output_path = os.path.join(output_dir_timeseries, 'transformed_labels_image.nii.gz')

    # Apply transformation of the labels image to MNI space
    transformed_labels_img_path = transform_labels_to_mni(labels_img_path, transform_path, mni_template_path, transformed_labels_output_path)

    # Load the transformed labels image in MNI space
    transformed_labels_img = image.load_img(transformed_labels_img_path)

    # Extract the coordinates of the centers of ROIs in MNI space
    mni_coords = plotting.find_parcellation_cut_coords(transformed_labels_img)

    # Plot the connectome on the standard MNI152 brain
    display = plotting.plot_glass_brain(None, display_mode='ortho')
    display.add_graph(matrix, mni_coords, edge_threshold=threshold)

    # Set the filename based on the matrix type
    filename = f'roi_{matrix_type}_connectome.png'

    # Save the figure as an image file
    output_path = os.path.join(output_dir_timeseries, filename)
    display.savefig(output_path, dpi=300)

    print(f"Connectome {matrix_type} saved as: {output_path}")



def prepare_for_mantel_test(correlation_matrix):
    """
    Prepares the correlation matrix for the Mantel test by removing self-connections
    and ensuring the matrix is symmetric.

    :param correlation_matrix: The input correlation matrix.
    :return: Symmetric distance matrix.
    """
    if correlation_matrix is None or not isinstance(correlation_matrix, np.ndarray):
        raise ValueError("Invalid input: correlation matrix must be a numpy array.")

    np.fill_diagonal(correlation_matrix, 0)  # Remove self-connections

    # Ensure symmetry only if needed
    if not np.allclose(correlation_matrix, correlation_matrix.T):
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    return 1 - correlation_matrix



def mantel_test(correlation_matrix_1, correlation_matrix_2, matrix_threshold=0):
    """
    Conducts the Mantel test to compare the similarity between two correlation matrices.

    :param correlation_matrix_1: First correlation matrix.
    :param correlation_matrix_2: Second correlation matrix.
    :param matrix_threshold: The threshold applied when calculating the matrix.
    :return: Mantel test correlation coefficient and p-value.
    """
    if correlation_matrix_1 is None or correlation_matrix_2 is None:
        raise ValueError("One of the input correlation matrices is None.")

    # Prepare matrices
    matrix_1 = prepare_for_mantel_test(correlation_matrix_1)
    matrix_2 = prepare_for_mantel_test(correlation_matrix_2)

    # Convert matrices to distance vectors
    distance_matrix_1 = squareform(matrix_1, checks=False).astype(np.double)
    distance_matrix_2 = squareform(matrix_2, checks=False).astype(np.double)

    # Conduct Mantel test
    mantel_r, mantel_p, _ = mantel(distance_matrix_1, distance_matrix_2, method='spearman', permutations=9999)

    print(f"Mantel Test: Correlation = {mantel_r}, p-value = {mantel_p}")

    # Save results
    mantel_results_df = pd.DataFrame({
        "Mantel Correlation": [mantel_r],
        "p-Value": [mantel_p]
    })
    output_path = os.path.join(output_dir_plots, f'mantel_test_results_correlation{matrix_threshold}.csv')
    mantel_results_df.to_csv(output_path, index=False)
    print(f"Mantel Test results saved to {output_path}")

    return mantel_r, mantel_p



def compute_degree_centrality(adj_matrix, output_dir=None):
    # Set a threshold for the correlation matrix
    threshold = 0.5  # Example threshold, change if needed
    adj_matrix_bin = (adj_matrix > threshold).astype(int)

    G = nx.from_numpy_array(adj_matrix_bin)
    degree_centrality = nx.degree_centrality(G)

    # Save as CSV if an output directory is given
    if output_dir:
        save_to_csv(degree_centrality, output_dir, "degree_centrality")

    return degree_centrality



def plot_connectome_with_metrics(matrix, labels_img_path, output_dir_timeseries, metric_type="degree", threshold="70%",
                                 transform_path=None):
    mni_template = datasets.load_mni152_template()
    mni_template_path = os.path.join(output_dir_timeseries, 'mni_template.nii.gz')
    nib.save(mni_template, mni_template_path)
    transformed_labels_output_path = os.path.join(output_dir_timeseries, 'transformed_labels_image.nii.gz')
    transformed_labels_img_path = transform_labels_to_mni(labels_img_path, transform_path, mni_template_path,
                                                          transformed_labels_output_path)
    transformed_labels_img = image.load_img(transformed_labels_img_path)
    mni_coords = plotting.find_parcellation_cut_coords(transformed_labels_img)

    if metric_type == "degree":
        metric_values = compute_degree_centrality(matrix, output_dir_timeseries)
        node_color = list(metric_values.values())
        node_size = [v * 200 for v in node_color]
    elif metric_type == "participation":
        metric_values = compute_participation_coefficient(matrix, output_dir_timeseries)
        node_color = list(metric_values.values())
        node_size = 100
    elif metric_type == "betweenness":
        metric_values = compute_betweenness_centrality(matrix, output_dir_timeseries)
        node_color = list(metric_values.values())
        node_size = [v * 300 for v in node_color]
    else:
        raise ValueError("Invalid metric_type. Choose from 'degree', 'participation', or 'betweenness'.")

    # Normalize node colors and apply a colormap
    norm = Normalize(vmin=min(node_color), vmax=max(node_color))
    cmap = cm.get_cmap('coolwarm')  # Other options: 'RdYlBu' or different colormaps
    node_color = cmap(norm(node_color))

    display = plotting.plot_glass_brain(None, display_mode='ortho')
    display.add_graph(matrix, mni_coords, node_color=node_color, edge_threshold=threshold, node_size=node_size)

    filename = f'roi_{metric_type}_connectome.png'
    output_path = os.path.join(output_dir_timeseries, filename)
    display.savefig(output_path, dpi=300)
    print(f"Connectome {metric_type} saved as: {output_path}")



def find_largest_correlation_differences(correlation_matrix_1, correlation_matrix_2, roi_names, top_n=100):
    """
    Finds the ROIs with the largest differences in correlation between two conditions and saves the results to a CSV file.

    :param correlation_matrix_1: Correlation matrix for Condition 1.
    :param correlation_matrix_2: Correlation matrix for Condition 2.
    :param roi_names: List of ROI names.
    :param top_n: Number of top differences to return.
    :param output_dir_plots: Directory to save the CSV file.
    :return: A list of tuples containing the ROI pairs with the largest differences and their difference values.
    """
    # Calculate the absolute difference between the two correlation matrices
    diff_matrix = np.abs(correlation_matrix_1 - correlation_matrix_2)

    # Get the indices of the top N largest differences
    flat_indices = np.argsort(diff_matrix, axis=None)[::-1]
    top_indices = np.unravel_index(flat_indices[:top_n], diff_matrix.shape)

    # Get the ROI pairs corresponding to the largest differences
    top_differences = []
    for i, j in zip(top_indices[0], top_indices[1]):
        roi_pair = (roi_names[i], roi_names[j])
        difference = diff_matrix[i, j]
        top_differences.append((roi_pair, difference))

    # Save the results to a CSV file
    if output_dir_plots:
        output_csv_path = os.path.join(output_dir_plots, "largest_correlation_differences.csv")
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROI Pair", "Difference"])
            for roi_pair, difference in top_differences:
                writer.writerow([f"{roi_pair[0]} - {roi_pair[1]}", difference])



def save_to_csv(data_dict, output_dir, metric_name):
    """
    Saves the values for the nodes in a CSV file.

    :param data_dict: Dictionary with values, where the key is the node and the value is the computed metric.
    :param output_dir: Directory where the CSV file should be saved.
    :param metric_name: Name of the computed metric, used as the filename.
    """
    # Create the path to the CSV file
    output_path = os.path.join(output_dir, f"{metric_name}.csv")

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Node", "Value"])
        for node, value in data_dict.items():
            writer.writerow([node, value])

    print(f"Results for {metric_name} saved in: {output_path}")




