import numpy as np
import torch
import os
import shutil

def get_bounds_normalization(data, quantiles=[0.25, 0.75], factors=[1, 1]):
    left_data = data[data < 0]
    right_data = data[data > 0]

    q1_left, q3_left = np.quantile(left_data, quantiles)
    q1_right, q3_right = np.quantile(right_data, quantiles)

    iqr_left = q3_left - q1_left
    iqr_right = q3_right - q1_right

    lower_bound = q1_left - factors[0] * iqr_left
    upper_bound = q3_right + factors[1] * iqr_right

    return lower_bound, upper_bound

def get_bounds_normalization_positive(data, quantiles=[0.25, 0.75], factor=3):
    q1, q3 = np.quantile(data, quantiles)
    iqr = q3 - q1
    lower_bound = max(q1 - factor * iqr, np.array(0))
    upper_bound = q3 + factor * iqr

    return lower_bound, upper_bound

def get_bins_histogram(lower_limit, upper_limit, num_bins):
    bin_size = (upper_limit - lower_limit) / num_bins
    bins = np.arange(lower_limit, upper_limit + bin_size, bin_size).tolist()
    return bins

def create_histogram(data, bins):
    histogram = np.histogram(data, bins=bins)[0].astype(int).tolist()
    return histogram

def get_outliers(data, quantiles=[0.25, 0.75], factor=3):
    median = np.median(data)
    
    left_data = data[data < median]
    right_data = data[data >= median]
    
    q1_left, q3_left = np.quantile(left_data, quantiles)
    q1_right, q3_right = np.quantile(right_data, quantiles)
    
    iqr_left = q3_left - q1_left
    iqr_right = q3_right - q1_right
    
    lower_bound_left = q1_left - factor * iqr_left
    upper_bound_right = q3_right + factor * iqr_right
    
    outliers_left = data < lower_bound_left
    outliers_right = data > upper_bound_right
    
    outliers = outliers_left | outliers_right
    
    return np.where(outliers)[0].tolist(), data[outliers].tolist()

def get_median_histogram(data, bins, max_rows=200):
    if data.shape[0] > max_rows:
        indices = torch.randperm(data.shape[0])[:max_rows]
        sampled_data = data[indices].cpu()
    else:
        sampled_data = data.cpu()
    histograms = [np.histogram(row, bins=bins)[0] for row in sampled_data]
    median_histogram = np.median(histograms, axis=0).astype(int).tolist()
    return median_histogram

def adjust_token_name(token_name):
    token_name = token_name.replace(" ", "_")
    token_name = token_name.replace("\n", "\\n")
    return token_name

def get_value_dict(dict, keys):
    for key in keys:
        dict = dict[key]
    return dict

def set_value_dict(dict, keys, value):
    for key in keys[:-1]:
        dict = dict[key]
    dict[keys[-1]] = value

    
def get_median_histogram_activations_layers(data, bins, max_rows=1000):
    if data.shape[0] > max_rows:
        indices = torch.randperm(data.shape[0])[:max_rows]
        sampled_data = data[indices]
    else:
        sampled_data = data
    histograms = [np.histogram(row, bins=bins)[0] for row in sampled_data]
    median_histogram = np.median(histograms, axis=0).astype(int).tolist()
    return median_histogram

def create_sample_data_directory(sample_idx):
    if os.path.exists(f"app/samples_data/sample_{sample_idx}"):
        shutil.rmtree(f"app/samples_data/sample_{sample_idx}")
    activations_types = ["embed", "input", "input_layernorm", "q", "k", "v", "q_rope", "k_rope", "o", "o_mm_dense", "dense_attention", "dense_attention_residual", "post_attention_layernorm",
                         "mlp_h_to_4", "mlp_4_to_h", "output"]
    for activation_type in activations_types:
        os.makedirs(f"app/samples_data/sample_{sample_idx}/{activation_type}")