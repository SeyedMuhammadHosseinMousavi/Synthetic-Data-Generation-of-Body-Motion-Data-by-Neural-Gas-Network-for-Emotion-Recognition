%reset -f
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial.distance import cdist
from scipy.fftpack import fft
from scipy.linalg import sqrtm
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw

warnings.filterwarnings("ignore")

# load BVH data
def load_bvh_data(file_path):
    """Loads numeric motion data from a BVH file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    motion_data = []
    motion_start = False

    for line in lines:
        if "MOTION" in line:
            motion_start = True
            continue
        if motion_start:
            numbers = np.fromstring(line, sep=' ')
            if numbers.size > 0:
                motion_data.append(numbers)

    return np.array(motion_data) if motion_data else np.array([])

# Load all BVH files in a folder
def load_data_from_folder(folder_path):
    """Loads all BVH motion data from the specified folder."""
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            motion_data = load_bvh_data(file_path)
            if motion_data.size > 0:
                all_data.append(motion_data)

    return all_data

# Load Original and Synthetic Data
original_folder = '100StyleBaseline'
synthetic_folder = '100StyleSynthetic'

original_data = []
synthetic_data = []

for class_folder in os.listdir(original_folder):
    class_path = os.path.join(original_folder, class_folder)
    original_data.extend(load_data_from_folder(class_path))

for class_folder in os.listdir(synthetic_folder):
    class_path = os.path.join(synthetic_folder, class_folder)
    synthetic_data.extend(load_data_from_folder(class_path))

# same shape via Interpolation
target_length = max(max(len(s) for s in original_data), max(len(s) for s in synthetic_data))

from scipy.interpolate import interp1d
def interpolate_data(data, target_length):
    """Interpolates each sequence to the target length."""
    interpolated_data = []
    for sequence in data:
        if len(sequence) == 0:  # Skip empty sequences
            continue
        x_old = np.linspace(0, 1, num=len(sequence))
        x_new = np.linspace(0, 1, num=target_length)
        interp_func = interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
        new_sequence = interp_func(x_new)
        interpolated_data.append(new_sequence)
    return np.array(interpolated_data)

original_data = interpolate_data(original_data, target_length)
synthetic_data = interpolate_data(synthetic_data, target_length)

# same feature dimensions
if original_data.shape[1] != synthetic_data.shape[1]:
    min_features = min(original_data.shape[1], synthetic_data.shape[1])
    original_data = original_data[:, :min_features]
    synthetic_data = synthetic_data[:, :min_features]

# Standardize Data
scaler = StandardScaler()
original_data = scaler.fit_transform(original_data.reshape(original_data.shape[0], -1))
synthetic_data = scaler.transform(synthetic_data.reshape(synthetic_data.shape[0], -1))

# **Apply PCA to Reduce Feature Dimensions**
def apply_pca(data, variance_threshold=0.99):
    """Reduces feature dimensions using PCA while keeping 99% of variance."""
    pca = PCA(n_components=min(data.shape[0], data.shape[1]))  # Dynamic component selection
    reduced_data = pca.fit_transform(data)
    
    # Keep only components that explain the variance_threshold
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= variance_threshold) + 1
    reduced_data = reduced_data[:, :optimal_components]
    
    return reduced_data

# Apply PCA to both datasets
original_data = apply_pca(original_data)
synthetic_data = apply_pca(synthetic_data)

# --- METRIC 1: Fréchet Inception Distance (FID) ---
def compute_fid(original_data, synthetic_data):
    """Computes Fréchet Inception Distance (FID) with PCA-reduced features."""
    # Compute mean and covariance
    mu1, sigma1 = np.mean(original_data, axis=0), np.cov(original_data, rowvar=False)
    mu2, sigma2 = np.mean(synthetic_data, axis=0), np.cov(synthetic_data, rowvar=False)

    # Compute sqrtm of covariance product
    covmean = sqrtm(sigma1 @ sigma2) if sigma1.shape == sigma2.shape else np.zeros_like(sigma1)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Ensure real values only

    fid = np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# --- METRIC 2: Dynamic Time Warping (DTW) (Fix for fastdtw error) ---
def compute_dtw(original_data, synthetic_data):
    """Computes DTW ensuring correct input dimensions."""
    dtw_distances = []
    
    for i in range(min(len(original_data), len(synthetic_data))):
        original_vector = original_data[i].flatten().reshape(-1, 1)  # Ensure it's 2D
        synthetic_vector = synthetic_data[i].flatten().reshape(-1, 1)  # Ensure it's 2D
        
        distance, _ = fastdtw(original_vector, synthetic_vector, dist=lambda x, y: norm(x - y))
        dtw_distances.append(distance)

    return np.mean(dtw_distances)

# --- METRIC 3: Mean Per-Joint Position Error (MPJPE) ---
def compute_mpjpe(original_data, synthetic_data):
    """Computes MPJPE ensuring input dimensions match before subtraction."""
    
    # both datasets have the same number of features (columns)
    min_features = min(original_data.shape[1], synthetic_data.shape[1])
    original_data = original_data[:, :min_features]  # Trim extra features
    synthetic_data = synthetic_data[:, :min_features]  # Trim extra features
    
    # If they have different samples (rows), take the minimum set
    min_samples = min(original_data.shape[0], synthetic_data.shape[0])
    original_data = original_data[:min_samples, :]
    synthetic_data = synthetic_data[:min_samples, :]

    # Compute MPJPE
    mpjpe_errors = np.linalg.norm(original_data - synthetic_data, axis=1)
    return np.mean(mpjpe_errors)


# Compute metrics
fid_score = compute_fid(original_data, synthetic_data)
dtw_score = compute_dtw(original_data, synthetic_data)
mpjpe_score = compute_mpjpe(original_data, synthetic_data)

# Print optimized results
print(f" FID with PCA: {fid_score:.4f}")
print(f" DTW with PCA: {dtw_score:.4f}")
print(f" MPJPE: {mpjpe_score:.4f}")
