
%reset -f
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import warnings
from scipy.fftpack import fft
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
# Necessary imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# parse BVH files
def parse_bvh(file_path):
    """Extracts only the numeric motion data from a BVH file."""
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

    return np.array(motion_data)

# standardize motion data
def standardize_data(data):
    """Applies standardization (zero mean, unit variance) to the data."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# smooth motion data using Gaussian filter
def smooth_motion_data_gaussian(motion_data, sigma=3.0):
    return gaussian_filter(motion_data, sigma=(sigma, 0))

# load and process motion data from a folder
def load_data_from_folder(folder_path):
    """Loads and standardizes motion data from BVH files in a given folder."""
    all_data = []
    max_length = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            motion_data = parse_bvh(file_path)
            if motion_data.shape[0] > max_length:
                max_length = motion_data.shape[0]
            all_data.append(motion_data)

    # Pad sequences to the max length
    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in all_data]
    return np.array(padded_data)

# extract kinematic features
def extract_kinematic_features(data, dt=1/30):  # frame rate of 30 FPS
    """Extracts kinematic features such as velocity, acceleration, jerk, angular velocity, and harmonics."""
    
    velocity = np.diff(data, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt

    # Pad to match original shape
    velocity = np.vstack([velocity, np.zeros((1, velocity.shape[1]))])
    acceleration = np.vstack([acceleration, np.zeros((2, acceleration.shape[1]))])
    jerk = np.vstack([jerk, np.zeros((3, jerk.shape[1]))])

    # angular velocity is valid (only if data is 3D)
    if velocity.shape[1] % 3 == 0:
        reshaped_velocity = velocity.reshape(velocity.shape[0], -1, 3)
        angular_velocity = np.cross(reshaped_velocity[:-1], reshaped_velocity[1:], axis=2)
        angular_velocity = np.vstack([angular_velocity, np.zeros((1, angular_velocity.shape[1], 3))])
        angular_velocity = angular_velocity.reshape(angular_velocity.shape[0], -1)
    else:
        angular_velocity = np.zeros_like(velocity)

    # Range of motion
    range_of_motion = np.max(data, axis=0) - np.min(data, axis=0)

    # Spatial path length
    spatial_path = np.sum(norm(np.diff(data, axis=0), axis=1))

    # Harmonics using FFT
    harmonics = np.abs(fft(data, axis=0))

    # Flatten features
    features = {
        'velocity_mean': np.mean(velocity, axis=0).flatten(),
        'acceleration_mean': np.mean(acceleration, axis=0).flatten(),
        'jerk_mean': np.mean(jerk, axis=0).flatten(),
        'angular_velocity_mean': np.mean(angular_velocity, axis=0).flatten(),
        'range_of_motion': range_of_motion.flatten(),
        'spatial_path': np.array([spatial_path]),  # Scalar value
        'harmonics_magnitude_mean': np.mean(harmonics, axis=0).flatten(),
    }
    return features

# combine all kinematic features into a single vector
def fuse_features(kinematic_features):
    return np.concatenate(list(kinematic_features.values()))

# Main process
base_folder = '100StyleSynthetic'
class_folders = ['Angry', 'Depressed', 'Neutral', 'Proud']
sigma = 3.0  # Standard deviation for Gaussian smoothing
features_dict = {}

for class_folder in class_folders:
    print(f"Processing class: {class_folder}")
    class_path = os.path.join(base_folder, class_folder)

    motion_data = load_data_from_folder(class_path)
    original_shape = motion_data.shape
    motion_data = motion_data.reshape((motion_data.shape[0], -1))
    standardized_data = standardize_data(motion_data)
    smoothed_data = smooth_motion_data_gaussian(standardized_data, sigma)

    # Extract kinematic features for each sample
    class_features = [fuse_features(extract_kinematic_features(sample.reshape(original_shape[1:]))) for sample in smoothed_data]
    features_dict[class_folder] = class_features

    print(f"Kinematic features extracted and stored for class {class_folder}")

# Prepare kinematic features for classification
X = []
y = []

for label, features in features_dict.items():
    for feature_vector in features:
        X.append(feature_vector)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Classification metrics storage
accuracies_rf, accuracies_dt = [], []
precision_rf, recall_rf, fscore_rf = [], [], []
precision_dt, recall_dt, fscore_dt = [], [], []
mcc_rf, mcc_dt = [], []

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    dt_classifier = DecisionTreeClassifier(random_state=i)

    rf_classifier.fit(X_train, y_train)
    dt_classifier.fit(X_train, y_train)

    rf_predictions = rf_classifier.predict(X_test)
    dt_predictions = dt_classifier.predict(X_test)

    accuracies_rf.append(accuracy_score(y_test, rf_predictions))
    accuracies_dt.append(accuracy_score(y_test, dt_predictions))

    prf, rrf, frf, _ = precision_recall_fscore_support(y_test, rf_predictions, average='weighted', zero_division=0)
    pdt, rdt, fdt, _ = precision_recall_fscore_support(y_test, dt_predictions, average='weighted', zero_division=0)

    precision_rf.append(prf)
    recall_rf.append(rrf)
    fscore_rf.append(frf)
    precision_dt.append(pdt)
    recall_dt.append(rdt)
    fscore_dt.append(fdt)

    mcc_rf.append(matthews_corrcoef(y_test, rf_predictions))
    mcc_dt.append(matthews_corrcoef(y_test, dt_predictions))

# Print aggregated results (one line per result)
print("\nFinal Aggregated Results:")

# Random Forest Results
print("Random Forest - Accuracy:", np.mean(accuracies_rf))
print("Random Forest - Std:", np.std(accuracies_rf))
print("Random Forest - Precision:", np.mean(precision_rf))
print("Random Forest - Recall:", np.mean(recall_rf))
print("Random Forest - F1-Score:", np.mean(fscore_rf))
print("Random Forest - MCC:", np.mean(mcc_rf))

# Decision Tree Results
print("\nDecision Tree - Accuracy:", np.mean(accuracies_dt))
print("Decision Tree - Std:", np.std(accuracies_dt))
print("Decision Tree - Precision:", np.mean(precision_dt))
print("Decision Tree - Recall:", np.mean(recall_dt))
print("Decision Tree - F1-Score:", np.mean(fscore_dt))
print("Decision Tree - MCC:", np.mean(mcc_dt))



# --------------------------------------
# Diversity and Fidelity Metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.fftpack import fft
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


def calculate_diversity(features):
    """Calculate the average Euclidean distance between all feature vectors."""
    distances = pairwise_distances(features, metric='euclidean')
    return np.mean(distances)

def calculate_fidelity(features):
    """Calculate the average cosine similarity between all feature vectors."""
    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0)  # Ignore self-comparisons
    return np.mean(similarities)

# Load BVH data function (assumes BVH files are being used)
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

# Processing folders
# base_folder = '100StyleSynthetic'
all_data = []

for class_folder in os.listdir(base_folder):
    class_path = os.path.join(base_folder, class_folder)
    all_data.extend(load_data_from_folder(class_path))

# Convert to uniform shape (interpolation)
target_length = max(len(sample) for sample in all_data)

# Interpolate data to the same length
from scipy.interpolate import interp1d
def interpolate_data(data, target_length):
    """Interpolates each sequence to the target length."""
    interpolated_data = []
    for sequence in data:
        x_old = np.linspace(0, 1, num=len(sequence))
        x_new = np.linspace(0, 1, num=target_length)
        interp_func = interp1d(x_old, sequence, axis=0, kind='linear', fill_value="extrapolate")
        new_sequence = interp_func(x_new)
        interpolated_data.append(new_sequence)
    return np.array(interpolated_data)

all_data = interpolate_data(all_data, target_length)

# Standardize Data
scaler = StandardScaler()
all_data = scaler.fit_transform(all_data.reshape(all_data.shape[0], -1))

# Compute Diversity & Fidelity
diversity_score = calculate_diversity(all_data)
fidelity_score = calculate_fidelity(all_data)

# Normalize Scores between 0 and 1 (Correct Fix)
diversity_normalized = (diversity_score - np.min(diversity_score)) / (np.max(diversity_score) - np.min(diversity_score))
fidelity_normalized = 1 - fidelity_score  # Fidelity should be low for good results

# Print Results
print(f"Raw Diversity Score: {diversity_score:.4f}")
print(f"Raw Fidelity Score: {fidelity_score:.4f}")



# Violin Plot
# Prepare data for violin plot
df = pd.DataFrame({
    'Accuracy': accuracies_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'F1-Score': fscore_rf,
    'MCC': mcc_rf
})

# Melt dataframe for visualization
df_melted = df.melt(var_name="Metric", value_name="Score")

# Plot violin plot
plt.figure(figsize=(14, 3))
sns.violinplot(x="Metric", y="Score", data=df_melted, inner="quartile")
# plt.title("\nComparison of Emotion Recognition Metrics using Random Forest\n", fontsize=18, fontweight='bold')
# plt.xlabel("Performance Metric", fontsize=14, fontweight='bold')
plt.ylabel("Score", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', rotation=0)
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()





