
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


# Prepare kinematic features for classification
X = []
y = []

selected_features = ['velocity_mean', 'acceleration_mean', 'jerk_mean', 'angular_velocity_mean', 'range_of_motion', 'spatial_path']

for label, features in features_dict.items():
    for feature_vector in features:
        selected_values = [feature_vector[i] for i in range(len(selected_features))]  # Extract indices instead of keys
        X.append(selected_values)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Convert to DataFrame
df = pd.DataFrame(X, columns=selected_features)
df['Class'] = y

# Set global font properties
plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})


# Plot distribution of each selected feature by class in a 3x3 grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    ax = axes[i]
    sns.kdeplot(data=df, x=feature, hue="Class", fill=True, common_norm=False, palette="tab10", ax=ax)
    ax.set_title(f"Distribution of {feature} by Class", fontsize=12, fontweight='bold')
    ax.set_xlabel("Value", fontsize=10, fontweight='bold')
    ax.set_ylabel("Density", fontsize=10, fontweight='bold')
    ax.legend(title="Class", fontsize=10, loc='upper right', labels=df['Class'].unique().tolist())  # Ensure class names appear in legend

# Hide unused subplots
for j in range(len(selected_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Correlation Heatmap for Selected Features
# Reduced figure size to better fit labels
plt.figure(figsize=(8, 6))
correlation_matrix = df[selected_features].corr()

# Using a dark-centered diverging color palette
cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)

# Setting a smaller font size for the title to conserve space
plt.title("Correlation Heatmap of Features", fontsize=10, fontweight='bold')

# Rotate the labels on the x and y axis to be diagonal
plt.xticks(rotation=45, fontsize=10, fontweight='bold', ha='right')  # Horizontal alignment right for x labels
plt.yticks(rotation=45, fontsize=10, fontweight='bold', va='center')  # Vertical alignment center for y labels

# Use tight layout to automatically adjust subplot parameters
plt.tight_layout()
plt.show()






