%reset -f
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import warnings
from scipy.fftpack import fft
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to parse BVH motion data
def parse_bvh(file_path):
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

# Standardization
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Apply Gaussian smoothing
def smooth_motion_data_gaussian(motion_data, sigma=3.0):
    return gaussian_filter(motion_data, sigma=(sigma, 0))

# Load motion data from a folder
def load_data_from_folder(folder_path):
    all_data = []
    max_length = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            motion_data = parse_bvh(file_path)
            if motion_data.shape[0] > max_length:
                max_length = motion_data.shape[0]
            all_data.append(motion_data)

    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in all_data]
    return np.array(padded_data)

# Extract kinematic features
def extract_kinematic_features(data, dt=1/30):
    velocity = np.diff(data, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt

    velocity = np.vstack([velocity, np.zeros((1, velocity.shape[1]))])
    acceleration = np.vstack([acceleration, np.zeros((2, acceleration.shape[1]))])
    jerk = np.vstack([jerk, np.zeros((3, jerk.shape[1]))])

    range_of_motion = np.max(data, axis=0) - np.min(data, axis=0)
    spatial_path = np.sum(norm(np.diff(data, axis=0), axis=1))
    harmonics = np.abs(fft(data, axis=0))

    features = {
        'Velocity': np.mean(velocity, axis=0),
        'Acceleration': np.mean(acceleration, axis=0),
        'Jerk': np.mean(jerk, axis=0),
        'Range of Motion': range_of_motion,
        'Spatial Path': np.array([spatial_path]),
        'Harmonics': np.mean(harmonics, axis=0),
    }
    return features

# Combine kinematic features
def fuse_features(kinematic_features):
    return np.concatenate(list(kinematic_features.values()))

# Processing
base_folder = '100StyleSynthetic'
class_folders = ['Angry', 'Depressed', 'Neutral', 'Proud']
sigma = 3.0
features_dict = {}

for class_folder in class_folders:
    print(f"Processing class: {class_folder}")
    class_path = os.path.join(base_folder, class_folder)
    motion_data = load_data_from_folder(class_path)
    original_shape = motion_data.shape
    motion_data = motion_data.reshape((motion_data.shape[0], -1))
    standardized_data = standardize_data(motion_data)
    smoothed_data = smooth_motion_data_gaussian(standardized_data, sigma)

    class_features = [fuse_features(extract_kinematic_features(sample.reshape(original_shape[1:]))) for sample in smoothed_data]
    features_dict[class_folder] = class_features
    print(f"Features extracted for class {class_folder}")

# Prepare dataset
X = np.vstack(list(features_dict.values()))
y = np.concatenate([[label] * len(features) for label, features in features_dict.items()])

# Define feature categories
feature_categories = ['Velocity', 'Acceleration', 'Jerk', 'Range of Motion', 'Spatial Path', 'Harmonics']

# Create 1x4 plot for feature importance of each emotion
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))

all_importances = {}

for idx, emotion in enumerate(class_folders):
    y_binary = (y == emotion).astype(int)

    # Train the model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=0)
    rf_classifier.fit(X_train, y_train)

    # Extract feature importances
    raw_importances = rf_classifier.feature_importances_

    # Aggregate feature importance by category
    feature_importance_by_category = {cat: 0 for cat in feature_categories}
    category_sizes = {cat: 0 for cat in feature_categories}

    extracted_features = extract_kinematic_features(np.random.random((100, 3)))

    index = 0
    for category in feature_categories:
        num_features = len(extracted_features[category])
        category_sizes[category] += num_features
        feature_importance_by_category[category] += np.sum(raw_importances[index:index + num_features])
        index += num_features

    # Normalize importances
    total_importance = sum(feature_importance_by_category.values())
    for cat in feature_categories:
        feature_importance_by_category[cat] /= total_importance

    # Store for printing
    all_importances[emotion] = feature_importance_by_category

    # Sort by importance
    sorted_categories = sorted(feature_importance_by_category.items(), key=lambda x: x[1], reverse=True)

    # Plot
    ax = axes[idx]
    ax.bar([cat for cat, _ in sorted_categories], [imp for _, imp in sorted_categories], color='#6A0DAD', align='center')
    ax.set_title(f"{emotion}", fontsize=14, fontweight='bold')
    # ax.set_xlabel('Feature Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels([cat for cat, _ in sorted_categories], rotation=45, fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.show()

# Print all feature importances for each emotion
print("\n=== Feature Importances for Each Emotion ===")
for emotion, importances in all_importances.items():
    print(f"\nFeature Importance for {emotion}:")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.3f}")
