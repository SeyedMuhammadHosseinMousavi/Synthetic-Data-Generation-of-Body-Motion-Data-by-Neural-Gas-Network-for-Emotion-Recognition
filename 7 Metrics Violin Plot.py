%reset -f
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

# Function to parse BVH files
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

# Gaussian smoothing
def smooth_motion_data_gaussian(motion_data, sigma=3.0):
    return gaussian_filter(motion_data, sigma=(sigma, 0))

# Load BVH files from a folder
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

    # Pad sequences to the max length
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
        'velocity_mean': np.mean(velocity, axis=0).flatten(),
        'acceleration_mean': np.mean(acceleration, axis=0).flatten(),
        'jerk_mean': np.mean(jerk, axis=0).flatten(),
        'range_of_motion': range_of_motion.flatten(),
        'spatial_path': np.array([spatial_path]),
        'harmonics_magnitude_mean': np.mean(harmonics, axis=0).flatten(),
    }
    return features

# Fuse features
def fuse_features(kinematic_features):
    return np.concatenate(list(kinematic_features.values()))

# Load and process data
base_folder = '100StyleSynthetic'
class_folders = ['Angry', 'Depressed', 'Neutral', 'Proud']
sigma = 3.0
features_dict = {}

for class_folder in class_folders:
    class_path = os.path.join(base_folder, class_folder)
    motion_data = load_data_from_folder(class_path)
    original_shape = motion_data.shape
    motion_data = motion_data.reshape((motion_data.shape[0], -1))
    standardized_data = standardize_data(motion_data)
    smoothed_data = smooth_motion_data_gaussian(standardized_data, sigma)

    class_features = [fuse_features(extract_kinematic_features(sample.reshape(original_shape[1:]))) for sample in smoothed_data]
    features_dict[class_folder] = class_features

X, y = [], []
for label, features in features_dict.items():
    for feature_vector in features:
        X.append(feature_vector)
        y.append(label)

X = np.array(X)
y = np.array(y)

accuracies_rf, precision_rf, recall_rf, fscore_rf, mcc_rf = [], [], [], [], []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    accuracies_rf.append(accuracy_score(y_test, rf_predictions))
    prf, rrf, frf, _ = precision_recall_fscore_support(y_test, rf_predictions, average='weighted', zero_division=0)
    precision_rf.append(prf)
    recall_rf.append(rrf)
    fscore_rf.append(frf)
    mcc_rf.append(matthews_corrcoef(y_test, rf_predictions))

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
plt.figure(figsize=(14, 8))
sns.violinplot(x="Metric", y="Score", data=df_melted, inner="quartile")
plt.title("\nComparison of Emotion Recognition Metrics using Random Forest\n", fontsize=18, fontweight='bold')
plt.xlabel("Performance Metric", fontsize=14, fontweight='bold')
plt.ylabel("Score", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold', rotation=45)
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
