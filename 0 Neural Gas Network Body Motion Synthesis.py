# About Avg Error----------------------------------------
# Avg err represents the average distance between the input samples 
# (joint data) and their closest neurons in the network. This value indicates how well 
# the NGN is capturing the structure of the input data, with lower values indicating a 
# better fit.
# -------------------------------------------------

%reset -f
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
import warnings
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the Neural Gas Network class
class NeuralGasNetwork:
    def __init__(self, num_neurons, input_dim, max_iterations, epsilon_initial, epsilon_final, lambda_initial, lambda_final):
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.max_iterations = max_iterations
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final
        self.neurons = np.random.rand(num_neurons, input_dim)
        self.neuron_ranks = np.zeros(num_neurons)
        self.errors = []

    def train(self, data):
        for iteration in range(self.max_iterations):
            epsilon = self.epsilon_initial * (self.epsilon_final / self.epsilon_initial) ** (iteration / self.max_iterations)
            lambda_ = self.lambda_initial * (self.lambda_final / self.lambda_initial) ** (iteration / self.max_iterations)
            total_error = 0

            for sample in data:
                distances = np.linalg.norm(self.neurons - sample, axis=1)
                self.neuron_ranks = np.argsort(distances)
                total_error += np.min(distances)  # Record the minimum distance as error

                for rank, neuron_index in enumerate(self.neuron_ranks):
                    influence = np.exp(-rank / lambda_)
                    self.neurons[neuron_index] += epsilon * influence * (sample - self.neurons[neuron_index])

            # Calculate average error for the iteration
            avg_error = total_error / len(data)
            self.errors.append(avg_error)
            print(f"Iteration {iteration + 1}: Avg Error {avg_error}")

    def generate_sample(self):
        random_neuron = self.neurons[np.random.randint(self.num_neurons)]
        random_noise = np.random.normal(scale=0.05, size=self.input_dim)
        return random_neuron + random_noise

# parse BVH files
def parse_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header, motion_data = [], []
    capture_data = False
    for line in lines:
        if "MOTION" in line:
            capture_data = True
        elif capture_data:
            if line.strip().startswith("Frames") or line.strip().startswith("Frame Time"):
                continue
            motion_data.append(np.fromstring(line, sep=' '))
        else:
            header.append(line)
    return header, np.array(motion_data)

# save BVH files
def save_bvh(header, motion_data, file_path):
    with open(file_path, 'w') as file:
        file.writelines(header)
        file.write("MOTION\n")
        file.write(f"Frames: {len(motion_data)}\n")
        file.write("Frame Time: 0.008333\n")
        for frame in motion_data:
            line = ' '.join(format(value, '.6f') for value in frame)
            file.write(line + '\n')

# normalize motion data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_shape = data.shape
    data_flattened = data.reshape(-1, data_shape[-1])
    data_normalized = scaler.fit_transform(data_flattened).reshape(data_shape)
    return data_normalized, scaler

# smooth motion data using Gaussian filter
def smooth_motion_data_gaussian(motion_data, sigma=1.0):
    smoothed_data = gaussian_filter(motion_data, sigma=(sigma, 0))
    return smoothed_data

# load data from a folder
def load_data_from_folder(folder_path):
    all_data = []
    max_length = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bvh'):
            file_path = os.path.join(folder_path, file_name)
            _, motion_data = parse_bvh(file_path)
            all_data.append(motion_data)
            if motion_data.shape[0] > max_length:
                max_length = motion_data.shape[0]
    # Pad all sequences to the max_length
    padded_data = [np.pad(data, ((0, max_length - data.shape[0]), (0, 0)), 'constant') for data in all_data]
    return np.array(padded_data)

# Main process
base_folder = '100styleEmotionForSynthesis'  
output_folder = '100StyleSynthetic'
num_neurons = 50
max_iterations = 50
epsilon_initial = 0.3
epsilon_final = 0.05
lambda_initial = 10
lambda_final = 0.1
num_new_samples = 10
sigma = 3.0  # Standard deviation for Gaussian smoothing

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class_folders = ['Angry', 'Depressed', 'Neutral','Proud']


ngn_errors = {}  # Dictionary to store errors for each class

for class_folder in class_folders:
    print(f"Processing class: {class_folder}")
    class_path = os.path.join(base_folder, class_folder)
    class_output_folder = os.path.join(output_folder, class_folder)
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)

    # Load and normalize data for the current class
    motion_data = load_data_from_folder(class_path)
    original_shape = motion_data.shape
    motion_data = motion_data.reshape((motion_data.shape[0], -1))  # Flatten data
    normalized_data, scaler = normalize_data(motion_data)

    # Train Neural Gas Network
    ngn = NeuralGasNetwork(num_neurons, normalized_data.shape[1], max_iterations, epsilon_initial, epsilon_final, lambda_initial, lambda_final)
    ngn.train(normalized_data)
    ngn_errors[class_folder] = ngn.errors  # Store errors

    # Generate new samples
    generated_samples = []
    for _ in range(num_new_samples):
        generated_sample = ngn.generate_sample()
        generated_sample = scaler.inverse_transform(generated_sample.reshape(-1, generated_sample.shape[-1]))
        generated_samples.append(generated_sample.reshape(original_shape[1:]))
    
    # Post-process and save samples
    for i, generated_sample in enumerate(generated_samples):
        # Apply Gaussian smoothing
        smoothed_sample = smooth_motion_data_gaussian(generated_sample, sigma=sigma)
        
        # Save the smoothed sample
        header, _ = parse_bvh(os.path.join(class_path, os.listdir(class_path)[0]))
        output_file_path = os.path.join(class_output_folder, f'{class_folder}_generated_{i + 1}.bvh')
        save_bvh(header, smoothed_sample, output_file_path)
        print(f"Generated and smoothed sample {i + 1} for class {class_folder}")

# Plot the cost (average error) over iterations for each class
plt.figure(figsize=(15, 5))
for i, class_folder in enumerate(class_folders):
    plt.subplot(1, len(class_folders), i + 1)
    plt.plot(range(1, max_iterations + 1), ngn_errors[class_folder], marker='o')
    plt.title(f"Avg Error over Iterations - {class_folder.capitalize()}")
    plt.xlabel("Iterations")
    plt.ylabel("Average Error")
    plt.grid(True)
plt.tight_layout()
plt.show()

print("Generation completed.")

# Runtime
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"Total runtime: {int(minutes)} minutes and {int(seconds)} seconds")
