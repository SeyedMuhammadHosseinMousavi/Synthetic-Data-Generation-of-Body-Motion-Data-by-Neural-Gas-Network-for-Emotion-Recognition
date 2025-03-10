%reset -f
import numpy as np
import os
import matplotlib.pyplot as plt

def parse_bvh(file_path):
    """Extracts only the positions from the hip joint (considering index 6, 7, 8 for x, y, z) from a BVH file."""
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
                motion_data.append(numbers[[6, 7, 8]])  # Only extract hip joint positions

    return np.array(motion_data)

def load_and_plot(base_folder):
    """Loads BVH files, extracts motion paths, and plots trajectories."""
    colors = {'Angry': 'red', 'Depressed': 'blue', 'Neutral': 'green', 'Proud': 'purple'}
    plt.figure(figsize=(12, 10))

    for emotion, color in colors.items():
        emotion_path = os.path.join(base_folder, emotion)
        if not os.path.exists(emotion_path):
            print(f"Path not found: {emotion_path}")
            continue

        print(f"Loading data for {emotion}...")
        for file_name in os.listdir(emotion_path):
            if file_name.endswith('.bvh'):
                file_path = os.path.join(emotion_path, file_name)
                print(f"Parsing file: {file_path}")
                motion_data = parse_bvh(file_path)
                plt.plot(motion_data[:, 0], motion_data[:, 2], color=color, alpha=0.5, label=emotion)  # Plot x vs z for 2D trajectory
                print(f"Data plotted for {file_name}")

    # To avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_font_props = {'weight': 'bold', 'size': 'large'}  # Font properties for the legend
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop=legend_font_props, title='Emotions', title_fontsize='large')

    plt.title('Trajectories of Body Motions by Emotion (Baseline)', fontsize=14, fontweight='bold')
    plt.xlabel('X Position', fontsize=14, fontweight='bold')
    plt.ylabel('Z Position', fontsize=14, fontweight='bold')
    plt.xticks(fontweight='bold')  # Bold X axis tick labels
    plt.yticks(fontweight='bold')  # Bold Y axis tick labels
    plt.grid(True)
    plt.show()

# Main process
base_folder = '100StyleSynthetic'

load_and_plot(base_folder)
