%reset -f
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Neural Gas class
class NeuralGas:
    def __init__(self, n_units=200, max_iter=300, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = np.random.rand(n_units, 3)  # Initialize units in 3D space

    def train_step(self, joint_positions, i):
        eta = self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)
        lambda_val = self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)
        
        for position in joint_positions:
            dists = np.linalg.norm(self.units - np.array(position), axis=1)
            ranking = np.argsort(dists)
            for rank, idx in enumerate(ranking):
                influence = np.exp(-rank / lambda_val)
                self.units[idx] += eta * influence * (np.array(position) - self.units[idx])

def euler_to_rotation_matrix(z, y, x):
    z_rad = np.radians(z)
    y_rad = np.radians(y)
    x_rad = np.radians(x)
    Rz = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    return Rz @ Ry @ Rx

def parse_bvh(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    hierarchy = {}
    motions = []
    current_joint = None
    joint_stack = []
    reading_hierarchy = False
    reading_motion = False
    for line in lines:
        stripped_line = line.strip()
        if "HIERARCHY" in stripped_line:
            reading_hierarchy = True
            continue
        if "MOTION" in stripped_line:
            reading_hierarchy = False
            reading_motion = True
            continue
        if reading_hierarchy:
            if "{" in stripped_line:
                joint_stack.append(current_joint)
            elif "}" in stripped_line:
                joint_stack.pop()
            elif "ROOT" in stripped_line or "JOINT" in stripped_line:
                joint_name = stripped_line.split()[-1]
                current_joint = joint_name
                hierarchy[joint_name] = {
                    "parent": joint_stack[-1] if joint_stack else None,
                    "channels": [],
                    "offset": [],
                    "children": []
                }
                if joint_stack:
                    hierarchy[joint_stack[-1]]["children"].append(joint_name)
            elif "OFFSET" in stripped_line:
                _, x, y, z = stripped_line.split()
                hierarchy[current_joint]["offset"] = [float(x), float(y), float(z)]
            elif "CHANNELS" in stripped_line:
                _, count, *channels = stripped_line.split()
                hierarchy[current_joint]["channels"] = channels
        elif reading_motion:
            if "Frames:" in stripped_line or "Frame Time:" in stripped_line:
                continue
            else:
                frame_data = list(map(float, stripped_line.split()))
                motions.append(frame_data)
    return hierarchy, np.array(motions)

def extract_joint_positions(hierarchy, motion_data, frame_idx):
    positions = {joint: np.zeros(3) for joint in hierarchy}
    rotations = {joint: np.zeros(3) for joint in hierarchy}
    channel_index = 0
    for joint in hierarchy:
        joint_data = motion_data[frame_idx, channel_index:channel_index+len(hierarchy[joint]['channels'])]
        rotations[joint] = joint_data[:3]
        if len(hierarchy[joint]['channels']) > 3:
            positions[joint] = joint_data[3:6]
        channel_index += len(hierarchy[joint]['channels'])
    for joint in hierarchy:
        local_offset = np.array(hierarchy[joint]['offset'])
        rotation_matrix = euler_to_rotation_matrix(*rotations[joint])
        local_position = rotation_matrix @ local_offset
        if hierarchy[joint]['parent']:
            parent_position = positions[hierarchy[joint]['parent']]
            positions[joint] = parent_position + local_position
        else:
            positions[joint] = local_position
    return positions

# Load and process data
file_path = 'jump.bvh'
hierarchy, motion_data = parse_bvh(file_path)
ng = NeuralGas(n_units=10, max_iter=10)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-50, 20])
ax.set_zlim([-10, 30])

def update_frame(num, hierarchy, motion_data, ng):
    positions = extract_joint_positions(hierarchy, motion_data, num)
    joint_positions = list(positions.values())
    ng.train_step(joint_positions, num)
    ax.clear()

    # Rotate positions around Z-axis by 45 degrees
    rotation_matrix = euler_to_rotation_matrix(-45, -45, 90)  # Only rotate around Z-axis

    for joint in positions:
        # Apply rotation to each joint position
        end = rotation_matrix @ positions[joint]  
        if hierarchy[joint]['parent']:
            start = rotation_matrix @ positions[hierarchy[joint]['parent']]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
        ax.scatter(end[0], end[1], end[2], color='green', s=100)
        ax.text(end[0], end[1], end[2], joint, color='blue', fontsize=8)  # Add joint names

    # Plot Neural Gas units, applying the same rotation
    rotated_units = np.dot(ng.units, rotation_matrix[:3, :3].T)
    ax.scatter(rotated_units[:, 0], rotated_units[:, 1], rotated_units[:, 2], c='blue')
    
    ax.set_title(f'Iteration {num + 1}')
    return ng,


ani = FuncAnimation(fig, update_frame, frames=len(motion_data), fargs=(hierarchy, motion_data, ng), interval=30, blit=False)
plt.show()
