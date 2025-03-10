import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NeuralGas:
    def __init__(self, joint_positions, n_units=200, max_iter=300, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.joint_positions = np.array(list(joint_positions.values()))
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = np.random.rand(n_units, 2)  # Initialize units in 2D space

    def train_step(self, i):
        eta = self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)
        lambda_val = self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)
        
        for point in self.joint_positions:
            dists = np.linalg.norm(self.units - point, axis=1)
            ranking = np.argsort(dists)
            for rank, idx in enumerate(ranking):
                influence = np.exp(-rank / lambda_val)
                self.units[idx] += eta * influence * (point - self.units[idx])

# Joint positions and skeletal connections
joint_positions = {
    "Hips": (0, 0),
    "LHipJoint": (-1, -1), "LeftUpLeg": (-2, -2), "LeftLeg": (-3, -3), "LeftFoot": (-3, -4), "LeftToeBase": (-3, -5),
    "RHipJoint": (1, -1), "RightUpLeg": (2, -2), "RightLeg": (3, -3), "RightFoot": (3, -4), "RightToeBase": (3, -5),
    "LowerBack": (0, 1), "Spine": (0, 2), "Spine1": (0, 3),
    "Neck": (0, 4), "Neck1": (0, 5), "Head": (0, 6),
    "LeftShoulder": (-1, 3), "LeftArm": (-2, 3), "LeftForeArm": (-3, 3), "LeftHand": (-4, 3),
    "RightShoulder": (1, 3), "RightArm": (2, 3), "RightForeArm": (3, 3), "RightHand": (4, 3)
}

skeletal_connections = [
    ("Hips", "LHipJoint"), ("LHipJoint", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), 
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RHipJoint"), ("RHipJoint", "RightUpLeg"), ("RightUpLeg", "RightLeg"), 
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
    ("Hips", "LowerBack"), ("LowerBack", "Spine"), ("Spine", "Spine1"), 
    ("Spine1", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Spine1", "LeftShoulder"), ("LeftShoulder", "LeftArm"), 
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightShoulder"), ("RightShoulder", "RightArm"), 
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand")
]

# Initialize Neural Gas
ng = NeuralGas(joint_positions, n_units=1, max_iter=300)

import matplotlib.pyplot as plt

# Set up the figure, the axis, and the plot elements
fig, ax = plt.subplots(figsize=(10, 8))

# Adjust the font properties for axis labels
ax.set_xlabel('X Axis Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Axis Label', fontsize=12, fontweight='bold')

# Loop through the joints and plot them
for joint, pos in joint_positions.items():
    ax.plot(pos[0], pos[1], 'ro', markersize=10)  # Red dots for joints
    ax.text(pos[0] + 0.1, pos[1] + 0.1, joint, fontsize=12, fontweight='bold')  # Bold labels for joints

# Loop through the connections to draw lines between joints
for connection in skeletal_connections:
    start_pos = joint_positions[connection[0]]
    end_pos = joint_positions[connection[1]]
    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', linewidth=2)  # Draw connections

# Bold and larger font for tick labels on both axes
ax.tick_params(axis='both', labelsize=12, labelweight='bold')

# Add legend with bold font
units_scatter, = ax.plot([], [], 'bo', label='NGN Units')  
ax.legend(fontsize=12, title_fontsize=12, title='Legend', fontweight='bold')

plt.show()


def init():
    units_scatter.set_data([], [])
    return units_scatter,

def update(frame):
    ng.train_step(frame)
    units_scatter.set_data(ng.units[:, 0], ng.units[:, 1])
    ax.set_title(f'Iteration {frame+1}/{ng.max_iter}')
    return units_scatter,

ani = FuncAnimation(fig, update, frames=range(ng.max_iter), init_func=init, blit=True, interval=50)

plt.show()
