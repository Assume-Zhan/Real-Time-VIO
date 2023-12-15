import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing .npy files
dir_path = '../datasets/KITTI/pose_GT/'

# List all .npy files in the directory
npy_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]

# Create a new figure
plt.figure()

# Loop over all .npy files
for npy_file in npy_files:
    # Load the .npy file
    poses = np.load(os.path.join(dir_path, npy_file))

    # Extract x and y coordinates
    x = poses[:, 3]
    y = poses[:, 4]

    # Plot x and y coordinates
    plt.plot(x, y)

# Show the plot
plt.show()