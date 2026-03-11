#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import sys
import os

def plot_csv(filename):
    x_data = []
    y_data = []

    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) >= 3:
                    x_data.append(float(row[1]))
                    y_data.append(float(row[2]))
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return

    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(x_data, y_data, label='Robot Path', color='blue', linewidth=2)
    
    # Mark the Start and End points
    plt.scatter(x_data[0], y_data[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_data[-1], y_data[-1], color='red', s=100, label='End', zorder=5)

    # Make the graph visually scale to the real world
    plt.title(f'Robot Trajectory: {os.path.basename(filename)}')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')  # Keeps the X and Y scale 1:1
    plt.legend()
    
    # RDC FIX: Save as a PNG instead of displaying a popup
    output_image = filename.replace('.csv', '.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ Graph successfully saved as: {output_image}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 plot_trajectory.py <your_file.csv>")
    else:
        plot_csv(sys.argv[1])
