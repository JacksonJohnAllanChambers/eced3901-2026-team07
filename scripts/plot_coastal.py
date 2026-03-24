#!/usr/bin/env python3

import csv
import glob
import os
import matplotlib.pyplot as plt

def main():
    # Automatically find the newest coastal CSV file in the current directory
    csv_files = glob.glob('real_coastal_trajectory_*.csv')
    if not csv_files:
        print("❌ No coastal trajectory CSV files found in this directory!")
        return

    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"📊 Plotting data from: {latest_csv}")

    x_vals = []
    y_vals = []

    # Read the data
    with open(latest_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            x_vals.append(float(row[1]))
            y_vals.append(float(row[2]))

    # Create the plot
    plt.figure(figsize=(6, 8))
    plt.plot(x_vals, y_vals, marker='.', markersize=4, linestyle='-', color='blue', label='Robot Odometry')

    # Draw the boundary brackets we tuned for the Left Coastal lane
    plt.axvline(x=0.86, color='red', linestyle='--', alpha=0.5, label='Left Gap Limit (0.86)')
    plt.axvline(x=0.96, color='green', linestyle='--', alpha=0.5, label='Right Gap Limit (0.96)')

    # Formatting
    plt.title('ECED3901 Coastal Navigation Verification')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True)
    plt.legend()
    
    # Force the axes to scale equally so the map doesn't look squished or stretched
    plt.axis('equal') 

    # Save the output PNG alongside the CSV
    output_img = latest_csv.replace('.csv', '_plot.png')
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully as: {output_img}")
    
    # Display the plot on the screen
    plt.show()

if __name__ == '__main__':
    main()
