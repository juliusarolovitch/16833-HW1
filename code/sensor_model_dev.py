#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
from map_reader import MapReader
from sensor_model import SensorModel

def main():
    parser = argparse.ArgumentParser(
        description="Click on a point in the map to ray trace and print the distances of the rays."
    )
    parser.add_argument('--path_to_map', default='../data/map/wean.dat',
                        help="Path to the occupancy map file")
    parser.add_argument('--theta', type=float, default=0.0,
                        help="Orientation angle (in degrees, counterclockwise from the positive x-axis) for ray tracing (default 0)")
    args = parser.parse_args()

    # Convert theta from degrees to radians.
    theta = math.radians(args.theta)
    print(f"Using orientation (theta) = {args.theta:.2f}° ({theta:.2f} rad)")

    # Load the map and initialize the sensor model.
    map_obj = MapReader(args.path_to_map)
    occupancy_map = map_obj.get_map()
    sensor_model = SensorModel(map_obj)
    resolution = sensor_model._map_resolution  # centimeters per cell

    # Display the occupancy map.
    plt.figure(figsize=(8, 8))
    plt.imshow(occupancy_map, cmap='Greys', origin='lower')
    plt.title("Click on a point to ray trace")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Let the user click on a point.
    print("Please click on a point on the map.")
    pts = plt.ginput(1)
    if not pts:
        print("No point selected. Exiting.")
        return
    # The clicked coordinates are in pixel units.
    selected_px, selected_py = pts[0]
    # Convert to world coordinates (centimeters).
    start_x = selected_px * resolution
    start_y = selected_py * resolution
    print(f"Selected point: x = {start_x:.2f} cm, y = {start_y:.2f} cm")
    
    # Compute sensor origin (25 cm ahead of the robot center).
    sensor_x = start_x + 25 * math.cos(theta)
    sensor_y = start_y + 25 * math.sin(theta)
    
    # Generate 180 ray angles (from -90° to +90° relative to heading).
    num_rays = 180
    angles = np.linspace(-np.pi/2, np.pi/2, num_rays)
    
    # Compute the hit locations using the sensor model's internal ray caster.
    x_hits, y_hits = sensor_model._ray_cast_all(start_x, start_y, theta, angles)
    
    # Compute distances from the sensor origin to the hit points.
    distances = np.sqrt((x_hits - sensor_x)**2 + (y_hits - sensor_y)**2)
    
    # Draw the sensor rays over the map.
    # Convert sensor and hit locations from centimeters back to pixel coordinates.
    sensor_px = sensor_x / resolution
    sensor_py = sensor_y / resolution
    for x_hit, y_hit in zip(x_hits, y_hits):
        hit_px = x_hit / resolution
        hit_py = y_hit / resolution
        plt.plot([sensor_px, hit_px], [sensor_py, hit_py], 'r-', linewidth=0.5)
    plt.scatter(sensor_px, sensor_py, c='blue', marker='o')
    plt.title("Ray Tracing from Selected Point")
    plt.show()
    
    # Print the distances of the 180 rays (from rightmost to leftmost).
    print("\nRay distances (from rightmost to leftmost):")
    for i, d in enumerate(distances):
        print(f"Ray {i+1:03d}: {d:.2f} cm")

if __name__ == '__main__':
    main()
