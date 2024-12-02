import numpy as np
import math


def get_camera_position_per_frame(viz_object, camera_trajectory, nr_frames,
                                  kwargs_camera_trajectory):
    # Create linearly spaced values for each axis from start to end points
    if camera_trajectory == 'linear':
        start_point = kwargs_camera_trajectory['start_point']
        end_point = kwargs_camera_trajectory['end_point']
        x_normalized = np.linspace(start_point[0], end_point[0], nr_frames)
        y_normalized = np.linspace(start_point[1], end_point[1], nr_frames)
        z_normalized = np.linspace(start_point[2], end_point[2], nr_frames)

        # Scale the values according to the scene range
        x_values, y_values, z_values = viz_object.get_absolute_coordinates(
            x_normalized, y_normalized, z_normalized)

    elif camera_trajectory == 'rotate':
        start_angle = kwargs_camera_trajectory['start_angle']
        end_angle = kwargs_camera_trajectory['end_angle']
        camera_height = kwargs_camera_trajectory['camera_height']
        camera_radius = kwargs_camera_trajectory['camera_radius']

        # Calculate the angle to increment for each frame to complete the rotation
        angle_per_frame = (end_angle - start_angle) / nr_frames

        # Initialize list to store camera positions
        x_normalized = []
        y_normalized = []
        z_normalized = []
        for i in range(nr_frames):
            # Convert the angle to radians for trigonometric functions
            radians = math.radians(start_angle + angle_per_frame * i)

            # Compute x and y position based on circular orbit formula
            x_normalized.append(math.sin(-radians) * camera_radius)
            y_normalized.append(math.cos(-radians) * camera_radius)
            z_normalized.append(camera_height)


        x_values, y_values, z_values = viz_object.get_absolute_coordinates(
            x_normalized=np.array(x_normalized),
            y_normalized=np.array(y_normalized),
            z_normalized=np.array(z_normalized)
        )

    else:
        start_point = kwargs_camera_trajectory['camera_point']


    # Combine the x, y, and z values into a list of tuples representing each frame's position
    return list(zip(x_values, y_values, z_values))


def get_rotating_camera_position(x_coordinates, y_coordinates, z_elevation,
                                 start_angle, end_angle, camera_height,
                                 camera_radius, nr_frames):
    # Calculate the range for each axis to scale the camera position
    x_range = abs(x_coordinates[-1] - x_coordinates[0])
    y_range = abs(y_coordinates[-1] - y_coordinates[0])
    z_range = abs(np.max(z_elevation) - np.min(z_elevation))

    # Calculate the angle to increment for each frame to complete the rotation
    angle_per_frame = (end_angle - start_angle) / nr_frames

    # Initialize list to store camera positions
    camera_positions = []
    for i in range(nr_frames):
        # Convert the angle to radians for trigonometric functions
        radians = math.radians(start_angle + angle_per_frame * i)

        # Compute x and y position based on circular orbit formula
        max_range = max(x_range, y_range)
        camera_x = math.sin(-radians) * max_range * camera_radius + np.mean(
            x_coordinates)
        camera_y = math.cos(-radians) * max_range * camera_radius + np.mean(
            y_coordinates)

        # Set z position based on specified height
        camera_z = z_range * camera_height

        # Append the calculated position for this frame
        camera_positions.append((camera_x, camera_y, camera_z))

    return camera_positions
