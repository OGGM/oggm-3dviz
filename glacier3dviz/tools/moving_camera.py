import numpy as np
import math


def get_camera_position_per_frame(viz_object, camera_trajectory, nr_frames,
                                  kwargs_camera_trajectory):

    if 'normalized_coordinates' in kwargs_camera_trajectory.keys():
        normalized_coords = kwargs_camera_trajectory['normalized_coordinates']
    else:
        normalized_coords = True

    # Create linearly spaced values for each axis from start to end points
    if camera_trajectory == 'linear':
        start_point = kwargs_camera_trajectory['start_point']
        end_point = kwargs_camera_trajectory['end_point']
        x_values = np.linspace(start_point[0], end_point[0], nr_frames)
        y_values = np.linspace(start_point[1], end_point[1], nr_frames)
        z_values = np.linspace(start_point[2], end_point[2], nr_frames)

        # Scale the values according to the scene range
        if normalized_coords:
            x_values, y_values, z_values = viz_object.get_absolute_coordinates(
                x_values, y_values, z_values)

    elif camera_trajectory == 'rotate':
        start_angle = kwargs_camera_trajectory['start_angle']
        end_angle = kwargs_camera_trajectory['end_angle']
        camera_height = kwargs_camera_trajectory['camera_height']
        camera_radius = kwargs_camera_trajectory['camera_radius']

        # Calculate the angle to increment for each frame to complete the rotation
        angle_per_frame = (end_angle - start_angle) / nr_frames

        # Initialize list to store camera positions
        x_values = []
        y_values = []
        z_values = []
        for i in range(nr_frames):
            # Convert the angle to radians for trigonometric functions
            radians = math.radians(start_angle + angle_per_frame * i)

            # Compute x and y position based on circular orbit formula
            x_values.append(math.sin(-radians) * camera_radius)
            y_values.append(math.cos(-radians) * camera_radius)
            z_values.append(camera_height)

        if normalized_coords:
            x_values, y_values, z_values = viz_object.get_absolute_coordinates(
                x_normalized=np.array(x_values),
                y_normalized=np.array(y_values),
                z_normalized=np.array(z_values)
            )

    else:
        raise NotImplementedError(f'Camera trajectory {camera_trajectory} not '
                                  f'implemented!')

    # Combine the x, y, and z values into a list of tuples representing each
    # frame's position
    return list(zip(x_values, y_values, z_values))
