import numpy as np
import math


def get_camera_position_per_frame(viz_object, nr_frames, camera_trajectory,
                                  kwargs_camera_trajectory):
    """Calculates a camera position per frame, for different camera trajectories.

    viz_object: viz.Glacier3D object
        The Glacier3D object to be visualized.
    nr_frames: int
        Number of frames.
    camera_trajectory: str
        Type of camera movement. Depending on the type of camera trajectory
        different kwargs need to be defined (see kwargs_camera_trajectory).
        Options are:
        - `'linear'`: Moves the camera along a straight line.
        - `'rotate'`: Rotates the camera around the glacier.
    kwargs_camera_trajectory: dict
        Additional keyword arguments to customize the animation based on the
        selected camera trajectory:

        - For all trajectories:
            - normalized_coordinates: bool
                Are provided coordinates normalized?
                Default is True.
        - For `'linear'` trajectory:
            - 'start_point': tuple
                Start point of camera, by default in normalized coordinates.
            - 'end_point': tuple
                End point of camera, by default in normalized coordinates.
            Examples:
                - `(0, 0, 0)`: The topography center.
                - `(1, 0, 0)`: At the edge.

        - For `'rotate'` trajectory:
            - 'rotate_camera_start_and_end_angle': tuple
                Start and end angles for the camera. Range: 0 to 360.
                Example: `[200, 220]`.
            - 'rotate_camera_height': int optional
                The height of the rotated camera, multiplied by the elevation
                range. Defaults to 5.
            - 'rotate_camera_radius': int optional
                The radius of the rotated camera, multiplied by the map
                dimensions, if normalized_coordinates is True. Defaults to 1.
            - 'camera_radius_ref_axis': str optional
                The reference axis to be used to get a circle movement instead
                of an ellipse. Default is 'x'.
    """

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
        camera_height = kwargs_camera_trajectory.get('camera_height', 5)
        camera_radius = kwargs_camera_trajectory.get('camera_radius', 1)
        radius_reference_axis = kwargs_camera_trajectory.get(
            'camera_radius_ref_axis', 'x')

        # Calculate the angle for each frame to complete the rotation
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
                z_normalized=np.array(z_values),
                reference_axis=radius_reference_axis
            )

    else:
        raise NotImplementedError(f'Camera trajectory {camera_trajectory} not '
                                  f'implemented!')

    # Combine the x, y, and z values into a list of tuples representing each
    # frame's position
    return list(zip(x_values, y_values, z_values))
