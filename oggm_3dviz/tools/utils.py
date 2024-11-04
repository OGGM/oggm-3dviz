import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pyvista as pv
import vtk


def resize_ds(
        ds: xr.Dataset,
        x_crop: int | float | None = None,
        y_crop: int | float | None = None,
        x: str = "x",
        y: str = "y",
) -> xr.Dataset:
    """
    Resize a given dataset in a 'centered' manner, e.g. if the number of grid
    points(set via 'x_crop' and 'y_crop') is given as 200, the dataset is
    resized to 200 grid points with 100 grid points on each side of the center.
    x_corp and y_crop can also be a crop factor. If x_crop is given as 0.5, the
    dataset is resized by half its width, always in a centered manner.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be resized.
    x_crop: float | int | None
        Number of grid points in x direction or x-crop factor between 0 and 1.
        If None, the complete extend is
        used.
    y_crop: float | int | None
        Number of grid points in y direction or y-crop factor between 0 and 1.
        If None, the complete extend is
        used.
    x: str
        Name of x coordinate of ds.
    y: str
        Name of y coordinate of ds.
    """
    # resize map to given extend, if None the complete extend is used
    if x_crop is not None:
        # doesn't make sense to use values bigger than 1.(100%) for cropping,
        # as the data outside is not directly available
        if 0. < x_crop <= 1.:
            x_nr_of_grid_points = x_crop * len(ds[x])
        else:
            x_nr_of_grid_points = x_crop
        x_middle_point = int(len(ds[x]) / 2)
        ds = ds.isel(
            {x: slice(x_middle_point - int(x_nr_of_grid_points / 2),
                      x_middle_point + int(x_nr_of_grid_points / 2))})

    if y_crop is not None:
        if 0. < y_crop < 1.:
            y_nr_of_grid_points = y_crop * len(ds[y])
        else:
            y_nr_of_grid_points = y_crop
        y_middle_point = int(len(ds[y]) / 2)
        ds = ds.isel(
            {y: slice(y_middle_point - int(y_nr_of_grid_points / 2),
                      y_middle_point + int(y_nr_of_grid_points / 2))})

    return ds


def check_color(
        color: str | tuple | list,
) -> tuple:
    """
    Checks if color is given as RGBA in 255 convention. Further it converts
    some str colors into RGBA.

    Parameters
    ----------
    color: str | tuple | list
        Color for checking or converting.
    """
    if isinstance(color, tuple) or isinstance(color, list):
        if (len(color) != 4 or
                not all(isinstance(c, int) for c in color)):
            raise ValueError('The color of the labels should be a tuple of'
                             'four digits (RGBA) in 255 convention! '
                             f'(Given {color})')
        return color
    elif isinstance(color, str):
        color_dict = {
            'black': [0, 0, 0, 255],
            'blue': [21, 41, 153, 255],
        }
        if color in color_dict:
            return color_dict[color]
        else:
            raise ValueError(f"Color '{color}' not found in color_dict! "
                             f"Available colors: {list(color_dict.keys())}")


def get_custom_colormap(cmap):
    def extract_part_of_cmap(cmap, start, end):
        cmap_orig = plt.colormaps[cmap]
        subset_colors = cmap_orig(np.linspace(start, end, 256))
        return ListedColormap(subset_colors, name=f'{cmap}_custom')

    if cmap == 'gist_earth':
        return extract_part_of_cmap(cmap, 0.3, 0.95)
    elif cmap == 'Blues':
        return extract_part_of_cmap(cmap, 0.2, 1)
    else:
        raise NotImplementedError


def get_nice_thickness_colorbar_labels(max_value, min_value=0.1,
                                       n_labels=5, rounding_value=20):
    # nice labels, only ending in 0 and 5, n_labels and rounding_value must fit
    max_value = np.ceil(max_value / rounding_value) * rounding_value
    ticks = np.linspace(min_value, max_value, n_labels)

    def custom_label(value):
        if value == min_value:
            return f"{min_value} m"
        else:
            return f"{value:.0f} m"

    annotations = {tick: custom_label(tick) for tick in ticks}

    return max_value, annotations


def add_main_title_for_subplots(plotter, title, kwargs_title):
    # Create an overlay renderer that spans the entire window
    overlay_renderer = vtk.vtkRenderer()
    overlay_renderer.SetViewport(0, 0, 1, 1)  # Full window
    overlay_renderer.SetLayer(1)  # Ensure it's on top of other renderers
    overlay_renderer.InteractiveOff()  # Disable interactivity for the overlay

    # Create the text actor for the heading
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(title)

    # set some default arguments for the title
    if kwargs_title is None:
        kwargs_title = {}
    kwargs_title.setdefault('fontsize', 24)
    kwargs_title.setdefault('color', (0, 0, 0))  # Black color
    kwargs_title.setdefault('position', (0.5, 0.97))  # X=0.5 (center), Y=0.97 (near top)
    kwargs_title.setdefault('bold', True)

    # Configure the text properties
    text_prop = text_actor.GetTextProperty()
    text_prop.SetFontSize(kwargs_title['fontsize'])
    text_prop.SetColor(kwargs_title['color'][0],
                       kwargs_title['color'][1],
                       kwargs_title['color'][2])
    if kwargs_title['bold']:
        text_prop.SetBold(True)
    text_prop.SetJustificationToCentered()
    text_prop.SetVerticalJustificationToTop()

    # Position the text actor using normalized viewport coordinates
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(kwargs_title['position'][0],
                           kwargs_title['position'][1])

    # Add the text actor to the overlay renderer
    overlay_renderer.AddActor(text_actor)

    # Access the render window and set the number of layers
    render_window = plotter.ren_win
    render_window.SetNumberOfLayers(2)  # Default is 1; we need at least 2 for the overlay

    # Add the overlay renderer to the render window
    render_window.AddRenderer(overlay_renderer)


def side_by_side_visualization(
        viz_objects: list,
        shape: tuple | None = None,
        filename_animation: str | None = "side_by_side_animation.mp4",
        filename_plot: str | None = None,
        plot_year: int | None = None,
        title: str | None = None,
        kwargs_title: dict | None = None,
        kwargs_plotter: dict | None = None,
        kwargs_light: dict | None = None,
        kwargs_screenshot: dict | None = None,
        framerate: int = 10,
        quality: int = 5,
):
    """
    Function for creating side by side animation and/or plots. It is assumed
    that they all use the same time. If a shape is provided it is not checked if
    it fits with the provided number of viz_objects.

    Parameters
    ----------
    viz_objects : list
        A list of Glacier3DViz objects to be shown side by side.
    shape : tuple
        Defines the arrangement of the provided viz_objects. By default, all
        will be arranged in one row (1, len(viz_objects)).
    filename_animation : str | None
        If not None, an animation is created and saved under this path. Default
        is None.
    filename_plot : str | None
        If not None, a plotter is created and saved under this path. Default is
        None.
    plot_year : int | None
        Define the year which should be plotted and saved under filename_plot.
        Default is to use the first year.
    title : str | None
        Add a centered title over all subplots. Default is None.
    kwargs_title : dict | None
        Kwargs passed to add_main_title_for_subplots, adding the main title.
        Options are 'fontsize', 'color' (R, G, B), 'position' and 'bold'.
    kwargs_plotter : dict | None
        Additional arguments for the pyvista plotter, see pyvista.Plotter.
    kwargs_light : dict | none
        Additional arguments for the pyvista light, see pyvista.Light.
    kwargs_screenshot : dict | None
        Additional arguments for the pyvista screenshot, see
        pyvista.Plotter.Screenshot.
    framerate : int
        Framerate for pyvista.Plotter.open_movie. Default is 10.
    quality : int
        Quality for pyvista.Plotter.open_movie. Default is 5.
    """

    # by default, we only use one row
    if shape is None:
        shape = (1, len(viz_objects))

    # create tuples for easy access of subplots
    subplots_ij = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            subplots_ij.append((i, j))

    # set some default plotter arguments
    if kwargs_plotter is None:
        kwargs_plotter = {}
    # defined window_size is used for each viz_object
    kwargs_plotter.setdefault('window_size', [960, 720])
    kwargs_plotter.setdefault('border', False)
    kwargs_plotter.setdefault('lighting', 'three lights')

    # set some additional light settings
    if kwargs_light is None:
        kwargs_light = {}
    kwargs_light.setdefault('position', (0, 1, 1))
    kwargs_light.setdefault('light_type', 'scene light')
    kwargs_light.setdefault('intensity', -0.3)

    # adapt that each element has the defined window size
    window_size_x = kwargs_plotter['window_size'][0] * shape[1]
    window_size_y = kwargs_plotter['window_size'][1] * shape[0]

    def init_side_by_side_plotter(initial_time_step=0):
        # initialize main plotter
        plotter = pv.Plotter(window_size=[window_size_x, window_size_y],
                             shape=shape,
                             **{key: value
                                for key, value in kwargs_plotter.items()
                                if key != 'window_size'}
                             )

        # initialize the individual plots with the provided viz_objects
        for viz_obj, (i, j) in zip(viz_objects, subplots_ij):
            plotter.subplot(i, j)

            light = pv.Light(**kwargs_light)
            plotter.add_light(light)

            _ = viz_obj.init_plotter(initial_time_step=initial_time_step,
                                     external_plotter=plotter,
                                     )

        # add a title, if provided
        if title:
            add_main_title_for_subplots(plotter, title, kwargs_title, )

        return plotter

    # create an animation
    if filename_animation:
        plotter = init_side_by_side_plotter()
        # create the movie frame by frame
        plotter.open_movie(filename_animation, framerate=framerate,
                           quality=quality)
        plotter.show(auto_close=False, jupyter_backend="static")
        for step in range(viz_objects[0].dataset[viz_objects[0].time].size):
            for viz_obj, (i, j) in zip(viz_objects, subplots_ij):
                plotter.subplot(i, j)
                viz_obj.update_glacier(step)

            plotter.write_frame()
        plotter.close()

    # only saving an image for one year
    if filename_plot:
        if plot_year:
            time_diff = np.abs(
                viz_objects[0].dataset[viz_objects[0].time].values - plot_year)
            time_index = np.argmin(time_diff)
        else:
            time_index = 0

        plotter = init_side_by_side_plotter(initial_time_step=time_index)
        if kwargs_screenshot is None:
            kwargs_screenshot = {}
        plotter.screenshot(filename_plot, **kwargs_screenshot)
        plotter.close()
