import numpy as np
import xarray as xr
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def resize_ds(
        ds: xr.Dataset,
        x_crop: int | float | None = None,
        y_crop: int | float | None = None,
        x: str = "x",
        y: str = "y",
) -> xr.Dataset:
    """
    Resize a given dataset in a 'centered' manner, e.g. if the number of grid
    points(set via 'x_crop' and 'y_crop') is given as 200, the dataset is resized to 200 grid points with 100
    grid points on each side of the center. x_corp and y_crop can also be a crop factor.
    if x_crop is given as 0.5, the dataset is resized by half its width, always in a centered manner.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be resized.
    x_crop: float | int | None
        Number of grid points in x direction or x-crop factor between 0 and 1. If None, the complete extend is
        used.
    y_crop: float | int | None
        Number of grid points in y direction or y-crop factor between 0 and 1. If None, the complete extend is
        used.
    x: str
        Name of x coordinate of ds.
    y: str
        Name of y coordinate of ds.
    """
    # resize map to given extend, if None the complete extend is used


    if x_crop is not None:
        # doesn't make sense to use values bigger than 1.(100%) for cropping, as the data outside is not directly
        # available
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


def animate_multiple(viz_objects, filepath, common_title='glacier plots', shape=None,
                     framerate=10, quality=5, resolution_factor=1
                     ):
    """
    this functions allows to animate multiple 3Dviz objects in one video.
    """

    res_x, res_y = (960, 720)
    # check if length of viz_objects and shape is matching
    if shape is None:
        shape= (1, len(viz_objects))
    if len(shape) != 2:
        raise ValueError('Shape should be a 2D array or tuple!')
    n_elements = shape[0]*shape[1]
    if len(viz_objects) != n_elements:
        raise ValueError('Number of Glacier3DViz-Obj. should equal the number of plot-elements defined by \'shape\'!')

    res_x = res_x * shape[1] * resolution_factor
    res_y = res_y * shape[0] * resolution_factor
    # Initialize plotter for side-by-side visualization
    plotter = pv.Plotter(window_size=[res_x, res_y], shape=shape,  border=False, lighting='three lights', )
                         ## these weights can change the size(width and height) of the subplots in relation to
                         ## but also haven't helped with the 'centered title' issue. 
                         # row_weights=[0.3], col_weights=[0.3, 0.7])

    k=0
    plotters = [None] * n_elements
    glacier_algos=[None] * n_elements
    for i in range(shape[0]):
        for j in range(shape[1]):
            plotter.subplot(i, j)

            plotters[k], glacier_algos[k] = viz_objects[k].init_plotter(external_plotter=plotter)
            # plotter.add_mesh(plotters[k].mesh)
            # counter for the n-th viz_object
            k+=1
    # static plot

    # THIS HAS BEEN SUGGESTED AS FINAL OPTION BY CHATGPT TO GET A CENTERED, OVERLAYING HEADING
    # # Use vtkTextActor for title (outside subplots)
    # from vtkmodules.vtkRenderingCore import vtkTextActor
    # title_actor = vtkTextActor()
    # title_actor.SetInput("My Centered Title")
    # title_actor.GetTextProperty().SetFontSize(24)
    # title_actor.GetTextProperty().SetColor(0, 0, 0)  # Set color to black
    # title_actor.SetPosition2(200, 570)  # Adjust this for fine-tuning the position
    # title_actor.GetTextProperty().SetJustificationToCentered()
    # # Add title actor to the renderer (not to a specific subplot)
    # plotter.renderer.AddActor2D(title_actor)


    ## idea was to make the subplots smaller so that a centered title can be added on top,
    ## but the 'set_viewport' method is not available
    # plotter.subplot(0, 0)
    # plotter.set_viewport(0.0, 0.5, 0.0, 0.85)
    # plotter.subplot(0, 1)
    # plotter.set_viewport(0.5, 1.0, 0.0, 0.85)
    plotter.add_text("My Centered Title", position=(-0.2, 0.9), font_size=20, color='black',
                     viewport=True)
    plotter.show(auto_close=False, jupyter_backend="static")

    plotter.open_movie(filepath, framerate=framerate, quality=quality)
    for step in range(viz_objects[0].dataset[viz_objects[0].time].size):
        for i in glacier_algos:
            glacier_algos[i].time_step = step
            glacier_algos[i].update()

        plotter.update()
        plotter.write_frame()
    plotter.close()


