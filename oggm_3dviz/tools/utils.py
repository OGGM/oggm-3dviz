import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def resize_ds_by_nr_of_grid_points(
        ds: xr.Dataset,
        x_nr_of_grid_points: int | None = None,
        y_nr_of_grid_points: int | None = None,
        x: str = "x",
        y: str = "y",
) -> xr.Dataset:
    """
    Resize a given dataset in a 'centered' manner, e.g. if the number of grid
    points is given as 200, the dataset is resized to 200 grid points with 100
    grid points on each side of the center.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be resized.
    x_nr_of_grid_points: int | None
        Number of grid points in x direction. If None, the complete extend is
        used.
    y_nr_of_grid_points: int | None
        Number of grid points in y direction. If None, the complete extend is
        used.
    x: str
        Name of x coordinate of ds.
    y: str
        Name of y coordinate of ds.
    """
    # resize map to given extend, if None the complete extend is used
    if x_nr_of_grid_points is not None:
        x_middle_point = int(len(ds[x]) / 2)
        ds = ds.isel(
            {x: slice(x_middle_point - int(x_nr_of_grid_points / 2),
                      x_middle_point + int(x_nr_of_grid_points / 2))})
    if y_nr_of_grid_points is not None:
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
