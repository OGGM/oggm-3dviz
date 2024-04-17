import ipywidgets as widgets
import numpy as np
import xarray as xr
import pyvista as pv

from .pyvista_xarray_ext import PyVistaGlacierSource
from .texture import get_topo_texture
from .utils import resize_ds_by_nr_of_grid_points, get_custom_colormap


class Glacier3DViz:
    def __init__(
        self,
        dataset: xr.Dataset,
        x: str = "x",
        y: str = "y",
        topo_bedrock: str = "bedrock",
        ice_thickness: str = 'simulated_thickness',
        time: str = "time",
        time_var_display: str = "calendar_year",
        x_nr_of_grid_points: int | None = None,
        y_nr_of_grid_points: int | None = None,
        additional_annotations: None | list = None,
        plotter_args: dict | None = None,
        add_mesh_topo_args: dict | None = None,
        add_mesh_ice_thick_args: dict | None = None,
        use_satellite_texture: bool = False,
        use_cache_for_satellite: bool = True,
        text_time_args: dict | None = None,
        light_args: dict | None = None,
        background_args: dict | None = None,
        camera_args: dict | None = None,
    ):
        """Class to visualize a glacier in 3D with pyvista.

        Parameters
        ----------
        dataset: xr.Dataset
            dataset containing the glacier data (topography, ice thickness,
            possibly more like outlines, etc.)
        x: str
            name of the x coordinate in the dataset
        y: str
            name of the y coordinate in the dataset
        topo_bedrock: str
            name of the topography in the dataset
        ice_thickness: str
            name of the ice thickness in the dataset
        time: str
            name of the time coordinate in the dataset
        time_var_display: str
            name of the time coordinate in the dataset to be displayed
        x_nr_of_grid_points: int | None
            number of grid points in x direction, if None the complete extend
            is used. See utils.resize_ds_by_nr_of_grid_points
        y_nr_of_grid_points: int | None
            number of grid points in y direction, if None the complete extend
            is used. See utils.resize_ds_by_nr_of_grid_points
        additional_annotations: None | list
            list of additional annotations to be added to the map, see
            oggm_3dviz.tools.map_annotations
        plotter_args: dict | None
            additional arguments for the pyvista plotter, see pyvista.Plotter
        add_mesh_topo_args: dict | None
            additional arguments for the mesh when adding topo_bedrock to the
            plotter, see pyvista.Plotter.add_mesh
        add_mesh_ice_thick_args: dict | None
            additional arguments for the mesh when adding ice_thickness to the
            plotter, see pyvista.Plotter.add_mesh
        use_satellite_texture: bool
            if True, a satellite texture is used for the topography
        use_cache_for_satellite: bool
            if True, satellite texture is cached,
            see texture.get_topo_texture
        text_time_args: dict | None
            additional arguments for the time text, at least it must contain
            'time' with a string on which .format(current_year) can be applied,
            e.g. 'time': 'year: {}', for other options see
            pyvista.Plotter.add_text
        light_args: dict | None
            additional arguments for the light, see pyvista.Light
        background_args: dict | None
            additional arguments for the background, see
            pyvista.plotter.set_background
        camera_args: dict | None
            additional arguments for the camera, see pyvista.Plotter.camera
        """
        # dataset coordinate names
        self.x = x
        self.y = y
        self.topo_bedrock = topo_bedrock

        self.additional_annotations_default = additional_annotations
        self.additional_annotations_use = additional_annotations

        # resize map to given extend, if None the complete extend is used
        self.dataset = resize_ds_by_nr_of_grid_points(
            dataset, x_nr_of_grid_points, y_nr_of_grid_points)

        # time_display for displaying total years only
        self.time = time
        self.time_var_display = time_var_display

        self.da_topo = self.dataset[self.topo_bedrock]
        self.da_glacier_surf = self.da_topo + self.dataset[ice_thickness]
        self.da_glacier_thick = self.dataset[ice_thickness]

        # add some default args for the plotter
        if plotter_args is None:
            plotter_args = {}
        self.plotter_args_default = {}
        self.plotter_args_use = None

        # add some default args for add_mesh_topo (cmap and colorbar)
        if add_mesh_topo_args is None:
            add_mesh_topo_args = {}
        self.add_mesh_topo_args_default = {}
        self.add_mesh_topo_args_use = None

        # add some default args for add_mesh_ice_thickness (cmap and colorbar)
        if add_mesh_ice_thick_args is None:
            add_mesh_ice_thick_args = {}
        self.add_mesh_ice_thick_args_default = {}
        self.add_mesh_ice_thick_args_use = None

        # add some default args for the time text
        if text_time_args is None:
            text_time_args = {}
        self.text_time_args_default = {}
        self.text_time_args_use = None

        # add some default args for light
        if light_args is None:
            light_args = {}
        self.light_args_default = {}
        self.light_args_use = None

        # add some default args for background
        if background_args is None:
            background_args = {}
        self.background_args_default = {}
        self.background_args_use = None

        # add some default camera args
        if camera_args is None:
            camera_args = {}
        self.camera_args_default = {}
        self.camera_args_use = None

        self.check_given_kwargs(set_default=True,
                                plotter_args=plotter_args,
                                add_mesh_topo_args=add_mesh_topo_args,
                                add_mesh_ice_thick_args=add_mesh_ice_thick_args,
                                text_time_args=text_time_args,
                                light_args=light_args,
                                background_args=background_args,
                                camera_args=camera_args)

        # here we add and potentially download satellite data
        if use_satellite_texture:
            self.set_topo_texture(use_cache_for_satellite)

        self.topo_mesh = None
        self.plotter = None
        self.glacier_algo = None
        self.widgets = None

    def check_given_kwargs(self, set_default=False, **kwargs):
        """Check and set the given kwargs.
        It is used to set the default values and makes it possible when calling
        .show or .plot_year or .export_animation to change the default values
        for more customization.
        """
        if 'additional_annotations' in kwargs:
            self.additional_annotations_use = kwargs['additional_annotations']
        else:
            self.additional_annotations_use = \
                self.additional_annotations_default

        if 'plotter_args' in kwargs:
            kwargs['plotter_args'].setdefault('window_size', [960, 720])
            kwargs['plotter_args'].setdefault('border', False)
            kwargs['plotter_args'].setdefault('lighting', 'three lights')

            if set_default:
                self.plotter_args_default = kwargs['plotter_args']
                self.plotter_args_use = self.plotter_args_default
            else:
                self.plotter_args_use = kwargs['plotter_args']
        else:
            self.plotter_args_use = self.plotter_args_default

        if 'add_mesh_topo_args' in kwargs:
            kwargs['add_mesh_topo_args'].setdefault(
                'cmap', get_custom_colormap('gist_earth'))
            kwargs['add_mesh_topo_args'].setdefault('scalar_bar_args', {})
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'title', 'Bedrock')
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'vertical', True)
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'fmt', '%.0f m')
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'position_x', 0.9)
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'position_y', 0.3)
            kwargs['add_mesh_topo_args']['scalar_bar_args'].setdefault(
                'height', 0.4)
            kwargs['add_mesh_topo_args'].setdefault('show_scalar_bar', True)

            if set_default:
                self.add_mesh_topo_args_default = kwargs['add_mesh_topo_args']
                self.add_mesh_topo_args_default.setdefault('texture', None)
                self.add_mesh_topo_args_use = self.add_mesh_topo_args_default
            else:
                self.add_mesh_topo_args_use = kwargs['add_mesh_topo_args']
        else:
            self.add_mesh_topo_args_use = self.add_mesh_topo_args_default
        self.add_mesh_topo_args_use['texture'] = \
            self.add_mesh_topo_args_default['texture']

        if 'add_mesh_ice_thick_args' in kwargs:
            kwargs['add_mesh_ice_thick_args'].setdefault(
                'cmap', get_custom_colormap('Blues'))
            kwargs['add_mesh_ice_thick_args'].setdefault(
                'clim', [0.1, self.da_glacier_thick.max().item()])
            kwargs['add_mesh_ice_thick_args'].setdefault('scalar_bar_args', {})
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'title', 'Ice Thickness')
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'vertical', True)
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'fmt', '%.1f m')
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'position_x', 0.03)
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'position_y', 0.3)
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'height', 0.4)
            kwargs['add_mesh_ice_thick_args'].setdefault(
                'show_scalar_bar', True)

            if set_default:
                self.add_mesh_ice_thick_args_default = \
                    kwargs['add_mesh_ice_thick_args']
                self.add_mesh_ice_thick_args_use = \
                    self.add_mesh_ice_thick_args_default
            else:
                self.add_mesh_ice_thick_args_use = \
                    kwargs['add_mesh_ice_thick_args']
        else:
            self.add_mesh_ice_thick_args_use = \
                self.add_mesh_ice_thick_args_default

        if 'text_time_args' in kwargs:
            kwargs['text_time_args'].setdefault('text', 'year: {:.0f}')
            kwargs['text_time_args'].setdefault('position', 'upper_right')
            kwargs['text_time_args'].setdefault('font_size', 12)
            # name for overwriting when updating time
            kwargs['text_time_args'].setdefault('name', 'current_year')

            if set_default:
                self.text_time_args_default = kwargs['text_time_args']
                self.text_time_args_use = self.text_time_args_default
            else:
                self.text_time_args_use = kwargs['text_time_args']
        else:
            self.text_time_args_use = self.text_time_args_default

        if 'light_args' in kwargs:
            kwargs['light_args'].setdefault('position', (0, 1, 1))
            kwargs['light_args'].setdefault('light_type', 'scene light')
            kwargs['light_args'].setdefault('intensity', 0.6)

            if set_default:
                self.light_args_default = kwargs['light_args']
                self.light_args_use = self.light_args_default
            else:
                self.light_args_use = kwargs['light_args']
        else:
            self.light_args_use = self.light_args_default

        if 'background_args' in kwargs:
            kwargs['background_args'].setdefault('color', 'white')
            kwargs['background_args'].setdefault('top', 'lightblue')

            if set_default:
                self.background_args_default = kwargs['background_args']
                self.background_args_use = self.background_args_default
            else:
                self.background_args_use = kwargs['background_args']
        else:
            self.background_args_use = self.background_args_default

        if 'camera_args' in kwargs:
            kwargs['camera_args'].setdefault('zoom', 1.)

            if set_default:
                self.camera_args_default = kwargs['camera_args']
                self.camera_args_use = self.camera_args_default
            else:
                self.camera_args_use = kwargs['camera_args']
        else:
            self.camera_args_use = self.camera_args_default

    def set_topo_texture(self, use_cache: bool = False):
        bbox = (
            self.dataset[self.x].min().item(),
            self.dataset[self.x].max().item(),
            self.dataset[self.y].min().item(),
            self.dataset[self.y].max().item(),
        )

        srs = self.dataset.attrs["pyproj_srs"]

        self.add_mesh_topo_args_default['texture'] = get_topo_texture(
            bbox, srs=srs, use_cache=use_cache)

    def _init_plotter(self, inital_time_step=0, **kwargs):
        self.check_given_kwargs(**kwargs)

        self.topo_mesh = self.da_topo.pyvista.mesh(x=self.x, y=self.y)
        self.topo_mesh = self.topo_mesh.warp_by_scalar()
        self.topo_mesh.texture_map_to_plane(use_bounds=True, inplace=True)

        glacier_algo = PyVistaGlacierSource(self.da_glacier_surf,
                                            self.da_glacier_thick,
                                            self.time,
                                            self.time_var_display,
                                            initial_time_step=inital_time_step)

        pl = pv.Plotter(**self.plotter_args_use)

        # add topography with texture (color)
        pl.add_mesh(self.topo_mesh, **self.add_mesh_topo_args_use)

        # add glacier surface, colored by thickness
        pl.add_mesh(glacier_algo, scalars='thickness',
                    **self.add_mesh_ice_thick_args_use)

        pl.add_text(
            self.text_time_args_use['text'].format(glacier_algo.time_display),
            **{key: value
               for key, value in self.text_time_args_use.items()
               if key != 'text'}
        )

        # here we add potential additional features
        if self.additional_annotations_use is not None:
            for annotation in self.additional_annotations_use:
                annotation.add_annotation(glacier_3dviz=self, plotter=pl)

        light = pv.Light(**self.light_args_use)
        pl.add_light(light)

        pl.set_background(**self.background_args_use)

        for key_cam, value_cam in self.camera_args_use.items():
            if key_cam == 'zoom':
                # zoom is not working with setattr()
                pl.camera.zoom(value_cam)
            else:
                setattr(pl.camera, key_cam, value_cam)

        return pl, glacier_algo

    def _init_widgets(self, plotter, glacier_algo):
        max_step = self.dataset[self.time].size - 1

        play = widgets.Play(
            value=0,
            min=0,
            max=max_step,
            step=1,
            interval=max_step,
            description="Press play",
            disabled=False,
        )
        slider = widgets.IntSlider(min=0, max=max_step, step=1)
        widgets.jslink((play, "value"), (slider, "value"))

        def update_glacier(change):
            glacier_algo.time_step = change["new"]
            glacier_algo.update()
            plotter.add_text(
                self.text_time_args_use['text'].format(glacier_algo.time_display),
                **{key: value
                   for key, value in self.text_time_args_use.items()
                   if key != 'text'}
            )

            plotter.update()

        slider.observe(update_glacier, names="value")

        output = widgets.Output()

        with output:
            plotter.show(jupyter_backend='trame')

        main = widgets.VBox([widgets.HBox([play, slider]), output])

        self.widgets = {
            "play": play,
            "slider": slider,
            "output": output,
            "main": main,
        }

        return main

    def show(self, **kwargs):
        self.plotter, self.glacier_algo = self._init_plotter(**kwargs)
        return self._init_widgets(self.plotter, self.glacier_algo)

    def close(self):
        if self.widgets is not None:
            for w in self.widgets.values():
                w.close()
        if self.plotter is not None:
            self.plotter.close()

    def export_animation(self, filename="animation.mp4", framerate=10,
                         **kwargs):
        plotter, glacier_algo = self._init_plotter(**kwargs)

        plotter.open_movie(filename, framerate=framerate)

        plotter.show(auto_close=False, jupyter_backend="static")

        for step in range(self.dataset[self.time].size):
            glacier_algo.time_step = step
            glacier_algo.update()
            plotter.add_text(
                self.text_time_args_use['text'].format(glacier_algo.time_display),
                **{key: value
                   for key, value in self.text_time_args_use.items()
                   if key != 'text'}
            )
            plotter.update()
            plotter.write_frame()

        plotter.close()

    def plot_year(self, time_given, filepath=None, show_plot=True,
                  kwargs_screenshot=None, **kwargs):
        # find index of closest time stamp matching the given time
        time_diff = np.abs(self.dataset[self.time].values - time_given)
        time_index = np.argmin(time_diff)

        plotter, glacier_algo = self._init_plotter(inital_time_step=time_index,
                                                   **kwargs)

        if show_plot:
            plotter.show(jupyter_backend="static")

        if filepath is not None:
            if kwargs_screenshot is None:
                kwargs_screenshot = {}
            plotter.screenshot(filepath, **kwargs_screenshot)
