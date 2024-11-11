import contextily as cx
import ipywidgets as widgets
import numpy as np
import xarray as xr
import pyvista as pv

from .pyvista_xarray_ext import PyVistaGlacierSource
from .texture import get_topo_texture
from .utils import (resize_ds, get_custom_colormap,
                    get_nice_thickness_colorbar_labels,
                    get_camera_position_per_frame)


class Glacier3DViz:
    def __init__(
        self,
        dataset: xr.Dataset,
        x: str = "x",
        y: str = "y",
        topo_bedrock: str = "bedrock",
        update_bedrock_with_time: bool = False,
        ice_thickness: str = 'simulated_thickness',
        time: str = "time",
        time_var_display: str = "calendar_year",
        x_crop: int | float | None = None,
        y_crop: int | float | None = None,
        additional_annotations: None | list = None,
        plotter_args: dict | None = None,
        add_mesh_topo_args: dict | None = None,
        add_mesh_ice_thick_args: dict | None = None,
        add_ice_thick_lookuptable_args: dict | None = None,
        use_texture: bool = False,
        show_topo_side_walls: bool = False,
        texture_args: dict | None = None,
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
        update_bedrock_with_time: bool
            update the bedrock with time
        ice_thickness: str
            name of the ice thickness in the dataset
        time: str
            name of the time coordinate in the dataset
        time_var_display: str
            name of the time coordinate in the dataset to be displayed
        x_crop: float| int | None
            number of grid points in x direction or crop factor between 0 and 1,
            if None the complete extend is used. See utils.resize_ds
        y_crop: float | int | None
            number of grid points in y direction or crop factor between 0 and 1,
            if None the complete extend is used. See utils.resize_ds
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
        add_ice_thick_lookuptable_args: dict | None
            additional arguments for the lookuptable when customizing the
            colorbar labels of the ice thickness, see pyvista.LookupTable
        use_texture: bool
            if True, a background texture is applied on the topography
        show_topo_side_walls: bool
            if True, the edges of the topography are set to the minimum elevation
            of the map, so that the map looks more like a solid.
        texture_args: dict | None
            additional arguments for the texture, see texture.get_topo_texture
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
        self.dataset = resize_ds(
            dataset, x_crop, y_crop)

        # time_display for displaying total years only
        self.time = time
        self.time_var_display = time_var_display

        # get topography
        if update_bedrock_with_time:
            raise NotImplementedError('Time update of bedrock not supported'
                                      'yet!')
        else:
            if len(self.dataset[self.topo_bedrock].coords) == 3:
                # ok the given topography has a time coordinate, just take the
                # first
                self.da_topo = self.dataset[self.topo_bedrock].isel(
                    {self.time: 0})
            else:
                self.da_topo = self.dataset[self.topo_bedrock]

        if show_topo_side_walls:
            min_elevation = np.min(self.da_topo)
            self.da_topo[0, :] = min_elevation
            self.da_topo[-1, :] = min_elevation
            self.da_topo[:, 0] = min_elevation
            self.da_topo[:, -1] = min_elevation

        # ignore ice thicknesses equal to 0
        self.da_glacier_thick = xr.where(
            self.dataset[ice_thickness] == 0.0,
            np.nan,
            self.dataset[ice_thickness])
        self.da_glacier_surf = self.da_topo + self.da_glacier_thick

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

        # add some default args for add_ice_thick_lookuptable_args
        if add_ice_thick_lookuptable_args is None:
            add_ice_thick_lookuptable_args = {}
        self.add_ice_thick_lookuptable_args_default = {}
        self.add_ice_thick_lookuptable_args_use = None

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

        # add some default texture args
        if texture_args is None:
            texture_args = {}
        self.texture_args_default = {}
        self.texture_args_use = None

        self.check_given_kwargs(
            set_default=True,
            plotter_args=plotter_args,
            add_mesh_topo_args=add_mesh_topo_args,
            add_mesh_ice_thick_args=add_mesh_ice_thick_args,
            add_ice_thick_lookuptable_args=add_ice_thick_lookuptable_args,
            text_time_args=text_time_args,
            light_args=light_args,
            background_args=background_args,
            camera_args=camera_args,
            texture_args=texture_args)

        # here we add and potentially download background map data
        # and apply it as the topographic texture
        if use_texture:
            self.set_topo_texture()

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
                'color', 'white')
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
            kwargs['add_mesh_ice_thick_args'].setdefault('scalar_bar_args', {})
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'title', 'Ice Thickness                      ')
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'n_labels', 0)
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'color', 'white')
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'vertical', True)
            kwargs['add_mesh_ice_thick_args']['scalar_bar_args'].setdefault(
                'position_x', 0.1)
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

        if 'add_ice_thick_lookuptable_args' in kwargs:
            kwargs['add_ice_thick_lookuptable_args'].setdefault(
                'cmap', get_custom_colormap('Blues'))
            kwargs['add_ice_thick_lookuptable_args'].setdefault(
                'n_labels', 5)
            max_value, annotations = get_nice_thickness_colorbar_labels(
                self.da_glacier_thick.max().load().item())
            kwargs['add_ice_thick_lookuptable_args'].setdefault(
                'scalar_range', [0.1, max_value])
            kwargs['add_ice_thick_lookuptable_args'].setdefault(
                'annotations', annotations)

            if set_default:
                self.add_ice_thick_lookuptable_args_default = \
                    kwargs['add_ice_thick_lookuptable_args']
                self.add_ice_thick_lookuptable_args_use = \
                    self.add_ice_thick_lookuptable_args_default
            else:
                self.add_ice_thick_lookuptable_args_use = \
                    kwargs['add_ice_thick_lookuptable_args']
        else:
            self.add_ice_thick_lookuptable_args_use = \
                self.add_ice_thick_lookuptable_args_default

        if 'text_time_args' in kwargs:
            kwargs['text_time_args'].setdefault('text', 'year: {:.0f}')
            kwargs['text_time_args'].setdefault('color', 'white')
            kwargs['text_time_args'].setdefault('position', 'upper_right')
            kwargs['text_time_args'].setdefault('viewport', True)
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
            kwargs['background_args'].setdefault('color', 'black')

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

        if 'texture_args' in kwargs:
            kwargs['texture_args'].setdefault('use_cache', True)
            kwargs['texture_args'].setdefault('background_source',
                                              cx.providers.Esri.WorldImagery)
            kwargs['texture_args'].setdefault('zoom_adjust', 1)
            kwargs['texture_args'].setdefault('remove_ice', True)

            if set_default:
                self.texture_args_default = kwargs['texture_args']
                self.texture_args_use = self.texture_args_default
            else:
                self.texture_args_use = kwargs['texture_args']
        else:
            self.texture_args_use = self.texture_args_default

    def set_topo_texture(self):
        bbox = (
            self.dataset[self.x].min().item(),
            self.dataset[self.y].min().item(),
            self.dataset[self.x].max().item(),
            self.dataset[self.y].max().item(),
        )

        srs = self.dataset.attrs["pyproj_srs"]

        self.add_mesh_topo_args_default['texture'] = get_topo_texture(
            bbox,
            srs=srs,
            **self.texture_args_use,
        )

    def _add_time_text(self, plotter, glacier_algo):
        text_actor_time = plotter.add_text(
            self.text_time_args_use['text'].format(glacier_algo.time_display),
            **{key: value
               for key, value in self.text_time_args_use.items()
               if key != 'text'}
        )
        # Center the text horizontally and vertically
        text_actor_time.GetTextProperty().SetJustificationToCentered()
        text_actor_time.GetTextProperty().SetVerticalJustificationToCentered()

    def _init_plotter(self, initial_time_step=0, external_plotter=None, **kwargs):
        self.check_given_kwargs(**kwargs)

        self.topo_mesh = self.da_topo.pyvista.mesh(x=self.x, y=self.y)
        self.topo_mesh = self.topo_mesh.warp_by_scalar()
        self.topo_mesh.texture_map_to_plane(use_bounds=True, inplace=True)

        glacier_algo = PyVistaGlacierSource(self.da_glacier_surf,
                                            self.da_glacier_thick,
                                            self.time,
                                            self.time_var_display,
                                            initial_time_step=initial_time_step)

        if external_plotter:
            pl = external_plotter
        else:
            pl = pv.Plotter(**self.plotter_args_use)

        # add topography with texture (color)
        pl.add_mesh(self.topo_mesh, **self.add_mesh_topo_args_use)

        # add glacier surface, colored by thickness, using custom colorbar
        custom_colorbar = pv.LookupTable(
            **{key: value
               for key, value in self.add_ice_thick_lookuptable_args_use.items()
               if key != 'n_labels'}
        )
        pl.add_mesh(glacier_algo, scalars='thickness',
                    cmap=custom_colorbar,
                    **self.add_mesh_ice_thick_args_use)

        # add text showing the current time
        self._add_time_text(pl, glacier_algo)

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

        self.glacier_algo = glacier_algo
        self.plotter = pl

        return pl, glacier_algo

    def init_plotter(self, initial_time_step=0, external_plotter=None,
                     **kwargs):
        pl, glacier_algo = self._init_plotter(
            initial_time_step=initial_time_step,
            external_plotter=external_plotter,
            **kwargs)
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
            # add text showing the current time
            self._add_time_text(plotter, glacier_algo)

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
        plotter, glacier_algo = self._init_plotter(**kwargs)
        return self._init_widgets(plotter, glacier_algo)

    def close(self):
        if self.widgets is not None:
            for w in self.widgets.values():
                w.close()
        if self.plotter is not None:
            self.plotter.close()

    def update_glacier(self, step, camera_position_per_step=None):
        self.glacier_algo.time_step = step
        self.glacier_algo.update()

        # add text showing the current time
        self._add_time_text(self.plotter, self.glacier_algo)

        if camera_position_per_step:
            self.plotter.camera.position = camera_position_per_step[step]

        self.plotter.update()

    def export_animation(self, filename="animation.mp4", framerate=10,
                         quality=5, moving_camera_start_and_end_point=None,
                         **kwargs):
        plotter, glacier_algo = self._init_plotter(**kwargs)

        if moving_camera_start_and_end_point:
            camera_position_per_frame = get_camera_position_per_frame(
                start_point=moving_camera_start_and_end_point[0],
                end_point=moving_camera_start_and_end_point[1],
                nr_frames=self.dataset[self.time].size,
            )
        else:
            camera_position_per_frame = None

        plotter.open_movie(filename, framerate=framerate, quality=quality)

        plotter.show(auto_close=False, jupyter_backend="static")

        for step in range(self.dataset[self.time].size):
            self.update_glacier(
                step,
                camera_position_per_step=camera_position_per_frame)
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

        plotter.close()
