import ipywidgets as widgets
import xarray as xr
import pyvista as pv

from .pyvista_xarray_ext import PyVistaGlacierSource
from .texture import get_topo_texture
from .utils import resize_ds_by_nr_of_grid_points


class Glacier3DViz:
    def __init__(
        self,
        dataset: xr.Dataset,
        ice_thickness: str,
        x: str = "x",
        y: str = "y",
        x_nr_of_grid_points: int | None = None,
        y_nr_of_grid_points: int | None = None,
        topo_bedrock: str = "bedrock",
        time: str = "time",
        time_display: str = "calendar_year",
        additional_annotations: None | list = None,
        plotter_args: dict | None = None,
        add_mesh_topo_args: dict | None = None,
        use_sentinal_texture: bool = False,
        use_cache_for_sentinal: bool = True,
        add_mesh_glacier_args: dict | None = None,
        text_time_args: dict | None = None,
        light_args: dict | None = None,
        background_args: dict | None = None,
        camera_args: dict | None = None,
    ):
        # dataset coordinate names
        self.x = x
        self.y = y
        self.topo_bedrock = topo_bedrock

        self.additional_annotations = additional_annotations

        # resize map to given extend, if None the complete extend is used
        self.dataset = resize_ds_by_nr_of_grid_points(
            dataset, x_nr_of_grid_points, y_nr_of_grid_points)

        # time_display for displaying total years only
        self.time = time
        self.time_display = time_display

        self.da_topo = self.dataset[self.topo_bedrock]
        self.da_glacier_surf = self.da_topo + self.dataset[ice_thickness]

        # add some default args for the plotter
        if plotter_args is None:
            plotter_args = {}
        plotter_args.setdefault('window_size', [960, 720])
        plotter_args.setdefault('border', False)
        plotter_args.setdefault('lighting', 'three lights')
        self.plotter_args = plotter_args

        # add some default args for add_mesh_topo (cmap and colorbar)
        if add_mesh_topo_args is None:
            add_mesh_topo_args = {}
        add_mesh_topo_args.setdefault('cmap', 'terrain')
        add_mesh_topo_args.setdefault(
            'scalar_bar_args',
            {'vertical': True,
             'fmt': '%.0f m',
             'position_y': 0.3,
             'height': 0.4})
        add_mesh_topo_args.setdefault('show_scalar_bar', True)
        self.add_mesh_topo_args = add_mesh_topo_args

        # here we add and potentially download sentinal data
        if use_sentinal_texture:
            self.set_topo_texture(use_cache_for_sentinal)

        # add some default args for add_mesh_glacier (color)
        if add_mesh_glacier_args is None:
            add_mesh_glacier_args = {}
        add_mesh_glacier_args.setdefault('color', '#CCCCCC')
        self.add_mesh_glacier_args = add_mesh_glacier_args

        # add some default args for the time text
        if text_time_args is None:
            text_time_args = {}
        text_time_args.setdefault('text', 'year: {:.0f}')
        text_time_args.setdefault('position', 'upper_right')
        text_time_args.setdefault('font_size', 12)
        text_time_args.setdefault('name', 'current_year')  # for overwriting
        self.text_time_args = text_time_args

        # add some default args for light
        if light_args is None:
            light_args = {}
        light_args.setdefault('position', (0, 1, 1))
        light_args.setdefault('light_type', 'scene light')
        light_args.setdefault('intensity', 0.6)
        self.light_args = light_args

        # add some default args for background
        if background_args is None:
            background_args = {}
        background_args.setdefault('color', 'white')
        background_args.setdefault('top', 'lightblue')
        self.background_args = background_args

        # add some default camera args
        if camera_args is None:
            camera_args = {}
        camera_args.setdefault('zoom', 1)
        self.camera_args = camera_args

        self.topo_texture = None
        self.topo_mesh = None
        self.plotter = None
        self.glacier_algo = None
        self.widgets = None

    def set_topo_texture(self, use_cache: bool = False):
        bbox = (
            self.dataset[self.x].min().item(),
            self.dataset[self.x].max().item(),
            self.dataset[self.y].min().item(),
            self.dataset[self.y].max().item(),
        )

        srs = self.dataset.attrs["pyproj_srs"]

        self.add_mesh_topo_args['texture'] = get_topo_texture(
            bbox, srs=srs, use_cache=use_cache)

    def _init_plotter(self):
        self.topo_mesh = self.da_topo.pyvista.mesh(x=self.x, y=self.y)
        self.topo_mesh = self.topo_mesh.warp_by_scalar()
        self.topo_mesh.texture_map_to_plane(use_bounds=True, inplace=True)

        glacier_algo = PyVistaGlacierSource(self.da_glacier_surf,
                                            self.time,
                                            self.time_display)

        pl = pv.Plotter(**self.plotter_args)

        # add topography with texture (color)
        pl.add_mesh(self.topo_mesh, **self.add_mesh_topo_args)

        # add glacier surface
        pl.add_mesh(glacier_algo, **self.add_mesh_glacier_args)

        pl.add_text(self.text_time_args['text'].format(glacier_algo.time),
                    **{key: value
                       for key, value in self.text_time_args.items()
                       if key != 'text'}
                    )

        # here we add potential additional features
        if self.additional_annotations is not None:
            for annotation in self.additional_annotations:
                annotation.add_annotation(glacier_3dviz=self, plotter=pl)

        light = pv.Light(**self.light_args)
        pl.add_light(light)

        pl.set_background(**self.background_args)

        for key_cam, value_cam in self.camera_args.items():
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
                self.text_time_args['text'].format(glacier_algo.time),
                **{key: value
                   for key, value in self.text_time_args.items()
                   if key != 'text'}
            )

            plotter.update()

        slider.observe(update_glacier, names="value")

        output = widgets.Output()

        with output:
            plotter.show()

        main = widgets.VBox([widgets.HBox([play, slider]), output])

        self.widgets = {
            "play": play,
            "slider": slider,
            "output": output,
            "main": main,
        }

        return main

    def show(self):
        self.plotter, self.glacier_algo = self._init_plotter()
        return self._init_widgets(self.plotter, self.glacier_algo)

    def close(self):
        if self.widgets is not None:
            for w in self.widgets.values():
                w.close()
        if self.plotter is not None:
            self.plotter.close()

    def export_animation(self, filename="animation.mp4", framerate=10):
        plotter, glacier_algo = self._init_plotter()

        plotter.open_movie(filename, framerate=framerate)

        plotter.show(auto_close=False, jupyter_backend="static")

        for step in range(self.dataset[self.time].size):
            glacier_algo.time_step = step
            glacier_algo.update()
            plotter.add_text(
                self.text_time_args['text'].format(glacier_algo.time),
                **{key: value
                   for key, value in self.text_time_args.items()
                   if key != 'text'}
            )
            plotter.update()
            plotter.write_frame()

        plotter.close()
