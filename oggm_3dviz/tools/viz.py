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
        zoom: float = 1.,
        azimuth: float | None = None,
        elevation: float | None = None,
        roll: float | None = None,
        topo_bedrock: str = "bedrock",
        time: str = "time",
        time_display: str = "calendar_year",
        additional_annotations: None | list = None,
    ):
        # dataset coordinates
        self.x = x
        self.y = y
        self.topo_bedrock = topo_bedrock

        # camera settings
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.roll = roll

        self.additional_annotations = additional_annotations

        # resize map to given extend, if None the complete extend is used
        self.dataset = resize_ds_by_nr_of_grid_points(
            dataset, x_nr_of_grid_points, y_nr_of_grid_points)

        # time_display for displaying total years only
        self.time = time
        self.time_display = time_display

        self.da_topo = self.dataset[self.topo_bedrock]
        self.da_glacier_surf = self.da_topo + self.dataset[ice_thickness]

        self.topo_texture = None
        self.topo_mesh = None
        self.plotter = None
        self.glacier_algo = None
        self.widgets = None

    def set_topo_texture(self, use_cache: bool = False):
        bbox = [
            self.dataset[self.x].min(),
            self.dataset[self.x].max(),
            self.dataset[self.y].min(),
            self.dataset[self.y].max(),
        ]

        srs = self.dataset.attrs["pyproj_srs"]

        self.topo_texture = get_topo_texture(bbox, srs=srs, use_cache=use_cache)

    def _init_plotter(self):
        self.topo_mesh = self.da_topo.pyvista.mesh(x=self.x, y=self.y)
        self.topo_mesh = self.topo_mesh.warp_by_scalar()
        self.topo_mesh.texture_map_to_plane(use_bounds=True, inplace=True)

        glacier_algo = PyVistaGlacierSource(self.da_glacier_surf,
                                            self.time,
                                            self.time_display)

        pl = pv.Plotter(
            window_size=[960, 720],
            border=False,
            lighting="three lights",
        )

        pl.add_mesh(self.topo_mesh, texture=self.topo_texture)
        pl.add_mesh(glacier_algo, color="#CCCCCC")

        pl.add_text(
            f"year: {glacier_algo.time:.0f}",
            position="upper_right",
            font_size=12,
            name="current_year",
        )

        # here we add potential additional features
        if self.additional_annotations is not None:
            for annotation in self.additional_annotations:
                annotation.add_annotation(glacier_3dviz=self, plotter=pl)

        light = pv.Light(
            position=(0, 1, 1),
            light_type="scene light",
            intensity=0.6,
        )
        pl.add_light(light)

        pl.set_background("white", top="lightblue")

        return pl, glacier_algo

    def _init_widgets(self, plotter, glacier_algo):
        max_step = self.dataset[self.time].size - 1

        play = widgets.Play(
            value=0,
            min=0,
            max=max_step,
            step=1,
            interval=200,
            description="Press play",
            disabled=False,
        )
        slider = widgets.IntSlider(min=0, max=max_step, step=1)
        widgets.jslink((play, "value"), (slider, "value"))

        def update_glacier(change):
            glacier_algo.time_step = change["new"]
            glacier_algo.update()
            plotter.add_text(
                f"year: {glacier_algo.time:.0f}",
                position="upper_right",
                font_size=12,
                name="current_year",
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

        plotter.camera_position = self.plotter.camera_position
        plotter.camera.zoom(self.zoom)
        if self.azimuth is not None:
            plotter.camera.azimuth = self.azimuth
        if self.elevation is not None:
            plotter.camera.elevation = self.elevation
        if self.roll is not None:
            plotter.camera.roll = self.roll
        plotter.open_movie(filename, framerate=framerate)

        plotter.show(auto_close=False, jupyter_backend="static")

        for step in range(self.dataset[self.time].size):
            glacier_algo.time_step = step
            glacier_algo.update()
            plotter.add_text(
                f"year: {glacier_algo.time:.0f}",
                position="upper_right",
                font_size=12,
                name="current_year",
            )
            plotter.update()
            plotter.write_frame()

        plotter.close()
