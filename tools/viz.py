import ipywidgets as widgets
import xarray as xr
import numpy as np
import pyvista as pv
import pvxarray

from .pyvista_xarray_ext import PyVistaGlacierSource
from .texture import get_topo_texture


class Glacier3DViz:
    def __init__(
        self,
        dataset: xr.Dataset,
        ice_thickness: str,
        x: str = "x",
        y: str = "y",
        x_border: int = 100,
        y_border: int = 100,
        zoom: float = 1.,
        azimuth: float | None = None,
        elevation: float | None = None,
        roll: float | None = None,
        topo_bedrock: str = "bedrock",
        time: str = "time",
        time_display: str = "calendar_year"
    ):
        self.x = x
        self.y = y
        self.zoom = zoom
        self.azimuth = azimuth
        self.elevation = elevation
        self.roll = roll

        # resize map to given border values
        x_middle_point = int(len(dataset[self.x]) / 2)
        y_middle_point = int(len(dataset[self.y]) / 2)
        self.dataset = dataset.isel({self.x: slice(x_middle_point - x_border,
                                                   x_middle_point + x_border),
                                     self.y: slice(y_middle_point - y_border,
                                                   y_middle_point + y_border)
                                     }).load()

        # time_display for displaying total years only for monthly timeseries
        self.time = time
        self.time_display = time_display

        self.da_topo = self.dataset[topo_bedrock]
        self.da_glacier_surf = self.da_topo + self.dataset[ice_thickness]

        self.topo_texture = None
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
        topo_mesh = self.da_topo.pyvista.mesh(x=self.x, y=self.y)
        topo_mesh = topo_mesh.warp_by_scalar()
        topo_mesh.texture_map_to_plane(use_bounds=True, inplace=True)

        glacier_algo = PyVistaGlacierSource(self.da_glacier_surf,
                                            self.time,
                                            self.time_display)

        pl = pv.Plotter(
            window_size=[960, 720],
            border=False,
            lighting="three lights",
        )

        pl.add_mesh(topo_mesh, texture=self.topo_texture)
        pl.add_mesh(glacier_algo, color="#CCCCCC")

        pl.add_text(
            f"year: {glacier_algo.time:.0f}",
            position="upper_right",
            font_size=12,
            name="current_year",
        )

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
