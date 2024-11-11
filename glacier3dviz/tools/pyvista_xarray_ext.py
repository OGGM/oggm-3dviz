import traceback

import pvxarray
from pvxarray.vtk_source import BaseSource
import xarray as xr


class PyVistaGlacierSource(BaseSource):
    """Simplified version of pvxarray.PyVistaXarraySource

    It adds a wrap_by_scalar filter to the generated pyvista mesh
    from the active DataArray (current selected time step).

    """

    def __init__(self, data_array_glacier_surf,
                 data_array_glacier_thick,
                 time_var_main,
                 time_var_display,
                 initial_time_step=0):
        BaseSource.__init__(
            self,
            nOutputPorts=1,
            outputType="vtkStructuredGrid",
        )
        self._data_array_glacier_surf = data_array_glacier_surf
        self._data_array_glacier_thick = data_array_glacier_thick
        self._time_step = initial_time_step
        self.time_var_main = time_var_main
        self.time_var_display = time_var_display

    @property
    def data_array_glacier_surf(self):
        return self._data_array_glacier_surf

    @property
    def data_array_glacier_thick(self):
        return self._data_array_glacier_thick

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, step: int):
        # TODO: hook into the VTK pipeling to get requested time
        self._time_step = step
        self.Modified()

    @property
    def time_display(self):
        return float(
            self._data_array_glacier_surf[self.time_var_display].isel(
                {self.time_var_main: self.time_step}))

    @property
    def time(self):
        return self._data_array_glacier_surf[self.time_var_main]

    def RequestData(self, request, inInfo, outInfo):
        try:
            da = self.data_array_glacier_surf.isel(
                {self.time_var_main: self.time_step})
            mesh = da.pyvista.mesh(x="x", y="y").warp_by_scalar()
            mesh['thickness'] = self.data_array_glacier_thick.isel(
                {self.time_var_main: self.time_step}).values.flatten()

            pdo = self.GetOutputData(outInfo, 0)
            pdo.ShallowCopy(mesh)
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1
