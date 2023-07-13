import traceback

import pvxarray
from pvxarray.vtk_source import BaseSource
import xarray as xr


class PyVistaGlacierSource(BaseSource):
    """Simplified version of pvxarray.PyVistaXarraySource

    It adds a wrap_by_scalar filter to the generated pyvista mesh
    from the active DataArray (current selected time step).

    """
    def __init__(self, data_array):
        BaseSource.__init__(
            self,
            nOutputPorts=1,
            outputType="vtkStructuredGrid",
        )
        self._data_array = data_array
        self._time_step = 0

    @property
    def data_array(self):
        return self._data_array

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, step: int):
        # TODO: hook into the VTK pipeling to get requested time
        self._time_step = step
        self.Modified()

    @property
    def time(self):
        return float(self._data_array.time.isel(time=self.time_step))

    def RequestData(self, request, inInfo, outInfo):
        try:
            da = self.data_array.isel(time=self.time_step)
            mesh = da.pyvista.mesh(x="x", y="y").warp_by_scalar()
    
            pdo = self.GetOutputData(outInfo, 0)
            pdo.ShallowCopy(mesh)
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1