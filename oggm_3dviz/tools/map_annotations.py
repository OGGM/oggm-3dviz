from pyproj import Proj
import numpy as np
import pyvista as pv

from . import viz
from .utils import check_color


class MapAnnotation(object):
    def __init__(self):
        pass

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter):
        raise NotImplementedError


class PointAnnotation(MapAnnotation):
    def __init__(
            self,
            latitude: float,
            longitude: float,
            height: float,
            text: str,
            **kwargs):
        """
        Add a point annotation to the map

        Parameters
        ----------
        latitude: float
            latitude of the point
        longitude: float
            longitude of the point
        height: float
            height of the point in meters
        text: str
            text to display at the point
        kwargs: dict
            additional keyword arguments for pv.Plotter.add_point_labels
        """
        super(PointAnnotation, self).__init__()

        self.latitude = latitude
        self.longitude = longitude
        self.height = height
        self.latitude_proj = None
        self.longitude_proj = None
        self.text = text
        self.kwargs = kwargs
        # set some defaults for add_point_labels kwargs
        self.kwargs.setdefault('font_size', 25)
        self.kwargs.setdefault('point_color', 'black')
        self.kwargs.setdefault('point_size', 10)
        self.kwargs.setdefault('shape', None)
        self.kwargs.setdefault('render_points_as_spheres', True)
        self.kwargs.setdefault('always_visible', True)

    def reproject_coords(self, glacier_3dviz: viz.Glacier3DViz):
        target_proj = Proj(glacier_3dviz.dataset.pyproj_srs)
        self.latitude_proj, self.longitude_proj = target_proj(
            self.longitude, self.latitude)

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter,
                       ):
        self.reproject_coords(glacier_3dviz)
        plotter.add_point_labels([
            (self.latitude_proj, self.longitude_proj, self.height)],
            [self.text],
            **self.kwargs)


class ArrowAnnotation(MapAnnotation):
    def __init__(self,
                 x_position: float = 1.05,
                 y_position: float = 0.5,
                 z_position: float = 0.5,
                 x_direction: float = 0.,
                 y_direction: float = 1.,
                 z_direction: float = 0.,
                 arrow_magnitude: float = 0.2,
                 text: str = 'N',
                 text_position_offset: list = None,
                 arrow_kwargs: dict | None = None,
                 text_kwargs: dict | None = None,
                 ):
        """
        Add an arrow annotation to the map

        Parameters
        ----------
        x_position: float
            position of the arrow in x direction, 0 is right, 1 is left
        y_position: float
            position of the arrow in y direction, 0 is back, 1 is front
        z_position: float
            position of the arrow in z direction, 0 is bottom, 1 is top
        x_direction: float
            direction of the arrow in x direction, if 0 the arrow is in
            yz-plane
        y_direction: float
            direction of the arrow in y direction, if 0 the arrow is in
            xz-plane
        z_direction: float
            direction of the arrow in z direction, if 0 the arrow is in
            xy-plane
        arrow_magnitude: float
            magnitude/size is relative to total length of y-axis, e.g. 0.5
            corresponds to half the length of the total y-axis
        text: str
            text to display at the arrow
        text_position_offset: list
            offset of the text from the arrow tip in [offset_x, offset_y,
            offset_z], it is given in relative length of the axis (e.g. 0.1 for
            offset_x means 10% of the total x-axis length,
            default is [-0.02, 0.01, 0]
        arrow_kwargs: dict
            additional keyword arguments for pv.Plotter.add_arrows
        text_kwargs: dict
            additional keyword arguments for pv.Plotter.add_point_labels
        """
        super(ArrowAnnotation, self).__init__()

        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position
        self.text = text
        if text_position_offset is None:
            text_position_offset = [-0.02, 0.01, 0]
        self.text_position_offset = text_position_offset
        self.absolute_text_offset = None
        self.arrow_cent = None

        self.arrow_direction = np.array([[x_direction, y_direction,
                                          z_direction]])

        # magnitude is given in relative length, needs to be calculated
        # depending on map
        self.arrow_magnitude_relative = arrow_magnitude
        self.arrow_magnitude = None

        # set some default kwargs for the arrow
        if arrow_kwargs is None:
            arrow_kwargs = {}
        arrow_kwargs.setdefault('show_scalar_bar', False)
        arrow_kwargs.setdefault('color', [0.2, 0.2, 0.2])
        self.arrow_kwargs = arrow_kwargs

        # set some default kwargs for the text
        if text_kwargs is None:
            text_kwargs = {}
        text_kwargs.setdefault('shape', None)
        text_kwargs.setdefault('show_points', False)
        text_kwargs.setdefault('font_size', 30)
        text_kwargs.setdefault('always_visible', True)
        self.text_kwargs = text_kwargs

    def set_arrow_position(self, glacier_3dviz: viz.Glacier3DViz):
        # calculate the position of the arrow, with (1, 0.5, 0.5) it is
        # located min_x, center_y, center_z
        def get_position(minimum, maximum, position):
            return minimum + position * (maximum - minimum)

        x_cent = get_position(
            glacier_3dviz.dataset[glacier_3dviz.x].min(),
            glacier_3dviz.dataset[glacier_3dviz.x].max(),
            self.x_position)
        y_cent = get_position(
            glacier_3dviz.dataset[glacier_3dviz.y].min(),
            glacier_3dviz.dataset[glacier_3dviz.y].max(),
            self.y_position)
        z_cent = get_position(
            glacier_3dviz.dataset[glacier_3dviz.topo_bedrock].min(),
            glacier_3dviz.dataset[glacier_3dviz.topo_bedrock].max(),
            self.z_position)
        self.arrow_cent = np.array([[x_cent, y_cent, z_cent]])

    def set_absolute_text_offset(self, glacier_3dviz: viz.Glacier3DViz):
        absolute_text_offset = np.array(self.text_position_offset)
        absolute_text_offset *= np.array([
            glacier_3dviz.dataset[glacier_3dviz.x].max().item() -
            glacier_3dviz.dataset[glacier_3dviz.x].min().item(),
            glacier_3dviz.dataset[glacier_3dviz.y].max().item() -
            glacier_3dviz.dataset[glacier_3dviz.y].min().item(),
            glacier_3dviz.dataset[glacier_3dviz.topo_bedrock].max().load().item() -
            glacier_3dviz.dataset[glacier_3dviz.topo_bedrock].min().load().item()
        ])
        self.absolute_text_offset = absolute_text_offset

    def set_arrow_magnitude(self, glacier_3dviz: viz.Glacier3DViz):
        self.arrow_magnitude = (
                (glacier_3dviz.dataset[glacier_3dviz.y].max().item() -
                 glacier_3dviz.dataset[glacier_3dviz.y].min().item()) *
                self.arrow_magnitude_relative)

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter,
                       ):
        # define arrow properties depending on the map
        self.set_arrow_position(glacier_3dviz)
        self.set_arrow_magnitude(glacier_3dviz)
        self.set_absolute_text_offset(glacier_3dviz)

        plotter.add_arrows(
            cent=self.arrow_cent,
            direction=self.arrow_direction,
            mag=self.arrow_magnitude,
            **self.arrow_kwargs,
        )

        plotter.add_point_labels(
            self.arrow_cent + self.arrow_direction *
            self.arrow_magnitude + self.absolute_text_offset,
            [self.text],
            **self.text_kwargs,
        )


class MaskAnnotation(MapAnnotation):
    def __init__(self,
                 mask_data: str = 'glacier_ext',
                 mask_color: str | list = 'black',
                 add_z: float = 0.,
                 ):
        """
        Adding a gridded mask to the map. All values equal 1 will be colored.

        Parameters
        ----------
        mask_data: str
            name of the mask data in the glacier_3dviz.dataset
        mask_color: str | list
            color of the mask, either a string or a list of rgba values in
            255 scale
        add_z: float
            Small increment to add to the height of used mesh. With this you
            can decide what is plotted on top of each other.
        """
        super(MaskAnnotation, self).__init__()

        self.mask_data = mask_data
        self.add_z = add_z

        # define color
        self.mask_color = check_color(mask_color)
        self.mask_texture = None

    def set_mask_texture(self, glacier_3dviz: viz.Glacier3DViz):
        # define the texture
        mask = glacier_3dviz.dataset[self.mask_data]
        mask_texture = np.zeros((*mask.shape, 4),
                                dtype=np.uint8)
        mask_texture[mask == 1, :] = self.mask_color
        self.mask_texture = pv.numpy_to_texture(mask_texture)

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter,
                       ):
        self.set_mask_texture(glacier_3dviz)

        topo_mesh = glacier_3dviz.topo_mesh.copy()
        topo_mesh.points[:, 2] += self.add_z

        plotter.add_mesh(topo_mesh,
                         texture=self.mask_texture)


class LegendAnnotation(MapAnnotation):
    def __init__(self,
                 labels: list | None = None,
                 **kwargs,
                 ):
        """
        Adding a legend to the map. This is a wrapper for
        pyvista.plotter.add_legend, see there for more details.

        Parameters
        ----------
        labels: list | None
            list of labels to be shown in the legend, each entry is a list
            with the label and the color, e.g. [['label1', 'black'], ['label2',
            'blue']]. If None, uses existing labels. Colors can be given as
            strings or as lists of rgba values in 255 scale.
        kwargs: dict
            additional keyword arguments for the legend, see
            pyvista.plotter.add_legend for more details.

        """
        super(LegendAnnotation, self).__init__()

        self.labels = labels
        # check if colors are valid or given as strings
        for label in self.labels:
            label[1] = check_color(label[1])

        # here setting some defaults for a single entry
        kwargs.setdefault('bcolor', (0.9, 0.9, 0.9))
        kwargs.setdefault('size', (0.2, 0.1))
        kwargs.setdefault('loc', 'lower center')
        kwargs.setdefault('face', 'rectangle')
        self.kwargs = kwargs

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter,
                       ):
        plotter.add_legend(labels=self.labels,
                           **self.kwargs)


class TextAnnotation(MapAnnotation):
    def __init__(
            self,
            text: str = '',
            position: str | list = 'upper_left',
            **kwargs):
        """
        Add a text annotation to the map

        Parameters
        ----------
        text: str
            text to display
        position: str | list
            Position to place the bottom left corner of the text box
        kwargs: dict
            additional keyword arguments for pv.Plotter.add_text
        """
        super(TextAnnotation, self).__init__()

        self.text = text
        self.position = position
        self.kwargs = kwargs

    def add_annotation(self,
                       glacier_3dviz: viz.Glacier3DViz,
                       plotter: pv.Plotter,
                       ):
        plotter.add_text(
            self.text,
            self.position,
            **self.kwargs)
