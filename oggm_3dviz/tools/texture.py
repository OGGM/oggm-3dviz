from typing import Any

import contextily as cx
import numpy as np
import pyproj
import skimage
import pyvista as pv
import xarray as xr
from skimage.exposure.exposure import rescale_intensity
from skimage.util import random_noise


def _ice_to_bedrock(
    img: np.ndarray, intensity_threshold: float = 0.45, noise_scale: float = 2e-4
) -> np.ndarray:
    """Simplistic processing to attenuate the ice texture
    and turn it into grey rock.

    Convert RGB to HSV, extract the intensity (V), rescale the
    highest intensity values, add some noise for a slightly more
    realistic bedrock texture and convert back into RGB.

    """
    img_hsv = skimage.color.rgb2hsv(img)
    hue = img_hsv[:, :, 0]
    saturation = img_hsv[:, :, 1]
    value = img_hsv[:, :, 2]

    noise_vars = np.where(value > intensity_threshold, noise_scale, 1e-10)

    value -= rescale_intensity(
        value,
        in_range=(intensity_threshold, 1.0),
        out_range=(0.0, intensity_threshold),
    )
    value = random_noise(value, mode="localvar", local_vars=noise_vars)

    new_hsv = np.stack([hue, saturation, value], axis=-1)
    new_rgb = skimage.color.hsv2rgb(new_hsv)

    return (new_rgb * 255).astype(np.uint8)


def get_topo_texture(
    bbox: tuple[float, float, float, float],
    srs: str | None = None,
    use_cache: bool = True,
    background_source: Any = cx.providers.Esri.WorldImagery,
    zoom_adjust: int = 1,
    remove_ice: bool = True,
) -> pv.Texture:
    """Get a texture for the bedrock surface topography from
    satellite imagery data.

    Downloading imagery data might take a while (>100s Mb).

    Parameters
    ----------
    bbox : tuple
        BBox coordinates (xmin, xmax, ymin, ymax).
    srs : str, optional
        The BBox (pyproj) coordinate reference system. If None,
        assumes lat/lon coordinates.
    use_cache : bool, optional
        If True, use previously downloaded background image data, if present
        (default: True).
    background_source : object, optional
        The provider used for downloading background map data
        (see https://contextily.readthedocs.io/en/latest/providers_deepdive.html).
        By default true-color satellite imagery is used.
    zoom_adjust : int, optional
        Can be used to adjust the zoom level of the downloaded background map
        data (higher value will result in more detail). Expects a value of either
        -1, 0 or 1 (default: 1). A positive value that is too high may cause
        downloading a great amount of data.
    remove_ice : bool, optional
        If True, processed the background image so that the intensity of
        white areas (snow, ice) is reduced (default: True). Relevant only
        when true-color satellite imagery is used as background source.

    Returns
    -------
    texture : :class:`pyvista.Texture`
        A pyvista texture of true-color (bedrock) terrain topography
        clipped to the input bbox.

    """
    if srs is None:
        # by default WGS84 lat-lon
        srs = "epsg:4326"

    # transformer input projection to Web Mercator
    p = pyproj.Proj.from_crs(crs_from=srs, crs_to="epsg:3857")

    west, south, east, north = bbox
    bbox_mercator  = p.transform_bounds(*bbox)

    raw_img, ext = cx.bounds2img(
        *bbox_mercator,
        source=background_source,
        zoom_adjust=zoom_adjust,
        use_cache=use_cache,
    )
    warped_img, warped_ext = cx.warp_tiles(raw_img, ext, srs)

    if remove_ice:
        processed_img = _ice_to_bedrock(warped_img[:, :, :-1])
    else:
        processed_img = warped_img[..., :-1] #_ice_to_bedrock(warped_img[:, :, :-1])

    x = np.linspace(warped_ext[0], warped_ext[1], warped_img.shape[1])
    y = np.linspace(warped_ext[3], warped_ext[2], warped_img.shape[0])

    da_img = xr.DataArray(processed_img, coords={"x": x, "y": y}, dims=("y", "x", "c"))
    da_img = da_img.sel(x=slice(west, east), y=slice(north, south))

    return pv.Texture(da_img.values)
