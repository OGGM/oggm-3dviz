import pathlib

import numpy as np
import pyproj
import pystac_client
import skimage
import pyvista as pv
import xarray as xr
from skimage.exposure.exposure import rescale_intensity
from skimage.util import random_noise
from xrspatial.multispectral import true_color


def _download_sentinel_data(bbox: tuple, srs: str = None) -> xr.DataArray:
    """Download raw sentinel data from a STAC catalog.

    Query parameters:

    - date range: summer 2020 to avoid snow cover
    - low cloud cover
    - visible bands

    TODO: only works for small bbox extent (within the same UTM zone)
    and in the nothern hemisphere.

    """
    if srs is not None:
        proj = pyproj.Proj(srs)
        lon_min, lat_min = proj(bbox[0], bbox[2], inverse=True)
        lon_max, lat_max = proj(bbox[1], bbox[3], inverse=True)
        bbox_lonlat = [lon_min, lat_min, lon_max, lat_max]
    else:
        bbox_lonlat = bbox

    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v0")
    catalog.add_conforms_to("ITEM_SEARCH")
    catalog.add_conforms_to("QUERY")

    search = catalog.search(
        collections=["sentinel-s2-l2a-cogs"],
        bbox=bbox_lonlat,
        query={
            "eo:cloud_cover": {"lt": 10},
            "sentinel:valid_cloud_cover": {"eq": True},
        },
        datetime="2020-07-01/2020-10-01",
        max_items=30,
    )

    return (
        xr.open_dataset(search, engine="stac")
        .sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))
        .get(["B04", "B03", "B02"])
        .to_array(dim="band", name="radiance")
    )


def _ice_to_bedrock(
    img: np.ndarray, intensity_threshold: float = 0.55, noise_scale: float = 2e-4
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

    noise_vars = value * noise_scale

    value -= rescale_intensity(
        value,
        in_range=(intensity_threshold, 1.0),
        out_range=(0.0, intensity_threshold),
    )
    value = random_noise(value, mode="localvar", local_vars=noise_vars)

    new_hsv = np.stack([hue, saturation, value], axis=-1)
    new_rgb = skimage.color.hsv2rgb(new_hsv)

    return (new_rgb * 255).astype(np.uint8)


def _xr_ice_to_berock(da: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(
        _ice_to_bedrock,
        da,
        input_core_dims=[["x", "y", "band"]],
        output_core_dims=[["x", "y", "band"]],
        vectorize=True,
    )


def _process_sentinel_data(da: xr.DataArray) -> xr.DataArray:
    """Process raw sentinel data and return a true-color RGB
    image that can be used as texture for the bedrock topography.

    """
    return (
        da.median(dim="time")  # smooth radiance and exclude clouds
        .pipe(lambda da: true_color(*da))
        .isel(band=[0, 1, 2])
        .pipe(lambda da: _xr_ice_to_berock(da))
    )


def get_topo_texture(
    bbox: tuple[float, float, float, float],
    srs: str | None = None,
    use_cache: bool = False,
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
        If True, use previously downloaded imagery data, if present
        (default: False).

    Returns
    -------
    texture : :class:`pyvista.Texture`
        A pyvista texture of true-color (bedrock) terrain topography
        clipped to the input bbox.

    """
    temp_zarr_dataset = "topo_texture_cache.zarr"

    if use_cache and pathlib.Path(temp_zarr_dataset).exists():
        da_img_raw = xr.open_zarr(temp_zarr_dataset)
        da_img_raw = da_img_raw.load().to_array().squeeze()
    else:
        da_img_raw = _download_sentinel_data(bbox, srs=srs)
        da_img_raw = da_img_raw.compute()
        da_img_raw.attrs.clear()
        da_img_raw.to_zarr(temp_zarr_dataset)

    da_img_processed = _process_sentinel_data(da_img_raw)

    return pv.Texture(da_img_processed.transpose("y", "x", "band").values)
