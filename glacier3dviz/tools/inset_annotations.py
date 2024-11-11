import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import subprocess
import os


def get_centered_lon_lat(ds):
    # Calculate the center x and y
    center_x = (ds['x'].min() + ds['x'].max()) / 2
    center_y = (ds['y'].min() + ds['y'].max()) / 2

    # Get the projection information
    proj_srs = ds.attrs['pyproj_srs']

    # Initialize the pyproj transformer from dataset projection to WGS84 (lat/lon)
    transformer = pyproj.Transformer.from_crs(proj_srs,
                                              "EPSG:4326", always_xy=True)

    # Convert center_x, center_y to lat and lon
    center_lon, center_lat = transformer.transform(center_x.values,
                                                   center_y.values)

    return center_lon, center_lat


def plot_centered_globe(ds,
                        filename=None,
                        figsize=(0.7, 0.7),
                        arrow_length=(15, 10),
                        ):
    center_lon, center_lat = get_centered_lon_lat(ds)

    fig = plt.figure(figsize=figsize)

    # Use Orthographic projection centered at location
    ortho = ccrs.Orthographic(central_longitude=center_lon,
                              central_latitude=center_lat)
    ax = plt.subplot(1, 1, 1, projection=ortho)

    # Add global features
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3,
                   edgecolor='black')

    # Add an red arrow for showing the glacier location
    ax.annotate('',
                xy=(center_lon, center_lat),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=arrow_length,
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                color='red',
                                lw=1.))

    # Add gridlines for lat and lon
    ax.gridlines(draw_labels=False, lw=0.2, color='black')

    ax.set_global()
    ax.axis("off")

    # Save the plot
    if filename:
        plt.savefig(filename,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=300)

    plt.show()


def add_inset_with_ffmpeg(main_video, inset_image, output_video,
                          position="W-w-10:H-h-10",
                          scale_factor=1,
                          ):
    if not os.path.exists(main_video):
        raise ValueError('main_video does not exist')
    if not os.path.exists(inset_image):
        raise ValueError('inset_image does not exist')

    filter_complex = (f"[1:v]scale=iw*{scale_factor}:ih*{scale_factor}[scaled];"
                      f"[0:v][scaled]overlay={position}")
    command = [
        'ffmpeg',
        '-y',
        '-i', main_video,
        '-i', inset_image,
        '-filter_complex', filter_complex,
        '-codec:a', 'copy',  # Copy audio without re-encoding (if any)
        output_video
    ]
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        print(f"Overlay added successfully! Output saved to '{output_video}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing: {e}")
