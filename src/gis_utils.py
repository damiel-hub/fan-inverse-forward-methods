from osgeo import gdal, osr, ogr
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
gdal.UseExceptions()

def read_geotiff(file_path):
    """
    Reads a GeoTIFF file and returns the mesh grids and elevation data.

    Parameters:
        file_path (str): The path to the GeoTIFF file.

    Returns:
        tuple:
            - xMesh (numpy.ndarray): 2D array of X-coordinates.
            - yMesh (numpy.ndarray): 2D array of Y-coordinates.
            - zMesh (numpy.ndarray): 2D array of elevation data.
    """
    dataset = gdal.Open(file_path)
    zMesh = dataset.GetRasterBand(1).ReadAsArray()
    transform = dataset.GetGeoTransform()
    x_pixels = zMesh.shape[1]
    y_pixels = zMesh.shape[0]
    x_orign = transform[0]
    y_orign = transform[3]
    x_pixel_size = transform[1]
    y_pixel_size = -transform[5]
    xMesh = np.arange(x_orign + x_pixel_size/2, x_orign + (x_pixels * x_pixel_size), x_pixel_size)
    yMesh = np.arange(y_orign - y_pixel_size/2, y_orign - (y_pixels * y_pixel_size), -y_pixel_size)
    xMesh, yMesh = np.meshgrid(xMesh, yMesh)
    zMesh = zMesh.astype(np.float64)
    return xMesh, yMesh, zMesh

def get_epsg_code(geotiff_path, debug=False):
    # Open the GeoTIFF file
    dataset = gdal.Open(geotiff_path)
    
    # Get the projection
    projection = dataset.GetProjection()
    
    # If there is no projection, return None
    if not projection:
        if debug:
            print("No projection found in the GeoTIFF file.")
        return None
    
    # Parse the projection information
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    
    # Get the EPSG code
    epsg_code = srs.GetAuthorityCode(None)
    
    if epsg_code is None:
        if debug:
            print("EPSG code not found. Here is the full projection info:")
            print(srs.ExportToPrettyWkt())
        return None
    
    if debug:
        print(f"The EPSG code for the GeoTIFF file is: {epsg_code}")
    
    return int(epsg_code)  # Safe to convert now

def write_geotiff(file_path, xMesh, yMesh, zMesh, epsg_code=None):
    """
    Writes the xMesh, yMesh, and zMesh to a GeoTIFF file.

    Parameters:
        file_path (str): Path to the output GeoTIFF file.
        xMesh (numpy.ndarray): The x-coordinates of the mesh.
        yMesh (numpy.ndarray): The y-coordinates of the mesh.
        zMesh (numpy.ndarray): The elevation data.
        epsg_code (int or None): The EPSG code for the coordinate reference system.
                                 If None, no spatial reference system is set.
    """
    # Expand the x and y coordinates while preserving the order
    flip_lr = xMesh[0, 0] > xMesh[0, -1]
    flip_ud = yMesh[0, 0] < yMesh[-1, 0]

    if flip_lr:
        xMesh = np.fliplr(xMesh)
        zMesh = np.fliplr(zMesh)
    if flip_ud:
        yMesh = np.flipud(yMesh)
        zMesh = np.flipud(zMesh)    
    rows, cols = zMesh.shape
    x_pixel_size = (xMesh[0, 1] - xMesh[0, 0])
    y_pixel_size = (yMesh[0, 0] - yMesh[1, 0])
    x_min = xMesh.min() - x_pixel_size/2
    y_max = yMesh.max() + y_pixel_size/2
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32)
    transform = (x_min, x_pixel_size, 0, y_max, 0, -y_pixel_size)
    dataset.SetGeoTransform(transform)
    if epsg_code is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        dataset.SetProjection(srs.ExportToWkt())

    dataset.GetRasterBand(1).WriteArray(zMesh)
    dataset.FlushCache()
    dataset = None


def clip_geotiff_with_shapefile(
    geotiff_path,
    shapefile_path,
    clipped_geotiff_path=None,
    center_coords=True,
    no_data_value=-9999,
    add_nan_padding = True
):
    """
    Clips a GeoTIFF by a shapefile boundary and returns its coordinate mesh and raster data.

    Parameters
    ----------
    geotiff_path : str
        Path to the input GeoTIFF file.
    shapefile_path : str
        Path to the clipping shapefile.
    clipped_geotiff_path : str, optional
        Path to save the clipped output GeoTIFF. If None, the clipping is done in memory.
    center_coords : bool, optional
        If True, xMesh and yMesh represent pixel centers rather than pixel corners.
    no_data_value : float, optional
        No-data value to assign after clipping.

    Returns
    -------
    (xMesh, yMesh, zMesh) : tuple of np.ndarray
        xMesh : 2D array, x-coordinates of each pixel
        yMesh : 2D array, y-coordinates of each pixel
        zMesh : 2D array, raster data with no-data replaced by np.nan
    """

    # Open the GeoTIFF file
    geotiff = gdal.Open(geotiff_path)
    if geotiff is None:
        raise IOError(f"Could not open GeoTIFF file: {geotiff_path}")

    # Define GDAL Warp options for clipping
    warp_options = gdal.WarpOptions(
        format='GTiff' if clipped_geotiff_path else 'MEM',  # Use MEM if no path is given
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=no_data_value
    )

    # Perform the clipping operation
    clipped_ds = gdal.Warp(
        destNameOrDestDS=(clipped_geotiff_path if clipped_geotiff_path else ''),
        srcDSOrSrcDSTab=geotiff,
        options=warp_options
    )

    if clipped_ds is None:
        raise RuntimeError("GDAL Warp operation failed. Please check your input files and paths.")

    # Read the data from the clipped raster band
    band = clipped_ds.GetRasterBand(1)
    zMesh = band.ReadAsArray().astype(float)

    # Replace the no-data value with np.nan
    zMesh[zMesh == no_data_value] = np.nan

    # Get the geotransform (affine transform coefficients)
    transform = clipped_ds.GetGeoTransform()
    x_min, pixel_width, _, y_max, _, pixel_height = transform

    # Raster size
    nX = clipped_ds.RasterXSize
    nY = clipped_ds.RasterYSize

    # If the user wants pixel-center coordinates, shift by half a pixel
    if center_coords:
        x_min += 0.5 * pixel_width
        y_max += 0.5 * pixel_height

    # Generate coordinate arrays
    xArray = np.linspace(x_min, x_min + (nX - 1) * pixel_width, nX)
    yArray = np.linspace(y_max, y_max + (nY - 1) * pixel_height, nY)

    # Create 2D meshgrids
    xMesh, yMesh = np.meshgrid(xArray, yArray)
    if add_nan_padding:
        # Original sizes
        old_nY, old_nX = zMesh.shape

        # Compute pixel steps (assuming > 1 pixel in each dimension).
        # - If your dataset is only 1 pixel wide in X or Y, you'll need special handling.
        deltaX = xArray[1] - xArray[0] if old_nX > 1 else 1.0
        deltaY = yArray[1] - yArray[0] if old_nY > 1 else -1.0  # often negative if north-up

        # New sizes
        new_nX = old_nX + 2
        new_nY = old_nY + 2

        # Extended coordinate range
        new_x_min = xArray[0] - deltaX
        new_x_max = xArray[-1] + deltaX

        new_y_min = yArray[0] - deltaY
        new_y_max = yArray[-1] + deltaY

        # Create the new coordinate arrays
        new_xArray = np.linspace(new_x_min, new_x_max, new_nX)
        new_yArray = np.linspace(new_y_min, new_y_max, new_nY)

        # Meshgrids with 1-pixel padding
        new_xMesh, new_yMesh = np.meshgrid(new_xArray, new_yArray)

        # Create a new z array of NaNs
        new_zMesh = np.full((new_nY, new_nX), np.nan, dtype=zMesh.dtype)
        # Insert the original data in the center
        new_zMesh[1:-1, 1:-1] = zMesh

        # Overwrite the original variables
        xMesh, yMesh, zMesh = new_xMesh, new_yMesh, new_zMesh
    # If you do not need the dataset object anymore, close it
    clipped_ds = None
    geotiff = None

    return xMesh, yMesh, zMesh


def read_shapefile_boundary(shapefile_path):
    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    
    all_fan_boundary_x = []
    all_fan_boundary_y = []

    # Helper function to check if points are in CCW order
    def is_ccw(x, y):
        # Calculate the signed area
        area = 0.0
        for i in range(len(x)):
            j = (i + 1) % len(x)
            area += x[i] * y[j] - y[i] * x[j]
        return area > 0

    # Iterate through each feature (polygon) in the layer
    for feature in layer:
        geom = feature.GetGeometryRef()

        # Extract the exterior ring (outer boundary)
        exterior_ring = geom.GetGeometryRef(0)
        exterior_x = []
        exterior_y = []
        for i in range(exterior_ring.GetPointCount()):
            lon, lat, _ = exterior_ring.GetPoint(i)
            exterior_x.append(lon)
            exterior_y.append(lat)
        
        # Check and ensure CCW order
        if not is_ccw(exterior_x, exterior_y):
            exterior_x.reverse()
            exterior_y.reverse()
        if exterior_x[0] != exterior_x[-1] or exterior_y[0] != exterior_y[-1]:
            exterior_x = np.append(interior_x, interior_x[0])
            exterior_y = np.append(exterior_y, exterior_y[0])          
        all_fan_boundary_x.append(exterior_x)
        all_fan_boundary_y.append(exterior_y)

        # Extract any interior rings (holes) and add them to the same lists
        for j in range(1, geom.GetGeometryCount()):
            interior_ring = geom.GetGeometryRef(j)
            interior_x = []
            interior_y = []
            for i in range(interior_ring.GetPointCount()):
                lon, lat, _ = interior_ring.GetPoint(i)
                interior_x.append(lon)
                interior_y.append(lat)
            
            # Check and ensure CLOCK-WISE(cw) 
            if is_ccw(interior_x, interior_y):
                interior_x.reverse()
                interior_y.reverse()
            
            if interior_x[0] != interior_x[-1] or interior_y[0] != interior_y[-1]:
                interior_x = np.append(interior_x, interior_x[0])
                interior_y = np.append(interior_y, interior_y[0])  
            all_fan_boundary_x.append(interior_x)
            all_fan_boundary_y.append(interior_y)
    
    return all_fan_boundary_x, all_fan_boundary_y

def plot_hull_and_boundary(hull, polygons, union_polygon):
    """
    Plots:
      - The convex hull (in red)
      - Each individual boundary polygon (in blue)
      - The union polygon boundary (in green, optional)
    """
    fig, ax = plt.subplots()

    # Plot each fan boundary polygon
    for i, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', label='Boundary' if i == 0 else "")

    # Plot the union boundary (optional, in green)
    if union_polygon.geom_type == 'Polygon':
        union_x, union_y = union_polygon.exterior.xy
        ax.plot(union_x, union_y, color='green', linestyle='--', label='Union')
    else:
        # If union is multi-polygon, plot each one
        for j, subpoly in enumerate(union_polygon):
            union_x, union_y = subpoly.exterior.xy
            ax.plot(union_x, union_y, color='green', linestyle='--',
                    label='Union' if j == 0 else "")

    # Plot the convex hull (in red)
    hull_x, hull_y = hull.exterior.xy
    ax.plot(hull_x, hull_y, color='red', linewidth=2, label='Convex Hull')

    ax.set_aspect('equal', 'datalim')
    ax.legend()
    plt.show()

def get_convex_hull_and_perimeter(shapefile_path, pltFlag = False):
    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()
    
    fan_boundary_x = []
    fan_boundary_y = []

    # Helper function to check if points are in CCW order
    def is_ccw(x, y):
        # Calculate the signed area
        area = 0.0
        for i in range(len(x)):
            j = (i + 1) % len(x)
            area += x[i] * y[j] - y[i] * x[j]
        return area > 0

    for feature in layer:
        geom = feature.GetGeometryRef()

        # Extract the exterior ring (outer boundary)
        exterior_ring = geom.GetGeometryRef(0)
        exterior_x = []
        exterior_y = []
        for i in range(exterior_ring.GetPointCount()):
            lon, lat, _ = exterior_ring.GetPoint(i)
            exterior_x.append(lon)
            exterior_y.append(lat)
        
        # Make sure it is CCW
        if not is_ccw(exterior_x, exterior_y):
            exterior_x.reverse()
            exterior_y.reverse()

        # Ensure closed ring
        if exterior_x[0] != exterior_x[-1] or exterior_y[0] != exterior_y[-1]:
            exterior_x = np.append(exterior_x, exterior_x[0])
            exterior_y = np.append(exterior_y, exterior_y[0])

        fan_boundary_x.append(exterior_x)
        fan_boundary_y.append(exterior_y)

    polygons = []
    
    # Convert each boundary list into a Shapely Polygon
    for x_coords, y_coords in zip(fan_boundary_x, fan_boundary_y):
        # zip the x/y together into (x,y) pairs
        coords = list(zip(x_coords, y_coords))
        polygon = Polygon(coords)
        polygons.append(polygon)
    
    # Merge (union) all polygons into one geometry
    union_polygon = unary_union(polygons)
    
    # Compute the convex hull of the union
    hull = union_polygon.convex_hull
    
    # Perimeter (length of the hull)
    hull_perimeter = hull.length
    boundary_perimeter = union_polygon.length
    if pltFlag:
        plot_hull_and_boundary(hull, polygons, union_polygon)
    
    return hull, hull_perimeter, boundary_perimeter



def save_resampled_geotiff(data, geo_transform, projection, output_path):
    # Get the dimensions of the data
    rows, cols = data.shape
    
    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    
    # Set the geotransform and projection
    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)
    
    # Write the data to the file
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(data)
    
    # Set no data value if necessary
    out_band.SetNoDataValue(np.nan)
    
    # Clean up
    out_band.FlushCache()
    out_raster = None


def get_gdalinfo(filepath):
  """
  Gets information about a GeoTIFF using gdalinfo.

  Args:
      filepath: Path to the GeoTIFF file.

  Returns:
      The output of the gdalinfo command as a string.
  """
  # Build the command
  command = ["gdalinfo", filepath]
  # Execute the command and capture output
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output, error = process.communicate()
  # Check for errors   

  if error:
    raise RuntimeError(f"Error running gdalinfo: {error.decode('utf-8')}")
  # Return the decoded output
  return output.decode('utf-8')

def create_mask_from_shapefile(geotiff_path, shapefile_path, burn_value=1):
    # Open the GeoTIFF to get geotransformation and projection
    geotiff = gdal.Open(geotiff_path)
    geo_transform = geotiff.GetGeoTransform()
    projection = geotiff.GetProjection()
    x_size = geotiff.RasterXSize
    y_size = geotiff.RasterYSize
    
    # Create an empty raster for the mask (same size as the GeoTIFF)
    mem_driver = gdal.GetDriverByName('MEM')  # In-memory raster
    mask_raster = mem_driver.Create('', x_size, y_size, 1, gdal.GDT_Byte)
    mask_raster.SetGeoTransform(geo_transform)
    mask_raster.SetProjection(projection)

    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Rasterize the shapefile into the mask raster
    gdal.RasterizeLayer(mask_raster, [1], layer, burn_values=[burn_value])

    # Read the mask as a numpy array
    mask_band = mask_raster.GetRasterBand(1)
    mask_array = mask_band.ReadAsArray()

    # Cleanup
    geotiff = None
    shapefile = None
    mask_raster = None

    return mask_array


def calculate_volume_difference_within_polygon(pre_tiff, post_tiff, shapefile_path=None, output_resampled_path = 'pre_resample.tif', pltFlag = 0, colorBarStr = None, vmin = None, vmax = None):
    """
    Calculates the volume difference between two GeoTIFF files within an optional polygon mask, with optional resampling and visualization.

    Parameters:
        pre_tiff (str): Path to the pre-event GeoTIFF file.
        post_tiff (str): Path to the post-event GeoTIFF file.
        shapefile_path (str, optional): Path to the shapefile for masking the area of interest.
        output_resampled_path (str, optional): Path to save the resampled pre-event GeoTIFF. Default is 'pre_resample.tif'.
        pltFlag (int, optional): Flag to plot the Difference of DEM (DoD) map. Default is 0 (no plot).
        vmin (float, optional): Colormap cmin
        vmax (float, optional): Colormap cmax

    Returns:
        float: The calculated volume difference within the specified area.
    """

    debug = 0
    if debug:
        pre_tiff_info = get_gdalinfo(pre_tiff)  # Replace with your actual filename
        print('pre_tiff_info')
        print(pre_tiff_info)
        post_tiff_info = get_gdalinfo(post_tiff)  # Replace with your actual filename
        print('post_tiff_info')
        print(post_tiff_info)

    # Open the first GeoTIFF
    ds1 = gdal.Open(post_tiff)
    band1 = ds1.GetRasterBand(1)
    data1 = band1.ReadAsArray()
    geo_transform1 = ds1.GetGeoTransform()

    # Open the second GeoTIFF
    ds2 = gdal.Open(pre_tiff)
    band2 = ds2.GetRasterBand(1)
    data2 = band2.ReadAsArray()
    geo_transform2 = ds2.GetGeoTransform()

    if geo_transform1[1]>0:
        xmin1 = geo_transform1[0]
        xmax1 = geo_transform1[0] + geo_transform1[1] * data1.shape[1]
    else:
        xmax1 = geo_transform1[0]
        xmin1 = geo_transform1[0] + geo_transform1[1] * data1.shape[1]            

    if geo_transform1[5]<0:
        ymin1 = geo_transform1[3] + geo_transform1[5] * data1.shape[0]
        ymax1 = geo_transform1[3]
    else:
        ymax1 = geo_transform1[3] + geo_transform1[5] * data1.shape[0]
        ymin1 = geo_transform1[3]

    # Resample ds2 to match ds1 if necessary
    if geo_transform1 != geo_transform2:
        print('Resample pre-event tif to match post-event tif is necessary')

        
        ds2_resampled = gdal.Warp('', ds2, format='MEM',
                                xRes=geo_transform1[1], yRes=geo_transform1[5],
                                outputBounds=[xmin1, ymin1, xmax1, ymax1],
                                resampleAlg=gdal.GRA_Bilinear
        )
        band2_resampled = ds2_resampled.GetRasterBand(1)
        data2_resampled = band2_resampled.ReadAsArray()

        if output_resampled_path:
            save_resampled_geotiff(data2_resampled, geo_transform1, ds1.GetProjection(), output_resampled_path)
            if debug:
                output_tiff_info = get_gdalinfo(output_resampled_path)  # Replace with your actual filename
                print('resample_tiff_info')
                print(output_tiff_info)
    else:
        data2_resampled = data2


    # Clip the rasters using the shapefile polygon if provided
    if shapefile_path:
        shapefile = ogr.Open(shapefile_path)
        layer = shapefile.GetLayer()

        # Create a mask for the area inside the polygon
        mem_driver = gdal.GetDriverByName('MEM')
        target_ds = mem_driver.Create('', ds1.RasterXSize, ds1.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo_transform1)
        target_ds.SetProjection(ds1.GetProjectionRef())

        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
        mask = target_ds.ReadAsArray()

        # Apply the mask to the data
        data1 = np.where(mask == 1, data1, np.nan)
        data2_resampled = np.where(mask == 1, data2_resampled, np.nan)

    # Calculate the difference
    difference = data1 - data2_resampled

    # Calculate the pixel area (assuming square pixels)
    pixel_area = abs(geo_transform1[1] * geo_transform1[5])

    # Calculate the volume difference
    volume_difference = np.nansum(difference) * pixel_area

    # Plot the DoD map if pltFlag is set to 1
    if pltFlag == 1:
        if vmin is not None and vmax is not None:
            plt.imshow(difference, extent= (xmin1, xmax1, ymin1, ymax1), cmap='RdBu_r',vmin=vmin, vmax=vmax)
        else:
            plt.imshow(difference, extent= (xmin1, xmax1, ymin1, ymax1), cmap='RdBu_r', vmin=-np.nanmax(abs(difference)), vmax=np.nanmax(abs(difference)))
        if colorBarStr is None:
            plt.colorbar(label='Elevation Difference (m)')
        else:
            plt.colorbar(label=colorBarStr)
        plt.title('DEM of Differences (DoD) Map')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')

    # Clean up
    ds1 = None
    ds2 = None
    if 'ds2_resampled' in locals():
        ds2_resampled = None

    return volume_difference


def calculate_polygon_area(shapefile_path, target_epsg=None):
    """
    Calculate the area of a single polygon in a shapefile.

    Parameters:
        shapefile_path (str): Path to the input shapefile.
        target_epsg (int, optional): EPSG code for the target CRS. If provided,
                                     reprojects the geometry to this CRS for area calculations.

    Returns:
        float: The area of the polygon.

    Raises:
        ValueError: If the shapefile does not contain exactly one feature.
        FileNotFoundError: If the shapefile cannot be opened.
    """
    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    if shapefile is None:
        raise FileNotFoundError(f"Could not open shapefile: {shapefile_path}")
    layer = shapefile.GetLayer()

    # Check the number of features in the layer
    feature_count = layer.GetFeatureCount()
    if feature_count != 1:
        raise ValueError(f"The shapefile must contain exactly one feature, but it contains {feature_count}.")

    # Get the spatial reference of the shapefile
    spatial_ref = layer.GetSpatialRef()

    # Initialize a transformation if a target EPSG is provided
    if target_epsg:
        target_ref = osr.SpatialReference()
        target_ref.ImportFromEPSG(target_epsg)
        transform = osr.CoordinateTransformation(spatial_ref, target_ref)
    else:
        transform = None

    # Get the single feature
    feature = layer.GetNextFeature()
    geometry = feature.GetGeometryRef()

    # Reproject geometry if a transformation is defined
    if transform:
        geometry.Transform(transform)

    # Calculate the area
    area = geometry.GetArea()  # Area in units of the CRS (e.g., square meters if projected)

    # Close the shapefile
    shapefile = None

    return area