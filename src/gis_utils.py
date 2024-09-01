from osgeo import gdal, osr, ogr
import numpy as np
import subprocess
import matplotlib.pyplot as plt
gdal.UseExceptions()

def read_geotiff(file_path):
    """
    Reads a GeoTIFF file and returns the xMesh, yMesh, and zMesh.

    Parameters:
        file_path (str): Path to the GeoTIFF file.

    Returns:
        xMesh (numpy.ndarray): The x-coordinates of the mesh.
        yMesh (numpy.ndarray): The y-coordinates of the mesh.
        zMesh (numpy.ndarray): The elevation data.
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


def clip_geotiff_with_shapefile(geotiff_path, shapefile_path, clipped_geotiff_path=None):
    # Open the GeoTIFF file
    geotiff = gdal.Open(geotiff_path)
    
    # Define GDAL Warp options for clipping
    warp_options = gdal.WarpOptions(
        format='GTiff' if clipped_geotiff_path else 'MEM',  # Use MEM for in-memory processing if no path is given
        cutlineDSName=shapefile_path,
        cropToCutline=True,
        dstNodata=-9999  # Specify no-data value
    )
    
    # Perform the clipping operation
    clipped_ds = gdal.Warp(clipped_geotiff_path if clipped_geotiff_path else '', geotiff, options=warp_options)
    
    if clipped_ds is None:
        raise RuntimeError("GDAL Warp operation failed. Please check your input files and paths.")
    
    # Extract the clipped raster data as a numpy array
    zMesh = clipped_ds.GetRasterBand(1).ReadAsArray()
    zMesh[zMesh == -9999] = np.nan
    # Obtain geo-transform information and generate coordinate meshes
    transform = clipped_ds.GetGeoTransform()
    x_min, pixel_width, _, y_max, _, pixel_height = transform
    
    xMesh, yMesh = np.meshgrid(
        np.arange(x_min, x_min + clipped_ds.RasterXSize * pixel_width, pixel_width),
        np.arange(y_max, y_max + clipped_ds.RasterYSize * pixel_height, pixel_height)
    )
    
    # Return the coordinate meshes and raster data
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



def calculate_volume_difference_within_polygon(pre_tiff, post_tiff, shapefile_path=None, output_resampled_path = 'pre_resample.tif', pltFlag = 0):
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

    # Resample ds2 to match ds1 if necessary
    if geo_transform1 != geo_transform2:
        print('Resample pre-event tif to match post-event tif is necessary')
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
        plt.figure(figsize=(10, 8))
        plt.imshow(difference, cmap='RdBu_r', vmin=-np.nanmax(abs(difference)), vmax=np.nanmax(abs(difference)))
        plt.colorbar(label='Elevation Difference (m)')
        plt.title('Difference of Differences (DoD) Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

    # Clean up
    ds1 = None
    ds2 = None
    if 'ds2_resampled' in locals():
        ds2_resampled = None

    return volume_difference