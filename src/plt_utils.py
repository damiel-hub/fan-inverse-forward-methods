from osgeo import ogr
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import matplotlib.pyplot as plt
from osgeo import ogr

def plot_polygon_boundary_gdal(shapefile_path, exterior_color='b', interior_color='r'):
    """
    Reads a shapefile using GDAL and plots the boundary of the polygon(s) it contains.

    Parameters:
        shapefile_path (str): Path to the shapefile.
        exterior_color (str): Color for the exterior polygon boundary.
        interior_color (str): Color for the interior polygon boundaries (holes).

    Returns:
        None
    """
    # Open the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Iterate over each feature in the shapefile
    for feature in layer:
        # Get the geometry of the feature
        geom = feature.GetGeometryRef()
        
        # Check if the geometry is a polygon or multipolygon
        if geom.GetGeometryType() == ogr.wkbPolygon:
            plot_polygon(geom, exterior_color, interior_color)
        elif geom.GetGeometryType() == ogr.wkbMultiPolygon:
            for i in range(geom.GetGeometryCount()):
                polygon = geom.GetGeometryRef(i)
                plot_polygon(polygon, exterior_color, interior_color)


def plot_polygon(polygon, exterior_color, interior_color):
    """
    Plots a polygon's boundary, including interior rings (holes).

    Parameters:
        polygon (ogr.Geometry): A polygon geometry.
        exterior_color (str): Color for the exterior polygon boundary.
        interior_color (str): Color for the interior polygon boundaries (holes).

    Returns:
        None
    """
    # Plot the exterior ring
    exterior_ring = polygon.GetGeometryRef(0)
    exterior_points = exterior_ring.GetPoints()
    x_exterior, y_exterior = zip(*exterior_points)
    plt.plot(x_exterior, y_exterior, color=exterior_color, linestyle='-', label='Exterior')

    # Plot interior rings (holes) if any
    num_interior_rings = polygon.GetGeometryCount() - 1
    for i in range(1, num_interior_rings + 1):
        interior_ring = polygon.GetGeometryRef(i)
        interior_points = interior_ring.GetPoints()
        x_interior, y_interior = zip(*interior_points)
        plt.plot(x_interior, y_interior, color=interior_color, linestyle='-', label='Interior')






def plot_hillshade(xMesh, yMesh, zMesh, azdeg=180, altdeg=45, vert_exag=1):
    """
    Plots a hillshade of the surface using the provided mesh grids and elevation data.

    Parameters:
        xMesh (numpy.ndarray): 2D array of X-coordinates.
        yMesh (numpy.ndarray): 2D array of Y-coordinates.
        zMesh (numpy.ndarray): 2D array of elevation values.
        azdeg (float, optional): Azimuth of the light source in degrees. Default is 315.
        altdeg (float, optional): Altitude angle of the light source in degrees. Default is 45.
        vert_exag (float, optional): Vertical exaggeration factor. Default is 1.

    Returns:
        None
    """
    # Create a LightSource object with the specified azimuth and altitude
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)

    # Calculate hillshade
    hillshade = ls.hillshade(zMesh, vert_exag=vert_exag, 
                             dx=xMesh[0, 1] - xMesh[0, 0], 
                             dy=yMesh[1, 0] - yMesh[0, 0])

    # Plot the hillshade
    plt.figure(figsize=(10, 8))
    plt.imshow(hillshade, cmap='gray', extent=[xMesh.min(), xMesh.max(), yMesh.min(), yMesh.max()])
    # plt.colorbar(label='Hillshade intensity')
    plt.title('Hillshade')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
