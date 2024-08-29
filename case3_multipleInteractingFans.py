import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('src/')

import gis_utils 
from fanTopo import fan_topo

if __name__ == "__main__":
    # Step 1: Read the GeoTIFF file
    file_path = 'datasets/raw/tif/zRock.tif'
    xMesh, yMesh, zMesh = gis_utils.read_geotiff(file_path)

    # Step 2: Define the apex coordinates
    xApexM = np.array([-40, -20, 0, 20, 40])
    yApexM = np.array([10, 15, 12, 14, 17]) + 15

    # Step 3: Create the interpolation function for zMesh
    interp_func = RegularGridInterpolator((xMesh[0, :], yMesh[:, 0]), zMesh.T)
    points = np.array([xApexM, yApexM]).T
    zApexM = interp_func(points)

    # Step 4: Define the slope angles (tanAlpha)
    tanAlpha = np.array([0.9, 1.2, 0.9, 1.2, 1]) * 0.5

    # Step 5: Compute the topography with the fan_topo function
    start_time = time.time()
    zTopo = fan_topo(xMesh, yMesh, zMesh, xApexM, yApexM, zApexM, {
        'caseName': 'cone',
        'tanAlphaM': tanAlpha,
        'dispflag': 1
    })
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # Step 6: Handle NaN values in the resulting topography map
    zMap = zTopo[0].copy()
    zMap[np.isnan(zMap)] = zMesh[np.isnan(zMap)]

    # Step 7: Plot the filled contour plot of the topography
    plt.figure(figsize=(10, 8))
    plt.contourf(xMesh, yMesh, zTopo[0], levels=50, cmap='viridis')
    plt.contour(xMesh, yMesh, zMap, levels=50, colors='k')
    plt.plot(xApexM, yApexM, 'r.', markersize=8, label='Apex Points')
    plt.title('Filled Contour Plot of zTopo')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Elevation')
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Step 8: Write the resulting topography to a GeoTIFF file
    output_file_path = 'results/zTopo.tif'
    gis_utils.write_geotiff(output_file_path, xMesh, yMesh, zTopo[0])