import sys
import numpy as np
import time

sys.path.append('src/')
from SPM import SPM
from EDR import EDR
import gis_utils
import reconstruct_fantopo

if __name__ == "__main__":
    tic = time.perf_counter()

    # Estimated Running Time for Different DEM Resolutions:
    # System 1: AMD Ryzen 9 5900X, NVIDIA GeForce RTX 3080, PYTHON 3.11.5
    # 10-meter: 158.0 seconds, 20-meter: 58.7 seconds

    dem_interval = 20

    topo_pre_event = r"datasets\raw\tif\topo_PuTunPuNas_min_before2014_riverbed2014_" + str(dem_interval) + r"m.tif"
    topo_post_event = r"datasets\raw\tif\topo_PuTunPuNas_2014_" + str(dem_interval) + r"m.tif"
    shape_fan_boundary = r"datasets\raw\shape\PT2014.shp"

    shortest_path_distance = SPM(topo_post_event, shape_fan_boundary)
    xMesh_crop, yMesh_crop, zMesh_crop, sMesh_crop = shortest_path_distance.within_boundary(pltFlag = False)
    boundary_x, boundary_y, boundary_z, boundary_s = shortest_path_distance.along_boundary(pltFlag = False)

    # Fit quadratic elevation-distance relationship
    bin_size = 100
    ds = 5
    outlength = 500
    fitting_s_z_within_boundary = EDR(sMesh_crop, zMesh_crop).medianFilter_on(bin_size, ds, outlength, pltFlag = False)
    fitting_s_z_along_boundary = EDR(boundary_s, boundary_z).medianFilter_off(ds, outlength, pltFlag = False)
    
  
    xApex, yApex = shortest_path_distance.xyApex()
    volume_expected = gis_utils.calculate_volume_difference_within_polygon(topo_pre_event, topo_post_event, shape_fan_boundary)
    guessHeightAboveGround_bottom = 1
    guessHeightAboveGround_top = 10
    xMesh, yMesh, zTopo, _ = reconstruct_fantopo.reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, fitting_s_z_within_boundary, fanBoundarySHP=shape_fan_boundary, tol=0.03, debug=False)
    epgs_code = gis_utils.get_epsg_code(topo_pre_event)
    gis_utils.write_geotiff('results/zTopo_2014.tif',xMesh, yMesh, zTopo, epgs_code)

    toc = time.perf_counter()
    elapsed_time = toc - tic
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


