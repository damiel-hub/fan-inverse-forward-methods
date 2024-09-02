import sys
sys.path.append('src/')
from SPM import SPM
from EDR import EDR
import gis_utils
import reconstruct_fantopo

if __name__ == "__main__":
    topo_pre_event = r"datasets\raw\tif\topo_10m_PuTunPuNas_pre_2024_min_base2024.tif"
    topo_post_event = r"datasets\raw\tif\topo_10m_PuTunPuNas_2024.tif"
    shape_fan_boundary_all = r"datasets\raw\shape\PT2024.shp"
    shape_fan_boundary_left = r"datasets\raw\shape\PT2024_left.shp"
    shape_fan_boundary_right = r"datasets\raw\shape\PT2024_right.shp"

    # Fan Boundary All
    shortest_path_distance = SPM(topo_post_event, shape_fan_boundary_all)
    xMesh_crop, yMesh_crop, zMesh_crop, sMesh_crop = shortest_path_distance.within_boundary(pltFlag = False)

    # Fit quadratic elevation-distance relationship
    bin_size = 100
    ds = 5
    outlength = 1000
    fitting_s_z_within_boundary = EDR(sMesh_crop, zMesh_crop).medianFilter_on(bin_size, ds, outlength, pltFlag = False)
    
    xApex, yApex = shortest_path_distance.xyApex()
    volume_expected = gis_utils.calculate_volume_difference_within_polygon(topo_pre_event, topo_post_event, shape_fan_boundary_all)
    guessHeightAboveGround_bottom = 15
    guessHeightAboveGround_top = 20
    xMesh, yMesh, zTopo, _ = reconstruct_fantopo.reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, fitting_s_z_within_boundary, fanBoundarySHP=shape_fan_boundary_all, tol=0.03, debug=False)
    epgs_code = gis_utils.get_epsg_code(topo_pre_event)
    gis_utils.write_geotiff('results/zTopo_2024_all.tif',xMesh, yMesh, zTopo, epgs_code)


    # Fan Boundary Left
    shortest_path_distance = SPM(topo_post_event, shape_fan_boundary_left)
    xMesh_crop, yMesh_crop, zMesh_crop, sMesh_crop = shortest_path_distance.within_boundary(pltFlag = False)

    # Fit quadratic elevation-distance relationship
    bin_size = 100
    ds = 5
    outlength = 1000
    fitting_s_z_within_boundary = EDR(sMesh_crop, zMesh_crop).medianFilter_on(bin_size, ds, outlength, pltFlag = False)
    
    xApex, yApex = shortest_path_distance.xyApex()
    volume_expected = gis_utils.calculate_volume_difference_within_polygon(topo_pre_event, topo_post_event, shape_fan_boundary_left)
    guessHeightAboveGround_bottom = 10
    guessHeightAboveGround_top = 30
    xMesh, yMesh, zTopo, _ = reconstruct_fantopo.reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, fitting_s_z_within_boundary, fanBoundarySHP=shape_fan_boundary_all, tol=0.03, debug=False)
    epgs_code = gis_utils.get_epsg_code(topo_pre_event)
    gis_utils.write_geotiff('results/zTopo_2024_left.tif',xMesh, yMesh, zTopo, epgs_code)


    # Fan Boundary Right
    shortest_path_distance = SPM(topo_post_event, shape_fan_boundary_right)
    xMesh_crop, yMesh_crop, zMesh_crop, sMesh_crop = shortest_path_distance.within_boundary(pltFlag = False)

    # Fit quadratic elevation-distance relationship
    bin_size = 100
    ds = 5
    outlength = 1000
    fitting_s_z_within_boundary = EDR(sMesh_crop, zMesh_crop).medianFilter_on(bin_size, ds, outlength, pltFlag = False)
    
    xApex, yApex = shortest_path_distance.xyApex()
    volume_expected = gis_utils.calculate_volume_difference_within_polygon(topo_pre_event, topo_post_event, shape_fan_boundary_right)
    guessHeightAboveGround_bottom = 5
    guessHeightAboveGround_top = 30
    xMesh, yMesh, zTopo, _ = reconstruct_fantopo.reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, fitting_s_z_within_boundary, fanBoundarySHP=shape_fan_boundary_all, tol=0.03, debug=False)
    epgs_code = gis_utils.get_epsg_code(topo_pre_event)
    gis_utils.write_geotiff('results/zTopo_2024_right.tif',xMesh, yMesh, zTopo, epgs_code)

