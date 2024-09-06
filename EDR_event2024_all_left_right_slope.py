import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

sys.path.append('src/')
from SPM import SPM
from EDR import EDR

if __name__ == "__main__":

    topo_post_event = r"datasets\raw\tif\topo_10m_PuTunPuNas_2024.tif"
    shape_fan_boundary_all = r"datasets\raw\shape\PT2024.shp"
    shape_fan_boundary_left = r"datasets\raw\shape\PT2024_left.shp"
    shape_fan_boundary_right = r"datasets\raw\shape\PT2024_right.shp"
    # Parameter Fit quadratic elevation-distance relationship
    bin_size = 100
    ds = 5
    outlength = 1000

    _, _, zMesh_crop_all, sMesh_crop_all = SPM(topo_post_event, shape_fan_boundary_all).within_boundary(pltFlag = False)
    _, _, zMesh_crop_left, sMesh_crop_left = SPM(topo_post_event, shape_fan_boundary_left).within_boundary(pltFlag = False)
    _, _, zMesh_crop_right, sMesh_crop_right = SPM(topo_post_event, shape_fan_boundary_right).within_boundary(pltFlag = False)

    xmin, xmax, ymin, ymax = -500, 3000, 560, 900

    plt.figure()
    EDR_all = EDR(sMesh_crop_all, zMesh_crop_all, drawFlag = True)
    EDR_all.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'm')
    plt.xlabel('Shortest path distance to all data points, s (m)')
    plt.ylabel('Elevation, z (m)')
    plt.gca().set_aspect(5)
    plt.grid(True)
    plt.box(True)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(r'results\event2024_all_left_right_slope\EDR_compare_all.png', dpi=300, bbox_inches='tight')

    plt.figure()
    EDR_left = EDR(sMesh_crop_left, zMesh_crop_left, drawFlag = True)
    EDR_left.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'c')
    plt.xlabel('Shortest path distance to all data points, s (m)')
    plt.ylabel('Elevation, z (m)')
    plt.gca().set_aspect(5)
    plt.grid(True)
    plt.box(True)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(r'results\event2024_all_left_right_slope\EDR_compare_left.png', dpi=300, bbox_inches='tight')

    plt.figure()
    EDR_right = EDR(sMesh_crop_right, zMesh_crop_right, drawFlag = True)
    EDR_right.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'y')
    plt.xlabel('Shortest path distance to all data points, s (m)')
    plt.ylabel('Elevation, z (m)')
    plt.gca().set_aspect(5)
    plt.grid(True)
    plt.box(True)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(r'results\event2024_all_left_right_slope\EDR_compare_right.png', dpi=300, bbox_inches='tight')

    plt.figure()
    EDR_all.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'm')
    EDR_left.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'c')
    EDR_right.medianFilter_on(bin_size, ds, outlength, pltFlag = False, drawFlag = 'y')
    plt.xlabel('Shortest path distance to all data points, s (m)')
    plt.ylabel('Elevation, z (m)')
    plt.gca().set_aspect(5)
    plt.grid(True)
    plt.box(True)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(r'results\event2024_all_left_right_slope\EDR_compare.png', dpi=300, bbox_inches='tight')
    plt.show(block = True)