import sys
sys.path.append('src/')
import gis_utils
import numpy as np
import matplotlib.pyplot as plt
import plt_utils

zTopo = [r"results\zTopo_2024_all.tif",
             r"results\zTopo_2024_left.tif",
             r"results\zTopo_2024_right.tif"]
fanBoundary = [r"datasets\raw\shape\PT2024.shp",
               r"datasets\raw\shape\PT2024_left.shp",
               r"datasets\raw\shape\PT2024_right.shp"]
color_sequence = ['m', 'c', 'y']

title_str = ['All', 'Left', 'Right']
zInit = r"datasets\raw\tif\topo_10m_PuTunPuNas_pre_2024_min_base2024.tif"
zPost = r"datasets\raw\tif\topo_10m_PuTunPuNas_2024.tif"
xMesh, yMesh, zInitMesh = gis_utils.read_geotiff(zInit)
_, _, zPostMesh = gis_utils.read_geotiff(zPost)

for i in range(len(zTopo)):
    xMesh, yMesh, zTopoMesh = gis_utils.read_geotiff(zTopo[i])
    nanIndexMesh = np.isnan(zTopoMesh)
    zMapMesh = zTopoMesh.copy()
    zMapMesh[nanIndexMesh] = zInitMesh[nanIndexMesh]
    plt_utils.plot_hillshade(xMesh, yMesh, zMapMesh, azdeg=180, altdeg=45, vert_exag=5)
    dxMesh = xMesh[0,1] - xMesh[0,0]
    dyMesh = yMesh[0,0] - yMesh[1,0]
    plt.imshow(zTopoMesh, extent=(xMesh.min()-dxMesh/2, xMesh.max()+dxMesh/2, yMesh.min()-dyMesh/2, yMesh.max()+dyMesh/2), origin='upper', cmap='turbo')
    plt.colorbar(label = 'Fan Topo elevation, z (m)')
    interval = 10
    levels = np.arange(zMapMesh.min(), zMapMesh.max(), interval)
    plt.contour(xMesh, yMesh, zMapMesh, levels=levels, colors='k', linewidths=0.7)
    plt.axis([226905, 229635, 2564093, 2567691])
    plt.title(title_str[i])
    plt.savefig('results/fill_fanTopo/' + title_str[i] + '_fig1.png', dpi=300)

    plt_utils.plot_hillshade(xMesh, yMesh, zPostMesh, azdeg=180, altdeg=45, vert_exag=5)
    plt.contour(xMesh, yMesh, zPostMesh, levels=levels, colors='k', linewidths=0.7)
    plt_utils.plot_polygon_boundary_gdal(fanBoundary[i], exterior_color=color_sequence[i], interior_color=color_sequence[i])
    plt.axis([226905, 229635, 2564700, 2567200])
    plt.colorbar(label='Hillshade intensity')
    plt.savefig('results/fill_fanTopo/' + title_str[i] + '_fig2.png', dpi=300)

    plt_utils.plot_hillshade(xMesh, yMesh, zMapMesh, azdeg=180, altdeg=45, vert_exag=5)
    gis_utils.calculate_volume_difference_within_polygon(zPost, zTopo[i], output_resampled_path=None, pltFlag=True, colorBarStr='z_sim - z_field', vmin = -20, vmax = 20)
    plt.contour(xMesh, yMesh, zMapMesh, levels=levels, colors='k', linewidths=0.7)
    plt_utils.plot_polygon_boundary_gdal(fanBoundary[0], exterior_color='b', interior_color='b')
    plt.title(title_str[i] + ' Simulation Error')
    plt.axis([226905, 229635, 2564093, 2567691])
    plt.savefig('results/fill_fanTopo/' + title_str[i] + '_fig3.png', dpi=300)
plt.show(block = True)
    
