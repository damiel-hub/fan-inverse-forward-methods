import numpy as np
import matplotlib.pyplot as plt
import fanTopo
import gis_utils
from scipy.interpolate import RegularGridInterpolator

class SPM:
    def __init__(self, topo_tiffile, fanboundary_shapefile, xyApex = None):

        self.xMesh_crop, self.yMesh_crop, self.zMesh_crop = gis_utils.clip_geotiff_with_shapefile(topo_tiffile, fanboundary_shapefile)
        self.xMesh, self.yMesh, self.zMesh = gis_utils.read_geotiff(topo_tiffile)
        self.all_fan_boundary_x, self.all_fan_boundary_y = gis_utils.read_shapefile_boundary(fanboundary_shapefile)

        if xyApex is None:
            # Find the apex (highest point) in the mesh grid
            iApex = np.nanargmax(self.zMesh_crop)  # Use nanargmax to ignore NaNs
            iApex_y, iApex_x = np.unravel_index(iApex, self.zMesh_crop.shape)  # Convert to 2D index

            # Get the coordinates of the apex
            self.xApex = self.xMesh_crop[iApex_y, iApex_x]
            self.yApex = self.yMesh_crop[iApex_y, iApex_x]
        else:
            self.xApex = xyApex[0]
            self.yApex = xyApex[1]

    def xyApex(self):
        return self.xApex, self.yApex

    def within_boundary(self, pltFlag=False, debug = False):
        '''
            Process the shortest path map within the boundary.
        '''
        # Calculate the diagonal length of the mesh grid
        diagonal_length = np.sqrt((self.xMesh_crop[0, 0] - self.xMesh_crop[0, -1])**2 + 
                                (self.yMesh_crop[0, 0] - self.yMesh_crop[-1, 0])**2)
        
        # Create a wall mesh with boundary values set to a high value
        wallMesh = np.zeros_like(self.zMesh_crop)
        wallMesh[np.isnan(self.zMesh_crop)] = diagonal_length * 10
        
        # Set the apex height for the shortest path calculation
        zApex_s = diagonal_length * 10 - 1e-4
        
        # Compute the shortest path topography using a hypothetical FanTopo function
        sTopo = fanTopo.fan_topo(self.xMesh_crop, self.yMesh_crop, wallMesh, [self.xApex], [self.yApex], [zApex_s], {
            'caseName': 'cone',
            'tanAlphaM': [1],
            'dispflag': debug
        })
        
        # Calculate the shortest path distance map
        sMap = zApex_s - sTopo[0]
        
        # Plot the shortest path distance map if pltFlag is true
        if pltFlag:
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(self.xMesh_crop, self.yMesh_crop, sMap, shading='auto')
            plt.colorbar(label='Shortest Path Distance (m)')
            plt.plot(self.xApex, self.yApex, 'r.', markersize=6)
            plt.contour(self.xMesh_crop, self.yMesh_crop, sMap, levels=np.arange(0, np.nanmax(sMap), 100), colors='k')
            plt.axis('equal')
            plt.title('Shortest Path Distance Map')
            plt.xlabel('Easting (m)')
            plt.ylabel('Northing (m)')
            plt.show(block = False)
            plt.pause(0.01)
            # plt.close()
        
        return self.xMesh_crop, self.yMesh_crop, self.zMesh_crop, sMap

    def resample_boundary_by_distance(self, all_fan_boundary_x, all_fan_boundary_y, ds):
        resampled_x_total = []
        resampled_y_total = []

        for segment_x, segment_y in zip(all_fan_boundary_x, all_fan_boundary_y):
            # Ensure the boundary segment is closed by appending the first point to the end if necessary
            if segment_x[0] != segment_x[-1] or segment_y[0] != segment_y[-1]:
                segment_x = np.append(segment_x, segment_x[0])
                segment_y = np.append(segment_y, segment_y[0])
            
            # Calculate cumulative distance along the boundary segment
            dist = np.sqrt(np.diff(segment_x)**2 + np.diff(segment_y)**2)
            cumdist = np.insert(np.cumsum(dist), 0, 0)  # Add a zero at the start

            # Generate points at intervals of ds along the cumulative distance
            new_cumdist = np.arange(0, cumdist[-1], ds)
            
            # If the last point doesn't fall exactly on a distance multiple of ds, include it
            if cumdist[-1] % ds != 0:
                new_cumdist = np.append(new_cumdist, cumdist[-1])
            
            # Use numpy.interp to perform the interpolation
            resampled_x = np.interp(new_cumdist, cumdist, segment_x)
            resampled_y = np.interp(new_cumdist, cumdist, segment_y)
            
            resampled_x_total.append(resampled_x)
            resampled_y_total.append(resampled_y)

        return resampled_x_total, resampled_y_total

    def interpolate_with_nan(self, x_total, y_total, interpolator):
        # Find the indices of NaNs
        nan_indices = np.isnan(x_total) | np.isnan(y_total)

        # Initialize an array to store the interpolated z values, with NaNs where appropriate
        z_total = np.full_like(x_total, np.nan)

        # Identify the indices where NaNs occur
        split_indices = np.where(nan_indices)[0]

        # Add the end of the array as a split point
        split_indices = np.concatenate(([0], split_indices, [len(x_total)]))

        # Use the split indices to extract segments without NaNs
        valid_segments = [np.arange(split_indices[i], split_indices[i+1])[~nan_indices[split_indices[i]:split_indices[i+1]]]
                  for i in range(len(split_indices) - 1)]

        valid_segments = [seg for seg in valid_segments if len(seg) > 0]

        # Perform interpolation on each valid segment
        for segment in valid_segments:
            segment_x = x_total[segment]
            segment_y = y_total[segment]

            if len(segment_x) > 0 and len(segment_y) > 0:
                points = np.array([segment_y, segment_x]).T  # Note: (y, x) order
                z_total[segment] = interpolator(points)

        return z_total
    
    def along_boundary(self, pltFlag = False):
        '''
            Process the shortest path distance along the boundary.
        '''
        dx = np.abs(np.diff(self.xMesh_crop[0, :]).mean())
        dy = np.abs(np.diff(self.yMesh_crop[:, 0]).mean())

        resampled_x_total, resampled_y_total = self.resample_boundary_by_distance(self.all_fan_boundary_x, self.all_fan_boundary_y, min(dx, dy))
        
        # add NaN values between the concatenated arrays to break the connections
        nan_array = np.array([np.nan])
        resampled_x_total = np.concatenate([np.concatenate([arr, nan_array]) for arr in resampled_x_total])[:-1]
        resampled_y_total = np.concatenate([np.concatenate([arr, nan_array]) for arr in resampled_y_total])[:-1]
        
        resampled_s_total = fanTopo.visi_polygon_shortest_path(self.all_fan_boundary_x, self.all_fan_boundary_y, self.xApex, self.yApex, resampled_x_total, resampled_y_total)

        x = self.xMesh[0, :]  # Extract the x-axis values (assuming xMesh is uniform)
        y = self.yMesh[:, 0]  # Extract the y-axis values (assuming yMesh is uniform)

        # Create the interpolator function
        interpolator = RegularGridInterpolator((y, x), self.zMesh)

        # Prepare the points for interpolation
        resampled_z_total = self.interpolate_with_nan(resampled_x_total, resampled_y_total, interpolator)      
        
        # Plot the shortest path distance map if pltFlag is true
        if pltFlag:
            # Create a scatter plot with colors based on shortest_path_length
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(resampled_x_total, resampled_y_total, c=resampled_s_total, cmap='viridis', s=50, edgecolor='k')

            # Add a colorbar to show the color scale
            plt.colorbar(scatter, label='Shortest Path Distance')

            # Add labels and title
            plt.xlabel('Resampled E Total')
            plt.ylabel('Resampled N Total')
            plt.title('Points Colored by Shortest Path Distance')

            # Show plot
            plt.show(block = False)
            plt.pause(0.01)
            # plt.close()
        
        return resampled_x_total, resampled_y_total, resampled_z_total, resampled_s_total