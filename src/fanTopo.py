import numpy as np
import matplotlib.pyplot as plt
import visilibity as vis
from inpoly_cython import inpoly2
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from cone_function import cone_function
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from matplotlib.path import Path


def fill_nans_with_nearest(zTopo):
    """
    Fills NaN values in a 2D array using the nearest neighbor interpolation.

    Parameters:
    zTopo (np.ndarray): 2D array containing NaN values to be filled.

    Returns:
    np.ndarray: A copy of the input array with NaN values filled using the nearest valid data point.
    """
    # Create a mask to identify NaNs
    nan_mask = np.isnan(zTopo)
    
    if not np.any(nan_mask):
        return zTopo  # Return original array if no NaNs are found

    # Copy the original array to avoid modifying it directly
    filled_zTopo = zTopo.copy()

    # Use distance transform to find nearest valid data for each NaN
    _, nearest_indices = distance_transform_edt(nan_mask, return_indices=True)

    # Fill NaNs with the nearest non-NaN values
    filled_zTopo[nan_mask] = filled_zTopo[tuple(nearest_indices[:, nan_mask])]
    
    return filled_zTopo



def find_contour_coordinates(xMesh, yMesh, zMesh, level):
    """
    Find contour coordinates at a specified level using matplotlib.
    
    Parameters:
    xMesh (numpy.ndarray): 2D array of x-coordinates
    yMesh (numpy.ndarray): 2D array of y-coordinates
    zMesh (numpy.ndarray): 2D array of z-values
    level (float): The level at which to find contours
    combine (bool): If True, combine all contours into a single array
    
    Returns:
    list or numpy.ndarray: List of contour arrays, or a single combined array if combine=True
    """
    # Create a new figure (it won't be displayed)
    fig, ax = plt.subplots()
    
    # Generate contour
    contour_set = ax.contour(xMesh, yMesh, zMesh, levels=[level])
    
    # Extract contour paths
    contours = contour_set.collections[0].get_paths()
    
    # Convert to list of numpy arrays
    contour_coords = [contour.vertices for contour in contours]
    
    # Close the figure to free up memory
    plt.close(fig)

    return contour_coords

def create_environment(polygonX, polygonY):
    polygonAll = [] # First polygon in the list is wall. Others are obstacles in the walls.

    for i in range(len(polygonX)):
        # Validate the input polygon
        if len(polygonX[i]) != len(polygonY[i]):
            raise ValueError("The length of polygonX and polygonY must be equal.")
        
        if len(polygonX[i]) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        
        # Create vis.Point objects for each vertex of the polygon
        points = [vis.Point(x, y) for x, y in zip(polygonX[i][:-1], polygonY[i][:-1])]
        
        # Create the outer boundary polygon, ensuring it is counter-clockwise
        polygon = vis.Polygon(points)
        polygonAll.append(polygon)

    return vis.Environment(polygonAll)

def visi_polygon(polygonX, polygonY, observeX, observeY, epsilon=1e-8, snap_dist=0.2):
    """
    Creates a visibility polygon from an observer point within a given polygon.
    Wall: Need to follow counter-clockwise (CCW).
    Hole: CLOCK-WISE (CW). 

    Parameters:
    polygonX (list of float): X coordinates of the polygon vertices.
    polygonY (list of float): Y coordinates of the polygon vertices.
    observeX (float): X coordinate of the observer point.
    observeY (float): Y coordinate of the observer point.
    epsilon (float): Tolerance for geometric calculations (default is 1e-9).
    snap_dist (float): Distance within which points are snapped to boundaries or vertices (default is 0.2).

    Returns:
    tuple: A tuple containing four lists:
        - xVisi (list of float): X coordinates of the visibility polygon vertices.
        - yVisi (list of float): Y coordinates of the visibility polygon vertices.
        - xChildApex (list of float): X coordinates of the growing vertices in the visibility polygon.
        - yChildApex (list of float): Y coordinates of the growing vertices in the visibility polygon.
    """

    env = create_environment(polygonX, polygonY) 

    # Define the observer point
    observer = vis.Point(observeX, observeY)
    
    # Snap the observer to the boundary or vertices of the environment if within snap_dist

    if observer._in(env, epsilon):
        observer.snap_to_boundary_of(env, epsilon)
        observer.snap_to_vertices_of(env, epsilon)
    else:
        observer.snap_to_boundary_of(env, snap_dist)
        observer.snap_to_vertices_of(env, snap_dist)        

    # Generate the visibility polygon from the observer's perspective
    isovist = vis.Visibility_Polygon(observer, env, epsilon)

    # Extract the coordinates of the visibility polygon vertices
    xVisi = [isovist[j].x() for j in range(isovist.n())] + [isovist[0].x()]
    yVisi = [isovist[j].y() for j in range(isovist.n())] + [isovist[0].y()]
    
    # Extract the coordinates of the growing vertices within the visibility polygon
    child = isovist.get_growing_vertices()
    xChildApex = [child[j].x() for j in range(child.n())]
    yChildApex = [child[j].y() for j in range(child.n())]
    observer_snap = isovist.observer()
    observer_snap_x = np.array(observer_snap.x())
    observer_snap_y = np.array(observer_snap.y())
    
    return xVisi, yVisi, xChildApex, yChildApex, observer_snap_x, observer_snap_y


def compute_shortest_path(polygonX, polygonY, observeX, observeY, end_x, end_y, epsilon, snap_dist):
    if np.isnan(end_x) or np.isnan(end_y):
        return np.nan
    
    # Recreate the environment and observer objects inside each worker process
    env = create_environment(polygonX, polygonY)
    observer = vis.Point(observeX, observeY)

    observer.snap_to_boundary_of(env, snap_dist)
    observer.snap_to_vertices_of(env, snap_dist)
    
    end = vis.Point(end_x, end_y)
    shortest_path = env.shortest_path(observer, end, epsilon)
    return shortest_path.length()

def visi_polygon_shortest_path(polygonX, polygonY, observeX, observeY, endX, endY, epsilon=1e-8, snap_dist=0.2, max_workers=12):
    shortest_path_length = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compute_shortest_path, polygonX, polygonY, observeX, observeY, endX[i], endY[i], epsilon, snap_dist)
            for i in range(len(endX))
        ]

        for future in tqdm(futures, desc="Processing shortest paths"):
            try:
                shortest_path_length.append(future.result())
            except Exception as e:
                print(f"Error: {e}")
                shortest_path_length.append(np.nan)
    
    shortest_path_length = np.array(shortest_path_length)
    
    return shortest_path_length



def fan_topo(xMesh, yMesh, zMesh, xApexM, yApexM, zApexM, options={}):
    """
    FAN_TOPO constructs the constant-slope or concave fan morphology, the apex positions, and source provenance.

    Inputs:
    xMesh - 2D numpy array of x-coordinates for mesh grid points.
    yMesh - 2D numpy array of y-coordinates for mesh grid points.
    zMesh - 2D numpy array of initial elevation values before fan aggradation.
    xApexM - List or numpy array of x-coordinates for fan apex(es).
    yApexM - List or numpy array of y-coordinates for fan apex(es).
    zApexM - List or numpy array of z-coordinates (elevations) for fan apex(es).
    options - Dictionary containing optional parameters:
        'caseName' - (string) Type of fan morphology to generate (e.g., 'cone', 'concave', 'infinite', 'myProfile'). Default is 'cone'.
        'tanAlphaM' - (list) Slope angles (tangents) for each apex, defining fan steepness. Default is NaN.
        'KM' - (list) Concavity factors for each apex, controlling the curvature of the fan. Default is NaN.
        'tanInfiniteM' - (list) Slope values for cases where the tangent approaches infinity. Default is NaN.
        'dz_interpM' - (list of numpy arrays) Interpolation values for elevation, used in spline-based morphologies. Default is NaN.
        'dispflag' - (boolean) Flag to display the generated topography (True for on, False for off). Default is False.
        'saveVisPolygon' - (boolean) Flag to save visibility polygons (True for yes, False for no). Default is False.

    Outputs:
    zTopo - 2D numpy array of final fan topography (elevation after aggradation).
    kTopoAll - 2D numpy array with indices of the apex dominating each mesh grid point.
    xyzkApexAll - List of apex coordinates and indices (including child apexes).
    xyzVisPolygon - List of 3D coordinates (`x`, `y`, `z`) for visibility polygons.
    xyVisPolygonAll - List of `x` and `y` coordinates for all visibility polygons.
    thetaMesh - 2D numpy array of angular distribution relative to apex(es).
    """
    # Set default options
    options.setdefault('caseName', 'cone')
    options.setdefault('tanAlphaM', [np.nan] * len(zApexM))
    options.setdefault('KM', [np.nan] * len(zApexM))
    options.setdefault('tanInfiniteM', [np.nan] * len(zApexM))
    options.setdefault('dz_interpM', [np.nan] * len(zApexM))
    options.setdefault('dispflag', False)
    options.setdefault('saveVisPolygon', False)

    # Expand the x and y coordinates while preserving the order
    flip_lr = xMesh[0, 0] > xMesh[0, -1]
    flip_ud = yMesh[0, 0] > yMesh[-1, 0]

    if flip_lr:
        xMesh = np.fliplr(xMesh)
        zMesh = np.fliplr(zMesh)
    if flip_ud:
        yMesh = np.flipud(yMesh)
        zMesh = np.flipud(zMesh)

    # Add a high wall around the domain
    zMax = max(np.nanmax(zMesh), np.nanmax(zApexM))

    # Calculate the mesh spacing
    dxMesh = (xMesh[0, -1] - xMesh[0, 0]) / (xMesh.shape[1] - 1)
    dyMesh = (yMesh[-1, 0] - yMesh[0, 0]) / (yMesh.shape[0] - 1)

    # Create new expanded xMesh and yMesh
    xMesh_expanded = np.linspace(xMesh[0, 0] - dxMesh, xMesh[0, -1] + dxMesh, xMesh.shape[1] + 2)
    yMesh_expanded = np.linspace(yMesh[0, 0] - dyMesh, yMesh[-1, 0] + dyMesh, yMesh.shape[0] + 2)
    xMesh, yMesh = np.meshgrid(xMesh_expanded, yMesh_expanded)

    # Create a new zMesh with high walls
    zMesh = np.pad(zMesh, pad_width=1, mode='constant', constant_values=zMax)

    # Get the size of the new zMesh
    nr, nc = zMesh.shape

    # Initialize topography and sorted apex list
    xyzkApexAll = []
    kTopoAll = np.full_like(zMesh, np.nan)
    xyzVisPolygon = []
    xyVisPolygonAll = []

    if options['dispflag']:
        fig, ax = plt.subplots()

    zTopo = np.full_like(zMesh, np.nan)
    thetaMesh = np.full_like(zMesh, np.nan)

    for jj in range(len(zApexM)):
        kTopo = np.zeros_like(zMesh)
        xyzkApex = np.array([[xApexM[jj], yApexM[jj], zApexM[jj], np.nan]])
        kApex = 1

        # Loop over apexes
        while kApex <= xyzkApex.shape[0]:
            xApex, yApex, zApex = xyzkApex[kApex - 1, :3]
            
            # Find intersection polygon of cone surface and boundary surface
            D = np.sqrt((xMesh - xApex) ** 2 + (yMesh - yApex) ** 2)
            zCone = cone_function(zApex, D, {
                'caseName': options['caseName'],
                'tanAlpha': options['tanAlphaM'][jj],
                'K': options['KM'][jj],
                'zApex0': zApexM[jj],
                'tanInfinite': options['tanInfiniteM'][jj],
                'dz_interp': options['dz_interpM'][jj]
            })
            
            contour_coords = find_contour_coordinates(xMesh, yMesh, zCone - zMesh, 1e-6)
            n_nodes = len(contour_coords[0])
            
            if n_nodes > 5:  # Ignore the apex whose impact is too small
                xContour = [contour_coords[i][:,0] for i in range(len(contour_coords))]
                yContour = [contour_coords[i][:,1] for i in range(len(contour_coords))]

                xVisi, yVisi, xChildApex, yChildApex, _, _ = visi_polygon(xContour, yContour, xApex, yApex, snap_dist = max(dxMesh,dyMesh)*np.sqrt(2))
                
                if len(xVisi) > 5:  # Ignore the apex whose impact is too small
                    if options['saveVisPolygon']:
                        D_Visi = np.sqrt((xVisi - xApex) ** 2 + (yVisi - yApex) ** 2)
                        zVisi = cone_function(zApex, D_Visi, {
                            'caseName': options['caseName'],
                            'tanAlpha': options['tanAlphaM'][jj],
                            'K': options['KM'][jj],
                            'zApex0': zApexM[jj],
                            'tanInfinite': options['tanInfiniteM'][jj],
                            'dz_interp': options['dz_interpM'][jj]
                        })
                        xyzVisPolygon.append(np.column_stack((xVisi, yVisi, zVisi)))

                    # Update fan surface to the visible sector occluded by boundary surface and other fan sectors
                    xyVisi = [[xVisi[i], yVisi[i]] for i in range(len(xVisi))]
                    
                    isVisible,_ = inpoly2(np.column_stack((xMesh.flatten(), yMesh.flatten())), xyVisi)
                    isVisible = isVisible.reshape(nr, nc)

                    thetaMesh_temp = np.arctan2(xMesh - xApex, yMesh - yApex)
                    mask = (isVisible & (zCone > zTopo)) | (isVisible & np.isnan(zTopo))
                    thetaMesh[mask] = thetaMesh_temp[mask]
                    zTopo[mask] = zCone[mask]
                    kTopo[mask] = kApex

                    if options['saveVisPolygon']:
                        if not xyVisPolygonAll:
                            xyVisPolygonAll = np.column_stack((xVisi, yVisi))
                        else:
                            isVisible,_ = inpoly2(np.column_stack((xyVisPolygonAll[:, 0], xyVisPolygonAll[:, 1])), xyVisi)
                            xyVisPolygonAll = np.delete(xyVisPolygonAll, np.where(isVisible), axis=0)
                            xyVisPolygonAll = np.vstack((xyVisPolygonAll, np.column_stack((xVisi, yVisi))))

                    # Add effective children apexes into the apex list
                    CTopo = find_contour_coordinates(xMesh, yMesh, kTopo, 0)                   
                    CTopo_withnan = CTopo.copy()
                    CTopo = np.vstack(CTopo).T
                    


                    min_d_xyVisi = np.min(np.sqrt(np.diff(xVisi) ** 2 + np.diff(yVisi) ** 2))  # Threshold for finding the semi-apexes that are too close
                    D = np.sqrt((xChildApex - xApex) ** 2 + (yChildApex - yApex) ** 2)
                    zConeChildApex = cone_function(zApex, D, {
                        'caseName': options['caseName'],
                        'tanAlpha': options['tanAlphaM'][jj],
                        'K': options['KM'][jj],
                        'zApex0': zApexM[jj],
                        'tanInfinite': options['tanInfiniteM'][jj],
                        'dz_interp': options['dz_interpM'][jj]
                    })

                    for i in range(len(xChildApex)):
                        dist_CTopo = np.min(np.sqrt((CTopo[0,:] - xChildApex[i]) ** 2 + (CTopo[1,:] - yChildApex[i]) ** 2))
                        if dist_CTopo < np.sqrt(2) * dxMesh * 2:  # Only keep the semi-apex that is close to the boundary of previous visibility polygons
                            dx_existChildApex = xyzkApex[:, 0] - xChildApex[i]
                            dy_existChildApex = xyzkApex[:, 1] - yChildApex[i]
                            ds_existChildApex = np.sqrt(dx_existChildApex ** 2 + dy_existChildApex ** 2)  # Distance to existing semi-apexes
                            isTooClose = np.where(ds_existChildApex < min_d_xyVisi / 4)[0]  # Find the existing semi-apexes that are too close to the new semi-apex
                            isTooClose = isTooClose[isTooClose != (kApex-1)]
                            isSameXorY = np.where((dx_existChildApex == 0) & (np.abs(dy_existChildApex) < dyMesh / 8) | (np.abs(dx_existChildApex) < dxMesh / 8) & (dy_existChildApex == 0))[0]  # Find the existing semi-apexes that have the same x or y coordinate as the new semi-apex
                            isSameXorY = isSameXorY[isSameXorY != (kApex-1)]
                            if len(isTooClose) > 0 or len(isSameXorY) > 0:
                                # Update the z value of the too-close/sameXorY existing semi-apexes
                                D = np.sqrt((xyzkApex[isTooClose, 0] - xApex) ** 2 + (xyzkApex[isTooClose, 1] - yApex) ** 2)
                                xyzkApex[isTooClose, 2] = np.maximum(xyzkApex[isTooClose, 2], cone_function(zApex, D, {
                                    'caseName': options['caseName'],
                                    'tanAlpha': options['tanAlphaM'][jj],
                                    'K': options['KM'][jj],
                                    'zApex0': zApexM[jj],
                                    'tanInfinite': options['tanInfiniteM'][jj],
                                    'dz_interp': options['dz_interpM'][jj]
                                }))
                                D = np.sqrt((xyzkApex[isSameXorY, 0] - xApex) ** 2 + (xyzkApex[isSameXorY, 1] - yApex) ** 2)
                                xyzkApex[isSameXorY, 2] = np.maximum(xyzkApex[isSameXorY, 2], cone_function(zApex, D, {
                                    'caseName': options['caseName'],
                                    'tanAlpha': options['tanAlphaM'][jj],
                                    'K': options['KM'][jj],
                                    'zApex0': zApexM[jj],
                                    'tanInfinite': options['tanInfiniteM'][jj],
                                    'dz_interp': options['dz_interpM'][jj]
                                }))
                            else:
                                # Add new semi-apex
                                zChildApex = zConeChildApex[i]
                                xyzkApex = np.vstack((xyzkApex, [xChildApex[i], yChildApex[i], zChildApex, kApex]))
                    
                    # Remove buried apexes
                    fill_nan_zTopo = fill_nans_with_nearest(zTopo)
                    interp_func = RegularGridInterpolator((xMesh[0, :], yMesh[:, 0]), fill_nan_zTopo.T)
                    zAtopo = interp_func(xyzkApex[:, 0:2])
                    zAtopo_vale = cone_function(zAtopo, 2*np.sqrt(2)*dxMesh, {
                        'caseName': options['caseName'],
                        'tanAlpha': options['tanAlphaM'][jj],
                        'K': options['KM'][jj],
                        'zApex0': zApexM[jj],
                        'tanInfinite': options['tanInfiniteM'][jj],
                        'dz_interp': options['dz_interpM'][jj]
                    })
                    xyzkApex = xyzkApex[~(xyzkApex[:, 2] <= zAtopo_vale), :]
                    # Sort the apexes by elevation
                    if xyzkApex.shape[0] > kApex:
                        xyzkApex[(kApex-1):, :] = xyzkApex[(kApex-1):, :][xyzkApex[(kApex-1):, 2].argsort()[::-1]]

                else:
                    if options['saveVisPolygon']:
                        xyzVisPolygon.append([np.nan, np.nan, np.nan])
            else:
                if options['saveVisPolygon']:
                    xyzVisPolygon.append([np.nan, np.nan, np.nan])

            # Show topography, active fan contour, and visible sector and apexes
            if options['dispflag']:
                if kApex is None:
                    fig, ax = plt.subplots()
                ax.clear()
                ax.contourf(xMesh,yMesh,zTopo)
                for i in range(len(xContour)):
                    ax.plot(xContour[i], yContour[i], 'b-')
                    # for j in range(len(xContour[i])):
                    #     ax.text(xContour[i][j], yContour[i][j], str(j), fontsize=12, color='red')
                ax.plot(xyzkApex[:,0],xyzkApex[:,1],'k.-')
                
                for i in range(len(CTopo_withnan)):
                    ax.plot(CTopo_withnan[i][:,0], CTopo_withnan[i][:,1], 'r--')
                
                ax.plot(xVisi, yVisi, 'r-')
                ax.plot(xChildApex, yChildApex, 'g*')
                ax.plot(xApex, yApex, 'ro')
                
                ax.set_xlim(np.nanmin(xMesh), np.nanmax(xMesh))
                ax.set_ylim(np.nanmin(yMesh), np.nanmax(yMesh))
                ax.set_aspect('equal')
                plt.title(str(kApex))
                plt.draw()
                if kApex is None:
                    plt.show(block=True)
                plt.pause(0.01)

            # Proceed to next apex on the list
            kApex += 1

        if jj > 0:
            xyzkApex[:, 3] += len(xyzkApexAll)
            kTopoAll[kTopo > 0] = kTopo[kTopo > 0] + len(xyzkApexAll)
        else:
            kTopoAll = kTopo
        xyzkApexAll.extend(xyzkApex.tolist())

        zMesh[~np.isnan(zTopo)] = zTopo[~np.isnan(zTopo)]

    if options['dispflag']:
        plt.close()

    # Remove the high wall
    zTopo = zTopo[1:-1, 1:-1]
    thetaMesh = thetaMesh[1:-1, 1:-1]
    kTopoAll = kTopoAll[1:-1, 1:-1]
    kTopoAll[kTopoAll == 0] = np.nan

    # Flip back if necessary
    if flip_ud:
        zTopo = np.flipud(zTopo)
        thetaMesh = np.flipud(thetaMesh)
        kTopoAll = np.flipud(kTopoAll)
    if flip_lr:
        zTopo = np.fliplr(zTopo)
        thetaMesh = np.fliplr(thetaMesh)
        kTopoAll = np.fliplr(kTopoAll)

    return zTopo, kTopoAll, xyzkApexAll, xyzVisPolygon, xyVisPolygonAll, thetaMesh

