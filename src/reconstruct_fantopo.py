import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RegularGridInterpolator, interp1d
import fanTopo
import gis_utils

# Function to simulate and calculate volume
def simulate_height_volume(params):
    guessHeight, xMesh, yMesh, zMesh, xApex, yApex, sz_profile, volumeCalculateMask, interpolator, debug, epsg_code = params

    zApex = interpolator((yApex, xApex)) + guessHeight
    zTopo = fanTopo.fan_topo(xMesh, yMesh, zMesh, [xApex], [yApex], [zApex], {
        'caseName': 'myProfile',
        'sz_interpM': [sz_profile],
        'dispflag':debug        
    })
    
    dod = zTopo[0] - zMesh
    dod[~volumeCalculateMask] = np.nan

    fanVolume = np.nansum(dod) * (xMesh[0, 1] - xMesh[0, 0])**2

    if np.any(~volumeCalculateMask):
        print(f'HAG = {guessHeight:.2f} [L], Volume = {fanVolume:.2f} [L^3], within given boundary')
    else:
        print(f'HAG = {guessHeight:.2f} [L], Volume = {fanVolume:.2f} [L^3], within simulation area')
    
    if debug:
        file_name = 'guessHeight' + f'{fanVolume:.0f}.tif'
        gis_utils.write_geotiff('datasets/processed/'+file_name, xMesh, yMesh, zTopo[0], epsg_code)
    
    return guessHeight, fanVolume

def fanTopoSimVolumeMask(interpV, minMaxInitialGuessHeightVolume, xMesh, yMesh, zMesh, xApex, yApex, volMask, sz_profile, epsg_code, tol=0.03, debug=False):
    """
    Simulates fan topography at the given target volumes (interpV) within a specified mask (volMask), based on the provided parameters.

    Parameters:
        interpV (numpy.ndarray): Array of target volumes to simulate.
        minMaxInitialGuessHeightVolume (numpy.ndarray): Initial guesses for height above ground (HAG) and volume relationships.
            Shape should be (n, 2), where the first column represents HAG and the second column represents volume.
        xMesh (numpy.ndarray): 2D array representing the x-coordinates of the mesh grid.
        yMesh (numpy.ndarray): 2D array representing the y-coordinates of the mesh grid.
        zMesh (numpy.ndarray): 2D array representing the z-coordinates (elevation) of the mesh grid.
        xApex (float): x-coordinate of the apex point.
        yApex (float): y-coordinate of the apex point.
        volMask (numpy.ndarray): Boolean mask array indicating the valid region for volume calculation.
        sz_profile (numpy.ndarray): elevation-distance relationship extracted using inverse method.
        epsg_code (int): EPSG code for the coordinate reference system used in geospatial data.
        tol (float, optional): Tolerance for volume matching. Default is 0.03 (3%).
        debug (bool, optional): If True, enables debug mode with additional outputs. Default is False.

    Returns:
        tuple:
            - numpy.ndarray: Simulated topography (zTopo) after the fan topography process.
            - numpy.ndarray: Updated height above ground (HAG) and volume relationship array.

    Raises:
        ValueError: If the initial apex height guesses are invalid (e.g., too high or too low).

    Notes:
        - The function iteratively adjusts the height above ground (HAG) to match the target volumes (interpV).
        - If the tolerance cannot be achieved, a warning is printed, and the process continues.
        - Debug mode saves intermediate results as GeoTIFF files for visualization.

    Example:
        zTopo, heightAG_Volume_All = fanTopoSimVolumeMask(
            interpV=np.array([1000, 2000, 3000]),
            minMaxInitialGuessHeightVolume=np.array([[10, 1000], [20, 2000]]),
            xMesh=x_mesh,
            yMesh=y_mesh,
            zMesh=z_mesh,
            xApex=500,
            yApex=500,
            volMask=volume_mask,
            sz_profile=s_z_profile,
            epsg_code=4326,
            tol=0.03,
            debug=True
        )
    """
    # Initial error check for apex heights
    minMaxVolumeDiff = minMaxInitialGuessHeightVolume[:, 1] - np.min(interpV)
    if np.all(minMaxVolumeDiff >= 0):
        raise ValueError('Bottom apex too high')
    elif np.all(minMaxVolumeDiff <= 0):
        raise ValueError('Bottom and Top apex too low')
    elif np.all(minMaxInitialGuessHeightVolume[:, 1] - np.max(interpV) >= 0):
        raise ValueError('Top apex too low')

    # Interpolate to find the initial zApexGround
    interpolator = RegularGridInterpolator((yMesh[:, 0], xMesh[0, :]), zMesh)
    zApexGround = interpolator((yApex, xApex))

    # Use the previously calculated height above ground and volume relationship
    if minMaxInitialGuessHeightVolume.shape[0] > 2:
        initialFlag = False
        heightAG_Volume_All = minMaxInitialGuessHeightVolume
    else:
        initialFlag = True

    prefanVolume = -9999
    while interpV.size > 0:
        # Remove negative volumes
        interpV = interpV[interpV >= 0]
        if interpV.size == 0:
            break

        if not initialFlag:
            _, uniqueIdx = np.unique(heightAG_Volume_All[:, 1], return_index=True)
            heightAG_Volume_All = heightAG_Volume_All[uniqueIdx, :]
        else:
            heightAG_Volume_All = minMaxInitialGuessHeightVolume
            initialFlag = False

        # Interpolate to get heights for the desired volumes
        interp1d_func = interp1d(heightAG_Volume_All[:, 1], heightAG_Volume_All[:, 0], fill_value="extrapolate")
        interpHAG = interp1d_func(interpV)

        for i in range(len(interpHAG)):
            # Run fan topo simulation process
            zApex = zApexGround + interpHAG[i]
            zTopo= fanTopo.fan_topo(xMesh, yMesh, zMesh, [xApex], [yApex], [zApex], {
            'caseName': 'myProfile',
            'sz_interpM': [sz_profile],
            'dispflag':debug
            })
            
            dod = zTopo[0] - zMesh
            dod[~volMask] = np.nan
            
            fanVolume = np.nansum(dod) * (xMesh[0, 1] - xMesh[0, 0])**2

            if np.any(~volMask):
                print(f'HAG = {interpHAG[i]:.2f} [L], Volume = {fanVolume:.2f} [L^3], within given boundary')
            else:
                print(f'HAG = {interpHAG[i]:.2f} [L], Volume = {fanVolume:.2f} [L^3], within simulation area')
            
            if debug:
                file_name = 'volume_' + f'{fanVolume:.0f}.tif'
                gis_utils.write_geotiff('datasets/processed/' + file_name, xMesh, yMesh, zTopo[0], epsg_code)

            heightAG_Volume_All = np.vstack([heightAG_Volume_All, [interpHAG[i], fanVolume]])
            
            if abs(fanVolume - interpV[i]) <= interpV[i] * tol:
                interpV[i] = -1  # Mark as processed
            elif round(prefanVolume) == round(fanVolume):
                print(f"The tolerance of {tol} could not be achieved. The resulting tolerance is {abs(fanVolume - interpV[i]) / interpV[i]:.6f}.")
                interpV[i] = -1
            prefanVolume = fanVolume

    return zTopo[0], heightAG_Volume_All



def reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, sz_profile, fanBoundarySHP=None, tol=0.03, debug=False):
    """
    Simulates and reconstructs a fan surface based on pre-event topography, apex location, 
    expected volume, and other parameters. The function uses parallel processing to optimize 
    the simulation of height-volume relationships and adjusts the surface to match the 
    expected volume within a defined boundary.
    Args:
        topo_pre_event (str): Filepath to the pre-event topography GeoTIFF file.
        xApex (float): X-coordinate of the fan apex.
        yApex (float): Y-coordinate of the fan apex.
        volume_expected (float): Expected volume of the fan.
        guessHeightAboveGround_top (float): Initial guess for the maximum height above ground.
        guessHeightAboveGround_bottom (float): Initial guess for the minimum height above ground.
        sz_profile (numpy.ndarray): elevation-distance relationship extracted using inverse method.
        fanBoundarySHP (str, optional): Filepath to the shapefile defining the boundary for calculating the volume. 
                                        If None, the entire area is considered. Defaults to None.
        tol (float, optional): Tolerance for volume matching. Defaults to 0.03.
        debug (bool, optional): If True, enables debug mode with additional outputs. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - xMesh (numpy.ndarray): X-coordinates of the mesh grid.
            - yMesh (numpy.ndarray): Y-coordinates of the mesh grid.
            - zTopo_sim (numpy.ndarray): Simulated topography after reconstruction.
            - heightAG_Volume_All (numpy.ndarray): Array of height above ground and corresponding volumes.
    """
    
    print(f'The expected simulated fan volume is {volume_expected:.2f} cubic meters (L^3) within the defined boundary.')
    
    xMesh, yMesh, zMesh = gis_utils.read_geotiff(topo_pre_event)
    epsg_code = gis_utils.get_epsg_code(topo_pre_event)

    # Handle fan boundary if provided
    if fanBoundarySHP is not None:
        volumeCalculateMask = gis_utils.create_mask_from_shapefile(topo_pre_event, fanBoundarySHP).astype(bool)
    else:
        volumeCalculateMask = np.ones_like(zMesh, dtype=bool).astype(bool)
    
    # Simulation Process
    heights = [guessHeightAboveGround_bottom, guessHeightAboveGround_top]
    minMaxInitialGuessHeightVolume = np.zeros((2, 2))

    # Create RegularGridInterpolator
    interpolator = RegularGridInterpolator((yMesh[:, 0], xMesh[0, :]), zMesh)

    # Prepare parameters for parallel processing
    params = [(height, xMesh, yMesh, zMesh, xApex, yApex, sz_profile, volumeCalculateMask, interpolator, debug, epsg_code) for height in heights]

    # Use ProcessPoolExecutor to parallelize the height simulation only for the two pre-guess heights
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_height_volume, params))
    
    minMaxInitialGuessHeightVolume = np.array(results)

    # Further processing with fanTopoSimVolumeMask
    zTopo_sim, heightAG_Volume_All = fanTopoSimVolumeMask(
        volume_expected, minMaxInitialGuessHeightVolume, xMesh, yMesh, zMesh, 
        xApex, yApex, volumeCalculateMask, sz_profile, epsg_code, tol, debug
    )

    return xMesh, yMesh, zTopo_sim, heightAG_Volume_All
