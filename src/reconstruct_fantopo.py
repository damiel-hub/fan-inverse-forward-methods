import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RegularGridInterpolator, interp1d
import fanTopo
from inpoly_cython import inpoly2
import gis_utils

# Function to simulate and calculate volume
def simulate_height_volume(params):
    guessHeight, xMesh, yMesh, zMesh, xApex, yApex, dz_profile, volumeCalculateMask, interpolator, debug, epsg_code = params

    zApex = interpolator((yApex, xApex)) + guessHeight
    zTopo = fanTopo.fan_topo(xMesh, yMesh, zMesh, [xApex], [yApex], [zApex], {
        'caseName': 'myProfile',
        'dz_interpM': [dz_profile],
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

def fanTopoSimVolumeMask(interpV, minMaxInitialGuessHeightVolume, xMesh, yMesh, zMesh, xApex, yApex, volMask, dz_profile, epsg_code, tol=0.03, debug=False):
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
            'dz_interpM': [dz_profile],
            'dispflag':debug
        })
            
            zMesh_sim = np.copy(zTopo[0])
            zMesh_sim[np.isnan(zMesh_sim)] = zMesh[np.isnan(zMesh_sim)]
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

    return zTopo[0], heightAG_Volume_All


def reconstruct_fan_surface(topo_pre_event, xApex, yApex, volume_expected, guessHeightAboveGround_top, guessHeightAboveGround_bottom, dz_profile, fanBoundarySHP=None, tol=0.03, debug=False):
    
    print(f'The expected simulated fan volume is {volume_expected:.2f} cubic meters (L^3) within the defined boundary.')
    
    xMesh, yMesh, zMesh = gis_utils.read_geotiff(topo_pre_event)
    epsg_code = gis_utils.get_epsg_code(topo_pre_event)

    # Handle fan boundary if provided
    if fanBoundarySHP is not None:
        fan_boundary_x, fan_boundary_y = gis_utils.read_shapefile_boundary(fanBoundarySHP)
        nan_array = np.array([np.nan])
        fan_boundary_x = np.concatenate([np.concatenate([arr, nan_array]) for arr in fan_boundary_x])[:-1]
        fan_boundary_y = np.concatenate([np.concatenate([arr, nan_array]) for arr in fan_boundary_y])[:-1]
        volumeCalculateExtentXY = np.column_stack((fan_boundary_x, fan_boundary_y))

        volumeCalculateMask,_ = inpoly2(np.column_stack((xMesh.flatten(), yMesh.flatten())), volumeCalculateExtentXY)
        volumeCalculateMask = volumeCalculateMask.reshape(xMesh.shape)
    else:
        volumeCalculateExtentXY = None
        volumeCalculateMask = np.ones_like(zMesh, dtype=bool)
    
    # Simulation Process
    heights = [guessHeightAboveGround_bottom, guessHeightAboveGround_top]
    minMaxInitialGuessHeightVolume = np.zeros((2, 2))

    # Create RegularGridInterpolator
    interpolator = RegularGridInterpolator((yMesh[:, 0], xMesh[0, :]), zMesh)

    # Prepare parameters for parallel processing
    params = [(height, xMesh, yMesh, zMesh, xApex, yApex, dz_profile, volumeCalculateMask, interpolator, debug, epsg_code) for height in heights]
    
    # Use ProcessPoolExecutor to parallelize the height simulation
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_height_volume, params))
    
    minMaxInitialGuessHeightVolume = np.array(results)

    # Further processing with fanTopoSimVolumeMask
    zTopo_sim, heightAG_Volume_All = fanTopoSimVolumeMask(
        volume_expected, minMaxInitialGuessHeightVolume, xMesh, yMesh, zMesh, 
        xApex, yApex, volumeCalculateMask, dz_profile, epsg_code, tol, debug
    )

    return xMesh, yMesh, zTopo_sim, heightAG_Volume_All
