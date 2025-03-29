import numpy as np
from scipy import interpolate

def cone_function(zApex, D, options={}):
    """
    Calculate the elevation at distance D based on the given elevation-distance relationship options.

    Parameters:
    zApex : float or numpy array
        Elevation of the apex.
    D : float or numpy array
        Distance from the apex.
    options : dict
        Dictionary containing optional parameters:
            'caseName' : str
                Type of surface to generate ('cone', 'concavity', 'infinite', 'myProfile'). Default is 'cone'.
            'tanAlpha' : float
                Slope angle (tangent) for the surface.
            'K' : float
                Concavity factor.
            'zApex0' : float
                Reference elevation for concave and infinite cases.
            'tanInfinite' : float
                Asymptotic slope for the infinite case (Under development).
            'sz_interp' : numpy array
                Elevation-distance relationship for 'myProfile' case.
    
    Returns:
    zCone : float or numpy array
        The calculated z-values of the surface.
    """
    options.setdefault('caseName', 'cone')
    options.setdefault('tanAlpha', np.nan)
    options.setdefault('K', np.nan)
    options.setdefault('zApex0', np.nan)
    options.setdefault('tanInfinite', np.nan)
    options.setdefault('sz_interp', np.nan)

    case_name = options['caseName']

    if case_name == 'cone':
        zCone = zApex - options['tanAlpha'] * D

    elif case_name == 'concavity':
        S = options['tanAlpha'] - options['K'] * (options['zApex0'] - zApex)
        zCone = zApex - S / options['K'] * (1 - np.exp(-options['K'] * D))

    elif case_name == 'infinite':
        S = options['tanAlpha'] - options['K'] * (options['zApex0'] - zApex)
        zCone = zApex - (S - options['tanInfinite']) / options['K'] * (1 - np.exp(-options['K'] * D)) - options['tanInfinite'] * D

    elif case_name == 'myProfile':
        sz_interp = options['sz_interp']
        interp_func = interpolate.interp1d(sz_interp[:, 1], sz_interp[:, 0], kind='linear', fill_value='extrapolate')
        dOffset = interp_func(zApex)

        
        if np.isscalar(zApex):
            interp_func = interpolate.interp1d(sz_interp[:, 0] - dOffset, sz_interp[:, 1], kind='linear', fill_value='extrapolate')
            zCone = interp_func(D)
            zCone = zCone.reshape(D.shape)
        else:
            zCone = np.full_like(zApex, np.nan)
            for i in range(len(zCone)):
                if np.isnan(zApex[i]):
                    zCone[i] = np.nan
                else:
                    interp_func = interpolate.interp1d(sz_interp[:, 0] - dOffset[i], sz_interp[:, 1], kind='linear', fill_value='extrapolate')
                    zCone[i] = interp_func(D)
    return zCone
