import numpy as np
from scipy import interpolate

def cone_function(zApex, D, options={}):
    """
    Constructs the z-values of a cone, concave, or custom profile surface based on the given options.

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
                Slope value for the infinite case.
            'dz_interp' : numpy array
                Interpolation values for the 'myProfile' case.
    
    Returns:
    zCone : float or numpy array
        The calculated z-values of the surface.
    """
    options.setdefault('caseName', 'cone')
    options.setdefault('tanAlpha', np.nan)
    options.setdefault('K', np.nan)
    options.setdefault('zApex0', np.nan)
    options.setdefault('tanInfinite', np.nan)
    options.setdefault('dz_interp', np.nan)

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
        out_linear = True
        dz_interp = options['dz_interp']
        
        if not out_linear:
            dOffset = interp1_extrap(dz_interp[:, 1], dz_interp[:, 0], zApex)
        else:
            interp_func = interpolate.interp1d(dz_interp[:, 1], dz_interp[:, 0], kind='linear', fill_value='extrapolate')
            dOffset = interp_func(zApex)

        
        if np.isscalar(zApex):
            if not out_linear:
                zCone = interp1_extrap(dz_interp[:, 0] - dOffset, dz_interp[:, 1], D)
            else:
                interp_func = interpolate.interp1d(dz_interp[:, 0] - dOffset, dz_interp[:, 1], kind='linear', fill_value='extrapolate')
                zCone = interp_func(D)
            zCone = zCone.reshape(D.shape)
        else:
            zCone = np.full_like(zApex, np.nan)
            for i in range(len(zCone)):
                if np.isnan(zApex[i]):
                    zCone[i] = np.nan
                else:
                    if not out_linear:
                        zCone[i] = interp1_extrap(dz_interp[:, 0] - dOffset[i], dz_interp[:, 1], D)
                    else:
                        interp_func = interpolate.interp1d(dz_interp[:, 0] - dOffset[i], dz_interp[:, 1], kind='linear', fill_value='extrapolate')
                        zCone[i] = interp_func(D)
    return zCone

def interp1_extrap(x, v, xq):
    """
    Perform 1D linear interpolation with extrapolation using numpy.interp. For query points outside
    the range of `x`, the corresponding values of `v` at the minimum or maximum of `x` are returned.

    Parameters:
    x : numpy array
        1D array of x-coordinates of the data points.
    v : numpy array
        1D array of y-coordinates (values) of the data points.
    xq : float or numpy array
        Query points at which to interpolate.

    Returns:
    vq : float or numpy array
        Interpolated values at the query points.
    """
    vq = np.interp(xq, x, v, left=v[0], right=v[-1])
    return vq
