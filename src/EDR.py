import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


class EDR: # Elevation-Distance Relationship
    """
    A class to compute and analyze the Elevation-Distance Relationship (EDR).
    This class provides methods to process elevation and distance data, apply median filtering,
    and perform quadratic fitting for extrapolation and visualization.
    Attributes:
        s_flatten_nan (np.ndarray): Flattened shortest path distance array with NaN values.
        z_flatten_nan (np.ndarray): Flattened elevation array with NaN values.
        s_flatten (np.ndarray): Flattened shortest path distance array without NaN values.
        z_flatten (np.ndarray): Flattened elevation array without NaN values.
    Methods:
        __init__(sMap, zMap, drawFlag=False):
            Initializes the EDR object with input distance and elevation maps.
        return_flatten_s_z():
            Returns the flattened distance and elevation arrays as a 2D array.
        medianFilter_on(bin_size, ds, outlength, pltFlag=False, drawFlag='b', return_fitting_coeff=False):
            Applies median filtering to the elevation data, performs polynomial fitting,
            and optionally visualizes the results.
        medianFilter_off(ds, outlength, pltFlag=False, drawFlag=False, return_fitting_coeff=False):
            Performs polynomial fitting on the raw elevation data without median filtering,
            and optionally visualizes the results.
    """
    def __init__(self, sMap, zMap, drawFlag = False):
        # Flatten the input matrices
        self.s_flatten_nan = sMap.flatten()
        self.z_flatten_nan = zMap.flatten()     
        nan_index = np.isnan(self.s_flatten_nan) | np.isnan(self.z_flatten_nan)
        self.s_flatten = self.s_flatten_nan[~nan_index]
        self.z_flatten = self.z_flatten_nan[~nan_index]
        if drawFlag:
            plt.scatter(self.s_flatten, self.z_flatten, color='k', marker='.')
    def return_flatten_s_z(self):
        s_z_flatten = np.vstack((self.s_flatten,self.z_flatten)).T
        return s_z_flatten

    def medianFilter_on(self, bin_size, ds, outlength, pltFlag = False, drawFlag = 'b', return_fitting_coeff = False):
        """
        Applies median filtering to the elevation data, performs quadratic fitting,
        and optionally visualizes the results.
        Args:
            bin_size (float): The size of bins for median filter.
            ds (float): The step size for interpolation and extrapolation.
            outlength (float): The length for extrapolation beyond 0 and s_T.
            pltFlag (bool or str, optional): If True, plots the results. If a string, saves the plot to the specified file.
            drawFlag (str, optional): The color for plotting (default is 'b').
            return_fitting_coeff (bool, optional): If True, returns additional fitting coefficients and morphometric characteristics.
        Returns:
            np.ndarray: A 2D array of fitted and extrapolated distance and elevation values.
            tuple (optional): If `return_fitting_coeff` is True, returns additional fitting coefficients and metrics:
                - slope (float): The slope of the fitted profile at midpoint.
                - dimensionless_drop (float): The dimensionless drop of the fan profile.
                - RMSE_value (float): The root mean square error of the fit.
                - L, S, P (float): Coefficients of the quadratic polynomial.
                - s_T (float): The farest distance point from the fan apex.
                - bin_pts_mid_s (np.ndarray): Midpoints of the distance bins.
                - Q2 (np.ndarray): Median elevation values for each bin corresponding to bin_pts_mid_s.
        """
        # Define bins for s values
        bin_pts_s = np.arange(np.min(self.s_flatten), np.max(self.s_flatten) + bin_size, bin_size)
        bin_pts_mid_s = []
        Q2 = []

        # Calculate the median z-value for each s bin, removing outliers
        for i in range(len(bin_pts_s) - 1):
            logic = (self.s_flatten > bin_pts_s[i]) & (self.s_flatten < bin_pts_s[i + 1])
            if np.any(logic):
                Q2.append(np.quantile(self.z_flatten[logic], 0.5))
                bin_pts_mid_s.append((bin_pts_s[i] + bin_pts_s[i + 1]) / 2)

        # Convert to numpy arrays for further processing
        bin_pts_mid_s = np.array(bin_pts_mid_s)
        Q2 = np.array(Q2)

        # Remove NaN values
        valid = ~np.isnan(Q2)
        bin_pts_mid_s = bin_pts_mid_s[valid]
        Q2 = Q2[valid]

        # Polynomial fitting and extrapolation
        s_T = np.max(self.s_flatten)
        ss_in = np.arange(0, s_T + ds, ds)
        
        if len(bin_pts_mid_s) > 0:
            p = Polynomial.fit(bin_pts_mid_s, Q2, 2).convert().coef
            L, S, P = p[2], p[1], p[0]
            zz_in = L * ss_in**2 + S * ss_in + P

            # Extrapolate for the upper and lower ranges
            ss_up = np.arange(-outlength, 0, ds)
            zz_up = S * ss_up + P
            ss_do = np.arange(ss_in[-1] + ds, ss_in[-1] + ds + outlength, ds)
            zz_do = (2 * L * s_T + S) * ss_do - L * s_T**2 + P
            
            if pltFlag:
                # Plot the data
                plt.figure()
                plt.scatter(self.s_flatten, self.z_flatten, color='k', marker='.')
                plt.plot(bin_pts_mid_s, Q2, drawFlag + '.')
                # Plot the fit and extrapolations
                plt.plot(ss_in, zz_in, drawFlag + '-')
                plt.plot(ss_up, zz_up, drawFlag + '--')
                plt.plot(ss_do, zz_do, drawFlag + '--')
                plt.plot([0, s_T], [zz_in[0], zz_in[-1]], drawFlag + 'o', markersize=6)
                plt.xlabel('Shortest path distance to all data points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                if isinstance(pltFlag, str):
                    plt.savefig(pltFlag, dpi=300, bbox_inches='tight')
                plt.show(block = False)
                plt.pause(0.01)
                # plt.close()

            ss = np.concatenate([ss_up, ss_in, ss_do])
            zz = np.concatenate([zz_up, zz_in, zz_do])
            fitting_s_z = np.vstack((ss, zz)).T
            if return_fitting_coeff:
                z_hat = np.zeros_like(self.s_flatten)
                z_hat[self.s_flatten<=0] = S*self.s_flatten[self.s_flatten<=0] + P
                z_hat[(self.s_flatten > 0) & (self.s_flatten < s_T)] = L*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < s_T)]**2 + S*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < s_T)] + P
                z_hat[self.s_flatten>=s_T] = (2*L*s_T + S)*self.s_flatten[self.s_flatten>=s_T] - L*s_T**2 + P
                RMSE_value = np.sqrt(np.sum((z_hat - self.z_flatten)**2)/z_hat.size)
                slope = -L*s_T - S
                dimensionless_drop = L*s_T/4
                return fitting_s_z, slope, dimensionless_drop, RMSE_value, L, S, P, s_T, bin_pts_mid_s, Q2
            else:
                return fitting_s_z
        
    def medianFilter_off(self, ds, outlength, pltFlag = False, drawFlag =  'b', return_fitting_coeff = False):    
        """
        Performs polynomial fitting on the raw elevation data without median filtering,
        and optionally visualizes the results.
        Args:
            ds (float): The step size for interpolation and extrapolation.
            outlength (float): The length for extrapolation beyond 0 and s_T..
            pltFlag (bool or str, optional): If True, plots the results. If a string, saves the plot to the specified file.
            drawFlag (str, optional): The color for plotting (default is False).
            return_fitting_coeff (bool, optional): If True, returns additional fitting coefficients and morphometric characteristics.
        Returns:
            np.ndarray: A 2D array of fitted and extrapolated distance and elevation values.
            tuple (optional): If `return_fitting_coeff` is True, returns additional fitting coefficients and metrics:
                - slope (float): The slope of the fitted profile at midpoint.
                - dimensionless_drop (float): The dimensionless drop of the fan profile.
                - RMSE_value (float): The root mean square error of the fit.
                - L, S, P (float): Coefficients of the quadratic polynomial.
                - s_T (float): The farest distance point from the fan apex.
        """        
        s_T = np.max(self.s_flatten)
        ss_in = np.arange(0, s_T + ds, ds)

        if len(self.s_flatten) > 0:
            p = Polynomial.fit(self.s_flatten, self.z_flatten, 2).convert().coef
            L, S, P = p[2], p[1], p[0]
            zz_in = L * ss_in**2 + S * ss_in + P

            # Extrapolate for the upper and lower ranges
            ss_up = np.arange(-outlength, 0, ds)
            zz_up = S * ss_up + P
            ss_do = np.arange(ss_in[-1] + ds, ss_in[-1] + ds + outlength, ds)
            zz_do = (2 * L * s_T + S) * ss_do - L * s_T**2 + P
            
            if pltFlag:
                # Plot the fit and extrapolations
                plt.figure()
                plt.plot(self.s_flatten_nan, self.z_flatten_nan, 'k-')
                plt.plot(ss_in, zz_in, drawFlag + '-')
                plt.plot(ss_up, zz_up, drawFlag + '--')
                plt.plot(ss_do, zz_do, drawFlag + '--')
                plt.plot([0, s_T], [zz_in[0], zz_in[-1]], drawFlag + 'o', markersize=6)
                plt.xlabel('Shortest path distance to boundary points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                # plt.show(block = False)
                # plt.pause(0.01)
                if isinstance(pltFlag, str):
                    plt.savefig(pltFlag, dpi=300, bbox_inches='tight')


            ss = np.concatenate([ss_up, ss_in, ss_do])
            zz = np.concatenate([zz_up, zz_in, zz_do])
            fitting_s_z = np.vstack((ss, zz)).T
            if return_fitting_coeff:
                z_hat = np.zeros_like(self.s_flatten)
                z_hat[self.s_flatten<=0] = S*self.s_flatten[self.s_flatten<=0] + P
                z_hat[(self.s_flatten > 0) & (self.s_flatten < s_T)] = L*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < s_T)]**2 + S*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < s_T)] + P
                z_hat[self.s_flatten>=s_T] = (2*L*s_T + S)*self.s_flatten[self.s_flatten>=s_T] - L*s_T**2 + P
                RMSE_value = np.sqrt(np.sum((z_hat - self.z_flatten)**2)/z_hat.size)
                slope = -L*s_T - S
                dimensionless_drop = L*s_T/4
                return fitting_s_z, slope, dimensionless_drop, RMSE_value, L, S, P, s_T
            else:
                return fitting_s_z
