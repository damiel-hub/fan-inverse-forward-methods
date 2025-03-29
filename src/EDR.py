import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


class EDR: # Elevation-Distance Relationship

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
        dd_max = np.max(self.s_flatten)
        dd_in = np.arange(0, dd_max + ds, ds)
        
        if len(bin_pts_mid_s) > 0:
            p = Polynomial.fit(bin_pts_mid_s, Q2, 2).convert().coef
            L, S, P = p[2], p[1], p[0]
            z_in = L * dd_in**2 + S * dd_in + P

            # Extrapolate for the upper and lower ranges
            dd_up = np.arange(-outlength, 0, ds)
            z_up = S * dd_up + P
            dd_do = np.arange(dd_in[-1] + ds, dd_in[-1] + ds + outlength, ds)
            z_do = (2 * L * dd_max + S) * dd_do - L * dd_max**2 + P
            
            if pltFlag:
                # Plot the data
                plt.figure()
                plt.scatter(self.s_flatten, self.z_flatten, color='k', marker='.')
                plt.plot(bin_pts_mid_s, Q2, drawFlag + '.')
                # Plot the fit and extrapolations
                plt.plot(dd_in, z_in, drawFlag + '-')
                plt.plot(dd_up, z_up, drawFlag + '--')
                plt.plot(dd_do, z_do, drawFlag + '--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], drawFlag + 'o', markersize=6)
                plt.xlabel('Shortest path distance to all data points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                if isinstance(pltFlag, str):
                    plt.savefig(pltFlag, dpi=300, bbox_inches='tight')
                else:
                    plt.savefig('EDR_median_on.png', dpi=300, bbox_inches='tight')
                plt.show(block = False)
                plt.pause(0.01)
                # plt.close()

            ss = np.concatenate([dd_up, dd_in, dd_do])
            zz = np.concatenate([z_up, z_in, z_do])
            fitting_s_z = np.vstack((ss, zz)).T
            if return_fitting_coeff:
                z_hat = np.zeros_like(self.s_flatten)
                z_hat[self.s_flatten<=0] = S*self.s_flatten[self.s_flatten<=0] + P
                z_hat[(self.s_flatten > 0) & (self.s_flatten < dd_max)] = L*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < dd_max)]**2 + S*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < dd_max)] + P
                z_hat[self.s_flatten>=dd_max] = (2*L*dd_max + S)*self.s_flatten[self.s_flatten>=dd_max] - L*dd_max**2 + P
                RMSE_value = np.sqrt(np.sum((z_hat - self.z_flatten)**2)/z_hat.size)
                slope = -L*dd_max - S
                dimensionless_drop = L*dd_max/4
                return fitting_s_z, slope, dimensionless_drop, RMSE_value, L, S, P, dd_max, bin_pts_mid_s, Q2
            else:
                return fitting_s_z
        
    def medianFilter_off(self, ds, outlength, pltFlag = False, drawFlag = False, return_fitting_coeff = False):    
        
        dd_max = np.max(self.s_flatten)
        dd_in = np.arange(0, dd_max + ds, ds)

        if len(self.s_flatten) > 0:
            p = Polynomial.fit(self.s_flatten, self.z_flatten, 2).convert().coef
            L, S, P = p[2], p[1], p[0]
            z_in = L * dd_in**2 + S * dd_in + P

            # Extrapolate for the upper and lower ranges
            dd_up = np.arange(-outlength, 0, ds)
            z_up = S * dd_up + P
            dd_do = np.arange(dd_in[-1] + ds, dd_in[-1] + ds + outlength, ds)
            z_do = (2 * L * dd_max + S) * dd_do - L * dd_max**2 + P
            
            if pltFlag:
                # Plot the fit and extrapolations
                plt.figure()
                plt.plot(self.s_flatten_nan, self.z_flatten_nan, 'k-')
                plt.plot(dd_in, z_in, 'b-')
                plt.plot(dd_up, z_up, 'b--')
                plt.plot(dd_do, z_do, 'b--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], 'bo', markersize=6)
                plt.xlabel('Shortest path distance to boundary points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                # plt.show(block = False)
                # plt.pause(0.01)
                if isinstance(pltFlag, str):
                    plt.savefig(pltFlag, dpi=300, bbox_inches='tight')
                else:
                    plt.savefig('EDR_median_off.png', dpi=300, bbox_inches='tight')
            if drawFlag:
                plt.plot(dd_in, z_in, drawFlag + '-')
                plt.plot(dd_up, z_up, drawFlag + '--')
                plt.plot(dd_do, z_do, drawFlag + '--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], drawFlag + 'o', markersize=6)


            ss = np.concatenate([dd_up, dd_in, dd_do])
            zz = np.concatenate([z_up, z_in, z_do])
            fitting_s_z = np.vstack((ss, zz)).T
            if return_fitting_coeff:
                z_hat = np.zeros_like(self.s_flatten)
                z_hat[self.s_flatten<=0] = S*self.s_flatten[self.s_flatten<=0] + P
                z_hat[(self.s_flatten > 0) & (self.s_flatten < dd_max)] = L*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < dd_max)]**2 + S*self.s_flatten[(self.s_flatten > 0) & (self.s_flatten < dd_max)] + P
                z_hat[self.s_flatten>=dd_max] = (2*L*dd_max + S)*self.s_flatten[self.s_flatten>=dd_max] - L*dd_max**2 + P
                RMSE_value = np.sqrt(np.sum((z_hat - self.z_flatten)**2)/z_hat.size)
                slope = -L*dd_max - S
                dimensionless_drop = L*dd_max/4
                return fitting_s_z, slope, dimensionless_drop, RMSE_value, L, S, P, dd_max
            else:
                return fitting_s_z
