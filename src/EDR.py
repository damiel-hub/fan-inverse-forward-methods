import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial


class EDR: # Elevation-Distance Relationship

    def __init__(self, sMap, zMap, drawFlag = False):
        # Flatten the input matrices
        self.s_flaten_nan = sMap.flatten()
        self.z_flaten_nan = zMap.flatten()     
        nan_index = np.isnan(self.s_flaten_nan) | np.isnan(self.z_flaten_nan)
        self.s_flaten = self.s_flaten_nan[~nan_index]
        self.z_flaten = self.z_flaten_nan[~nan_index]
        if drawFlag:
            plt.scatter(self.s_flaten, self.z_flaten, color='k', marker='.')

    def medianFilter_on(self, bin_size, ds, outlength, pltFlag = False, drawFlag = False):

        # Define bins for s values
        bin_pts_s = np.arange(np.min(self.s_flaten), np.max(self.s_flaten) + bin_size, bin_size)
        bin_pts_mid_s = []
        Q2 = []

        # Calculate the median z-value for each s bin, removing outliers
        for i in range(len(bin_pts_s) - 1):
            logic = (self.s_flaten > bin_pts_s[i]) & (self.s_flaten < bin_pts_s[i + 1])
            if np.any(logic):
                Q2.append(np.quantile(self.z_flaten[logic], 0.5))
                bin_pts_mid_s.append((bin_pts_s[i] + bin_pts_s[i + 1]) / 2)

        # Convert to numpy arrays for further processing
        bin_pts_mid_s = np.array(bin_pts_mid_s)
        Q2 = np.array(Q2)

        # Remove NaN values
        valid = ~np.isnan(Q2)
        bin_pts_mid_s = bin_pts_mid_s[valid]
        Q2 = Q2[valid]

        # Polynomial fitting and extrapolation
        dd_max = np.max(bin_pts_mid_s)
        dd_in = np.arange(0, dd_max + ds, ds)
        
        if len(bin_pts_mid_s) > 0:
            p = Polynomial.fit(bin_pts_mid_s, Q2, 2).convert().coef
            a, b, c = p[2], p[1], p[0]
            z_in = a * dd_in**2 + b * dd_in + c

            # Extrapolate for the upper and lower ranges
            dd_up = np.arange(-outlength, 0, ds)
            z_up = b * dd_up + c
            dd_do = np.arange(dd_in[-1] + ds, dd_in[-1] + ds + outlength, ds)
            z_do = (2 * a * dd_max + b) * dd_do - a * dd_max**2 + c
            
            if pltFlag:
                # Plot the data
                plt.figure()
                plt.scatter(self.s_flaten, self.z_flaten, color='k', marker='.')
                plt.plot(bin_pts_mid_s, Q2, 'b.')
                # Plot the fit and extrapolations
                plt.plot(dd_in, z_in, 'b-')
                plt.plot(dd_up, z_up, 'b--')
                plt.plot(dd_do, z_do, 'b--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], 'bo', markersize=6)
                plt.xlabel('Shortest path distance to all data points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                plt.show(block = False)
                plt.pause(0.01)
                if isinstance(pltFlag, str):
                    plt.savefig(pltFlag, dpi=300, bbox_inches='tight')
                else:
                    plt.savefig('EDR_median_on.png', dpi=300, bbox_inches='tight')
                # plt.close()

            if drawFlag:
                plt.plot(bin_pts_mid_s, Q2, drawFlag + '.')
                # Plot the fit and extrapolations
                plt.plot(dd_in, z_in, drawFlag + '-')
                plt.plot(dd_up, z_up, drawFlag + '--')
                plt.plot(dd_do, z_do, drawFlag + '--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], drawFlag + 'o', markersize=6)

            ss = np.concatenate([dd_up, dd_in, dd_do])
            zz = np.concatenate([z_up, z_in, z_do])
            fitting_s_z = np.vstack((ss, zz)).T
            return fitting_s_z
        
    def medianFilter_off(self, ds, outlength, pltFlag = False, drawFlag = False):    
        
        dd_max = np.max(self.s_flaten)
        dd_in = np.arange(0, dd_max + ds, ds)

        if len(self.s_flaten) > 0:
            p = Polynomial.fit(self.s_flaten, self.z_flaten, 2).convert().coef
            a, b, c = p[2], p[1], p[0]
            z_in = a * dd_in**2 + b * dd_in + c

            # Extrapolate for the upper and lower ranges
            dd_up = np.arange(-outlength, 0, ds)
            z_up = b * dd_up + c
            dd_do = np.arange(dd_in[-1] + ds, dd_in[-1] + ds + outlength, ds)
            z_do = (2 * a * dd_max + b) * dd_do - a * dd_max**2 + c
            
            if pltFlag:
                # Plot the fit and extrapolations
                plt.figure()
                plt.plot(self.s_flaten_nan, self.z_flaten_nan, 'k-')
                plt.plot(dd_in, z_in, 'b-')
                plt.plot(dd_up, z_up, 'b--')
                plt.plot(dd_do, z_do, 'b--')
                plt.plot([0, dd_max], [z_in[0], z_in[-1]], 'bo', markersize=6)
                plt.xlabel('Shortest path distance to boundary points, s (m)')
                plt.ylabel('Elevation, z (m)')
                plt.gca().set_aspect(5)
                plt.grid(True)
                plt.box(True)
                plt.show(block = False)
                plt.pause(0.01)
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
            return fitting_s_z
