import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['text.usetex'] = True
plt.rc('font', family = 'roman')

import SMBHBpy
from SMBHBpy import constants as c
from SMBHBpy import merger_system as ms

from scipy.interpolate import interp1d
import scipy.integrate as integrate
from matplotlib.colors import LogNorm
import math
import warnings

def compute_fmin_for_mz(m, z,
                        R0_increase_factor = 5.,
                        max_R0_iterations = 3,
                        opt_ev = SMBHBpy.inspiral.Classic.EvolutionOptions(accuracy=1e-10),
                        iota = np.pi/2,
                        sigma = 200 *10**3*c.m_to_pc/(c.s_to_pc),
                        Coulomb_log = 10):
        

    # Luminosity distance
    def D_L(z_):
        hubble_const = 67.66 *1e-3 * c.m_to_pc / c.s_to_pc
        def E(z__):
            omega_M = 0.3111
            omega_Lambda = 1. - omega_M
            return (omega_M*(1+z__)**3 + omega_Lambda)**(-1/2)
        result, _ = integrate.quad(lambda x: E(x), 0, z_)
        return (1+z_) / hubble_const * result

    D_lum = D_L(z)

    a_gamma = np.array([0,1,2,3,4])
    alpha_SIDM = (3 + a_gamma[4]) / 4.   # fix 7/4
    r_sp = 0.2 * m / sigma**2
    rho_sp_SIDM = (3 - alpha_SIDM) * 0.2**(3 - alpha_SIDM) * m / (2 * np.pi * r_sp**3)

    # Create spike & SystemProp
    spike_SIDM = SMBHBpy.halo.Spike(rho_sp_SIDM, r_sp, alpha_SIDM)
    sp_SIDM = ms.SystemProp(m, m, spike_SIDM, spike_SIDM, sigma, Coulomb_log, D=D_lum, inclination_angle = iota)

    R0_base = 150. * (sp_SIDM.r_isco_1() + sp_SIDM.r_isco_2())
    R0 = R0_base

    for attempt in range(max_R0_iterations):
        try:
            # Evolution
            R_fin = sp_SIDM.r_isco_1() + sp_SIDM.r_isco_2()
            ev_SIDM = SMBHBpy.inspiral.Classic.Evolve(sp_SIDM, R0, a_fin = R_fin, opt = opt_ev)

            # GW signal
            f_gw_SIDM, h_2_plus_SIDM, h_2_cross_SIDM, N_cyc_SIDM = SMBHBpy.waveform.h_2(sp_SIDM, ev_SIDM)
            h_c_SIDM_full = 2. * f_gw_SIDM * h_2_plus_SIDM

            Lisa = SMBHBpy.detector.Lisa()
            f_gw_LISA = np.geomspace(Lisa.Bandwith()[0], Lisa.Bandwith()[1], 1000)
            f_LISA_full = f_gw_LISA

            # Binning and interpolation
            mask_SIDM = (f_gw_SIDM >= np.min(f_LISA_full)) & (f_gw_SIDM <= np.max(f_LISA_full))
            f_gw_SIDM_cut = f_gw_SIDM[mask_SIDM]
            h_c_SIDM_cut = h_c_SIDM_full[mask_SIDM]

            y_LISA_full = Lisa.NoiseStrain(f_LISA_full)
            interp_LISA_to_SIDM = interp1d(f_LISA_full, y_LISA_full, kind='linear', bounds_error=False, fill_value=np.nan)
            y_LISA_on_SIDM = interp_LISA_to_SIDM(f_gw_SIDM_cut)

            sort_idx = np.argsort(f_gw_SIDM_cut)
            f_sorted = f_gw_SIDM_cut[sort_idx]                              # observed GW frequency of SMBHB with SIDM spikes
            h_sorted = h_c_SIDM_cut[sort_idx]                               # observed GW signal (h_c) of SMBHB with SIDM spikes
            y_sorted = y_LISA_on_SIDM[sort_idx]                             # LISA's sensitivity curve only in the range where h_c is also defined

            h0 = h_sorted[0]
            y0 = y_sorted[0]
            f0 = f_sorted[0]

            # --- 1st case: h0 < y0 ---
            if h0 < y0:
                comp = h_sorted > y_sorted
                if np.any(comp):
                    j = np.where(comp)[0][0]
                    f_min = f_sorted[j]
                    return f_min
                else:
                    # Never exceeds -> final result for this m-z (not detectable -> gray area)
                    f_min = np.nan
                    return f_min

            # --- 2nd case: h0 == y0 ---
            elif np.isclose(h0, y0, rtol=1e-12, atol=0):
                f_min = f0
                return f_min

            # --- 3rd case: h0 > y0 --- (-> initially selected R0 could be to small)
            else:
                f_min_SIDM = np.min(f_gw_SIDM)
                f_min_LISA = np.min(f_LISA_full)
                if f_min_SIDM < f_min_LISA:
                    f_min = f_min_LISA
                    return f_min
                else:
                    # Selected R0 too small -> increase and retry
                    R0 *= R0_increase_factor
                    continue  # <-- crucial: retry next attempt

        except Exception as e:
            warnings.warn(f"Attempt {attempt} failed for m={m}, z={z} with R0={R0:.3e}: {e}")
            R0 *= R0_increase_factor
            continue

    # If all attempts fail, return NaN
    return np.nan


# Fine grid
N_fine = 300
m_grid_vals = np.logspace(4, 7.2, N_fine) * c.solar_mass_to_pc 
z_grid_vals = np.logspace(-2, 1, N_fine)

M_mesh, Z_mesh = np.meshgrid(m_grid_vals, z_grid_vals, indexing='xy')
fmin_grid = np.full(M_mesh.shape, np.nan)

total = M_mesh.size
count = 0

# Define evolution options globally
opt_ev = SMBHBpy.inspiral.Classic.EvolutionOptions(accuracy=1e-10)

for i in range(M_mesh.shape[0]):
    for j in range(M_mesh.shape[1]):
        m_val = M_mesh[i, j]
        z_val = Z_mesh[i, j]
        count += 1

        print(f"Processing {count}/{total}", end="\r")

        fmin = compute_fmin_for_mz(
            m_val, z_val,
            opt_ev=opt_ev,     
            R0_increase_factor=5.,
            max_R0_iterations=3,
            iota=np.pi/2,
            Coulomb_log=10
        )

        # Store the result in the grid
        fmin_grid[i, j] = fmin

# Save the results in a .npz file
np.savez("fmin_LISA_results.npz", M_mesh=M_mesh, Z_mesh=Z_mesh, fmin_grid=fmin_grid)