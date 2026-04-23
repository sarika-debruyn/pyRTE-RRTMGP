"""
This module contains the core mathematical functions of the SSM, translated directly from mo_optics_ssm_kernels.F90.

References
----------
Williams, A. I. L. (2026). Bridging clarity and accuracy: A simple spectral
longwave radiation scheme for idealized climate modeling.
Journal of Advances in Modeling Earth Systems, 18, e2025MS005405.
https://doi.org/10.1029/2025MS005405
"""

import numpy as np
from .defaults import PLANCK_H, LIGHTSPEED, BOLTZMANN_K, GRAV

"""
compute_absorption_coeffs() Evaluates the reference absorption coeffecients on the wavenumber grid

Each triangle contributes:
  kappa(nu) = kappa0 * exp(-|nu - nu0| / l)

Multiple triangles for the same gas are summed.

Parameters:
  triangle_params: np.ndarray, shape(n_triangles, 4)
      Each row is [gas_index, kappa0, nu0, l] # one triangle
  nus: np.ndarray, shape(n_nu) 
      Wavenumber grid [cm^-1] 
  n_gases: int 
      Total number of gases

Returns:
  absorption_coeffs: np_ndarray, shape (n_gases, n_nu)
        Reference absorption coefficients [m^2 kg^-1] at each wavenumber,
        evaluated at the reference pressure and temperature defined in defaults.py.
"""
def compute_absorption_coeffs(triangle_params: np.ndarray, nus: np.ndarray. n_gases: int) -> np.ndarray:
  
  # set up output array
  n_nu = len(nus)
  absorption_coeffs = np.zeros((n_gases, n_nu), dtype=float)

  # loop over triangles to obtain their parameters
  for row in triangle_params:
    gas_idx = int(row[0]) - 1 #convert from 1-based Fortran to 0-based Python
    kappa0 = row[1]
    nu0 = row[2]
    l = row[3]

    # derived from lines 300-307 in mo_optics_ssm.F90
    absorption_coeffs[gas_idx, :] += kappa0 * np.exp(-np.abs(nus - nu0) / l)

  return absorption_coeffs


"""
compute_layer_mass() converts volume mixing ratios (VMR) and pressure levels to layer masses so it can be used to compute optical depth
by employing a direct translation of compute_layer_mass() from lines 162 - 186 in mo_optics_ssm_kernels.F90

Parameters:
  vmr: np.ndarray, shape (n_gases, n_col, n_lay)
      Volume mixing rations of each gas [mol/mol]
  plev: np.ndarray, shape (n_col, n_lay+1)
      Pressure at layer interfaces (levels) [Pa]
  mol_weights: np.ndarray, shape (n_gases)
      Molecular weight of each gas [kg/mol]
  m_dry: float
      Molecular weight of dry air [kg/mol]

Returns: 
  layer_mass: np.ndarray, shape (n_gases, n_col, n_lay)
      Mass of each gas in each layer [kg m^-2]
"""
def compute_layer_mass(vmr: np.ndarray, plev: np.ndarray, mol_weights: np.ndarray, m_dry: float = 0.029) -> np.ndarray:
    # pressure thickness of each layer
    dp = np.abs(plev[:, 1:] - plev[:, :-1]) #always positive

    # mol_weights has shape (n_gases,); broadcast to (n_gases, n_col, n_lay)
    # dp has shape (n_col, n_lay); broadcast by adding a leading axis
    mmr = vmr * (mol_weights[:, np.newaxis, np.newaxis] / m_dry)
    layer_mass = mmr * dp[np.newaxis, :, :] / GRAV
 
    return layer_mass


"""
compute_tau() computes the absorption optical depth for each column, layer, and wavelength number.

   The optical depth of a layer is computed by:
     tau(col, lay, nu) = (p / p_ref) * sum_gas[ layer_mass(gas, col, lay) * kappa_ref(gas, nu) ]
 
  The factor p/p_ref is the pressure-broadening scaling described in Section 2.2 of Williams (2026), Equations 7 and 9.  It accounts for the
  fact that at higher pressures, more frequent molecular collisions broaden the absorption lines, increasing the effective absorption.
 
  This is a direct translation of compute_tau() from lines 40-96 in mo_optics_ssm_kernels.F90 

Parameters:
  absorption_coeffs: np.ndarray, shape (n_gases, n_nu)
      Reference absorption coefficients [m^2 kg^-1] from compute_absorption_coeffs()
  play: np.ndarray, shape (n_col, n_lay)
      Layer pressures [Pa]
  pref: float
      Reference pressure [Pa]
      If zero, pressure broadening is disabled
  layer_mass: np.ndarray, shape (n_gases, n_col, n_lay)
      Gas layer masses [kg m^-2] from compute_layer_mass()

Returns:
  tau: np.ndarray, shape (n_col, n_lay, n_nu)
      Absorption optical depth [dimensionless]
"""
def compute_tau(absorption_coeffs: np.ndarray, play: np.ndarray, pref: float, layer_mass: np.ndarray) -> np.ndarray:

  #pressure broadening factor: shape (n_col, n_lay)
  if pref != 0.0:
    p_scaling = play / pref
  else:
    p_scaling = np.ones_like(play)

  # sum over gases: layer_mass (n_gases, n_col, n_lay), absorption_coeffs (n_gases, n_nu)
  # we want result (n_col, n_lay, n_nu)
  # for each gas g: contribution(col, lay, nu) = layer_mass(g, col, lay) * kappa(g, nu)
  # use Einstein summation: 'gcl, gn -> cln'
  gas_weighted = np.einsum('gcl,gn->cln', layer_mass, absorption_coeffs)

  # apply pressure broadening: p_scaling is (n_col, n_lay), result is (l, n_lay, n_nu)
  tau = p_scaling[:, :, np.newaxis] * gas_weighted

  return tau


"""
compute_band_limits() constructs wavenumber band edges from a set of central wavenumbers

  This is a direct translation of lines 258-277 from mo_optics_ssm.F90
  The Fortran places band edges at the midpoints between adjacent nus, with the first band starting at nu_min and the last ending at nu_max

Parameters:
  nus: np.ndarray, shape (n_nu)
      Central wavenumbers [cm^-1]
  nu_min: float
      Lower bound of the spectrum [cm^-1]
  nu_max: float
      Upper bound of the spectrum [cm^-1]

Returns:
  band_lims: np.ndarray, shape (2, n_nu)
      band_lims[0, i] = lower edge of band i
      band_lims[1, i] = upper edge of band i
"""
def compute_band_limits(nus: np.ndarray, nu_min: float, nu_max: float) -> np.ndarray:
  n_nu = len(nus) #default case is 42 wavenumber points
  band_lims = np.empty((2, n_nu), dtype=float) #empty 2D array, two rows and one column per wavenumber point

  # lower edges: nu_min for the first band, midpoints for the rest
  band_lims[0, 0] = nu_min
  band_lims[0, 1:] = 0.5 * (nus[:-1] + nus[1:])

  # upper edges: midpoints for all but last, nu_max for last
  band_lims[1, :-1] = 0.5 * (nus[:-1] + nus[1:])
  band_lims[1, -1] = nu_max

  return band_lims

"""
planck_function() calculates how much radiation a perfect blackbody emits at wavelength nu given temperature T, 
per unit wavelength interval, per unit solid angle. In other words, the spectral radiance of a blackbody per unit wavenumber:
    B_nu(T, nu) = 100 * 2*h*c^2 * (100*nu)^3 
                  / [exp(h*c*100*nu / (k_B * T)) - 1]

    This is an exact translation of the elemental function on B_nu() from lines 101-106 in mo_optics_ssm_kernels.F90

Parameters:
  T: np.ndarray, any shape
      Temperature [k]
  nu: float
      Wavelength [cm^-1]

Returns: 
  B: np.ndarray, same shape as T
    Spectral radiance [ W m^-2 sr^-1 (cm^-1)^-1]
"""
def planck_function(T: np.ndarray, nu: float) -> np.ndarray:
  nu_si = nu 100.0 #convert cm^-1 to m^-1
  numerator = 100.0 * 2.0 * PLANCK_H * (nu_si ** 3) * (LIGHTSPEED ** 2)
  
  exponent = (PLANC_H * LIGHTSPEED * nu_si) / (BOLTZMANN_K * T)
  denominator = np.exp(exponent) - 1.0 
  
  return numerator / denominator


"""
compute_planck_source() computes the spectrally integrated Planck source function.
  For each wavenumber band, the source is:
      source(..., nu) = B_nu(T, nu) * dnu

      where dnu is the width of the spectral band. This is the band-integrated radiance, or the total emission in the wavenumber interval

  This handles both the 1D (surface/TOA) and 2D (layer) cases from the Fortran compute_Planck_source_1D and 
  compute_Planck_source_2D functions in lines 108-160 from mo_optics_ssm_kernels.F90

Parameters:
  T: np.ndarray
      Temperature [K], can be shape (n_col) for 1D or (n_col, n_lay) for 2D
  nus: np.ndarray, shape (n_nu)
      Wavenumber at each spectral point [cm^-1]
  dnus: np.ndarray, shape (n_nu)
      Width of each spectral band [cm^-1]

Returns:
  source: np.ndarray
      Shape (..., n_nu) where ... matches the shape of T
      Band-integrated Planck radiance [W m^-2 sr^-1]   
"""
def compute_planck_source(T: np.ndarray, nus: np.ndarray, dnus: np.ndarray) -> np.ndarray:
  # T has shape (...), we need to evalute B_nu at each wavenumber
  # Add a trailing axist to T so it broadcasts against nus
  T_expanded = T[..., np.newaxis] #(..., 1)

  #nus and dnus have shape (n_nu), broadcast against T_expanded -> (..., n_nu)
  source = planck_function(T_expanded, nus) * dnus

  return source



  
