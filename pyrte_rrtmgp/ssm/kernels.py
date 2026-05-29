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
import xarray as xr

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

def compute_absorption_coeffs(
    triangle_params: xr.DataArray,
    nus: xr.DataArray,
    gas_names,
) -> xr.DataArray:
    absorption_coeffs = xr.DataArray(
        np.zeros((len(gas_names), nus.sizes["gpt"]), dtype=float),
        dims=("gas", "gpt"),
        coords={"gas": list(gas_names), "gpt": nus["gpt"]},
        name="absorption_coeffs",
        attrs={"units": "m2 kg-1"},
    )

    for row in triangle_params.values:
        gas = gas_names[int(row[0]) - 1]
        kappa0 = row[1]
        nu0 = row[2]
        ell = row[3]

        absorption_coeffs.loc[dict(gas=gas)] += (
            kappa0 * np.exp(-abs(nus - nu0) / ell)
        )

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

def compute_layer_mass(
    vmr: xr.DataArray,
    plev: xr.DataArray,
    play: xr.DataArray,
    mol_weights: xr.DataArray,
    m_dry: float = 0.029,
) -> xr.DataArray:
    _, lev_dim = plev.dims
    _, lay_dim = play.dims

    dp = abs(plev.diff(lev_dim))
    dp = dp.rename({lev_dim: lay_dim})

    if lay_dim in play.coords:
        dp = dp.assign_coords({lay_dim: play[lay_dim]})

    mmr = vmr * (mol_weights / m_dry)
    layer_mass = mmr * dp / GRAV

    layer_mass.name = "layer_mass"
    layer_mass.attrs["units"] = "kg m-2"

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
def compute_tau(
    absorption_coeffs: xr.DataArray,
    play: xr.DataArray,
    pref: float,
    layer_mass: xr.DataArray,
) -> xr.DataArray:
    if pref != 0.0:
        p_scaling = play / pref
    else:
        p_scaling = xr.ones_like(play)

    tau = p_scaling * (layer_mass * absorption_coeffs).sum("gas")
    tau.name = "tau"

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
def compute_band_limits(
    nus: xr.DataArray,
    nu_min: float,
    nu_max: float,
) -> xr.DataArray:
    nu_values = nus.values
    midpoints = 0.5 * (nu_values[:-1] + nu_values[1:])

    lower = np.empty_like(nu_values, dtype=float)
    upper = np.empty_like(nu_values, dtype=float)

    lower[0] = nu_min
    lower[1:] = midpoints

    upper[:-1] = midpoints
    upper[-1] = nu_max

    return xr.DataArray(
        np.stack([lower, upper], axis=0),
        dims=("band_edge", "gpt"),
        coords={
            "band_edge": ["lower", "upper"],
            "gpt": nus["gpt"],
        },
        name="band_lims",
        attrs={"units": "cm^-1"},
    )


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
def planck_function(T: xr.DataArray, nu: xr.DataArray) -> xr.DataArray:
    nu_si = nu / 100.0
    numerator = 100.0 * 2.0 * PLANCK_H * (nu_si ** 3) * (LIGHTSPEED ** 2)
    exponent = (PLANCK_H * LIGHTSPEED * nu_si) / (BOLTZMANN_K * T)
    return numerator / (np.exp(exponent) - 1.0)


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
def compute_planck_source(
    T: xr.DataArray,
    nus: xr.DataArray,
    dnus: xr.DataArray,
) -> xr.DataArray:
    source = planck_function(T, nus) * dnus
    source.name = "planck_source"
    source.attrs["units"] = "W m-2 sr-1"
    return source


  
