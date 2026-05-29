import numpy as np
import xarray as xr
from .defaults import MOL_WEIGHTS
from .kernels import (
    compute_absorption_coeffs,
    compute_band_limits,
    compute_layer_mass,
    compute_planck_source,
    compute_tau,
)



class GasOptics:
    
    def __init__(
        self,
        gas_names,
        triangle_params,
        nus,
        nu_min,
        nu_max,
        pref=1.0e5,
    ):

        # Validate the Input
        self._validate_inputs()
        # initialize input as in xarray dataset
        self._init_inputs(gas_names, triangle_params, nus, nu_min, nu_max, pref)

        # Compute derived quantities (using function from kernel.py)
        self.band_lims = compute_band_limits(
            self.nus,
            self.nu_min,
            self.nu_max,
        )

        self.dnus = (
            self.band_lims.sel(band_edge="upper")
            - self.band_lims.sel(band_edge="lower")
        )

        self.absorption_coeffs = compute_absorption_coeffs(
            triangle_params=self.triangle_params,
            nus=self.nus,
            gas_names=self.gas_names,
        )
    
    # Unit and parameters might not be correct
    def _init_inputs(self, gas_names, triangle_params, nus, nu_min, nu_max, pref):
        self.gas_names = tuple(g.lower() for g in gas_names)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.pref = float(pref)

        self.nus = xr.DataArray(
            nus,
            dims=("gpt",),
            name="nus",
            attrs={"units": "cm^-1"},
        )

        self.triangle_params = xr.DataArray(
            triangle_params,
            dims=("triangle", "triangle_param"),
            coords={
                "triangle_param": ["gas_index", "kappa0", "nu0", "ell"],
            },
            name="triangle_params",
        )

        self.mol_weights = xr.DataArray(
            [MOL_WEIGHTS[gas] for gas in self.gas_names],
            dims=("gas",),
            coords={"gas": self.gas_names},
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )
    # 
    def compute(self, atmos: xr.Dataset) -> xr.Dataset:

        play = atmos["play"]
        plev = atmos["plev"]
        tlay = atmos["tlay"]
        tlev = atmos["tlev"]
        tsfc = atmos["tsfc"]

    
        vmr = xr.concat(
            [atmos[gas] for gas in self.gas_names],
            dim=xr.IndexVariable("gas", self.gas_names),
        )

        layer_mass = compute_layer_mass(
            vmr=vmr,
            plev=plev,
            play=play,
            mol_weights=self.mol_weights,
        )

        tau = compute_tau(
            absorption_coeffs=self.absorption_coeffs,
            play=play,
            pref=self.pref,
            layer_mass=layer_mass,
        )

        return atmos.assign(
            tau=tau,
            lay_source=compute_planck_source(tlay, self.nus, self.dnus),
            lev_source=compute_planck_source(tlev, self.nus, self.dnus),
            sfc_source=compute_planck_source(tsfc, self.nus, self.dnus),
            nus=self.nus,
            dnus=self.dnus,
        )


    def _validate_inputs(self):
        if len(self.gas_names) == 0:
            raise ValueError("gas_names must not be empty")

        if len(set(self.gas_names)) != len(self.gas_names):
            raise ValueError("gas_names must be unique")

        for gas in self.gas_names:
            if gas not in MOL_WEIGHTS:
                raise ValueError(f"Unknown gas name: {gas}")

        if not np.isfinite(self.nu_min) or not np.isfinite(self.nu_max):
            raise ValueError("nu_min and nu_max must be finite")

        if self.nu_min >= self.nu_max:
            raise ValueError("nu_min must be less than nu_max")

        if self.nus.ndim != 1:
            raise ValueError("nus must be 1D")

        if self.nus.sizes["gpt"] < 2:
            raise ValueError("nus must contain at least two points")

        if not bool(np.isfinite(self.nus).all()):
            raise ValueError("nus must be finite")

        if not np.all(np.diff(self.nus.values) > 0):
            raise ValueError("nus must be strictly increasing")

        if not bool(((self.nus > self.nu_min) & (self.nus < self.nu_max)).all()):
            raise ValueError("all nus must satisfy nu_min < nus < nu_max")

        tri = self.triangle_params

        if tri.ndim != 2 or tri.sizes["triangle_param"] != 4:
            raise ValueError("triangle_params must have shape (ntriangles, 4)")

        if not bool(np.isfinite(tri).all()):
            raise ValueError("triangle_params must be finite")

        gas_idx = tri.sel(triangle_param="gas_index")
        kappa0 = tri.sel(triangle_param="kappa0")
        nu0 = tri.sel(triangle_param="nu0")
        ell = tri.sel(triangle_param="ell")

        if not bool((gas_idx == np.floor(gas_idx)).all()):
            raise ValueError("triangle gas indices must be integers")

        if not bool(((gas_idx >= 1) & (gas_idx <= len(self.gas_names))).all()):
            raise ValueError("triangle gas indices must lie between 1 and n_gases")

        if not bool((kappa0 >= 0).all()):
            raise ValueError("kappa0 must be >= 0")

        if not bool(((nu0 >= self.nu_min) & (nu0 <= self.nu_max)).all()):
            raise ValueError("triangle nu0 must satisfy nu_min <= nu0 <= nu_max")

        if not bool((ell > 0).all()):
            raise ValueError("triangle ell must be > 0")

        if not np.isfinite(self.pref) or self.pref < 0:
            raise ValueError("pref must be finite and >= 0")
